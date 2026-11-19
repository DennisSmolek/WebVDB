/**
 * Demo 10 scenes: what each one loads and how it becomes a `VDBFrameVolume` (or a level set / point
 * sprite), written against the public API only (`@pmndrs/vdb`, `@pmndrs/vdb/tools`, `@pmndrs/vdb/three`).
 *
 * Every object is placed as his examples page places it: the longest side 100 units, the box standing
 * on the ground plane at y = -50, centred on the origin.
 */
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import {
  NvdbFile,
  buildFromVdb,
  parseVdb,
  readWvdb,
  vdbPointRenderData,
  vdbToWvdb,
  wvdbDenseToNvdb,
  wvdbDenseVolume,
  type VdbPointDataGrid,
  type WvdbDenseVolume,
} from "@pmndrs/vdb/tools";
import {
  VDBFrameEmissionLight,
  VDBFrameVolume,
  VDBGridNode,
  VDBLevelSetMaterial,
  VDBPoints,
  VDBSequence,
  getDevice,
  vdbDenseTextures,
  vdbFrameBlackbodyEmission,
  vdbProxyGeometry,
  vdbSparseSource,
  type VDBFrameRenderer,
  type VDBGridNodeOptions,
  type VDBFrameVolumeProps,
  type VDBSparseLod,
  type VDBThreeDenseSource,
  type VDBThreeVolumeSource,
} from "@pmndrs/vdb/three";

export type SourceKind = "dense" | "sparse";
export type SceneKind = "froxel" | "levelset" | "points";

export const DISPLAY_SIZE = 100;
export const GROUND_LEVEL = -50;

export interface SceneContext {
  renderer: THREE.WebGPURenderer;
  frameRenderer: VDBFrameRenderer;
  status(text: string): void;
  /** Explosion sequence: start paused (deterministic test runs). */
  paused: boolean;
  /** Explosion sequence: the frame to show first. */
  startFrame: number;
  /** Emission light probes (quality preset). */
  emissionResolution: number;
}

export interface SceneDef {
  id: string;
  label: string;
  kind: SceneKind;
  /** Probed with HEAD; the scene is offered only when all exist. */
  files: string[];
  /** How to get the files when they are missing. */
  fetchHint: string;
  /** Source kinds the scene can build (froxel scenes). */
  sourceKinds: SourceKind[];
  /** Has an emission channel and the emission light. */
  emission?: boolean;
  load(ctx: SceneContext, kind: SourceKind | null, lod: VDBSparseLod): Promise<LoadedScene>;
}

export interface LoadedScene {
  readonly group: THREE.Group;
  /** The froxel volume (froxel scenes). */
  readonly volume: VDBFrameVolume | null;
  readonly sourceKind: SourceKind | null;
  /** Rebuild the source (another kind or LOD) and the volume in place. */
  rebuild?(kind: SourceKind, lod: VDBSparseLod): Promise<void>;
  /** Per frame: sequence playback. */
  tick?(now: number): void;
  /** Sequence frames and the current one. */
  sequence?: { count: number; index(): number; seek(i: number): Promise<void>; playing: boolean };
  /** One line for the stats: grid size and where it lives. */
  info(): string;
  dispose(): void;
}

const MiB = (bytes: number) => `${(bytes / 1048576).toFixed(bytes < 10 * 1048576 ? 2 : 1)} MiB`;

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

/** Place an object so a world box stands on the ground, longest side DISPLAY_SIZE (his frame.js placement). */
export function placeOnGround(object: THREE.Object3D, box: THREE.Box3, zUp = false): void {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const s = DISPLAY_SIZE / Math.max(size.x, size.y, size.z);
  const height = zUp ? size.z : size.y;
  object.scale.setScalar(s);
  object.rotation.set(zUp ? -Math.PI / 2 : 0, 0, 0);
  // Rx(-90°) takes (x, y, z) to (x, z, -y).
  const c = zUp ? new THREE.Vector3(center.x, center.z, -center.y) : center;
  object.position.copy(c).multiplyScalar(-s);
  object.position.y += GROUND_LEVEL + (height * s) / 2;
  object.updateMatrix();
}

const boxOf = (b: readonly (readonly number[])[]): THREE.Box3 =>
  new THREE.Box3(new THREE.Vector3(b[0]![0], b[0]![1], b[0]![2]), new THREE.Vector3(b[1]![0], b[1]![1], b[1]![2]));

/** One pool and one `VDBGridNode` (march profile, the sparse source's helpers) for a scene. */
class SparsePool {
  private constructor(
    readonly pool: VDBGridPool,
    readonly grids: VDBGridNode,
    readonly id: number,
    readonly image: Uint32Array,
  ) {}

  static async create(renderer: THREE.WebGPURenderer, image: Uint32Array, options: VDBGridNodeOptions): Promise<SparsePool> {
    const pool = await VDBGridPool.create(getDevice(renderer), { label: "demo-10" });
    const id = pool.add(image);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, options);
    return new SparsePool(pool, grids, id, image);
  }

  info(): string {
    const i = this.pool.info(this.id);
    const box = i.indexBBox;
    const dims = box.max.map((v, k) => v - box.min[k]! + 1).join("×");
    return `sparse ${i.type} ${dims}, ${MiB(i.imageBytes)} in pool (pool ${MiB(this.pool.byteLength)})`;
  }

  dispose(): void {
    this.grids.dispose();
    this.pool.dispose();
  }
}

/** The sparse source's helpers: HDDA bound, cell end, the node-average LOD and the fast trilinear path. */
const SPARSE_NODE = { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"] } as const;

interface MadeSource {
  source: VDBThreeVolumeSource;
  info: string;
}

/** A froxel asset: builds a source of a kind on demand (lazy: a dense bake runs only when picked). */
interface VolumeAsset {
  make(kind: SourceKind, lod: VDBSparseLod): Promise<MadeSource>;
  zUp?: boolean;
  props?: Partial<VDBFrameVolumeProps>;
  dispose(): void;
}

const denseInfo = (dense: WvdbDenseVolume, note = ""): string => {
  const bytes = dense.dimensions[0] * dense.dimensions[1] * dense.dimensions[2] + (dense.density.mip?.data.byteLength ?? 0);
  return `dense r8 ${dense.dimensions.join("×")} texture + mips, ${MiB(bytes)}${note}`;
};

/** A froxel scene: one volume over a source, rebuilt when the source kind or LOD changes. */
class FroxelScene implements LoadedScene {
  readonly group = new THREE.Group();
  volume: VDBFrameVolume | null = null;
  sourceKind: SourceKind | null = null;
  private made: MadeSource | null = null;

  constructor(
    private readonly ctx: SceneContext,
    private readonly asset: VolumeAsset,
  ) {}

  async rebuild(kind: SourceKind, lod: VDBSparseLod): Promise<void> {
    this.ctx.status(`building the ${kind} source…`);
    const made = await this.asset.make(kind, lod);
    const old = this.volume;
    const oldSource = this.made?.source;
    const volume = new VDBFrameVolume(made.source, {
      frameRenderer: this.ctx.frameRenderer,
      extinction: 100,
      angstromExponent: 0,
      normalizeDensity: true,
      ...this.asset.props,
    });
    placeOnGround(volume, volume.bbox, this.asset.zUp);
    if (old) {
      this.group.remove(old);
      old.dispose();
    }
    if (oldSource && oldSource !== made.source) oldSource.dispose();
    this.group.add(volume);
    this.volume = volume;
    this.made = made;
    this.sourceKind = kind;
    this.ctx.status("");
  }

  info(): string {
    return this.made?.info ?? "";
  }

  dispose(): void {
    this.volume?.dispose();
    this.made?.source.dispose();
    this.asset.dispose();
  }
}

// --- Assets --------------------------------------------------------------------------------------------

/** A NanoVDB / OpenVDB fog grid: sparse from the pool, dense from a 128³ `.wvdb` bake made in the page. */
function gridAsset(ctx: SceneContext, image: Uint32Array, bakeDense: (() => WvdbDenseVolume) | null): VolumeAsset {
  let sparse: SparsePool | null = null;
  let dense: WvdbDenseVolume | null = null;
  return {
    async make(kind, lod) {
      if (kind === "dense") {
        if (!bakeDense) throw new Error("this scene has no dense source");
        if (!dense) {
          ctx.status("baking a 128³ .wvdb in the page (vdbToWvdb)…");
          await new Promise((r) => setTimeout(r, 0));
          dense = bakeDense();
        }
        return { source: vdbDenseTextures(dense), info: denseInfo(dense, " (baked in the page)") };
      }
      sparse ??= await SparsePool.create(ctx.renderer, image, SPARSE_NODE);
      const source = vdbSparseSource(sparse.grids, { density: sparse.id }, { lod, images: { density: image } });
      return { source, info: sparse.info() };
    },
    dispose() {
      sparse?.dispose();
    },
  };
}

/** A dense `.wvdb` level: dense textures, or the same texels as a NanoVDB FLOAT grid in the pool (voxel-exact). */
function wvdbAsset(ctx: SceneContext, dense: WvdbDenseVolume, props?: Partial<VDBFrameVolumeProps>, zUp = false): VolumeAsset {
  let sparse: SparsePool | null = null;
  let grid: ReturnType<typeof wvdbDenseToNvdb> | null = null;
  return {
    zUp,
    ...(props ? { props } : {}),
    async make(kind, lod) {
      if (kind === "dense") return { source: vdbDenseTextures(dense), info: denseInfo(dense) };
      grid ??= wvdbDenseToNvdb(dense);
      sparse ??= await SparsePool.create(ctx.renderer, grid.image, SPARSE_NODE);
      // The texture's index box, the dense mean and the .wvdb box: both kinds line up and differ only in the sampler.
      const source = vdbSparseSource(sparse.grids, { density: sparse.id }, { ...grid.sparseSourceOptions, lod, images: { density: grid.image } });
      return { source, info: sparse.info() };
    },
    dispose() {
      sparse?.dispose();
    },
  };
}

async function froxelScene(ctx: SceneContext, asset: VolumeAsset, kind: SourceKind, lod: VDBSparseLod): Promise<FroxelScene> {
  const scene = new FroxelScene(ctx, asset);
  await scene.rebuild(kind, lod);
  return scene;
}

// --- The explosion sequence ------------------------------------------------------------------------------

const EXPLOSION_DIR = fixtureUrl("k3/ground_explosion/");
const EXPLOSION_FPS = 24;
/**
 * His frame.js explosion: the emission grid "flames" ÷ emissiveScale 6. Early frames have no flames grid: the
 * dense source's replace reads their emission as 0.
 */
const explosionDense = (bytes: ArrayBuffer): WvdbDenseVolume => {
  const file = readWvdb(new Uint8Array(bytes));
  const flames = file.grids.some((g) => g.name === "flames");
  return wvdbDenseVolume(file, { ...(flames ? { emission: "flames" } : {}), scales: { flames: 6.0 } });
};

class ExplosionScene implements LoadedScene {
  readonly group = new THREE.Group();
  readonly sourceKind = "dense" as const;
  readonly volume: VDBFrameVolume;
  readonly sequence: NonNullable<LoadedScene["sequence"]>;
  private readonly light: VDBFrameEmissionLight;
  /** Frame playback into the dense source (`VDBSequence` with a `{ source }` target, K5.9). */
  private readonly player: VDBSequence<WvdbDenseVolume>;
  private readonly bytes = new Map<number, Promise<ArrayBuffer>>();
  private readonly start: number;

  private constructor(
    private readonly frames: string[],
    private readonly source: VDBThreeDenseSource,
    first: WvdbDenseVolume,
    start: number,
    paused: boolean,
    emissionResolution: number,
    frameRenderer: VDBFrameRenderer,
  ) {
    this.start = start;
    // His frame.js explosion: base colour #555555, 0–3000 K, emissive intensity 15, 'planck' blackbody colour.
    this.volume = new VDBFrameVolume(source, {
      frameRenderer,
      baseColor: "#555555",
      extinction: 100,
      angstromExponent: 0,
      normalizeDensity: true,
      // One mean for every frame (his sequences divide by the mean of all frames; ours: the first frame shown), so
      // the extinction scale does not pump from frame to frame.
      densityMean: first.meanDensity,
      temperatureRange: [0, 3000],
      emissiveIntensity: 15,
      emissionColor: vdbFrameBlackbodyEmission("planck", 1.39),
    });
    placeOnGround(this.volume, this.volume.bbox, true);
    this.group.add(this.volume);
    // His emission light: the volume's flames light the scene's meshes (intensity 25, probe grid around the emitters).
    this.light = new VDBFrameEmissionLight({ intensity: 25, resolution: emissionResolution, cascades: 1, range: 3, distance: 512, steps: 24 });
    this.group.add(this.light);
    // A frame without the flames channel reads emission 0 (the dense source's replace, K5.9).
    this.player = new VDBSequence<WvdbDenseVolume>({
      urls: frames.map((f) => EXPLOSION_DIR + f),
      fps: EXPLOSION_FPS,
      prefetch: 6,
      loader: async (_url, index) => explosionDense(await this.fetchFrame(index)),
      onError: (err, index) => console.warn(`demo 10: explosion frame ${index}: ${String(err)}`),
    });
    this.player.start({ source });
    this.player.seek(start);
    if (paused) this.player.stop();
    const player = this.player;
    this.sequence = {
      count: frames.length,
      index: () => (player.stats.frame >= 0 ? player.stats.frame : this.start),
      seek: async (i: number) => player.seek(i),
      get playing() {
        return player.isPlaying;
      },
      set playing(v: boolean) {
        if (v) player.resume();
        else player.stop();
      },
    };
  }

  static async load(ctx: SceneContext): Promise<ExplosionScene> {
    const r = await fetch(`${EXPLOSION_DIR}frames.json`);
    if (!r.ok) throw new Error(`fetch ${EXPLOSION_DIR}frames.json: ${r.status}`);
    const frames = ((await r.json()) as { frames: string[] }).frames;
    const start = Math.min(Math.max(0, ctx.startFrame), frames.length - 1);
    ctx.status(`loading explosion frame ${start}…`);
    const first = explosionDense(await fetchBytes(EXPLOSION_DIR + frames[start]));
    const scene = new ExplosionScene(frames, vdbDenseTextures(first), first, start, ctx.paused, ctx.emissionResolution, ctx.frameRenderer);
    ctx.status("");
    return scene;
  }

  private fetchFrame(i: number): Promise<ArrayBuffer> {
    let p = this.bytes.get(i);
    if (!p) {
      p = fetchBytes(EXPLOSION_DIR + this.frames[i]);
      p.catch(() => this.bytes.delete(i));
      this.bytes.set(i, p);
    }
    return p;
  }

  tick(now: number): void {
    this.player.update(now);
  }

  info(): string {
    return `${denseInfo(this.source.volume)} · frame ${this.sequence.index()} / ${this.frames.length}`;
  }

  dispose(): void {
    this.player.dispose();
    this.volume.dispose();
    this.source.dispose();
    this.light.dispose();
  }
}

// --- Level set and points ------------------------------------------------------------------------------

class MeshScene implements LoadedScene {
  readonly group = new THREE.Group();
  readonly volume = null;
  readonly sourceKind = null;
  constructor(
    private readonly infoText: string,
    private readonly cleanup: () => void,
  ) {}
  info(): string {
    return this.infoText;
  }
  dispose(): void {
    this.cleanup();
  }
}

async function levelSetScene(ctx: SceneContext, url: string): Promise<MeshScene> {
  const file = parseVdb(await fetchBytes(url));
  const grid = file.grids.find((g) => g.valueType === "float");
  if (!grid) throw new Error(`${url} has no float grid`);
  ctx.status("building the NanoVDB grid…");
  const image = buildFromVdb(grid);
  const sparse = await SparsePool.create(ctx.renderer, image, { profile: "march", helpers: ["levelSet"] });
  const info = sparse.pool.info(sparse.id);
  // Place the grid: its world box on the ground, through the pool transform (the proxy box follows it).
  const placed = new THREE.Object3D();
  placeOnGround(placed, boxOf([info.worldBBox.min, info.worldBBox.max]));
  sparse.pool.setTransform(sparse.id, placed.matrix.elements);
  sparse.pool.flush();
  await sparse.pool.settled();
  const material = new VDBLevelSetMaterial(ctx.renderer, sparse.grids, sparse.id, { color: 0xd9d4c7, roughness: 0.45, metalness: 0.0 });
  const mesh = new THREE.Mesh(vdbProxyGeometry(sparse.pool.info(sparse.id), { pad: 1 }), material);
  const scene = new MeshScene(`level set ${sparse.info()}`, () => {
    mesh.geometry.dispose();
    material.dispose();
    sparse.dispose();
  });
  scene.group.add(mesh);
  ctx.status("");
  return scene;
}

async function pointsScene(ctx: SceneContext, url: string): Promise<MeshScene> {
  const file = parseVdb(await fetchBytes(url));
  const grid = file.grids.find((g) => g.valueType === "ptdataidx32") as VdbPointDataGrid | undefined;
  if (!grid) throw new Error(`${url} has no point data grid`);
  const data = vdbPointRenderData(grid);
  const points = new VDBPoints(data, { color: 0xfff4e0 });
  const box = new THREE.Box3().setFromArray(data.positions.subarray(0, data.count * 3));
  placeOnGround(points, box);
  const scene = new MeshScene(`${data.count.toLocaleString()} of ${data.totalCount.toLocaleString()} points as one instanced sprite`, () => points.dispose());
  scene.group.add(points);
  ctx.status("");
  return scene;
}

// --- The catalogue -----------------------------------------------------------------------------------------

const VDB_SAMPLES_HINT = "pnpm fixtures:vdb-samples";
const HIS_CLONE_HINT = "copy it from a mjurczyk/openvdb clone (not fetched by a script)";

export const SCENES: readonly SceneDef[] = [
  {
    id: "wdas",
    label: "WDAS cloud (quarter, Fp8)",
    kind: "froxel",
    files: [fixtureUrl("wdas/wdas_cloud_quarter_fp8.nvdb")],
    fetchHint: "pnpm fixtures:wdas, then nanovdb_convert --fp8 (fixtures/README.md)",
    sourceKinds: ["sparse"],
    async load(ctx, _kind, lod) {
      ctx.status("loading wdas_cloud_quarter_fp8.nvdb (28 MB)…");
      const image = NvdbFile.fromArrayBuffer(await fetchBytes(this.files[0]!)).gridImage(0);
      return froxelScene(ctx, gridAsset(ctx, image, null), "sparse", lod);
    },
  },
  {
    id: "smoke",
    label: "smoke.vdb",
    kind: "froxel",
    files: [fixtureUrl("vdb-samples/smoke.vdb")],
    fetchHint: VDB_SAMPLES_HINT,
    sourceKinds: ["sparse", "dense"],
    async load(ctx, kind, lod) {
      const file = parseVdb(await fetchBytes(this.files[0]!));
      const grid = file.grids.find((g) => g.valueType === "float");
      if (!grid) throw new Error("smoke.vdb has no float grid");
      const image = buildFromVdb(grid);
      const bake = () => wvdbDenseVolume(readWvdb(vdbToWvdb(file, { resolution: 128 }).bytes));
      return froxelScene(ctx, gridAsset(ctx, image, bake), kind ?? "sparse", lod);
    },
  },
  {
    id: "bunny",
    label: "bunny_cloud.wvdb",
    kind: "froxel",
    files: [fixtureUrl("k3/bunny_cloud.wvdb")],
    fetchHint: `fixtures/k3/bunny_cloud.wvdb: ${HIS_CLONE_HINT}`,
    sourceKinds: ["dense", "sparse"],
    async load(ctx, kind, lod) {
      const dense = wvdbDenseVolume(readWvdb(new Uint8Array(await fetchBytes(this.files[0]!))));
      return froxelScene(ctx, wvdbAsset(ctx, dense), kind ?? "dense", lod);
    },
  },
  {
    id: "explosion",
    label: "EmberGen explosion (emission)",
    kind: "froxel",
    files: [`${EXPLOSION_DIR}frames.json`, `${EXPLOSION_DIR}ground_explosion_0030.wvdb`],
    fetchHint: `fixtures/k3/ground_explosion/ (EmberGen, CC0): ${HIS_CLONE_HINT}`,
    sourceKinds: ["dense"],
    emission: true,
    load(ctx) {
      return ExplosionScene.load(ctx);
    },
  },
  {
    id: "teapot",
    label: "utahteapot.vdb (level set)",
    kind: "levelset",
    files: [fixtureUrl("vdb-samples/utahteapot.vdb")],
    fetchHint: VDB_SAMPLES_HINT,
    sourceKinds: [],
    load(ctx) {
      ctx.status("parsing utahteapot.vdb (19 MB)…");
      return levelSetScene(ctx, this.files[0]!);
    },
  },
  {
    id: "points",
    label: "sphere_points.vdb (points)",
    kind: "points",
    files: [fixtureUrl("vdb-samples/sphere_points.vdb")],
    fetchHint: `fixtures/vdb-samples/sphere_points.vdb: ${HIS_CLONE_HINT}`,
    sourceKinds: [],
    load(ctx) {
      return pointsScene(ctx, this.files[0]!);
    },
  },
];

/** HEAD-probe a scene's files. */
export async function probeScene(def: SceneDef): Promise<boolean> {
  for (const f of def.files) {
    try {
      const r = await fetch(f, { method: "HEAD" });
      if (!r.ok) return false;
    } catch {
      return false;
    }
  }
  return true;
}
