/**
 * Demo 13's sculptures: OpenVDB level sets parsed in the page (`parseVdb`), built to NanoVDB at full resolution
 * (`buildFromVdb`), one `VDBGridPool` each, traced by `VDBLevelSetMaterial` on a proxy box. Public API only.
 *
 * One pool per sculpture keeps every pool under the device's storage-binding limit (the four images are 6–81 MB);
 * a material reads one `VDBGridNode`, so pools never share a shader.
 */
import { fixtureUrl } from "../assets.js";
import * as THREE from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { buildFromVdb, parseVdb, type VdbFloatGrid } from "@pmndrs/vdb/tools";
import { VDBGridNode, VDBLevelSetMaterial, getDevice, vdbProxyGeometry } from "@pmndrs/vdb/three";
import { fetchWithProgress, probe } from "../showcase-shell.js";

export interface SculptureDef {
  id: string;
  label: string;
  /** Candidate URLs, first that exists wins. */
  urls: string[];
  /** Approximate download, for the status line. */
  mb: number;
  /** Longest side in metres on the plinth. */
  size: number;
  /** Turn about +y (radians) so the piece faces the visitor. */
  yaw: number;
}

export const SCULPTURES: readonly SculptureDef[] = [
  { id: "icosahedron", label: "Icosahedron", urls: [fixtureUrl("vdb-samples/icosahedron.vdb")], mb: 1, size: 0.85, yaw: 0.35 },
  { id: "teapot", label: "Utah teapot", urls: [fixtureUrl("vdb-samples/utahteapot.vdb")], mb: 19, size: 1.15, yaw: -0.5 },
  { id: "venus", label: "Venus statue", urls: [fixtureUrl("k3/venusstatue.vdb"), fixtureUrl("vdb-samples/venusstatue.vdb")], mb: 27, size: 1.55, yaw: 0 },
  { id: "bunny", label: "Stanford bunny", urls: [fixtureUrl("vdb-samples/bunny.vdb")], mb: 16, size: 1.05, yaw: 0.4 },
];

/** First existing URL of a sculpture, or null. */
export async function resolveSculpture(def: SculptureDef): Promise<string | null> {
  for (const u of def.urls) if (await probe(u)) return u;
  return null;
}

export interface SculptureStats {
  activeVoxels: number;
  /** Index box size (voxels). */
  dims: [number, number, number];
  /** NanoVDB image bytes in the pool (the full-resolution sparse grid). */
  sparseBytes: number;
  /** A dense f32 grid over the same box. */
  denseF32Bytes: number;
  /** His 200³ 8-bit bake (`resolution: 200`): bytes and how much coarser its voxels are than the source's. */
  bake200Bytes: number;
  bake200Coarser: number;
}

export class Sculpture {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, VDBLevelSetMaterial>;
  /** World box of the placed surface (the proxy's). */
  readonly box: THREE.Box3;

  private constructor(
    readonly def: SculptureDef,
    readonly pool: VDBGridPool,
    readonly grids: VDBGridNode,
    readonly gridId: number,
    readonly stats: SculptureStats,
    readonly material: VDBLevelSetMaterial,
  ) {
    const info = pool.info(gridId);
    this.mesh = new THREE.Mesh(vdbProxyGeometry(info, { pad: 1 }), material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    // The proxy is in world space (the pool record holds the placement): culling by its own bounds is right.
    this.mesh.geometry.computeBoundingBox();
    this.box = this.mesh.geometry.boundingBox!.clone();
    this.mesh.name = def.id;
  }

  /**
   * Downloads, parses and builds a sculpture, and stands it on `base` (the plinth top centre). `progress(text, f)`
   * reports the download fraction, then the parse and build steps (fraction null).
   */
  static async load(
    renderer: THREE.WebGPURenderer,
    def: SculptureDef,
    url: string,
    base: THREE.Vector3,
    progress: (text: string, fraction: number | null) => void,
  ): Promise<Sculpture> {
    const bytes = await fetchWithProgress(url, (got, total) =>
      progress(`${def.label}: downloading ${(got / 1048576).toFixed(1)}${total ? ` / ${(total / 1048576).toFixed(1)}` : ""} MB`, total ? got / total : null),
    );
    progress(`${def.label}: parsing the .vdb (our parser)…`, null);
    await nextFrame();
    const file = parseVdb(bytes);
    const grid = file.grids.find((g): g is VdbFloatGrid => g.valueType === "float");
    if (!grid) throw new Error(`${url} has no float level set grid`);
    progress(`${def.label}: building the NanoVDB tree…`, null);
    await nextFrame();
    const image = buildFromVdb(grid);

    const pool = await VDBGridPool.create(getDevice(renderer), { label: `demo-13 ${def.id}` });
    const id = pool.add(image);
    pool.flush();
    await pool.settled();
    // Place it: scale the world box to `size`, turn by `yaw`, bottom centre on `base` (the pool record's transform).
    const raw = pool.info(id);
    const min = new THREE.Vector3(...raw.worldBBox.min);
    const max = new THREE.Vector3(...raw.worldBBox.max);
    const size = max.clone().sub(min);
    const center = min.clone().add(max).multiplyScalar(0.5);
    const s = def.size / Math.max(size.x, size.y, size.z);
    const m = new THREE.Matrix4()
      .makeTranslation(base.x, base.y + (size.y * s) / 2, base.z)
      .multiply(new THREE.Matrix4().makeRotationY(def.yaw))
      .multiply(new THREE.Matrix4().makeScale(s, s, s))
      .multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
    pool.setTransform(id, m.elements);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["levelSet"] });
    const material = new VDBLevelSetMaterial(renderer, grids, id, { color: 0xd8d2c4, roughness: 0.8, metalness: 0 });

    const ib = pool.info(id).indexBBox;
    const dims: [number, number, number] = [ib.max[0] - ib.min[0] + 1, ib.max[1] - ib.min[1] + 1, ib.max[2] - ib.min[2] + 1];
    const longest = Math.max(...dims);
    const stats: SculptureStats = {
      activeVoxels: Number(grid.activeVoxelCount),
      dims,
      sparseBytes: pool.info(id).imageBytes,
      denseF32Bytes: dims[0] * dims[1] * dims[2] * 4,
      bake200Bytes: 200 ** 3,
      bake200Coarser: longest / 200,
    };
    return new Sculpture(def, pool, grids, id, stats, material);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.grids.dispose();
    this.pool.dispose();
  }
}

const nextFrame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
