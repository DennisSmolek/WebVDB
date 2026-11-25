/**
 * K6.1c: the `k6-core` harness's raw `?scene=` mode. His K3 Step 1 ('volume' lighting) and Step 3 ('frame' lighting)
 * scenes rendered by the core frame renderer on raw WebGPU, no three.js: the host frame loop of design §2.4 with the
 * host pieces of demo 17 (`examples/src/demos/17-raw-webgpu/raw-frame.ts`). The result is the same linear frame the
 * `k3-frame` harness reads back (HalfFloat target, his background 0x598eff, NoToneMapping, sRGB on the CPU), so the two
 * compare directly (`e2e/k6-core.spec.ts`: raw vs `k3-frame ?backend=core`), and the PNGs go to
 * `scripts/k3/compare.mjs` against his captures.
 *
 * Scenes (`?scene=`):
 * - `smoke`: k3-frame's own CC0 scene (the committed `cloud_04_variant_0000.wvdb`, his sun, 'volume' lighting,
 *   extinction 40, camera at 0.75 box diagonals; 320 × 180, froxels 128 × 128 × 24). Runs on CI.
 * - his isolated parity scenes on his CC BY-SA bunny (`fixtures/k3/bunny_cloud.wvdb`, git-ignored): `step1-volume-dir-ms-off-isolated`,
 *   `step1-volume-dir-ms-on-isolated` ('volume'), `step3-default-dir-isolated`, `step3-shadow-cascades-4-dir-isolated`,
 *   `step3-shadow-grid-high-dir-isolated`, `step3-two-bunnies-dir-isolated` ('frame'); camera and volume matrices from his
 *   capture manifest when present (`fixtures/k3-reference/75e7345/manifest.json`), else his frame.js placement.
 *   1280 × 720, froxels 512 × 512 × 24. Not here: `step3-environment-sky-midday-env-isolated` (his UltraHDR map needs a
 *   gain-map decoder; three's `UltraHDRLoader` is the one the repo has) and the as-is scenes (his skybox and ground).
 *
 * Query: `&w=&h=`, `&froxels=WxHxD`, `&source=dense|sparse` (sparse: the same texels as a NanoVDB FLOAT grid through
 * `vdbSparseSourceGpu`, LOD 'coarse-pyramid'), `&lighting=volume|frame`, `&ms=0|1`.
 * Results in `window.__VDB_K6_CORE__.scene` (+ `width`, `height`, `rgba`: base64 sRGB RGBA8, top row first).
 */
import { VDBGridPool } from "@pmndrs/vdb";
import { vdbCreateFrameRenderer, vdbDenseSource, vdbSparseSourceGpu, type VDBFrameLight, type VDBFrameVolumeParams, type VDBVolumeSlot } from "../../../packages/vdb/src/core/internal.js";
import { readWvdb, vdbHalfToFloat, vdbSrgbEotf, vdbSrgbOetf, wvdbDenseToNvdb, wvdbDenseVolume } from "../../../packages/vdb/src/tools/internal.js";
import { boxMatrix, lookAt, multiply, normalize, perspective, rawCamera, RawTargets, readTextureRows, type Vec3 } from "../demos/17-raw-webgpu/raw-frame.js";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

export interface RawSceneResult {
  name: string;
  lighting: "volume" | "frame";
  source: "dense" | "sparse";
  /** 'manifest' (his captured camera and volume matrices) or 'frame.js' (his demo placement). */
  placement: "manifest" | "frame.js" | "smoke";
  volumes: number;
  froxels: number[];
  settings: Record<string, unknown>;
  /** Pixels farther than 6 (sum of |d|, sRGB bytes) from his background. */
  nonBackground: number;
  /** A second frame read back byte-identical. */
  inPageIdentical: boolean;
  /** Pipelines and shader modules created by the second update (must be 0). */
  secondBuilds: number;
  redraw: unknown;
}

interface SceneDef {
  lighting: "volume" | "frame";
  ms: boolean;
  shadowCascades?: number;
  shadowResolution?: number;
  twoBunnies?: boolean;
}

/** His isolated scenes this mode renders (scripts/k3/capture-reference.mjs names). */
export const RAW_SCENES: Record<string, SceneDef> = {
  "step1-volume-dir-ms-off-isolated": { lighting: "volume", ms: false },
  "step1-volume-dir-ms-on-isolated": { lighting: "volume", ms: true },
  "step3-default-dir-isolated": { lighting: "frame", ms: true },
  "step3-shadow-cascades-4-dir-isolated": { lighting: "frame", ms: true, shadowCascades: 4 },
  "step3-shadow-grid-high-dir-isolated": { lighting: "frame", ms: true, shadowResolution: 160 },
  "step3-two-bunnies-dir-isolated": { lighting: "frame", ms: true, twoBunnies: true },
};

interface ManifestVolume {
  matrixWorld: number[];
  visible?: boolean;
}
interface ManifestScenario {
  name: string;
  metadata?: { camera?: { matrixWorld: number[]; projectionMatrix: number[]; near: number; far: number }; volumes?: ManifestVolume[] };
}

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status} (git-ignored fixtures: copy them from the main checkout)`);
  return r.arrayBuffer();
}

async function manifestScenario(name: string): Promise<ManifestScenario | null> {
  const r = await fetch("/fixtures/k3-reference/75e7345/manifest.json");
  if (!r.ok) return null;
  const m = (await r.json()) as { scenarios: ManifestScenario[] };
  return m.scenarios.find((s) => s.name === name) ?? null;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** Counts pipelines and shader modules the device creates while `fn` runs. */
async function countBuilds(device: GPUDevice, fn: () => Promise<void>): Promise<number> {
  const d = device as unknown as Record<string, (...a: unknown[]) => unknown>;
  const names = ["createShaderModule", "createComputePipeline", "createComputePipelineAsync", "createRenderPipeline", "createRenderPipelineAsync"];
  const saved = names.map((n) => d[n]!);
  let count = 0;
  names.forEach((n, i) => {
    d[n] = (...a: unknown[]) => {
      count++;
      return saved[i]!.apply(device, a);
    };
  });
  try {
    await fn();
  } finally {
    names.forEach((n, i) => (d[n] = saved[i]!));
  }
  return count;
}

/**
 * Renders one raw scene and fills `out` (the harness result): `scene`, `width`, `height`, `rgba`.
 */
export async function runRawScene(device: GPUDevice, name: string, params: URLSearchParams, out: { scene?: RawSceneResult; width?: number; height?: number; rgba?: string }): Promise<void> {
  const smoke = name === "smoke";
  const def: SceneDef | undefined = smoke ? { lighting: "volume", ms: true } : RAW_SCENES[name];
  if (!def) throw new Error(`unknown ?scene=${name} (raw mode: smoke, ${Object.keys(RAW_SCENES).join(", ")})`);
  const W = Number(params.get("w") ?? (smoke ? 320 : 1280));
  const H = Number(params.get("h") ?? (smoke ? 180 : 720));
  const froxels = (params.get("froxels") ?? (smoke ? "128x128x24" : "512x512x24")).split("x").map(Number);
  const lighting = (params.get("lighting") ?? def.lighting) as "volume" | "frame";
  const ms = params.has("ms") ? params.get("ms") === "1" : def.ms;
  const sourceKind = (params.get("source") ?? "dense") as "dense" | "sparse";

  // The volume: his bunny or the CC0 cloud, as his dense source (r8unorm textures + box mip atlas) or the same texels
  // as a sparse NanoVDB grid (k3-frame's `?source=sparse`)
  const bytes = await fetchBytes(smoke ? cloudWvdbUrl : "/fixtures/k3/bunny_cloud.wvdb");
  const dense = wvdbDenseVolume(readWvdb(new Uint8Array(bytes)), { mipmaps: true });
  const [lo, hi] = (dense.bbox ?? [
    [-0.5, -0.5, -0.5],
    [0.5, 0.5, 0.5],
  ]) as [Vec3, Vec3];
  const box = boxMatrix(lo, hi);
  let slot: VDBVolumeSlot;
  let pool: VDBGridPool | null = null;
  let sparse: ReturnType<typeof vdbSparseSourceGpu> | null = null;
  if (sourceKind === "sparse") {
    const grid = wvdbDenseToNvdb(dense);
    pool = await VDBGridPool.create(device, { label: "k6.1c raw" });
    const id = pool.add(grid.image);
    pool.flush();
    await pool.settled();
    sparse = vdbSparseSourceGpu(pool, { lod: "coarse-pyramid", types: ["float"], label: "k6.1c raw sparse" });
    slot = sparse.volume({ density: id }, { indexBox: grid.textureIndexBox, stats: { meanNonZero: dense.meanDensity, max: dense.maxDensity, activeVoxels: dense.activeTexels } });
  } else {
    slot = vdbDenseSource(device, dense, { label: "k6.1c raw dense" });
  }

  // His camera and volume placement: the capture's matrices, else frame.js (the volume centred on the stage at
  // groundLevel + height / 2; camera (−100, −11, 80) at the origin, fov 60, near 0.01, far 10000)
  const capture = smoke ? null : await manifestScenario(name);
  const size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  let cameraWorld: number[];
  let projection: number[];
  let near: number;
  let far: number;
  const volumeMatrices: number[][] = [];
  let placement: RawSceneResult["placement"];
  if (smoke) {
    placement = "smoke";
    const diag = Math.hypot(...size);
    near = diag * 0.01;
    far = diag * 20;
    const eye = center.map((c, k) => c + normalize([0.9, 0.35, 1.1])[k]! * diag * 0.75);
    cameraWorld = lookAt(eye, center);
    projection = perspective(Math.PI / 3, W / H, near, far);
    volumeMatrices.push([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  } else if (capture?.metadata?.camera && capture.metadata.volumes?.length) {
    placement = "manifest";
    const c = capture.metadata.camera;
    cameraWorld = c.matrixWorld;
    projection = c.projectionMatrix;
    near = c.near;
    far = c.far;
    const shown = capture.metadata.volumes.filter((v) => v.visible !== false);
    volumeMatrices.push(shown[0]!.matrixWorld);
    if (def.twoBunnies && capture.metadata.volumes[1]) volumeMatrices.push(capture.metadata.volumes[1].matrixWorld);
  } else {
    placement = "frame.js";
    near = 0.01;
    far = 10000;
    cameraWorld = lookAt([-100, -11, 80], [0, 0, 0]);
    projection = perspective(Math.PI / 3, W / H, near, far);
    const first = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -center[0]!, -center[1]! - 50 + size[1]! / 2, -center[2]!, 1];
    volumeMatrices.push(first);
    if (def.twoBunnies) volumeMatrices.push([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, first[12]! + size[0]! * 0.6, first[13]! + size[1]! * 0.1, first[14]! - size[2]! * 0.2, 1]);
  }
  const camera = rawCamera(cameraWorld, projection, near, far);

  // The renderer: his settings for the scene (k3-frame's), the core composite into our HalfFloat target
  const frame = vdbCreateFrameRenderer(device, { composite: { colorFormat: RawTargets.COLOR_FORMAT, depthFormat: RawTargets.DEPTH_FORMAT }, label: "k6.1c raw" });
  Object.assign(frame.settings, {
    width: froxels[0],
    height: froxels[1],
    depth: froxels[2],
    subSteps: 32,
    jitter: false,
    temporal: false,
    lighting,
    multipleScattering: ms,
    shadowCascades: def.shadowCascades ?? 1,
    shadowResolution: def.shadowResolution ?? 64,
    shadowDistance: 512,
    phase: "dual-lobe",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    phaseGExponent: 0,
    environmentLighting: "directional",
  });
  // His frame.js volume: dual-lobe 0.8 / −0.3 / 0.2, extinction 100 (the smoke 40), normalizeDensity, mip maps
  const params0: Partial<VDBFrameVolumeParams> = {
    baseColor: [1, 1, 1],
    extinction: smoke ? 40 : 100,
    densityScale: 1,
    normalizeDensity: true,
    absorbance: 1,
    opacity: 1,
    angstromExponent: 0,
    phase: "dual-lobe",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    phaseGExponent: 0,
    shadowSteps: 32,
    mipmaps: true,
  };
  volumeMatrices.forEach((m, i) => frame.addVolume(slot, i === 0 ? { ...params0, matrixWorld: multiply(m, box) } : { extinction: 100, phase: "dual-lobe", shadowSteps: 32, mipmaps: true, matrixWorld: multiply(m, box) }));
  // His sun: DirectionalLight(0xffffff, 1) at (100, 200, 100) toward the origin (frame.js L191–194)
  const lights: VDBFrameLight[] = [{ id: 1, kind: "directional", direction: normalize([-100, -200, -100]), color: [1, 1, 1] }];

  // His background: scene.background 0x598eff (sRGB) as linear in the HalfFloat target
  const background = [0x59, 0x8e, 0xff].map((b) => vdbSrgbEotf(b / 255));
  const targets = new RawTargets(device, "k6.1c raw");
  targets.resize(W, H);
  let frameId = 0;
  const render = async (): Promise<Uint8Array> => {
    const encoder = device.createCommandEncoder({ label: "k6.1c raw frame" });
    pool?.update(encoder);
    sparse?.update(encoder);
    frame.update(encoder, {
      frameId: frameId++,
      camera: { view: camera.view, projection: camera.projection, near, far, id: "k6.1c" },
      viewport: { drawingBuffer: [W, H], canvas: [W, H] },
      lights,
    });
    // The opaque scene: only his background (an empty depth buffer)
    const pass = encoder.beginRenderPass({
      label: "k6.1c background",
      colorAttachments: [{ view: targets.hdr.createView(), loadOp: "clear", clearValue: [background[0]!, background[1]!, background[2]!, 1], storeOp: "store" }],
      depthStencilAttachment: { view: targets.depth.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "store" },
    });
    pass.end();
    targets.copySceneColor(encoder);
    // NoToneMapping: his clamp to [0, 1] in the sRGB blend
    frame.render(encoder, targets.composite(true));
    device.queue.submit([encoder.finish()]);
    const halves = new Uint16Array((await readTextureRows(device, targets.hdr, 8)).buffer);
    const rgba = new Uint8Array(W * H * 4);
    for (let i = 0; i < W * H; i++) {
      for (let k = 0; k < 3; k++) rgba[i * 4 + k] = toByte(vdbSrgbOetf(vdbHalfToFloat(halves[i * 4 + k]!)));
      rgba[i * 4 + 3] = 255;
    }
    return rgba;
  };
  const first = await render();
  let second = first;
  const secondBuilds = await countBuilds(device, async () => {
    second = await render();
  });
  const bg = [0x59, 0x8e, 0xff];
  let nonBackground = 0;
  for (let i = 0; i < first.length; i += 4) if (Math.abs(first[i]! - bg[0]!) + Math.abs(first[i + 1]! - bg[1]!) + Math.abs(first[i + 2]! - bg[2]!) > 6) nonBackground++;
  out.width = W;
  out.height = H;
  out.rgba = toBase64(first);
  out.scene = {
    name,
    lighting,
    source: sourceKind,
    placement,
    volumes: volumeMatrices.length,
    froxels,
    settings: { ...frame.settings },
    nonBackground,
    inPageIdentical: first.every((v, i) => v === second[i]),
    secondBuilds,
    redraw: JSON.parse(JSON.stringify(frame.redrawStats)),
  };
  frame.dispose();
  targets.destroy();
  sparse?.dispose();
  pool?.dispose();
}
