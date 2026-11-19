/**
 * K4.16 / K4.18 browser gate: coarse-first `.nvdb` delivery
 * (docs/dev/design/K4-5-coarse-first.md, option a).
 *
 * 1. Bakes `smoke.vdb` in the page with `vdbBakeCoarseFirst` (`[density.lod48,
 *    density]`, box, FpN, codec ZIP), and the same two images with codec NONE.
 * 2. Streams each through a throttled, pull-driven `ReadableStream` (fixed-size
 *    chunks, one at a time) into `vdbStreamCoarseFirst` on a real
 *    `VDBGridPool`, and counts the transferred bytes when the preview is added:
 *    - `zipRaw`: codec ZIP, no HTTP compression (1 KiB chunks);
 *    - `noneGzip` / `zipGzip`: the file gzipped in the page
 *      (`CompressionStream('gzip')`), streamed in 256-byte chunks through a
 *      `DecompressionStream`, counting compressed bytes (an upper bound: the
 *      pipe buffers up to a couple of chunks ahead).
 * 3. On the `zipRaw` run the stream is held at the preview: the preview is
 *    rendered (`VDBVolumeMaterial`, density scaled by the voxel ratio), then
 *    the stream resumes, the full grid replaces the preview in place, and the
 *    final frame is rendered. A fresh pool with the full grid alone (read from
 *    a single-grid file by `NvdbFile`) renders the reference frame with the
 *    same camera and uniforms. Gate: final == reference, byte for byte.
 *
 * Results go to `window.__VDB_COARSE_FIRST__` for e2e/coarse-first.spec.ts.
 * Query: `?w=&h=` (default 256x256), `?file=` (default /fixtures/vdb-samples/smoke.vdb).
 */
import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, VDBVolumeMaterial, vdbProxyGeometry, vdbStreamCoarseFirst, type VDBCoarseFirstEvent } from "@pmndrs/vdb/three";
import { vdbBakeCoarseFirst, NvdbFile, parseVdb, writeNvdb, type NvdbGridMetadata } from "@pmndrs/vdb/tools";
import { createDeviceRenderer } from "../demos/device-renderer";

interface StreamRun {
  /** Transferred bytes (compressed when gzipped) delivered when the preview was added to the pool. */
  firstPaintTransfer: number;
  /** The reader's exact `.nvdb` byte offset at the end of the preview. */
  firstPaintFile: number;
  /** Total transferred bytes of the whole file. */
  transferTotal: number;
  chunk: number;
  grids: string[];
  id: number;
  actions: string[];
}

interface Frame {
  rgba: string;
  nonBackground: number;
}

interface Result {
  done: boolean;
  error?: string;
  missing?: string;
  adapter?: string;
  width?: number;
  height?: number;
  sizes?: { zipFile: number; noneFile: number; zipGzip: number; noneGzip: number };
  grids?: { name: string; gridType: string; voxelCount: number; gridByteSize: number; voxelSize: number[] }[];
  runs?: { zipRaw: StreamRun; noneGzip: StreamRun; zipGzip: StreamRun };
  placement?: {
    sameId: boolean;
    inPlace: boolean;
    finalWorldBBox: number[][];
    fullWorldBBox: number[][];
    previewCellMin: number[];
    fullCellMin: number[];
  };
  preview?: Frame & { densityScale: number };
  final?: Frame;
  reference?: Frame;
  finalEqualsReference?: boolean;
  previewDiffersFromFinal?: boolean;
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_COARSE_FIRST__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const W = Number(params.get("w") ?? 256);
const H = Number(params.get("h") ?? 256);
const FILE = params.get("file") ?? "/fixtures/vdb-samples/smoke.vdb";
const DENSITY = 4;
const BACKGROUND = 0x223049;

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, uncaptured: [] };
window.__VDB_COARSE_FIRST__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/** A pull-driven stream of `step`-byte chunks, one per pull, that waits on `gate()` before each. */
function throttled(bytes: Uint8Array, step: number, counter: { sent: number }, gate: () => Promise<void> = async () => undefined) {
  let at = 0;
  return new ReadableStream<Uint8Array>(
    {
      async pull(c) {
        await gate();
        await tick();
        if (at >= bytes.length) return c.close();
        const chunk = bytes.slice(at, at + step);
        at += chunk.length;
        counter.sent += chunk.length;
        c.enqueue(chunk);
      },
    },
    { highWaterMark: 0 },
  );
}

async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function streamRun(
  device: GPUDevice,
  file: Uint8Array,
  opts: { gzipped?: Uint8Array; chunk: number; pool?: VDBGridPool; onGrid?: (e: VDBCoarseFirstEvent) => void; gate?: () => Promise<void> },
): Promise<StreamRun & { pool: VDBGridPool }> {
  const pool = opts.pool ?? (await VDBGridPool.create(device));
  const counter = { sent: 0 };
  const transfer = opts.gzipped ?? file;
  let source = throttled(transfer, opts.chunk, counter, opts.gate);
  if (opts.gzipped) source = source.pipeThrough(new DecompressionStream("gzip") as unknown as ReadableWritablePair<Uint8Array, Uint8Array>);
  let firstPaintTransfer = -1;
  const res = await vdbStreamCoarseFirst(pool, source, {
    onGrid: (e) => {
      if (e.action === "add") firstPaintTransfer = counter.sent;
      opts.onGrid?.(e);
    },
  });
  return {
    pool,
    firstPaintTransfer,
    firstPaintFile: res.firstBytes,
    transferTotal: transfer.length,
    chunk: opts.chunk,
    grids: res.shown.map((e) => e.grid.metadata.name),
    id: res.id,
    actions: res.shown.map((e) => e.action),
  };
}

async function run(): Promise<void> {
  const resp = await fetch(FILE);
  if (!resp.ok) {
    result.missing = `${FILE} -> ${resp.status}`;
    log(`fixture missing: ${result.missing}`, "fail");
    return;
  }
  const vdb = parseVdb(await resp.arrayBuffer());

  // 1. Bake: ZIP (the default) with vdbBakeCoarseFirst; the same images with codec NONE.
  const zipFile = vdbBakeCoarseFirst(vdb, { previewResolution: 48, filter: "box", quantize: "fpn" });
  const parsed = NvdbFile.fromArrayBuffer(zipFile.slice().buffer);
  const previewImage = parsed.gridImage(0);
  const fullImage = parsed.gridImage(1);
  const noneFile = new Uint8Array(writeNvdb([previewImage, fullImage]));
  const [zipGz, noneGz] = await Promise.all([gzip(zipFile), gzip(noneFile)]);
  result.sizes = { zipFile: zipFile.length, noneFile: noneFile.length, zipGzip: zipGz.length, noneGzip: noneGz.length };
  result.grids = parsed.grids.map((g: NvdbGridMetadata) => ({
    name: g.name,
    gridType: g.gridType,
    voxelCount: g.voxelCount,
    gridByteSize: g.gridByteSize,
    voxelSize: [...g.voxelSize],
  }));
  log(`baked ${parsed.grids.map((g) => g.name).join(" + ")}: zip ${zipFile.length} B, none ${noneFile.length} B`);

  // 2. Device-first renderer and the frame helpers.
  const { renderer, device, adapterName } = await createDeviceRenderer();
  result.adapter = adapterName;
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));
  result.width = W;
  result.height = H;

  const fullMeta = parsed.grids[1]!;
  const box = new THREE.Box3(new THREE.Vector3(...fullMeta.worldBBox.min), new THREE.Vector3(...fullMeta.worldBBox.max));
  const center = box.getCenter(new THREE.Vector3());
  const diag = box.getSize(new THREE.Vector3()).length();
  const camera = new THREE.PerspectiveCamera(40, W / H, diag * 0.01, diag * 10);
  camera.position.set(center.x + diag * 0.9, center.y + diag * 0.35, center.z + diag * 1.1);
  camera.lookAt(center);
  camera.updateMatrixWorld();

  const rt = new THREE.RenderTarget(W, H);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  const bg = new THREE.Color(BACKGROUND);
  const bgBytes = [bg.clone().convertLinearToSRGB()].map((c) => [Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255)])[0]!;

  async function frame(scene: THREE.Scene, r: WebGPURenderer): Promise<Frame> {
    r.setRenderTarget(rt);
    r.render(scene, camera);
    const pixels = (await r.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as Uint8Array;
    r.setRenderTarget(null);
    const packed = W * 4;
    const stride = H > 1 ? (pixels.byteLength - packed) / (H - 1) : packed;
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(pixels.subarray((H - 1 - y) * stride, (H - 1 - y) * stride + packed), y * packed);
    let nonBackground = 0;
    for (let i = 0; i < out.length; i += 4) {
      if (Math.abs(out[i]! - bgBytes[0]!) + Math.abs(out[i + 1]! - bgBytes[1]!) + Math.abs(out[i + 2]! - bgBytes[2]!) > 6) nonBackground++;
    }
    let bin = "";
    for (let i = 0; i < out.length; i += 0x8000) bin += String.fromCharCode(...out.subarray(i, i + 0x8000));
    return { rgba: btoa(bin), nonBackground };
  }

  function sceneFor(pool: VDBGridPool, id: number) {
    const grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
    const material = new VDBVolumeMaterial(renderer, grids, id, { jitter: false, densityScale: DENSITY });
    const scene = new THREE.Scene();
    scene.background = bg;
    const mesh = new THREE.Mesh(vdbProxyGeometry(pool.info(id)), material);
    scene.add(mesh);
    return { grids, material, scene, mesh };
  }

  // 3. zipRaw, held at the preview: render it, then let the full grid in.
  const pool = await VDBGridPool.create(device);
  let release!: () => void;
  let held: Promise<void> | null = null;
  let previewWork: Promise<void> | null = null;
  let view: ReturnType<typeof sceneFor> | null = null;
  let previewEvent: VDBCoarseFirstEvent | null = null;
  let versionAtPreview = -1;
  const zipRaw = await streamRun(device, zipFile, {
    chunk: 1024,
    pool,
    gate: async () => {
      if (previewWork) await previewWork;
      if (held) await held;
    },
    onGrid: (e) => {
      if (e.action !== "add") return;
      previewEvent = e;
      held = new Promise<void>((r) => (release = r));
      previewWork = (async () => {
        pool.flush();
        await pool.settled();
        versionAtPreview = pool.version;
        view = sceneFor(pool, e.id);
        view.material.densityScale.value = DENSITY * e.voxelRatio;
        result.preview = { ...(await frame(view.scene, renderer)), densityScale: DENSITY * e.voxelRatio };
        log(`preview ${e.grid.metadata.name} added at ${e.bytesReceived} file bytes; rendered (${result.preview.nonBackground} px)`);
        release();
      })();
    },
  });
  await previewWork;
  pool.flush();
  await pool.settled();
  const v = view! as ReturnType<typeof sceneFor>;
  v.grids.update();
  v.material.densityScale.value = DENSITY;
  v.mesh.geometry.dispose();
  v.mesh.geometry = vdbProxyGeometry(pool.info(zipRaw.id));
  result.final = await frame(v.scene, renderer);
  const info = pool.info(zipRaw.id);
  const pe = previewEvent! as VDBCoarseFirstEvent;
  const cellMin = (m: { worldBBox: { min: number[] }; voxelSize: number[] }) => m.worldBBox.min.map((x, a) => x - m.voxelSize[a]! / 2);
  result.placement = {
    sameId: zipRaw.id === pe.id,
    inPlace: pool.version === versionAtPreview,
    finalWorldBBox: [[...info.worldBBox.min], [...info.worldBBox.max]],
    fullWorldBBox: [[...fullMeta.worldBBox.min], [...fullMeta.worldBBox.max]],
    previewCellMin: cellMin(pe.grid.metadata),
    fullCellMin: cellMin(fullMeta),
  };

  // 4. Reference: the full grid alone, from a single-grid file, in a fresh pool.
  const alone = NvdbFile.fromArrayBuffer(writeNvdb([fullImage]));
  const refPool = await VDBGridPool.create(device);
  const refId = refPool.add(alone.gridImage(0));
  refPool.flush();
  await refPool.settled();
  const ref = sceneFor(refPool, refId);
  result.reference = await frame(ref.scene, renderer);
  result.finalEqualsReference = result.final.rgba === result.reference.rgba;
  result.previewDiffersFromFinal = result.preview!.rgba !== result.final.rgba;
  log(`final == full-grid-alone reference: ${result.finalEqualsReference}`, result.finalEqualsReference ? "ok" : "fail");

  // 5. Byte counts with gzip on the wire (no rendering).
  const noneGzip = await streamRun(device, noneFile, { gzipped: noneGz, chunk: 256 });
  const zipGzip = await streamRun(device, zipFile, { gzipped: zipGz, chunk: 256 });
  noneGzip.pool.dispose();
  zipGzip.pool.dispose();
  const strip = ({ pool: _p, ...r }: StreamRun & { pool: VDBGridPool }): StreamRun => r;
  result.runs = { zipRaw: strip(zipRaw), noneGzip: strip(noneGzip), zipGzip: strip(zipGzip) };
  for (const [k, r] of Object.entries(result.runs)) {
    log(`${k}: preview added after ${r.firstPaintTransfer} transferred bytes (file offset ${r.firstPaintFile}); whole transfer ${r.transferTotal} B`);
  }
}

run()
  .catch((err: unknown) => {
    result.error = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
