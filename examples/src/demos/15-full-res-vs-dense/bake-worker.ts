/**
 * Demo 15's CPU side, off the main thread: load an asset (the full-resolution grid image and its file size), bake its
 * dense `.wvdb` at a resolution (his `wvdb` bake: point sampling, 8-bit, clamped to 0–1), then measure both sides
 * against the source with the comparison report's metric (`vdbResampleError`, docs/dev/research/INDEX) and the file's gzip
 * size. Requests run in order: the page asks for `load`, `bake`, then `measure`, so a bake shows before the numbers.
 * Public tools API only.
 *
 * The dense bake is his default: the grid's box, `resolution` voxels on the longest side, the other sides rounded on
 * their own (so voxels are not quite cubic; `wvdbDenseToNvdb` reads each axis back with its own voxel size), and the
 * page renders that same bake.
 */
import {
  NvdbFile,
  buildFromVdb,
  nvdbToWvdb,
  parseVdb,
  readWvdb,
  vdbResampleError,
  vdbToWvdb,
  writeNvdb,
  wvdbDenseToNvdb,
  wvdbDenseVolume,
  wvdbSourceFromNvdb,
  wvdbSourceFromVdbGrid,
  type VdbFile,
} from "@pmndrs/vdb/tools";

import { ASSETS, type AssetId, type WorkerRequest, type WorkerResponse } from "./protocol.js";

interface Loaded {
  /** What the bake reads. */
  bakeFrom: { kind: "vdb"; file: VdbFile } | { kind: "nvdb"; image: Uint32Array };
  /** The full-resolution grid (the sparse side). */
  image: Uint32Array;
  /** The sparse side's file (an .nvdb). */
  fileBytes: Uint8Array;
  /** What the errors are measured against, once `measure` has looked for it. */
  reference: { grid: VdbFile | Uint32Array; name: string } | null;
  sparseMeasured: boolean;
  bakes: Map<number, Uint8Array>;
}

const assets = new Map<AssetId, Loaded>();
const post = (m: WorkerResponse, transfer: Transferable[] = []): void => (self as unknown as Worker).postMessage(m, transfer);
const fileName = (url: string): string => url.split("/").pop()!;

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

async function gzipSize(bytes: Uint8Array): Promise<number> {
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new CompressionStream("gzip"));
  const reader = stream.getReader();
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return n;
    n += value.byteLength;
  }
}

async function load(asset: AssetId): Promise<Loaded> {
  const have = assets.get(asset);
  if (have) return have;
  const def = ASSETS[asset];
  post({ type: "status", text: `loading ${fileName(def.url)}…` });
  const bytes = await fetchBytes(def.url);
  let a: Loaded;
  if (asset === "smoke") {
    const file = parseVdb(bytes);
    const grid = file.grids.find((g) => g.valueType === "float");
    if (!grid) throw new Error("smoke.vdb has no float grid");
    const image = buildFromVdb(grid);
    if (!wvdbSourceFromVdbGrid(grid).bbox) throw new Error("smoke.vdb: empty grid");
    // The sparse side ships the NanoVDB grid as an .nvdb file.
    a = { bakeFrom: { kind: "vdb", file }, image, fileBytes: new Uint8Array(writeNvdb([image])), reference: null, sparseMeasured: false, bakes: new Map() };
  } else {
    const image = NvdbFile.fromArrayBuffer(bytes).gridImage(0);
    if (!wvdbSourceFromNvdb(image).bbox) throw new Error("WDAS: empty grid");
    a = { bakeFrom: { kind: "nvdb", image }, image, fileBytes: new Uint8Array(bytes), reference: null, sparseMeasured: false, bakes: new Map() };
  }
  assets.set(asset, a);
  const meta = NvdbFile.fromArrayBuffer(a.fileBytes.slice().buffer).grids[0]!;
  const ib = meta.indexBBox;
  const indexDims = [0, 1, 2].map((k) => ib.max[k]! - ib.min[k]! + 1) as [number, number, number];
  const image = a.image.slice();
  post(
    { type: "loaded", asset, image, gridType: meta.gridType, activeVoxels: meta.voxelCount, indexDims, fileName: fileName(def.url), fileBytes: a.fileBytes.byteLength },
    [image.buffer],
  );
  post({ type: "status", text: "" });
  return a;
}

async function bake(asset: AssetId, resolution: number): Promise<void> {
  const a = await load(asset);
  let bytes = a.bakes.get(resolution);
  const options = { resolution, fileName: fileName(ASSETS[asset].url) };
  if (!bytes) {
    post({ type: "status", text: `baking a ${resolution} .wvdb (his bake: point sampling, 8-bit)…` });
    bytes = (a.bakeFrom.kind === "vdb" ? vdbToWvdb(a.bakeFrom.file, options) : nvdbToWvdb(a.bakeFrom.image, options)).bytes;
    a.bakes.set(resolution, bytes);
  }
  const wvdb = bytes.slice();
  post({ type: "baked", asset, resolution, wvdb }, [wvdb.buffer]);
  post({ type: "status", text: "" });
}

async function measure(asset: AssetId, resolution: number): Promise<void> {
  const a = await load(asset);
  const def = ASSETS[asset];
  if (!a.reference) {
    a.reference = { grid: a.bakeFrom.kind === "vdb" ? a.bakeFrom.file : a.image, name: asset === "smoke" ? "smoke.vdb" : "the Fp8 grid" };
    if (def.reference) {
      post({ type: "status", text: `loading ${fileName(def.reference)} (the source, for the error)…` });
      try {
        a.reference = { grid: parseVdb(await fetchBytes(def.reference)), name: "the .vdb" };
      } catch {
        // keep the Fp8 grid as the reference
      }
    }
  }
  const against = a.reference.name;
  if (!a.sparseMeasured) {
    post({ type: "status", text: "measuring the full-resolution grid…" });
    const fileGz = await gzipSize(a.fileBytes);
    const e = vdbResampleError(a.reference.grid, a.image, { stride: def.stride });
    a.sparseMeasured = true;
    post({ type: "sparse-stats", asset, fileGz, error: { percent: e.percent, count: e.count, stride: def.stride, against } });
  }
  const bytes = a.bakes.get(resolution);
  if (bytes) {
    post({ type: "status", text: `measuring the ${resolution} bake…` });
    // The dense level as stored (8-bit, clamped to 0–1), read back trilinearly at each sampled source voxel.
    const dense = wvdbDenseVolume(readWvdb(bytes));
    const e = vdbResampleError(a.reference.grid, wvdbDenseToNvdb(dense).image, { stride: def.stride });
    post({ type: "dense-stats", asset, resolution, error: { percent: e.percent, count: e.count, stride: def.stride, against } });
  }
  post({ type: "status", text: "" });
}

let queue: Promise<void> = Promise.resolve();
self.addEventListener("message", (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  queue = queue
    .then(() => (req.type === "load" ? load(req.asset).then(() => undefined) : req.type === "bake" ? bake(req.asset, req.resolution) : measure(req.asset, req.resolution)))
    .catch((err: unknown) => post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err) }));
});
