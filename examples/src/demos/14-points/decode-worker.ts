/**
 * Demo 14's decode worker: downloads an OpenVDB point data grid, parses it with our parser (`parseVdb`), decodes
 * its points and attributes (`VdbPointDataGrid.points()`), and hands back render-ready arrays, so a 157 MB file
 * never blocks the page. Public API only.
 *
 * The points come back in "subsample order": every 10th point first, then the rest of the even points, then the
 * odd ones. A prefix of 10% (or 50%) of the arrays is then an even subsample of the whole cloud, and the page
 * changes the density by changing the instance count, without touching the buffers.
 */
import { parseVdb, type VdbPointDataGrid } from "@pmndrs/vdb/tools";

export interface DecodeRequest {
  url: string;
}

export type DecodeMessage =
  | { type: "progress"; stage: "download"; received: number; total: number }
  | { type: "progress"; stage: "parse" | "decode" | "order"; count?: number }
  | { type: "done"; result: DecodedPoints }
  | { type: "error"; message: string };

export interface DecodedPoints {
  count: number;
  /** Subsample order (see the file comment): xyz per point, grid world space. */
  positions: Float32Array;
  /** `Cd` in the same order (linear RGB), or null. */
  cd: Float32Array | null;
  /** `|v|` in the same order, or null. */
  speed: Float32Array | null;
  /** 2nd and 98th percentile of `|v|` (colour ramp range). */
  speedRange: [number, number];
  /** Prefix counts: 10%, 50%, 100%. */
  prefix: { p10: number; p50: number; p100: number };
  /** Mean spacing of all points in world units (`voxel / cbrt(points per active voxel)`). */
  spacing: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  /** Attribute names the file has, and the ones the decoder skipped. */
  attributes: string[];
  skipped: string[];
  fileBytes: number;
  activeVoxels: number;
  ms: { download: number; parse: number; decode: number; order: number };
}

/** Residue classes mod 10 in output order: 0 (10%), then 2, 4, 6, 8 (to 50%), then the odd ones. */
const CLASS_ORDER = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9] as const;
const classCount = (n: number, c: number): number => (c < n ? Math.floor((n - 1 - c) / 10) + 1 : 0);

const post = (m: DecodeMessage, transfer: Transferable[] = []): void => (self as unknown as Worker).postMessage(m, transfer);

self.onmessage = async (e: MessageEvent<DecodeRequest>) => {
  try {
    const t0 = performance.now();
    const r = await fetch(e.data.url);
    if (!r.ok || !r.body) throw new Error(`fetch ${e.data.url}: ${r.status}`);
    const total = Number(r.headers.get("content-length") ?? 0);
    const reader = r.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    let lastPost = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      const now = performance.now();
      if (now - lastPost > 50) {
        lastPost = now;
        post({ type: "progress", stage: "download", received, total });
      }
    }
    const bytes = new Uint8Array(received);
    let o = 0;
    for (const c of chunks) {
      bytes.set(c, o);
      o += c.byteLength;
    }
    chunks.length = 0;
    const t1 = performance.now();

    post({ type: "progress", stage: "parse" });
    const file = parseVdb(bytes.buffer);
    const grid = file.grids.find((g): g is VdbPointDataGrid => g.valueType === "ptdataidx32");
    if (!grid) throw new Error(`${e.data.url} has no point data grid`);
    const t2 = performance.now();

    post({ type: "progress", stage: "decode", count: grid.pointCount });
    const list = grid.points();
    const t3 = performance.now();

    post({ type: "progress", stage: "order", count: list.count });
    const n = list.count;
    const real3 = (name: string): Float32Array | Float64Array | null => {
      const a = list.attributes[name];
      const shape = list.attributeShapes[name];
      return a && shape?.components === 3 && shape.stride === 1 && (a instanceof Float32Array || a instanceof Float64Array) ? a : null;
    };
    const cdSrc = real3("Cd");
    const vSrc = real3("v");
    const positions = new Float32Array(n * 3);
    const cd = cdSrc ? new Float32Array(n * 3) : null;
    const speed = vSrc ? new Float32Array(n) : null;
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const src = list.positions;
    let j = 0;
    for (const c of CLASS_ORDER) {
      for (let i = c; i < n; i += 10, j++) {
        for (let k = 0; k < 3; k++) {
          const p = src[i * 3 + k]!;
          positions[j * 3 + k] = p;
          if (p < min[k]!) min[k] = p;
          if (p > max[k]!) max[k] = p;
          if (cd) cd[j * 3 + k] = cdSrc![i * 3 + k]!;
        }
        if (speed) {
          const vx = vSrc![i * 3]!;
          const vy = vSrc![i * 3 + 1]!;
          const vz = vSrc![i * 3 + 2]!;
          speed[j] = Math.sqrt(vx * vx + vy * vy + vz * vz);
        }
      }
    }
    // Percentiles of |v| from an even sample (the 10% prefix is one).
    let speedRange: [number, number] = [0, 1];
    if (speed && n > 0) {
      const sample = speed.slice(0, Math.min(n, 200_000)).sort();
      speedRange = [sample[Math.floor(sample.length * 0.02)]!, sample[Math.floor(sample.length * 0.98)]!];
      if (!(speedRange[1] > speedRange[0])) speedRange = [speedRange[0], speedRange[0] + 1];
    }
    const p10 = classCount(n, 0);
    const p50 = [0, 2, 4, 6, 8].reduce((s, c) => s + classCount(n, c), 0);
    const [sx, sy, sz] = grid.transform.voxelSize;
    const voxel = Math.cbrt(Math.abs(sx * sy * sz));
    const activeVoxels = Number(grid.activeVoxelCount);
    const spacing = voxel / Math.cbrt(activeVoxels > 0 ? n / activeVoxels : 1);
    const t4 = performance.now();

    const result: DecodedPoints = {
      count: n,
      positions,
      cd,
      speed,
      speedRange,
      prefix: { p10, p50, p100: n },
      spacing,
      bounds: { min, max },
      attributes: Object.keys(list.attributes),
      skipped: list.skipped.map((s) => s.name),
      fileBytes: received,
      activeVoxels,
      ms: { download: t1 - t0, parse: t2 - t1, decode: t3 - t2, order: t4 - t3 },
    };
    const transfer: Transferable[] = [positions.buffer];
    if (cd) transfer.push(cd.buffer);
    if (speed) transfer.push(speed.buffer);
    post({ type: "done", result }, transfer);
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err) });
  }
};
