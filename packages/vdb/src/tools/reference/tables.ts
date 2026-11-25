/**
 * K6.R1: the reference renderer's tables. Tabulated phase functions (his Lorenz-Mie and Mie RGB tables, sampled by
 * inverse CDF), the blackbody emission colour (the renderer's LUT over log temperature) and per-cell majorant grids
 * (piecewise-constant density bounds for delta and ratio tracking: dense texels or the sparse pool's node stats).
 * The GPU kernel reads the same numbers (`core/reference/renderer.ts` packs them into its table and majorant
 * textures), so the CPU twin and the kernel make the same decisions.
 */

import { nodeStats, treeNodes } from "../cpu/march.js";
import { frameBlackbodyMaxTemperature, frameBlackbodyMinTemperature, frameBlackbodyTableSize, vdbCie1931Fit, vdbFrameBlackbodyColor, type VDBColorMatchingFunctions } from "../frame/blackbody.js";
import { frameMieRgbTable, frameMieTable, frameMieTableSize } from "../frame/mie-table.js";
import type { WvdbDenseVolume } from "../wvdb/dense.js";
import type { VDBRefVec3, VDBReferenceColorTable, VDBReferenceMajorantGrid, VDBReferencePhaseTable } from "./scene.js";

// ── tabulated phase ──────────────────────────────────────────────────────────────────────────────────

/**
 * The sampling CDF of a phase table: the marginal density of u (θ = π u²) of the channel mean m,
 * p(u) = m(u) · π u · sin(π u²) (phase / isotropic → per steradian, dω = 4π² u sin θ du), trapezoid weights per
 * table interval, normalised; stored in f32 (the GPU table's precision) so the twin and the kernel pick the same
 * interval.
 */
export function vdbReferencePhaseTableCdf(values: ArrayLike<number>, size: number): Float32Array {
  const q = (i: number) => {
    const u = i / (size - 1);
    const m = (values[i * 3]! + values[i * 3 + 1]! + values[i * 3 + 2]!) / 3;
    return m * Math.PI * u * Math.sin(Math.PI * u * u);
  };
  const acc = new Float64Array(size);
  for (let i = 0; i + 1 < size; i++) acc[i + 1] = acc[i]! + (q(i) + q(i + 1)) / (2 * (size - 1));
  const total = acc[size - 1]!;
  const cdf = new Float32Array(size);
  for (let i = 0; i < size; i++) cdf[i] = acc[i]! / total;
  cdf[size - 1] = 1;
  return cdf;
}

/** A phase table from RGB values (phase / isotropic at θ = π u², `size` angles). */
export function vdbReferencePhaseTable(values: ArrayLike<number>, size: number): VDBReferencePhaseTable {
  if (values.length !== size * 3 || size < 2) throw new RangeError(`vdbReferencePhaseTable: ${values.length} values for ${size} angles`);
  const v = Float32Array.from(values);
  return { size, values: v, cdf: vdbReferencePhaseTableCdf(v, size) };
}

/**
 * His Lorenz-Mie tables (the committed `mie-table.ts`, `vdbFrameMieTexels`): 'mono' ('lorenz-mie', one table for
 * every channel) or 'rgb' ('lorenz-mie-rgb', one per channel).
 */
export function vdbReferenceMiePhaseTable(kind: "mono" | "rgb"): VDBReferencePhaseTable {
  const n = frameMieTableSize;
  const values = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) values[i * 3 + c] = kind === "mono" ? frameMieTable[i]! : frameMieRgbTable[i * 3 + c]!;
  }
  return vdbReferencePhaseTable(values, n);
}

/** The table's RGB phase / isotropic at u ∈ [0, 1] (texel u·(N − 1), linear blend: `vdbFrameMieTexel`). */
function tableAt(t: VDBReferencePhaseTable, u: number): VDBRefVec3 {
  const texel = Math.min(Math.max(u, 0), 1) * (t.size - 1);
  const i = Math.min(Math.trunc(texel), t.size - 2);
  const f = texel - i;
  const a = i * 3;
  return [0, 1, 2].map((c) => t.values[a + c]! + (t.values[a + 3 + c]! - t.values[a + c]!) * f) as VDBRefVec3;
}

/** The tabulated phase per steradian of each channel at cos θ. */
export function vdbRefPhaseTableEval(t: VDBReferencePhaseTable, cosTheta: number): VDBRefVec3 {
  const u = Math.sqrt(Math.acos(Math.min(Math.max(cosTheta, -1), 1)) / Math.PI);
  const v = tableAt(t, u);
  return [v[0] / (4 * Math.PI), v[1] / (4 * Math.PI), v[2] / (4 * Math.PI)];
}

/**
 * Sample cos θ from the table's CDF with ξ ∈ [0, 1): the interval j with cdf[j] ≤ ξ < cdf[j+1] (binary search),
 * u uniform in it. `weight` per channel = phase / sampling density (1 for a table whose channels are equal, up to the
 * piecewise-constant density's error): the estimate stays unbiased per channel.
 */
export function vdbRefPhaseTableSample(t: VDBReferencePhaseTable, xi: number): { cosTheta: number; weight: VDBRefVec3; pdfU: number } {
  let lo = 0;
  let hi = t.size - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t.cdf[mid]! <= xi) lo = mid;
    else hi = mid;
  }
  const c0 = t.cdf[lo]!;
  const c1 = t.cdf[lo + 1]!;
  const f = (xi - c0) / (c1 - c0);
  const u = (lo + f) / (t.size - 1);
  const pdfU = (c1 - c0) * (t.size - 1);
  const theta = Math.PI * u * u;
  const s = (Math.PI * u * Math.sin(theta)) / pdfU;
  const v = tableAt(t, u);
  return { cosTheta: Math.cos(theta), weight: [v[0] * s, v[1] * s, v[2] * s], pdfU };
}

// ── blackbody emission colour ────────────────────────────────────────────────────────────────────────

/**
 * The renderer's blackbody emission colour (`vdbFrameBlackbodyEmission(spectrum, sootExponent)`, three/frame) as a
 * reference colour table: its 256 log-spaced columns at the soot exponent (his texel row blend included), read with
 * the same linear blend in log temperature, so the reference's colour equals `vdbFrameBlackbodyColor` (f32 table).
 * `exponent`: 0 for 'planck', the soot exponent for 'soot' (`vdbFrameEmissionSpectrum`).
 */
export function vdbReferenceBlackbodyTable(exponent = 0, cmf: VDBColorMatchingFunctions = vdbCie1931Fit): VDBReferenceColorTable {
  const n = frameBlackbodyTableSize;
  const values = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const temperature = frameBlackbodyMinTemperature * (frameBlackbodyMaxTemperature / frameBlackbodyMinTemperature) ** (i / (n - 1));
    values.set(vdbFrameBlackbodyColor(temperature, exponent, cmf), i * 3);
  }
  return { size: n, values, minTemperature: frameBlackbodyMinTemperature, maxTemperature: frameBlackbodyMaxTemperature };
}

/** A colour table read at `temperature` (K): log-temperature column, linear blend, clamped to the table's range. */
export function vdbRefColorTableAt(t: VDBReferenceColorTable, temperature: number): VDBRefVec3 {
  const x = Math.min(1, Math.max(0, Math.log(Math.max(temperature, 1e-3) / t.minTemperature) / Math.log(t.maxTemperature / t.minTemperature))) * (t.size - 1);
  const i = Math.min(Math.trunc(x), t.size - 2);
  const f = x - i;
  return [0, 1, 2].map((c) => t.values[i * 3 + c]! + (t.values[i * 3 + 3 + c]! - t.values[i * 3 + c]!) * f) as VDBRefVec3;
}

// ── majorant grids ───────────────────────────────────────────────────────────────────────────────────

/**
 * The cell of a majorant grid the ray o + t·d (unit-box local coordinates) is in at t, and the t where it leaves
 * it. Each coordinate is nudged by 1e-4 cell in the direction of travel, so a point on a cell face belongs to the
 * cell ahead and the exit is past t. Twin of the kernel's `vdbRefMajorantAt`.
 */
export function vdbRefMajorantCell(g: VDBReferenceMajorantGrid, lo: readonly number[], ld: readonly number[], t: number): { index: number; exit: number; cell: [number, number, number] } {
  let exit = Infinity;
  let index = 0;
  let stride = 1;
  const cell: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const n = g.dims[k]!;
    const q = (lo[k]! + ld[k]! * t + 0.5) * n;
    const nudge = ld[k]! > 0 ? 1e-4 : ld[k]! < 0 ? -1e-4 : 0;
    const c = Math.min(Math.max(Math.floor(q + nudge), 0), n - 1);
    if (Math.abs(ld[k]!) > 1e-12) {
      const face = (c + (ld[k]! > 0 ? 1 : 0)) / n - 0.5;
      exit = Math.min(exit, (face - lo[k]!) / ld[k]!);
    }
    index += c * stride;
    stride *= n;
    cell[k] = c;
  }
  return { index, exit, cell };
}

/**
 * Majorants of a dense source from its texels: per cell of `cellTexels`³ texels, the largest density texel a
 * trilinear read inside the cell can touch (the cell's texel range ± one texel, clamped), times the mask's when the
 * volume has one. Exact bound of the hardware-trilinear read (convex weights).
 */
export function vdbReferenceMajorantGridDense(volume: Pick<WvdbDenseVolume, "density" | "mask">, cellTexels = 8): VDBReferenceMajorantGrid {
  const D = volume.density.dimensions;
  const dims = [0, 1, 2].map((k) => Math.max(1, Math.ceil(D[k]! / cellTexels))) as [number, number, number];
  const blockMax = (tex: NonNullable<WvdbDenseVolume["mask"]>) => {
    const out = new Float32Array(dims[0] * dims[1] * dims[2]);
    const range = (k: number, c: number) => {
      const n = dims[k]!;
      const lo = Math.floor((c * D[k]!) / n - 0.5);
      const hi = Math.floor(((c + 1) * D[k]!) / n - 0.5) + 1;
      return [Math.max(lo, 0), Math.min(hi, D[k]! - 1)] as const;
    };
    const stride = tex.texelBytes;
    for (let cz = 0; cz < dims[2]; cz++) {
      const [z0, z1] = range(2, cz);
      for (let cy = 0; cy < dims[1]; cy++) {
        const [y0, y1] = range(1, cy);
        for (let cx = 0; cx < dims[0]; cx++) {
          const [x0, x1] = range(0, cx);
          let m = 0;
          for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) {
            const row = (z * D[1]! + y) * D[0]!;
            for (let x = x0; x <= x1; x++) m = Math.max(m, tex.data[(row + x) * stride]!);
          }
          out[(cz * dims[1] + cy) * dims[0] + cx] = m / 255;
        }
      }
    }
    return out;
  };
  const values = blockMax(volume.density);
  if (volume.mask) {
    const mask = blockMax(volume.mask);
    for (let i = 0; i < values.length; i++) values[i] = values[i]! * mask[i]!;
  }
  return { dims, values };
}

/** Options of {@link vdbReferenceMajorantGridSparse}. */
export interface VDBReferenceMajorantGridSparseOptions {
  /** The source's `normalize: 'clamp01'` (bounds capped at 1). Default false. */
  clamp01?: boolean;
  /** Index voxels per cell side. Default 8 (a leaf). */
  cellVoxels?: number;
  /**
   * 'stats' (default): leaf max from the node stats (ACTIVE values only, as the pool's `nodeStats`), active tiles by
   * their value. 'voxels': every leaf voxel decoded (also inactive ones), for checking the stats bound.
   */
  from?: "stats" | "voxels";
  /** Leaf voxel values for `from: 'voxels'` (`leafVoxels` of tools/cpu/march; injected to keep this module light). */
  leafValues?: (image: Uint32Array, leafAddress: number) => ArrayLike<number>;
}

/**
 * Majorants of one sparse volume (a NanoVDB float grid read through `indexBox`, `vdbSparseSourceGpu` at level 0) from
 * the tree: each leaf's max (node stats) and each active tile's value bound every cell its voxels can reach under a
 * trilinear read (the node's index box dilated by one voxel; uvw 0 is index min − ½). Cells no node reaches are 0
 * (the background is assumed 0: fog volumes; a non-zero background is a violation the twin counts).
 */
export function vdbReferenceMajorantGridSparse(image: Uint32Array, indexBox: { min: readonly number[]; max: readonly number[] }, o: VDBReferenceMajorantGridSparseOptions = {}): VDBReferenceMajorantGrid {
  const cell = o.cellVoxels ?? 8;
  const size = [0, 1, 2].map((k) => indexBox.max[k]! - indexBox.min[k]! + 1);
  const dims = size.map((s) => Math.max(1, Math.ceil(s / cell))) as [number, number, number];
  const values = new Float32Array(dims[0] * dims[1] * dims[2]);
  const put = (origin: readonly number[], dim: number, value: number) => {
    if (!(value > 0)) return;
    const r = [0, 1, 2].map((k) => {
      // voxels origin … origin + dim − 1 reach index coordinates (origin − 1, origin + dim); uvw = (x − min + ½) / size
      const u0 = (origin[k]! - 1 - indexBox.min[k]! + 0.5) / size[k]!;
      const u1 = (origin[k]! + dim - indexBox.min[k]! + 0.5) / size[k]!;
      return [Math.max(Math.floor(u0 * dims[k]!), 0), Math.min(Math.floor(u1 * dims[k]!), dims[k]! - 1)] as const;
    });
    for (let z = r[2]![0]; z <= r[2]![1]; z++) for (let y = r[1]![0]; y <= r[1]![1]; y++) for (let x = r[0]![0]; x <= r[0]![1]; x++) {
      const i = (z * dims[1] + y) * dims[0] + x;
      if (value > values[i]!) values[i] = value;
    }
  };
  for (const node of treeNodes(image)) {
    for (const t of node.activeTiles) put(t.origin, t.dim, t.value);
    if (node.level !== 0) continue;
    let m: number;
    if (o.from === "voxels") {
      if (!o.leafValues) throw new RangeError("vdbReferenceMajorantGridSparse: from 'voxels' needs leafValues");
      m = -Infinity;
      for (const v of Array.from(o.leafValues(image, node.address))) m = Math.max(m, v);
    } else m = nodeStats(image, 0, node.address).max;
    put(node.origin, 8, m);
  }
  if (o.clamp01) for (let i = 0; i < values.length; i++) values[i] = Math.min(values[i]!, 1);
  return { dims, values };
}

/** The share of a majorant grid's cells that are 0 (empty space delta tracking skips) and the mean bound. */
export function vdbReferenceMajorantGridStats(g: VDBReferenceMajorantGrid): { cells: number; empty: number; mean: number; max: number } {
  let empty = 0;
  let sum = 0;
  let max = 0;
  for (const v of g.values) {
    if (!(v > 0)) empty++;
    sum += v;
    max = Math.max(max, v);
  }
  return { cells: g.values.length, empty: empty / g.values.length, mean: sum / g.values.length, max };
}
