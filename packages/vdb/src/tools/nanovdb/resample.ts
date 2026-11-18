/**
 * resample.ts — resample a sparse grid to a new resolution (K4.4) and build a
 * NanoVDB FLOAT image from it, optionally quantized to Fp8/FpN.
 *
 * ## Conventions
 *
 * Voxels are centred at integer index coordinates, the convention of the CPU
 * oracle's `readValue`/`sampleTrilinear` (and the WGSL sampler), and of
 * mjurczyk's `.wvdb` bake (`Math.round` in index space). Voxel `i` therefore
 * covers the index interval `[i - 0.5, i + 0.5]`.
 *
 * - **Box.** A world-space box `[min, max]`. Default: the cell box of the
 *   source's active voxels, `[imin - 0.5, imax + 0.5]` in source index space,
 *   mapped to world. Pass `bbox` for a shared box (a sequence) or to reproduce
 *   a `.wvdb` bake (his box is the voxel-centre box, `wvdbSourceFromVdbGrid(g).bbox`).
 * - **Target lattice.** `dims` voxels per axis with voxel size `vs` per axis;
 *   target voxel `j` is centred at `min + ((j + 0.5) / dims) * (dims * vs)`, the
 *   arithmetic of his `forEachTexel`. The output Map is `world = vs * ijk + t`
 *   with `t = min + vs / 2`, so the target index box is `[0, dims - 1]`.
 * - **Resolution.** Exactly one of `resolution` (voxels on the longest box side,
 *   uniform voxels), `voxelSize` (number: uniform; `[x, y, z]`: anisotropic) or
 *   `dimensions` (voxels per axis; voxel size = box / dims, anisotropic unless
 *   the box is proportional). With none, the source voxel size (an identity
 *   resample). With `resolution` or `voxelSize`, `dims = ceil(side / vs)` and the
 *   box grows at its max side to a whole number of voxels; with `dimensions`
 *   the box is kept exactly.
 *
 * ## Filters
 *
 * The map between target and source index space is separable (both maps are
 * axis-aligned), so every filter is a per-axis weight table applied as three
 * 1D passes over a gathered block of source voxels.
 *
 * - `'point'`: the nearest source voxel (`Math.round`), no prefilter. Same
 *   positions and arithmetic as the `.wvdb` bake; aliases when downsampling.
 * - `'box'` (default): the area-weighted average of the source voxels the
 *   target voxel's cell overlaps (inactive voxels count with their stored
 *   value, normally the background). It is the right default for
 *   downsampling: it is the integral of the source field over the target
 *   voxel, so mass (density times volume) is conserved, the coarse grid is the
 *   source seen at the coarse scale, and thin features fade instead of
 *   flickering in or out with the sampling phase. A downsample by 2 on an
 *   aligned box is the exact 2x2x2 mean.
 * - `'trilinear'`: the trilinear interpolation of the source at the target
 *   voxel centre (`sampleTrilinear`). Smooth for upsampling and small ratios;
 *   aliases like `'point'` at large downsampling ratios.
 *
 * ## Sparsity
 *
 * Only target leaves whose filter footprint overlaps a source leaf or active
 * tile are visited (empty space costs nothing), and a target voxel is active
 * only when its footprint touches an active source voxel and, given `cutoff`,
 * `|value - background| > cutoff`. Leaves left with no active voxel are dropped.
 * Inactive voxels inside an emitted leaf keep their filtered value (the field a
 * texture sampler would see).
 */

import { readValue } from "../cpu/read-value.js";
import { gridTypeConstantsForId } from "../cpu/stride-tables.js";
import { leafVoxels, treeNodes } from "../cpu/march.js";
import type { VdbFile, VdbFloatGrid, VdbGrid } from "../vdb/types.js";
import { sampleTrilinear } from "../cpu/sample-trilinear.js";
import {
  GRID_NAME_MAX,
  GRID_OFF_GRID_CLASS,
  GRID_OFF_GRID_NAME,
  GRID_OFF_GRID_TYPE,
  GRID_OFF_MAP,
  GRID_SIZE,
  MAP_OFF_INVMATD,
  MAP_OFF_MATD,
  MAP_OFF_VECD,
  TREE_OFF_NODE_OFFSET_ROOT,
} from "./bytes.js";
import { FLOAT_LEAF_CODEC } from "./leaf-codec.js";
import { quantizeDetailed, type QuantizeMode } from "./quantize.js";
import { buildFromLeavesDetailed, type BuiltGrid, type GridClassName } from "./serialize.js";
import type { LeafInput } from "./tree.js";

type Vec3 = [number, number, number];

export type ResampleFilter = "point" | "box" | "trilinear";

/** A NanoVDB FLOAT/Fp8/FpN grid image, a parsed `.vdb` float grid, or a parsed `.vdb` file. */
export type ResampleSource = Uint32Array | VdbGrid | VdbFile;

export interface ResampleOptions {
  /** Voxels on the longest side of the box; uniform voxels. */
  resolution?: number;
  /** Target voxel size in world units: uniform, or `[x, y, z]` (anisotropic). */
  voxelSize?: number | Vec3;
  /** Target voxels per axis; voxel size = box size / dims per axis. */
  dimensions?: Vec3;
  /** Default `'box'`. */
  filter?: ResampleFilter;
  /** World box `[min, max]`. Default: the source's active cell box. */
  bbox?: [Vec3, Vec3];
  /**
   * Keep a target voxel active only if `|value - background| > cutoff` (as
   * well as touching an active source voxel). Default: no value test.
   */
  cutoff?: number;
  /** Quantize the result (via `quantize`). Default: FLOAT. */
  quantize?: QuantizeMode;
  /** FpN tolerance (`quantize`'s, default 0.01). */
  tolerance?: number;
  /** Which grid of a `VdbFile` to resample. Default: the first float grid. */
  sourceGridName?: string;
  /** Output grid name. Default: the source grid's name. */
  gridName?: string;
  /** Output background. Default: the source's root background. */
  background?: number;
  /**
   * Collapse uniform target leaves and nodes into tiles (OpenVDB `prune`,
   * tolerance 0; K4.6), before any quantization. Default true: a resample is a
   * new build with no native topology to keep, and a uniform interior (a
   * source tile, or a saturated core) costs one tile instead of leaves.
   */
  collapse?: boolean;
}

export interface ResampledGrid extends BuiltGrid {
  /** Target voxels per axis (the index box is `[0, dims - 1]`). */
  dimensions: Vec3;
  /** The world box that was resampled (grown to whole voxels for `resolution`/`voxelSize`). */
  bbox: [Vec3, Vec3];
  filter: ResampleFilter;
  /** Source voxel size per axis. */
  sourceVoxelSize: Vec3;
}

// ---------------------------------------------------------------------------
// Source access
// ---------------------------------------------------------------------------

interface SourceLeaf {
  origin: Vec3;
  values: Float32Array;
  /** 1 per active voxel, x-major (`n = x*64 + y*8 + z`). */
  active: Uint8Array;
}

/** An axis-aligned sparse float field, indexed by source voxel. */
interface SourceField {
  name: string;
  gridClass: GridClassName;
  background: number;
  /** index → world: `world = scale * ijk + translation` (per axis). */
  scale: Vec3;
  translation: Vec3;
  /** world → index multiplier, as the source's own lookup computes it. */
  inverse: Vec3;
  leaves: Map<number, SourceLeaf>;
  tiles: { origin: Vec3; dim: number; value: number }[];
  /** Value and active state of an 8^3 block that has no leaf (constant over the block). */
  blockValue(bx: number, by: number, bz: number): { value: number; active: boolean };
}

const KEY_OFF = 1 << 16;
const KEY_SPAN = 1 << 17;

/** Packs a leaf-block coordinate (index >> 3) into a number key (|block| < 2^16, i.e. |index| < 2^19). */
function blockKey(bx: number, by: number, bz: number): number {
  const x = bx + KEY_OFF;
  const y = by + KEY_OFF;
  const z = bz + KEY_OFF;
  if (x < 0 || y < 0 || z < 0 || x >= KEY_SPAN || y >= KEY_SPAN || z >= KEY_SPAN) {
    throw new Error(`resampleGrid: index coordinate block [${bx}, ${by}, ${bz}] is outside the supported range (|ijk| < 2^19)`);
  }
  return (x * KEY_SPAN + y) * KEY_SPAN + z;
}

function cachedBlocks(lookup: (ijk: Vec3) => { value: number; active: boolean }): SourceField["blockValue"] {
  const cache = new Map<number, { value: number; active: boolean }>();
  return (bx, by, bz) => {
    const key = blockKey(bx, by, bz);
    let hit = cache.get(key);
    if (!hit) {
      const r = lookup([bx * 8, by * 8, bz * 8]);
      hit = { value: r.value, active: r.active };
      cache.set(key, hit);
    }
    return hit;
  };
}

function nanoClass(id: number): GridClassName {
  return id === 1 ? "LevelSet" : id === 2 ? "FogVolume" : id === 3 ? "Staggered" : "Unknown";
}

function vdbClass(grid: VdbGrid): GridClassName {
  const c = typeof grid.metadata["class"] === "string" ? (grid.metadata["class"] as string).toLowerCase() : "";
  if (c === "level set" || c === "levelset") return "LevelSet";
  if (c === "fog volume" || c === "fogvolume") return "FogVolume";
  if (c === "staggered") return "Staggered";
  return "Unknown";
}

function fieldFromNanoVdb(image: Uint32Array): SourceField {
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const gridType = view.getUint32(GRID_OFF_GRID_TYPE, true);
  const nameBytes = new Uint8Array(image.buffer, image.byteOffset + GRID_OFF_GRID_NAME, GRID_NAME_MAX);
  const nul = nameBytes.indexOf(0);
  const name = new TextDecoder().decode(nameBytes.subarray(0, nul < 0 ? GRID_NAME_MAX : nul));
  const mat = Array.from({ length: 9 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_MATD + i * 8, true));
  const inv = Array.from({ length: 9 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_INVMATD + i * 8, true));
  const vec = Array.from({ length: 3 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_VECD + i * 8, true));
  if ([1, 2, 3, 5, 6, 7].some((i) => mat[i] !== 0) || !(mat[0]! > 0 && mat[4]! > 0 && mat[8]! > 0)) {
    throw new Error(`resampleGrid: grid "${name}" has a rotated, sheared or flipped Map; only positive axis-aligned maps are supported`);
  }
  const root = GRID_SIZE + Number(view.getBigUint64(GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT, true));
  const background = view.getFloat32(root + gridTypeConstantsForId(gridType).root_off_background, true);

  const leaves = new Map<number, SourceLeaf>();
  const tiles: SourceField["tiles"] = [];
  for (const node of treeNodes(image)) {
    for (const t of node.activeTiles) tiles.push({ origin: [...t.origin], dim: t.dim, value: t.value });
    if (node.level !== 0) continue;
    const { values, active } = leafVoxels(image, node.address);
    const o = node.origin;
    leaves.set(blockKey(o[0] >> 3, o[1] >> 3, o[2] >> 3), { origin: [...o], values, active });
  }
  return {
    name,
    gridClass: nanoClass(view.getUint32(GRID_OFF_GRID_CLASS, true)),
    background,
    scale: [mat[0]!, mat[4]!, mat[8]!],
    translation: [vec[0]!, vec[1]!, vec[2]!],
    inverse: [inv[0]!, inv[4]!, inv[8]!],
    leaves,
    tiles,
    blockValue: cachedBlocks((ijk) => readValue(image, ijk)),
  };
}

function fieldFromVdbGrid(grid: VdbGrid): SourceField {
  if (grid.valueType !== "float") {
    throw new Error(`resampleGrid: grid "${grid.name}" is a ${grid.valueType} grid; only float grids can be resampled`);
  }
  const m = grid.transform.matrix;
  if ([m[1], m[2], m[4], m[6], m[8], m[9]].some((v) => v !== 0)) {
    throw new Error(`resampleGrid: grid "${grid.name}" has a rotated or sheared map; only axis-aligned maps are supported`);
  }
  const scale: Vec3 = [m[0]!, m[5]!, m[10]!];
  if (!scale.every((v) => v > 0)) {
    throw new Error(`resampleGrid: grid "${grid.name}" has a non-positive scale ${JSON.stringify(scale)}; only positive axis-aligned maps are supported`);
  }
  const leaves = new Map<number, SourceLeaf>();
  for (const leaf of grid.iterLeaves()) {
    const active = new Uint8Array(512);
    for (let n = 0; n < 512; n++) active[n] = (leaf.valueMask[n >>> 5]! >>> (n & 31)) & 1;
    const o = leaf.origin;
    leaves.set(blockKey(o[0] >> 3, o[1] >> 3, o[2] >> 3), {
      origin: [o[0], o[1], o[2]],
      values: leaf.values.length === 512 ? leaf.values : leaf.values.slice(0, 512),
      active,
    });
  }
  const tiles: SourceField["tiles"] = [];
  for (const t of grid.iterTiles()) tiles.push({ origin: [...t.origin], dim: t.dim, value: t.value as number });
  const g = grid as VdbFloatGrid;
  return {
    name: grid.name,
    gridClass: vdbClass(grid),
    background: grid.background as number,
    scale,
    translation: [m[3]!, m[7]!, m[11]!],
    inverse: [1 / scale[0], 1 / scale[1], 1 / scale[2]],
    leaves,
    tiles,
    blockValue: cachedBlocks((ijk) => g.readValue(ijk)),
  };
}

function toField(source: ResampleSource, sourceGridName?: string): SourceField {
  if (source instanceof Uint32Array) return fieldFromNanoVdb(source);
  if ("grids" in source) {
    const grid = sourceGridName
      ? source.grids.find((g) => g.name === sourceGridName)
      : source.grids.find((g) => g.valueType === "float");
    if (!grid) {
      throw new Error(
        sourceGridName ? `resampleGrid: no grid named "${sourceGridName}"` : "resampleGrid: the file has no float grid",
      );
    }
    return fieldFromVdbGrid(grid);
  }
  return fieldFromVdbGrid(source);
}

/** Inclusive index box of the active voxels (leaves and active tiles), or null. */
function activeIndexBox(field: SourceField): { min: Vec3; max: Vec3 } | null {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const leaf of field.leaves.values()) {
    for (let n = 0; n < 512; n++) {
      if (!leaf.active[n]) continue;
      const p = [leaf.origin[0] + (n >> 6), leaf.origin[1] + ((n >> 3) & 7), leaf.origin[2] + (n & 7)];
      for (let a = 0; a < 3; a++) {
        if (p[a]! < min[a]!) min[a] = p[a]!;
        if (p[a]! > max[a]!) max[a] = p[a]!;
      }
    }
  }
  for (const t of field.tiles) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a]!, t.origin[a]!);
      max[a] = Math.max(max[a]!, t.origin[a]! + t.dim - 1);
    }
  }
  return min[0] <= max[0] ? { min, max } : null;
}

// ---------------------------------------------------------------------------
// Per-axis filter tables
// ---------------------------------------------------------------------------

/** For each target index j on one axis: source taps `start[j] .. start[j] + count[j] - 1` with `weights`. */
interface AxisTable {
  start: Int32Array;
  count: Int32Array;
  /** Weight offset of target j in `weights`. */
  offset: Int32Array;
  weights: Float64Array;
}

const SNAP = 1e-9;

function snap(v: number): number {
  const r = Math.round(v);
  return Math.abs(v - r) < SNAP ? r : v;
}

function axisTable(
  filter: ResampleFilter,
  dims: number,
  boxMin: number,
  size: number,
  srcTranslation: number,
  srcInverse: number,
  ratio: number,
): AxisTable {
  const start = new Int32Array(dims);
  const count = new Int32Array(dims);
  const offset = new Int32Array(dims);
  const w: number[] = [];
  for (let j = 0; j < dims; j++) {
    // His forEachTexel arithmetic, then his world -> index map.
    const world = ((j + 0.5) / dims) * size + boxMin;
    const c = (world - srcTranslation) * srcInverse;
    offset[j] = w.length;
    if (filter === "point") {
      start[j] = Math.round(c);
      count[j] = 1;
      w.push(1);
    } else if (filter === "trilinear") {
      const cs = snap(c);
      const base = Math.floor(cs);
      const f = cs - base;
      start[j] = base;
      if (f === 0) {
        count[j] = 1;
        w.push(1);
      } else {
        count[j] = 2;
        w.push(1 - f, f);
      }
    } else {
      const half = Math.abs(ratio) / 2;
      const lo = snap(c - half);
      const hi = snap(c + half);
      const i0 = Math.floor(lo + 0.5);
      const i1 = Math.ceil(hi - 0.5);
      let first = 0;
      const taps: number[] = [];
      for (let i = i0; i <= i1; i++) {
        const ov = Math.min(hi, i + 0.5) - Math.max(lo, i - 0.5);
        if (ov <= SNAP * Math.max(1, half)) {
          if (taps.length === 0) continue;
          break;
        }
        if (taps.length === 0) first = i;
        taps.push(ov);
      }
      if (taps.length === 0) {
        // Degenerate (zero-width cell): nearest voxel.
        first = Math.round(c);
        taps.push(1);
      }
      const sum = taps.reduce((s, v) => s + v, 0);
      start[j] = first;
      count[j] = taps.length;
      for (const t of taps) w.push(t / sum);
    }
  }
  return { start, count, offset, weights: Float64Array.from(w) };
}

/** Target index range `[j0, j1]` whose footprints overlap source indices `[i0, i1]` (empty when j0 > j1). */
function targetRange(t: AxisTable, i0: number, i1: number): [number, number] {
  const n = t.start.length;
  // end[j] = start[j] + count[j] - 1 and start[j] are both non-decreasing in j.
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t.start[mid]! + t.count[mid]! - 1 >= i0) hi = mid;
    else lo = mid + 1;
  }
  const j0 = lo;
  lo = 0;
  hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t.start[mid]! <= i1) lo = mid + 1;
    else hi = mid;
  }
  return [j0, lo - 1];
}

// ---------------------------------------------------------------------------
// resampleGrid
// ---------------------------------------------------------------------------

/**
 * Resamples a sparse grid to a new resolution over a world box and builds a
 * NanoVDB FLOAT image (optionally quantized to Fp8/FpN). See the module doc for
 * the conventions, filters and sparsity rule.
 */
export function resampleGrid(source: ResampleSource, options: ResampleOptions = {}): ResampledGrid {
  const filter = options.filter ?? "box";
  if (filter !== "point" && filter !== "box" && filter !== "trilinear") {
    throw new Error(`resampleGrid: unknown filter "${String(filter)}"`);
  }
  const given = [options.resolution, options.voxelSize, options.dimensions].filter((v) => v !== undefined).length;
  if (given > 1) throw new Error("resampleGrid: pass at most one of resolution, voxelSize, dimensions");

  const field = toField(source, options.sourceGridName);
  const background = options.background ?? field.background;
  const srcVoxel: Vec3 = [Math.abs(field.scale[0]), Math.abs(field.scale[1]), Math.abs(field.scale[2])];

  // World box.
  let boxMin: Vec3;
  let boxMax: Vec3;
  if (options.bbox) {
    boxMin = [...options.bbox[0]];
    boxMax = [...options.bbox[1]];
  } else {
    const ib = activeIndexBox(field);
    if (!ib) {
      boxMin = [0, 0, 0];
      boxMax = [0, 0, 0];
    } else {
      boxMin = [0, 0, 0];
      boxMax = [0, 0, 0];
      for (let a = 0; a < 3; a++) {
        const p = (ib.min[a]! - 0.5) * field.scale[a]! + field.translation[a]!;
        const q = (ib.max[a]! + 0.5) * field.scale[a]! + field.translation[a]!;
        boxMin[a] = Math.min(p, q);
        boxMax[a] = Math.max(p, q);
      }
    }
  }
  const boxSize: Vec3 = [boxMax[0] - boxMin[0], boxMax[1] - boxMin[1], boxMax[2] - boxMin[2]];
  if (boxSize.some((s) => !(s >= 0) || !Number.isFinite(s))) {
    throw new Error(`resampleGrid: invalid bbox ${JSON.stringify([boxMin, boxMax])}`);
  }

  // Target lattice.
  let dims: Vec3;
  let size: Vec3; // world extent of the lattice (dims * voxel size)
  if (options.dimensions) {
    dims = [...options.dimensions];
    if (!dims.every((d) => Number.isInteger(d) && d >= 1)) {
      throw new Error(`resampleGrid: dimensions must be positive integers, got ${JSON.stringify(dims)}`);
    }
    if (boxSize.some((s) => s <= 0)) throw new Error("resampleGrid: dimensions need a box with positive size");
    size = [...boxSize];
  } else {
    let vs: Vec3;
    if (options.resolution !== undefined) {
      const r = options.resolution;
      if (!Number.isInteger(r) || r < 1) throw new Error(`resampleGrid: invalid resolution ${r}`);
      const longest = Math.max(...boxSize);
      const v = longest > 0 ? longest / r : srcVoxel[0];
      vs = [v, v, v];
    } else if (options.voxelSize !== undefined) {
      const v = options.voxelSize;
      vs = typeof v === "number" ? [v, v, v] : [...v];
    } else {
      vs = [...srcVoxel];
    }
    if (!vs.every((v) => Number.isFinite(v) && v > 0)) throw new Error(`resampleGrid: invalid voxel size ${JSON.stringify(vs)}`);
    dims = [0, 1, 2].map((a) => Math.max(1, Math.ceil(boxSize[a]! / vs[a]! - 1e-9))) as Vec3;
    size = [0, 1, 2].map((a) => dims[a]! * vs[a]!) as Vec3;
  }
  const voxel: Vec3 = [size[0] / dims[0], size[1] / dims[1], size[2] / dims[2]];
  const lattice: [Vec3, Vec3] = [[...boxMin], [boxMin[0] + size[0], boxMin[1] + size[1], boxMin[2] + size[2]]];
  const leaves = filterLeaves(field, filter, dims, boxMin, size, background, options.cutoff);

  const uniform = voxel[0] === voxel[1] && voxel[0] === voxel[2];
  const worldOrigin: Vec3 = [boxMin[0] + voxel[0] / 2, boxMin[1] + voxel[1] / 2, boxMin[2] + voxel[2] / 2];
  let built = buildFromLeavesDetailed(leaves, FLOAT_LEAF_CODEC, {
    voxelSize: uniform ? voxel[0] : voxel,
    worldOrigin,
    background,
    gridName: options.gridName ?? field.name ?? "grid",
    gridClass: field.gridClass,
    collapse: options.collapse ?? true,
  });
  if (options.quantize) built = quantizeDetailed(built, options.quantize, options.tolerance);
  return { ...built, dimensions: dims, bbox: lattice, filter, sourceVoxelSize: srcVoxel };
}

/**
 * Filters every target leaf whose footprint overlaps a source leaf or active
 * tile, on the lattice of `dims` voxels over `[boxMin, boxMin + size]` (voxel
 * `j` centred at `boxMin + ((j + 0.5) / dims) * size`). A zero `size` on an
 * axis samples the nearest source voxel there. Leaves are sorted by origin.
 */
function filterLeaves(
  field: SourceField,
  filter: ResampleFilter,
  dims: Vec3,
  boxMin: Vec3,
  size: Vec3,
  background: number,
  cutoff: number | undefined,
): LeafInput[] {
  const srcVoxel: Vec3 = [Math.abs(field.scale[0]), Math.abs(field.scale[1]), Math.abs(field.scale[2])];
  const voxel: Vec3 = [size[0] / dims[0], size[1] / dims[1], size[2] / dims[2]];
  // Per-axis tables: target index -> source taps.
  const tables = [0, 1, 2].map((a) =>
    axisTable(filter, dims[a]!, boxMin[a]!, size[a]!, field.translation[a]!, field.inverse[a]!, voxel[a]! / srcVoxel[a]!),
  ) as [AxisTable, AxisTable, AxisTable];

  // Candidate target leaves: those whose footprint overlaps a source leaf or active tile.
  const candidates = new Map<number, Vec3>();
  const mark = (lo: Vec3, hi: Vec3): void => {
    const r = [0, 1, 2].map((a) => targetRange(tables[a]!, lo[a]!, hi[a]!));
    if (r.some(([j0, j1]) => j0 > j1)) return;
    for (let bx = r[0]![0] >> 3; bx <= r[0]![1] >> 3; bx++) {
      for (let by = r[1]![0] >> 3; by <= r[1]![1] >> 3; by++) {
        for (let bz = r[2]![0] >> 3; bz <= r[2]![1] >> 3; bz++) {
          const key = blockKey(bx, by, bz);
          if (!candidates.has(key)) candidates.set(key, [bx * 8, by * 8, bz * 8]);
        }
      }
    }
  };
  for (const leaf of field.leaves.values()) {
    if (!leaf.active.some((v) => v !== 0)) continue;
    const o = leaf.origin;
    mark(o, [o[0] + 7, o[1] + 7, o[2] + 7]);
  }
  for (const t of field.tiles) {
    const o = t.origin;
    mark(o, [o[0] + t.dim - 1, o[1] + t.dim - 1, o[2] + t.dim - 1]);
  }

  // Filter each candidate leaf.
  const leaves: LeafInput[] = [];
  for (const origin of candidates.values()) {
    const leaf = filterLeaf(field, tables, dims, origin, background, cutoff);
    if (leaf) leaves.push(leaf);
  }
  leaves.sort((p, q) => p.origin[0] - q.origin[0] || p.origin[1] - q.origin[1] || p.origin[2] - q.origin[2]);
  return leaves;
}

export interface ResampleDenseOptions {
  /** World box `[min, max]`: target voxel `j` is centred at `min + ((j + 0.5) / dims) * (max - min)`. */
  bbox: [Vec3, Vec3];
  /** Target voxels per axis. */
  dimensions: Vec3;
  /** Default `'box'`. */
  filter?: ResampleFilter;
  /** Which grid of a `VdbFile`. Default: the first float grid. */
  sourceGridName?: string;
}

/**
 * {@link resampleGrid}'s filters onto a dense lattice (the `.wvdb` bake's,
 * K4.13): `values` x fastest (`x + y*X + z*X*Y`), `active` 1 where the target
 * voxel's footprint touches an active source voxel. Voxels in no emitted leaf
 * (the footprint touches no active source voxel) hold the source background.
 * Unlike `resampleGrid`, the box may have zero size on an axis (one voxel
 * thick); that axis samples the nearest source voxel.
 */
export function resampleDense(
  source: ResampleSource,
  options: ResampleDenseOptions,
): { values: Float32Array; active: Uint8Array; background: number } {
  const filter = options.filter ?? "box";
  if (filter !== "point" && filter !== "box" && filter !== "trilinear") {
    throw new Error(`resampleDense: unknown filter "${String(filter)}"`);
  }
  const dims: Vec3 = [...options.dimensions];
  if (!dims.every((d) => Number.isInteger(d) && d >= 1)) {
    throw new Error(`resampleDense: dimensions must be positive integers, got ${JSON.stringify(dims)}`);
  }
  const boxMin: Vec3 = [...options.bbox[0]];
  const size: Vec3 = [0, 1, 2].map((a) => options.bbox[1][a]! - boxMin[a]!) as Vec3;
  if (!size.every((v) => v >= 0 && Number.isFinite(v)) || !boxMin.every(Number.isFinite)) {
    throw new Error(`resampleDense: invalid bbox ${JSON.stringify(options.bbox)}`);
  }
  const field = toField(source, options.sourceGridName);
  const background = field.background;
  const [X, Y, Z] = dims;
  const values = new Float32Array(X * Y * Z).fill(background);
  const active = new Uint8Array(X * Y * Z);
  for (const leaf of filterLeaves(field, filter, dims, boxMin, size, background, undefined)) {
    const [ox, oy, oz] = leaf.origin;
    for (let i = 0; i < 8 && ox + i < X; i++) {
      for (let j = 0; j < 8 && oy + j < Y; j++) {
        for (let k = 0; k < 8 && oz + k < Z; k++) {
          const n = (i << 6) | (j << 3) | k;
          const d = ox + i + (oy + j) * X + (oz + k) * X * Y;
          values[d] = leaf.values[n]!;
          active[d] = (leaf.valueMask[n >>> 5]! >>> (n & 31)) & 1;
        }
      }
    }
  }
  return { values, active, background };
}

/** Gathers the source block a target leaf needs, runs the three 1D passes, and returns the leaf (or null if no voxel is active). */
function filterLeaf(
  field: SourceField,
  tables: [AxisTable, AxisTable, AxisTable],
  dims: Vec3,
  origin: Vec3,
  background: number,
  cutoff: number | undefined,
): LeafInput | null {
  // Valid target range on each axis inside this leaf, and the source range it reads.
  const jn: number[] = [];
  const s0: number[] = [];
  const sn: number[] = [];
  for (let a = 0; a < 3; a++) {
    const t = tables[a]!;
    const j0 = origin[a]!;
    const j1 = Math.min(origin[a]! + 7, dims[a]! - 1);
    if (j1 < j0) return null;
    jn.push(j1 - j0 + 1);
    const lo = t.start[j0]!;
    const hi = t.start[j1]! + t.count[j1]! - 1;
    s0.push(lo);
    sn.push(hi - lo + 1);
  }
  const [Sx, Sy, Sz] = sn as Vec3;
  const vals = new Float64Array(Sx * Sy * Sz);
  const act = new Uint8Array(Sx * Sy * Sz);

  // Gather, one source 8^3 block at a time.
  const bx0 = Math.floor(s0[0]! / 8);
  const by0 = Math.floor(s0[1]! / 8);
  const bz0 = Math.floor(s0[2]! / 8);
  const bx1 = Math.floor((s0[0]! + Sx - 1) / 8);
  const by1 = Math.floor((s0[1]! + Sy - 1) / 8);
  const bz1 = Math.floor((s0[2]! + Sz - 1) / 8);
  for (let bx = bx0; bx <= bx1; bx++) {
    const x0 = Math.max(bx * 8, s0[0]!);
    const x1 = Math.min(bx * 8 + 7, s0[0]! + Sx - 1);
    for (let by = by0; by <= by1; by++) {
      const y0 = Math.max(by * 8, s0[1]!);
      const y1 = Math.min(by * 8 + 7, s0[1]! + Sy - 1);
      for (let bz = bz0; bz <= bz1; bz++) {
        const z0 = Math.max(bz * 8, s0[2]!);
        const z1 = Math.min(bz * 8 + 7, s0[2]! + Sz - 1);
        const leaf = field.leaves.get(blockKey(bx, by, bz));
        if (leaf) {
          for (let x = x0; x <= x1; x++) {
            for (let y = y0; y <= y1; y++) {
              let d = ((x - s0[0]!) * Sy + (y - s0[1]!)) * Sz + (z0 - s0[2]!);
              let n = ((x - bx * 8) << 6) | ((y - by * 8) << 3) | (z0 - bz * 8);
              for (let z = z0; z <= z1; z++, d++, n++) {
                vals[d] = leaf.values[n]!;
                act[d] = leaf.active[n]!;
              }
            }
          }
        } else {
          const { value, active } = field.blockValue(bx, by, bz);
          const a = active ? 1 : 0;
          for (let x = x0; x <= x1; x++) {
            for (let y = y0; y <= y1; y++) {
              let d = ((x - s0[0]!) * Sy + (y - s0[1]!)) * Sz + (z0 - s0[2]!);
              for (let z = z0; z <= z1; z++, d++) {
                vals[d] = value;
                act[d] = a;
              }
            }
          }
        }
      }
    }
  }

  const [nx, ny, nz] = jn as Vec3;
  const [tx, ty, tz] = tables;
  // Pass x: [Sx][Sy][Sz] -> [nx][Sy][Sz]
  const v1 = new Float64Array(nx * Sy * Sz);
  const a1 = new Uint8Array(nx * Sy * Sz);
  for (let i = 0; i < nx; i++) {
    const j = origin[0] + i;
    const st = tx.start[j]! - s0[0]!;
    const cnt = tx.count[j]!;
    const off = tx.offset[j]!;
    for (let yz = 0; yz < Sy * Sz; yz++) {
      let sum = 0;
      let any = 0;
      for (let k = 0; k < cnt; k++) {
        const d = (st + k) * Sy * Sz + yz;
        sum += tx.weights[off + k]! * vals[d]!;
        any |= act[d]!;
      }
      v1[i * Sy * Sz + yz] = sum;
      a1[i * Sy * Sz + yz] = any;
    }
  }
  // Pass y: [nx][Sy][Sz] -> [nx][ny][Sz]
  const v2 = new Float64Array(nx * ny * Sz);
  const a2 = new Uint8Array(nx * ny * Sz);
  for (let i = 0; i < nx; i++) {
    for (let jy = 0; jy < ny; jy++) {
      const j = origin[1] + jy;
      const st = ty.start[j]! - s0[1]!;
      const cnt = ty.count[j]!;
      const off = ty.offset[j]!;
      for (let z = 0; z < Sz; z++) {
        let sum = 0;
        let any = 0;
        for (let k = 0; k < cnt; k++) {
          const d = (i * Sy + st + k) * Sz + z;
          sum += ty.weights[off + k]! * v1[d]!;
          any |= a1[d]!;
        }
        v2[(i * ny + jy) * Sz + z] = sum;
        a2[(i * ny + jy) * Sz + z] = any;
      }
    }
  }
  // Pass z -> leaf
  const values = new Float32Array(512).fill(background);
  const valueMask = new Uint32Array(16);
  let anyActive = false;
  for (let i = 0; i < nx; i++) {
    for (let jy = 0; jy < ny; jy++) {
      const row = (i * ny + jy) * Sz;
      for (let jz = 0; jz < nz; jz++) {
        const j = origin[2] + jz;
        const st = tz.start[j]! - s0[2]!;
        const cnt = tz.count[j]!;
        const off = tz.offset[j]!;
        let sum = 0;
        let any = 0;
        for (let k = 0; k < cnt; k++) {
          sum += tz.weights[off + k]! * v2[row + st + k]!;
          any |= a2[row + st + k]!;
        }
        const n = (i << 6) | (jy << 3) | jz;
        values[n] = sum;
        const v = values[n]!;
        if (any && (cutoff === undefined || Math.abs(v - background) > cutoff)) {
          valueMask[n >>> 5]! |= 1 << (n & 31);
          anyActive = true;
        }
      }
    }
  }
  return anyActive ? { origin: [...origin], values, valueMask } : null;
}

// ---------------------------------------------------------------------------
// Error metric (research/INDEX comparison)
// ---------------------------------------------------------------------------

export interface ResampleErrorOptions {
  /** Use every `stride`-th active source voxel (leaf order). Default 7, as in the comparison report. */
  stride?: number;
  /** Which grid of a `VdbFile`. Default: the first float grid. */
  sourceGridName?: string;
}

export interface ResampleErrorResult {
  /** Mean |target − source| over the sampled active source voxels. */
  meanAbsError: number;
  /** Mean source value over the same voxels. */
  meanValue: number;
  /** `100 * meanAbsError / meanValue`. */
  percent: number;
  /** Voxels compared. */
  count: number;
}

/**
 * The error metric of the mjurczyk comparison (research/INDEX): mean absolute
 * difference over every `stride`-th active source voxel, as a percentage of the
 * mean source value there. The target (any NanoVDB FLOAT/Fp8/FpN image with an
 * axis-aligned Map) is read back with `sampleTrilinear` at each source voxel
 * centre. Active tiles are not sampled (leaves only).
 */
export function resampleError(
  source: ResampleSource,
  target: Uint32Array,
  options: ResampleErrorOptions = {},
): ResampleErrorResult {
  const stride = options.stride ?? 7;
  const field = toField(source, options.sourceGridName);
  const view = new DataView(target.buffer, target.byteOffset, target.byteLength);
  const tInv = [0, 4, 8].map((i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_INVMATD + i * 8, true));
  const tVec = [0, 1, 2].map((i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_VECD + i * 8, true));
  const sorted = [...field.leaves.values()].sort(
    (p, q) => p.origin[0] - q.origin[0] || p.origin[1] - q.origin[1] || p.origin[2] - q.origin[2],
  );
  let k = 0;
  let count = 0;
  let sumAbs = 0;
  let sumVal = 0;
  const xyz: Vec3 = [0, 0, 0];
  for (const leaf of sorted) {
    for (let n = 0; n < 512; n++) {
      if (!leaf.active[n]) continue;
      if (k++ % stride !== 0) continue;
      const ijk = [leaf.origin[0] + (n >> 6), leaf.origin[1] + ((n >> 3) & 7), leaf.origin[2] + (n & 7)];
      for (let a = 0; a < 3; a++) {
        const world = ijk[a]! * field.scale[a]! + field.translation[a]!;
        xyz[a] = (world - tVec[a]!) * tInv[a]!;
      }
      const src = leaf.values[n]!;
      sumAbs += Math.abs(sampleTrilinear(target, xyz) - src);
      sumVal += src;
      count++;
    }
  }
  const meanAbsError = count ? sumAbs / count : 0;
  const meanValue = count ? sumVal / count : 0;
  return { meanAbsError, meanValue, percent: meanValue !== 0 ? (100 * meanAbsError) / meanValue : 0, count };
}
