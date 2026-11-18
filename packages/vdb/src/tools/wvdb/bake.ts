/**
 * bake.ts — resample sparse grids (a parsed `.vdb`, or a NanoVDB grid image
 * through the CPU oracle) to dense levels and write a `.wvdb`.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345:
 * `src/wvdb/bake.js` (getWVDBBox, getFinestDimensions, forEachTexel,
 * sampleScalarGrid, the level-set band limit) and `src/wvdb/cli.js` (the
 * `source` member). His sampling is reproduced, including its arithmetic order:
 *  - one shared world box for all grids: the union of the grids' active-voxel
 *    index boxes (voxel centres, so `max` is the last voxel, not one past it)
 *    mapped to world space, unless `bbox` is given;
 *  - the finest LOD has `resolution` voxels on the longest box side, the other
 *    sides `max(1, round(side * resolution / longest))` (§3.2.1);
 *  - **point sampling, no prefilter** (§3.2.2): world position of voxel centre
 *    `min + ((i + 0.5) / X) * size`, mapped to index space with the inverse
 *    map, rounded with `Math.round` (halves toward +infinity) and read with
 *    the grid's own value lookup (inactive values included). A source much
 *    finer than the target resolution aliases; that is his behaviour and the
 *    default here (`sampling: 'nearest'`).
 *
 * Options beyond his CLI (K4.13):
 *  - `sampling: 'box'` prefilters: each texel is the area-weighted mean of the
 *    source voxels its cell covers, on the same lattice (`vdbResampleDense`, the
 *    K4.4 `vdbResampleGrid` box filter). Mass is conserved and thin features fade
 *    instead of flickering in and out;
 *  - `levelSetSampler: 'surface-mean'` samples level sets with his CLI's
 *    `sampleLevelSet` (`level-set.ts`), which reproduces his level-set files.
 *
 * Differences from his CLI (also in docs/handoffs/K4-3.md):
 *  - by default (`levelSetSampler: 'spec'`) level sets are point-sampled and
 *    then encoded by the §8 rule (active: clamp to `±(b - 2b/maxCode)`;
 *    inactive: `±b` by sign). His CLI uses `sampleLevelSet` (a surface-aware
 *    mean) instead, so level-set bytes differ (DIVERGES V17);
 *  - `dimensions` can override the finest LOD size (his pre-0.11 bakes used
 *    `resolution`³ for every box, e.g. the committed `smoke.wvdb` and
 *    `bunny_cloud.wvdb`; this reproduces them).
 */

import { VDBToolsError, VDBUnsupportedError } from "../errors.js";
import { vdbReadValueCpu } from "../cpu/read-value.js";
import { vdbResampleDense } from "../nanovdb/resample.js";
import { gridTypeConstantsForId } from "../cpu/stride-tables.js";
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
  ROOT_OFF_BBOX_MAX,
  ROOT_OFF_BBOX_MIN,
  TREE_OFF_NODE_OFFSET_ROOT,
} from "../nanovdb/bytes.js";
import { parseVdb } from "../vdb/parse.js";
import type { VdbFile, VdbGrid } from "../vdb/types.js";
import type { WvdbBits, WvdbBox, WvdbSource, WvdbVec3 } from "./format.js";
import { sampleWvdbLevelSetSurfaceMean, type WvdbIndexAccess } from "./level-set.js";
import { sha256Hex } from "./sha256.js";
import { writeWvdb, type WriteWvdbOptions, type WriteWvdbResult, type WvdbVolumeGridInput } from "./write.js";

/** A grid the baker can point-sample in world space. */
export interface WvdbSampleSource {
  name: string;
  /** `.wvdb` class string: "fog volume", "level set", "staggered" or "unknown". */
  class: string;
  background: number;
  /** World box of the active voxels (voxel centres), or null when the grid has none. */
  bbox: WvdbBox | null;
  /** The grid value (and active state) at the voxel nearest to a world position. */
  sample(world: WvdbVec3): { value: number; active: boolean };
  /** Index-space access, for `levelSetSampler: 'surface-mean'`. The built-in sources have it. */
  index?: WvdbIndexAccess;
  /** The sparse grid behind the source, for `sampling: 'box'`. The built-in sources have it. */
  grid?: Uint32Array | VdbGrid;
}

export interface BakeWvdbOptions extends Omit<WriteWvdbOptions, "resolution" | "bakeBBox"> {
  /** Voxels on the longest box side of the finest LOD. Default 128. */
  resolution?: number;
  /** One world box for all grids (a sequence box, §10; recorded as `bake.bbox`). Default: union of the grids' boxes. */
  bbox?: WvdbBox | null;
  /** Override the finest LOD dimensions (default from `resolution` and the box). */
  dimensions?: WvdbVec3;
  /**
   * Grids to keep, by name; also recorded as `bake.grids`. Default null (all).
   */
  gridNames?: string[] | null;
  /**
   * Resampling filter. `"nearest"` (default): his point sampling. `"box"`: the
   * area-weighted mean over each texel's cell (K4.4's box filter; conserves
   * mass, no aliasing when the source is finer than the texels). Needs a source
   * with `grid` (the built-in sources).
   */
  sampling?: WvdbSampling;
  /**
   * How level sets are sampled. `"spec"` (default): `sampling`, then the spec
   * §8 rule by active state. `"surface-mean"`: his CLI's `sampleLevelSet`
   * (trilinear near the surface, a 4x4x4 mean in texels that cross it), which
   * reproduces his level-set `.wvdb` files; it ignores `sampling`.
   */
  levelSetSampler?: WvdbLevelSetSampler;
}

export type WvdbSampling = "nearest" | "box";
export type WvdbLevelSetSampler = "spec" | "surface-mean";

function unionBox(boxes: WvdbBox[]): WvdbBox | null {
  if (!boxes.length) return null;
  const min = [...boxes[0]![0]] as WvdbVec3;
  const max = [...boxes[0]![1]] as WvdbVec3;
  for (const [lo, hi] of boxes.slice(1)) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a]!, lo[a]!);
      max[a] = Math.max(max[a]!, hi[a]!);
    }
  }
  return [min, max];
}

/** §3.2.1, his `getFinestDimensions`. */
export function wvdbFinestDimensions(bbox: WvdbBox | null, resolution: number): WvdbVec3 {
  const size = bbox ? ([0, 1, 2].map((a) => bbox[1][a]! - bbox[0][a]!) as WvdbVec3) : ([0, 0, 0] as WvdbVec3);
  const longest = Math.max(...size);
  if (!(longest > 0)) return [resolution, resolution, resolution];
  return size.map((side) => Math.max(1, Math.round((side * resolution) / longest))) as WvdbVec3;
}

/** Point-samples one source over the box (his `forEachTexel` + `sampleScalarGrid`). */
export function sampleWvdbDense(
  source: WvdbSampleSource,
  bbox: WvdbBox | null,
  dimensions: WvdbVec3,
  bits: WvdbBits = 8,
): Float32Array {
  const [w, h, d] = dimensions;
  const values = new Float32Array(w * h * d);
  const background = source.background;
  if (!source.bbox || !bbox) return values.fill(background);
  const min = bbox[0];
  const size = [0, 1, 2].map((a) => bbox[1][a]! - min[a]!);
  const isLevelSet = source.class === "level set";
  const band = background - (2 * background) / ((1 << bits) - 1);
  const world: WvdbVec3 = [0, 0, 0];
  let o = 0;
  for (let z = 0; z < d; z++) {
    world[2] = ((z + 0.5) / d) * size[2]! + min[2];
    for (let y = 0; y < h; y++) {
      world[1] = ((y + 0.5) / h) * size[1]! + min[1];
      for (let x = 0; x < w; x++) {
        world[0] = ((x + 0.5) / w) * size[0]! + min[0];
        const { value, active } = source.sample(world);
        let v = Number.isFinite(value) ? value : background;
        if (isLevelSet) v = active ? Math.min(Math.max(v, -band), band) : v < 0 ? -background : background;
        values[o++] = v;
      }
    }
  }
  return values;
}

/**
 * Box-filters one source over the box (`sampling: 'box'`): `vdbResampleDense` on
 * his lattice, then the level-set rule of {@link sampleWvdbDense} by the
 * filtered active state.
 */
export function sampleWvdbDenseBox(
  source: WvdbSampleSource,
  bbox: WvdbBox | null,
  dimensions: WvdbVec3,
  bits: WvdbBits = 8,
): Float32Array {
  const background = source.background;
  if (!source.bbox || !bbox) return new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]).fill(background);
  if (!source.grid) {
    throw new VDBToolsError("VDB_BAD_OPTION", `bakeWvdb: sampling "box" needs the source grid; "${source.name}" has none (use wvdbSourceFromVdbGrid or wvdbSourceFromNvdb)`);
  }
  const { values, active } = vdbResampleDense(source.grid, { bbox, dimensions, filter: "box" });
  if (source.class === "level set") {
    const band = background - (2 * background) / ((1 << bits) - 1);
    for (let i = 0; i < values.length; i++) {
      const v = Number.isFinite(values[i]!) ? values[i]! : background;
      values[i] = active[i] ? Math.min(Math.max(v, -band), band) : v < 0 ? -background : background;
    }
  } else {
    for (let i = 0; i < values.length; i++) if (!Number.isFinite(values[i]!)) values[i] = background;
  }
  return values;
}

/** Bakes sources to a `.wvdb` with his CLI defaults (128, 3 LODs, 8 bits, sparse, 1024 bricks per page). */
export function bakeWvdb(sources: readonly WvdbSampleSource[], options: BakeWvdbOptions = {}): WriteWvdbResult {
  const {
    resolution = 128,
    bbox: bakeBBox = null,
    dimensions,
    gridNames = null,
    sampling = "nearest",
    levelSetSampler = "spec",
    ...write
  } = options;
  if (sampling !== "nearest" && sampling !== "box") throw new VDBToolsError("VDB_BAD_OPTION", `bakeWvdb: unknown sampling "${String(sampling)}"`);
  if (levelSetSampler !== "spec" && levelSetSampler !== "surface-mean") {
    throw new VDBToolsError("VDB_BAD_OPTION", `bakeWvdb: unknown levelSetSampler "${String(levelSetSampler)}"`);
  }
  if (!Number.isInteger(resolution) || resolution < 1) throw new VDBToolsError("VDB_BAD_OPTION", `bakeWvdb: invalid resolution ${resolution}`);
  const selected = gridNames ? sources.filter((s) => gridNames.includes(s.name)) : [...sources];
  const box = bakeBBox ?? unionBox(selected.flatMap((s) => (s.bbox ? [s.bbox] : [])));
  const dims = dimensions ?? wvdbFinestDimensions(box, resolution);
  const bits = write.bits ?? 8;
  const grids: WvdbVolumeGridInput[] = selected.map((s) => ({
    name: s.name,
    class: s.class,
    background: s.background,
    bbox: box,
    finest: {
      dimensions: dims,
      values:
        s.class === "level set" && levelSetSampler === "surface-mean"
          ? sampleWvdbLevelSetSurfaceMean(s, box, dims, bits)
          : sampling === "box"
            ? sampleWvdbDenseBox(s, box, dims, bits)
            : sampleWvdbDense(s, box, dims, bits),
    },
  }));
  return writeWvdb(grids, { ...write, resolution, gridNames, bakeBBox });
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/** A parsed `.vdb` grid (axis-aligned map) as a sample source. */
export function wvdbSourceFromVdbGrid(grid: VdbGrid): WvdbSampleSource {
  if (grid.valueType !== "float") {
    throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", 
      `wvdbSourceFromVdbGrid: grid "${grid.name}" is a ${grid.valueType} grid; only float grids can be baked ` +
        `(vdbToWvdb skips the others)`,
    );
  }
  const m = grid.transform.matrix;
  if (m[1] !== 0 || m[2] !== 0 || m[4] !== 0 || m[6] !== 0 || m[8] !== 0 || m[9] !== 0) {
    throw new VDBUnsupportedError(`wvdbSourceFromVdbGrid: grid "${grid.name}" has a rotated map; only axis-aligned maps are supported`);
  }
  const scale = [m[0]!, m[5]!, m[10]!];
  const translation = [m[3]!, m[7]!, m[11]!];
  const inverse = scale.map((s) => 1 / s);
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const leaf of grid.iterLeaves()) {
    for (let offset = 0; offset < 512; offset++) {
      if (((leaf.valueMask[offset >>> 5]! >>> (offset & 31)) & 1) === 0) continue;
      const ijk = [leaf.origin[0] + (offset >> 6), leaf.origin[1] + ((offset >> 3) & 7), leaf.origin[2] + (offset & 7)];
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a]!, ijk[a]!);
        hi[a] = Math.max(hi[a]!, ijk[a]!);
      }
    }
  }
  let bbox: WvdbBox | null = null;
  if (lo[0]! <= hi[0]!) {
    const a = [0, 1, 2].map((i) => lo[i]! * scale[i]! + translation[i]!);
    const b = [0, 1, 2].map((i) => hi[i]! * scale[i]! + translation[i]!);
    bbox = [a.map((v, i) => Math.min(v, b[i]!)) as WvdbVec3, a.map((v, i) => Math.max(v, b[i]!)) as WvdbVec3];
  }
  const cls = grid.metadata["class"];
  const ijk: [number, number, number] = [0, 0, 0];
  return {
    name: grid.name,
    class: typeof cls === "string" ? cls : "unknown",
    background: grid.background,
    bbox,
    sample(world) {
      for (let a = 0; a < 3; a++) ijk[a] = Math.round((world[a]! - translation[a]!) * inverse[a]!);
      return grid.readValue(ijk);
    },
    index: {
      // His GridTransform scale-translate maps: (v - translation) * scaleInverse.
      toIndex(world, out) {
        for (let a = 0; a < 3; a++) out[a] = (world[a]! - translation[a]!) * inverse[a]!;
        return out;
      },
      toIndexVector(v, out) {
        for (let a = 0; a < 3; a++) out[a] = v[a]! * inverse[a]!;
        return out;
      },
      voxel: (p) => grid.readValue(p),
    },
    grid,
  };
}

const NANO_CLASS: Record<number, string> = { 1: "level set", 2: "fog volume", 3: "staggered" };

/** A NanoVDB FLOAT/Fp8/FpN grid image as a sample source, read through the CPU oracle (`readValue`). */
export function wvdbSourceFromNvdb(gridWords: Uint32Array): WvdbSampleSource {
  const view = new DataView(gridWords.buffer, gridWords.byteOffset, gridWords.byteLength);
  const nameBytes = new Uint8Array(gridWords.buffer, gridWords.byteOffset + GRID_OFF_GRID_NAME, GRID_NAME_MAX);
  const nul = nameBytes.indexOf(0);
  const name = new TextDecoder().decode(nameBytes.subarray(0, nul < 0 ? GRID_NAME_MAX : nul));
  const gridType = view.getUint32(GRID_OFF_GRID_TYPE, true);
  const cls = NANO_CLASS[view.getUint32(GRID_OFF_GRID_CLASS, true)] ?? "unknown";
  const mat = Array.from({ length: 9 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_MATD + i * 8, true));
  const inv = Array.from({ length: 9 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_INVMATD + i * 8, true));
  const vec = Array.from({ length: 3 }, (_, i) => view.getFloat64(GRID_OFF_MAP + MAP_OFF_VECD + i * 8, true));
  const root = GRID_SIZE + Number(view.getBigUint64(GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT, true));
  const lo = [0, 1, 2].map((i) => view.getInt32(root + ROOT_OFF_BBOX_MIN + i * 4, true));
  const hi = [0, 1, 2].map((i) => view.getInt32(root + ROOT_OFF_BBOX_MAX + i * 4, true));
  const background = view.getFloat32(root + gridTypeConstantsForId(gridType).root_off_background, true);

  const toWorld = (p: number[]): number[] =>
    [0, 1, 2].map((r) => mat[r * 3]! * p[0]! + mat[r * 3 + 1]! * p[1]! + mat[r * 3 + 2]! * p[2]! + vec[r]!);
  let bbox: WvdbBox | null = null;
  if (lo[0]! <= hi[0]! && lo[1]! <= hi[1]! && lo[2]! <= hi[2]!) {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let c = 0; c < 8; c++) {
      const w = toWorld([c & 1 ? hi[0]! : lo[0]!, c & 2 ? hi[1]! : lo[1]!, c & 4 ? hi[2]! : lo[2]!]);
      for (let a = 0; a < 3; a++) {
        min[a] = Math.min(min[a]!, w[a]!);
        max[a] = Math.max(max[a]!, w[a]!);
      }
    }
    bbox = [min as WvdbVec3, max as WvdbVec3];
  }
  const ijk: [number, number, number] = [0, 0, 0];
  return {
    name,
    class: cls,
    background,
    bbox,
    sample(world) {
      const p = [world[0] - vec[0]!, world[1] - vec[1]!, world[2] - vec[2]!];
      for (let r = 0; r < 3; r++) ijk[r] = Math.round(inv[r * 3]! * p[0]! + inv[r * 3 + 1]! * p[1]! + inv[r * 3 + 2]! * p[2]!);
      return vdbReadValueCpu(gridWords, ijk);
    },
    index: {
      toIndex(world, out) {
        const p0 = world[0] - vec[0]!;
        const p1 = world[1] - vec[1]!;
        const p2 = world[2] - vec[2]!;
        for (let r = 0; r < 3; r++) out[r] = inv[r * 3]! * p0 + inv[r * 3 + 1]! * p1 + inv[r * 3 + 2]! * p2;
        return out;
      },
      toIndexVector(v, out) {
        const [v0, v1, v2] = v;
        for (let r = 0; r < 3; r++) out[r] = inv[r * 3]! * v0 + inv[r * 3 + 1]! * v1 + inv[r * 3 + 2]! * v2;
        return out;
      },
      voxel: (p) => vdbReadValueCpu(gridWords, p),
    },
    grid: gridWords,
  };
}

export interface VdbToWvdbOptions extends BakeWvdbOptions {
  /** `source.file` (the name without directories). Default "". */
  fileName?: string;
}

/**
 * `.vdb` → `.wvdb`, like his `wvdb` CLI with the same options. Given the file
 * bytes, `source.sha256` is their digest. Only float grids are baked: vector
 * and point data grids are skipped (his CLI bakes vectors as 3 channels and
 * point data grids as points grids; not ported yet).
 */
export function vdbToWvdb(input: ArrayBuffer | Uint8Array | VdbFile, options: VdbToWvdbOptions = {}): WriteWvdbResult {
  const { fileName = "", ...bake } = options;
  let file: VdbFile;
  let source: WvdbSource | undefined = bake.source;
  if (input instanceof ArrayBuffer || input instanceof Uint8Array) {
    const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    const buffer = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength && bytes.buffer instanceof ArrayBuffer
      ? bytes.buffer
      : bytes.slice().buffer;
    file = parseVdb(buffer);
    source ??= { file: fileName, sha256: sha256Hex(bytes) };
  } else {
    file = input;
  }
  const floatGrids = file.grids.filter((g) => g.valueType === "float");
  if (floatGrids.length === 0) throw new VDBToolsError("VDB_UNSUPPORTED_GRID_TYPE", "vdbToWvdb: the file has no float grids to bake");
  return bakeWvdb(floatGrids.map(wvdbSourceFromVdbGrid), { ...bake, ...(source ? { source } : {}) });
}

/** NanoVDB grid image(s) → `.wvdb`, sampled through the CPU oracle. */
export function nvdbToWvdb(images: Uint32Array | readonly Uint32Array[], options: BakeWvdbOptions = {}): WriteWvdbResult {
  const list = images instanceof Uint32Array ? [images] : images;
  return bakeWvdb(list.map(wvdbSourceFromNvdb), options);
}
