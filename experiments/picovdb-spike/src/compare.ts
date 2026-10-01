/**
 * Oracle (a): value round-trip against our proven CPU NanoVDB reader.
 *
 * For a coordinate set, compare PicoVDB reads (via the WGSL-mirroring
 * {@link PicoVDBAccessor}) against `readValue` on the source FLOAT NanoVDB
 * grid, after applying PicoVDB's documented value mapping:
 *
 * - active voxel/tile -> value == source (fog), or source / voxelSize (SDF,
 *   f32 division); u8 SDF within half a quantization step (6/255/2), clamped
 *   to [-3, 3];
 * - inactive -> PicoVDB keeps no value, only "outside" (background, slot 0) or
 *   "inside" (slot 1: 1.0 for fog, -background for SDF), chosen by the
 *   source value's sign. `inactiveLossy` counts inactive coords whose
 *   *source* value differs from that collapsed value (information PicoVDB
 *   drops by design), which is separate from a reader/converter mismatch.
 */

import { readValue } from "../../../packages/nanovdb-wgsl/src/cpu/read-value.js";
import { defineNumber, gridTypeConstantsFor } from "../../../packages/nanovdb-wgsl/src/cpu/stride-tables.js";
import { GRID_TYPE_FOG_FLOAT, GRID_TYPE_SDF_UINT8 } from "./convert.js";
import { PicoVDBAccessor, PicoVDBView } from "./reader.js";

export type Coord = [number, number, number];

export interface CompareStats {
  checked: number;
  active: number;
  /** Coords whose PicoVDB value disagrees with the mapped source value beyond tolerance. */
  valueMismatch: number;
  /** Coords whose PicoVDB active-ness (index >= 2) disagrees with the source active flag. */
  activeMismatch: number;
  /** Max |pico - mapped source| over all checked coords. */
  maxAbsDelta: number;
  /** Inactive coords whose source value is not the background/inside value PicoVDB collapses it to. */
  inactiveLossy: number;
  firstMismatches: { ijk: Coord; pico: number; expected: number; active: boolean; level: number }[];
}

const GT = gridTypeConstantsFor("FLOAT");
const GRID_SIZE = defineNumber("PNANOVDB_GRID_SIZE");
const TREE_OFF_NODE_OFFSET_ROOT = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");
const ROOT_OFF_TABLE_SIZE = defineNumber("PNANOVDB_ROOT_OFF_TABLE_SIZE");
const ROOT_TILE_OFF_KEY = defineNumber("PNANOVDB_ROOT_TILE_OFF_KEY");
const ROOT_TILE_OFF_CHILD = defineNumber("PNANOVDB_ROOT_TILE_OFF_CHILD");
const ROOT_TILE_OFF_STATE = defineNumber("PNANOVDB_ROOT_TILE_OFF_STATE");
const UPPER_OFF_VALUE_MASK = defineNumber("PNANOVDB_UPPER_OFF_VALUE_MASK");
const UPPER_OFF_CHILD_MASK = defineNumber("PNANOVDB_UPPER_OFF_CHILD_MASK");
const LOWER_OFF_VALUE_MASK = defineNumber("PNANOVDB_LOWER_OFF_VALUE_MASK");
const LOWER_OFF_CHILD_MASK = defineNumber("PNANOVDB_LOWER_OFF_CHILD_MASK");
const LEAF_OFF_VALUE_MASK = defineNumber("PNANOVDB_LEAF_OFF_VALUE_MASK");

function i64(w: Uint32Array, addr: number): number {
  return (w[(addr >>> 2) + 1]! | 0) * 0x100000000 + w[addr >>> 2]!;
}
function bit(w: Uint32Array, maskAddr: number, n: number): boolean {
  return ((w[(maskAddr >>> 2) + (n >>> 5)]! >>> (n & 31)) & 1) === 1;
}

export interface ActiveCoords {
  /** Every active leaf voxel. */
  leafVoxels: Coord[];
  /** For each active internal/root tile: its 8 corner voxels + centre. */
  tileSamples: Coord[];
  activeTiles: { root: number; upper: number; lower: number };
}

/**
 * Enumerates active voxels of a FLOAT NanoVDB image by walking its tree:
 * every active leaf voxel, plus corner and centre samples of every active tile.
 */
export function enumerateActive(image: Uint32Array): ActiveCoords {
  const w = image;
  const root = GRID_SIZE + i64(w, GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT);
  const leafVoxels: Coord[] = [];
  const tileSamples: Coord[] = [];
  const activeTiles = { root: 0, upper: 0, lower: 0 };
  const pushTile = (o: Coord, dim: number) => {
    for (let c = 0; c < 8; c++) {
      tileSamples.push([o[0] + (c & 4 ? dim - 1 : 0), o[1] + (c & 2 ? dim - 1 : 0), o[2] + (c & 1 ? dim - 1 : 0)]);
    }
    tileSamples.push([o[0] + (dim >> 1), o[1] + (dim >> 1), o[2] + (dim >> 1)]);
  };
  const tiles = w[(root + ROOT_OFF_TABLE_SIZE) >>> 2]!;
  for (let t = 0; t < tiles; t++) {
    const tile = root + GT.root_size + t * GT.root_tile_size;
    const k0 = w[(tile + ROOT_TILE_OFF_KEY) >>> 2]!;
    const k1 = w[(tile + ROOT_TILE_OFF_KEY + 4) >>> 2]!;
    const ku = k0 & 0x1fffff;
    const ju = ((k0 >>> 21) | ((k1 & 0x3ff) << 11)) >>> 0;
    const iu = k1 >>> 10;
    const uo: Coord = [(iu << 12) | 0, (ju << 12) | 0, (ku << 12) | 0];
    const child = i64(w, tile + ROOT_TILE_OFF_CHILD);
    if (child === 0) {
      if (w[(tile + ROOT_TILE_OFF_STATE) >>> 2]) {
        activeTiles.root++;
        pushTile(uo, 4096);
      }
      continue;
    }
    const upper = root + child;
    for (let n = 0; n < 32768; n++) {
      const lo: Coord = [uo[0] + ((n >>> 10) & 31) * 128, uo[1] + ((n >>> 5) & 31) * 128, uo[2] + (n & 31) * 128];
      if (!bit(w, upper + UPPER_OFF_CHILD_MASK, n)) {
        if (bit(w, upper + UPPER_OFF_VALUE_MASK, n)) {
          activeTiles.upper++;
          pushTile(lo, 128);
        }
        continue;
      }
      const lower = upper + i64(w, upper + GT.upper_off_table + GT.table_stride * n);
      for (let m = 0; m < 4096; m++) {
        const leo: Coord = [lo[0] + ((m >>> 8) & 15) * 8, lo[1] + ((m >>> 4) & 15) * 8, lo[2] + (m & 15) * 8];
        if (!bit(w, lower + LOWER_OFF_CHILD_MASK, m)) {
          if (bit(w, lower + LOWER_OFF_VALUE_MASK, m)) {
            activeTiles.lower++;
            pushTile(leo, 8);
          }
          continue;
        }
        const leaf = lower + i64(w, lower + GT.lower_off_table + GT.table_stride * m);
        for (let v = 0; v < 512; v++) {
          if (bit(w, leaf + LEAF_OFF_VALUE_MASK, v)) {
            leafVoxels.push([leo[0] + (v >>> 6), leo[1] + ((v >>> 3) & 7), leo[2] + (v & 7)]);
          }
        }
      }
    }
  }
  return { leafVoxels, tileSamples, activeTiles };
}

/** Root background of a FLOAT NanoVDB image (what native stores in PicoVDB slot 0, before SDF scaling). */
export function sourceBackground(image: Uint32Array): number {
  const root = GRID_SIZE + i64(image, GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT);
  return new Float32Array(image.buffer, image.byteOffset + root + GT.root_off_background, 1)[0]!;
}

/** Maps a source NanoVDB read to what PicoVDB should return for it. */
export function expectedPico(
  src: { value: number; active: boolean },
  ctx: { isFog: boolean; voxelSize: number; background: number },
): number {
  const scale = (v: number) => (ctx.isFog ? v : Math.fround(v / Math.fround(ctx.voxelSize)));
  if (src.active) return scale(src.value);
  const bg = scale(ctx.background);
  return src.value < 0 ? (ctx.isFog ? 1.0 : Math.fround(-bg)) : bg;
}

export function comparePico(
  image: Uint32Array,
  pvdb: Uint8Array,
  coords: Iterable<Coord>,
  opts: { voxelSize: number; maxRecorded?: number } ,
): CompareStats {
  const view = new PicoVDBView(pvdb);
  const acc = new PicoVDBAccessor(view, 0);
  const g = view.grid(0);
  const isFog = g.gridType === GRID_TYPE_FOG_FLOAT;
  const u8 = g.gridType === GRID_TYPE_SDF_UINT8;
  const background = sourceBackground(image);
  const ctx = { isFog, voxelSize: opts.voxelSize, background };
  const tol = u8 ? 3 / 255 + 1e-6 : 0;

  const stats: CompareStats = {
    checked: 0, active: 0, valueMismatch: 0, activeMismatch: 0, maxAbsDelta: 0, inactiveLossy: 0, firstMismatches: [],
  };
  const maxRec = opts.maxRecorded ?? 8;
  for (const ijk of coords) {
    const src = readValue(image, ijk);
    let expected = expectedPico(src, ctx);
    if (u8) expected = Math.min(3, Math.max(-3, expected));
    const got = acc.read(ijk);
    const d = Math.abs(got.value - expected);
    stats.checked++;
    if (src.active) stats.active++;
    else if (src.value !== (src.value < 0 ? (isFog ? 1 : -background) : background)) stats.inactiveLossy++;
    if (Number.isNaN(d) || d > tol) {
      stats.valueMismatch++;
      if (stats.firstMismatches.length < maxRec) {
        stats.firstMismatches.push({ ijk, pico: got.value, expected, active: src.active, level: got.level });
      }
    }
    if (!Number.isNaN(d)) stats.maxAbsDelta = Math.max(stats.maxAbsDelta, d);
    if (got.active !== src.active) stats.activeMismatch++;
  }
  return stats;
}
