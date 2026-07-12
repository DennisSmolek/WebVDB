/**
 * leaf-walk.ts — enumerates NanoVDB LEAF node origins (index space) for demo
 * 06 (explorer)'s "node-bbox wireframe" panel.
 *
 * A grid-type-independent tree walk: root tile -> upper node (skip if no
 * child) -> lower node (skip if no child) -> LEAF. Node layout (bbox-min
 * field position, table strides, child-mask bit offsets) is shared across
 * FLOAT/Fp8/FpN — `packages/vdb-web-tools/src/nanovdb/inspect.ts` relies on
 * the same fact ("root/tile/upper/lower strides are shared across grid
 * types") — so this walk only needs `nanovdb-wgsl`'s package-exported
 * `defineNumber`/`gridTypeConstantsForId` (the same layout-constant
 * accessors `readValue` is built on — `gridTypeConstantsForId` is the
 * "any grid type, not just FLOAT/FP8/FPN" variant this generic walk needs,
 * see its doc comment in `packages/nanovdb-wgsl/src/cpu/stride-tables.ts`).
 *
 * Three numbers below are structural (identical for every grid type, not
 * looked up from the layout tables) — the same convention `read-value.ts`
 * uses for its own `MASK_WORD_BITS` literal:
 *   - `NODE_OFF_BBOX_MIN = 0`: the first field of every internal node.
 *   - `UPPER_TABLE_COUNT = 32768` (32^3), `LOWER_TABLE_COUNT = 4096` (16^3):
 *     fixed by NanoVDB's 5-4-3 tree configuration.
 *
 * Verified (scratch, not checked in) against both a `buildFromVdb`-built
 * FLOAT image (smoke.vdb: 3117/3117 leaves, matching `inspect()`'s node
 * count) and the native `sphere_fog_fp8.nvdb` fixture (788/788 leaves,
 * matching its sidecar) before being written here.
 *
 * Uniform ACTIVE TILES (a solid region an encoder collapsed into one
 * upper/lower table slot instead of a real leaf — see
 * `packages/vdb-web-tools/src/nanovdb/quantize.ts`'s `traverseUpper`/
 * `traverseLower` for the same case) are intentionally NOT expanded into
 * synthetic per-voxel leaf boxes here: they aren't leaves, and this panel is
 * "leaf-level bboxes" specifically. Grids this project builds itself never
 * emit active tiles; native fixtures occasionally do (see
 * docs/handoffs/PHASE-4.md's demo 04 notes on `box_fog_float`) — those
 * regions are simply absent from the wireframe, a documented, minor gap.
 */
import { defineNumber, gridTypeConstantsForId } from "nanovdb-wgsl";

export type IndexCoord = readonly [number, number, number];

const NODE_OFF_BBOX_MIN = 0;
const UPPER_TABLE_COUNT = 32768;
const LOWER_TABLE_COUNT = 4096;

// ------------------------- layout constants (via nanovdb-wgsl) -------------
const GRID_SIZE = defineNumber("PNANOVDB_GRID_SIZE");
const TREE_OFF_NODE_OFFSET_ROOT = defineNumber("PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT");
const ROOT_OFF_TABLE_SIZE = defineNumber("PNANOVDB_ROOT_OFF_TABLE_SIZE");
const ROOT_TILE_OFF_CHILD = defineNumber("PNANOVDB_ROOT_TILE_OFF_CHILD");
const UPPER_OFF_CHILD_MASK = defineNumber("PNANOVDB_UPPER_OFF_CHILD_MASK");
const LOWER_OFF_CHILD_MASK = defineNumber("PNANOVDB_LOWER_OFF_CHILD_MASK");

function readU32(words: Uint32Array, address: number): number {
  return words[address >>> 2]!;
}

function readI64(words: Uint32Array, address: number): bigint {
  const lo = readU32(words, address);
  const hi = readU32(words, address + 4);
  const u = (BigInt(hi) << 32n) | BigInt(lo);
  return u >= 1n << 63n ? u - (1n << 64n) : u;
}

function getMaskBit(words: Uint32Array, maskBase: number, bitIndex: number): boolean {
  const word = readU32(words, maskBase + 4 * (bitIndex >>> 5));
  return ((word >>> (bitIndex & 31)) & 1) !== 0;
}

function floorDiv(v: number, block: number): number {
  return Math.floor(v / block) * block;
}

export interface LeafWalkResult {
  /** Index-space min corner of each enumerated LEAF (a real 8^3 node), capped at `cap`. */
  origins: IndexCoord[];
  /** True total LEAF count in the tree (may exceed `origins.length` when capped). */
  total: number;
}

/** Enumerates real LEAF node origins (index space), capped at `cap` entries. */
export function walkLeafOrigins(image: Uint32Array, gridTypeId: number, cap: number): LeafWalkResult {
  const gt = gridTypeConstantsForId(gridTypeId);
  const i32 = new Int32Array(image.buffer, image.byteOffset, image.length);
  const readI32 = (address: number): number => i32[address >>> 2]!;

  const treeOff = GRID_SIZE;
  const rootOff = treeOff + Number(readI64(image, treeOff + TREE_OFF_NODE_OFFSET_ROOT));
  const nTiles = readU32(image, rootOff + ROOT_OFF_TABLE_SIZE);

  const origins: IndexCoord[] = [];
  let total = 0;

  for (let t = 0; t < nTiles; t++) {
    const tileOff = rootOff + gt.root_size + t * gt.root_tile_size;
    const child = readI64(image, tileOff + ROOT_TILE_OFF_CHILD);
    if (child === 0n) continue; // background tile, or an (unhandled) active root-level tile

    const upperOff = rootOff + Number(child);
    const upperOrigin: IndexCoord = [
      floorDiv(readI32(upperOff + NODE_OFF_BBOX_MIN), 4096),
      floorDiv(readI32(upperOff + NODE_OFF_BBOX_MIN + 4), 4096),
      floorDiv(readI32(upperOff + NODE_OFF_BBOX_MIN + 8), 4096),
    ];

    for (let n = 0; n < UPPER_TABLE_COUNT; n++) {
      if (!getMaskBit(image, upperOff + UPPER_OFF_CHILD_MASK, n)) continue;
      const slotOff = upperOff + gt.upper_off_table + gt.table_stride * n;
      const lowerOff = upperOff + Number(readI64(image, slotOff));
      const lowerOrigin: IndexCoord = [
        floorDiv(readI32(lowerOff + NODE_OFF_BBOX_MIN), 128),
        floorDiv(readI32(lowerOff + NODE_OFF_BBOX_MIN + 4), 128),
        floorDiv(readI32(lowerOff + NODE_OFF_BBOX_MIN + 8), 128),
      ];

      for (let m = 0; m < LOWER_TABLE_COUNT; m++) {
        if (!getMaskBit(image, lowerOff + LOWER_OFF_CHILD_MASK, m)) continue;
        total++;
        if (origins.length < cap) {
          origins.push([
            lowerOrigin[0] + (((m >> 8) & 15) << 3),
            lowerOrigin[1] + (((m >> 4) & 15) << 3),
            lowerOrigin[2] + ((m & 15) << 3),
          ]);
        }
      }
    }
  }

  return { origins, total };
}
