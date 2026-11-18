/**
 * quantize.ts — re-encode a FLOAT NanoVDB grid image as Fp8 or FpN.
 *
 * Quantization is a leaf-value-representation change only: topology, masks,
 * internal-node tables/stats, GridData and the transform are unchanged. So
 * `quantize` reads the source FLOAT image's leaves (each: 8-aligned origin,
 * all 512 values, value mask), its active tiles (root, upper, lower; kept as
 * tiles since K4.1 — tile values stay plain f32 in Fp8/FpN grids) and its
 * GridData transform/background, then re-runs the leaf-iterator build path
 * ({@link buildFromLeavesDetailed}) with the chosen quantizing {@link LeafCodec}. The result satisfies the repo's proven decoders
 * (`read-value.ts` `leafFp*ReadFloat`, `pnanovdb.wgsl`).
 *
 * - `fp8` — fixed 8-bit codes; per-leaf quantum = (max-min)/255.
 * - `fpn` — per-leaf variable bit-width, smallest of 1/2/4/8/16 bits whose
 *   dequantized values stay within `tolerance` (default: native's FogVolume 0.01).
 *
 * v1 quantizes FLOAT sources only (throws otherwise) — that is the conversion
 * the toolset exposes (`float -> fp8/fpn`), and it is what the native
 * `nanovdb_convert --fp8/--fpn` pipeline does too.
 */

import {
  FLOAT_LAYOUT,
  GRID_OFF_GRID_CLASS,
  GRID_OFF_GRID_NAME,
  GRID_OFF_GRID_TYPE,
  GRID_OFF_MAP,
  GRID_OFF_VOXEL_SIZE,
  GRID_SIZE,
  GRID_TYPE_FLOAT,
  LEAF_OFF_VALUE_MASK,
  LEAF_TABLE_COUNT,
  LOWER_OFF_CHILD_MASK,
  LOWER_TABLE_COUNT,
  MAP_OFF_VECD,
  NODE_OFF_VALUE_MASK,
  ROOT_OFF_TABLE_SIZE,
  ROOT_TILE_OFF_CHILD,
  ROOT_TILE_OFF_KEY,
  ROOT_TILE_OFF_STATE,
  TREE_OFF_NODE_OFFSET_ROOT,
  UPPER_OFF_CHILD_MASK,
  UPPER_TABLE_COUNT,
} from "./bytes.js";
import { FP8_LEAF_CODEC, makeFpNLeafCodec, type LeafCodec } from "./leaf-codec.js";
import { buildFromLeavesDetailed, type BuiltGrid } from "./serialize.js";
import type { LeafInput, TileInput } from "./tree.js";

export type QuantizeMode = "fp8" | "fpn";

/** Quantizes a FLOAT grid image (or built grid) to Fp8/FpN. Returns a new image. */
export function quantize(
  input: Uint32Array | BuiltGrid,
  mode: QuantizeMode,
  tolerance?: number,
): Uint32Array {
  return quantizeDetailed(input, mode, tolerance).image;
}

/** As {@link quantize}, but returns the computed metadata alongside the image. */
export function quantizeDetailed(
  input: Uint32Array | BuiltGrid,
  mode: QuantizeMode,
  tolerance?: number,
): BuiltGrid {
  const image = input instanceof Uint32Array ? input : input.image;
  const src = readFloatGrid(image);
  const codec: LeafCodec = mode === "fp8" ? FP8_LEAF_CODEC : makeFpNLeafCodec(tolerance);
  return buildFromLeavesDetailed(src.leaves, codec, {
    voxelSize: src.voxelSize,
    worldOrigin: src.translation,
    background: src.background,
    gridName: src.gridName,
    gridClass:
      src.gridClassId === 1 ? "LevelSet" : src.gridClassId === 2 ? "FogVolume" : "Unknown",
    tiles: src.tiles,
  });
}

interface FloatGridView {
  voxelSize: [number, number, number];
  translation: [number, number, number];
  background: number;
  gridName: string;
  gridClassId: number;
  leaves: Iterable<LeafInput>;
  tiles: TileInput[];
}

function floorDiv(v: number, block: number): number {
  return Math.floor(v / block) * block;
}

function readI64(view: DataView, off: number): number {
  return Number(view.getBigInt64(off, true));
}

function maskBit(view: DataView, maskBase: number, n: number): boolean {
  return ((view.getUint32(maskBase + 4 * (n >>> 5), true) >>> (n & 31)) & 1) !== 0;
}

/** Reads a FLOAT grid image's transform/background and yields its leaves. */
function readFloatGrid(image: Uint32Array): FloatGridView {
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const gridType = view.getUint32(GRID_OFF_GRID_TYPE, true);
  if (gridType !== GRID_TYPE_FLOAT) {
    throw new Error(
      `quantize: source grid type ${gridType} is not FLOAT (${GRID_TYPE_FLOAT}); ` +
        `v1 quantizes float grids only (float -> fp8/fpn).`,
    );
  }
  // All three: a resampled grid can be anisotropic (K4.4), and the Map must survive quantization.
  const voxelSize: [number, number, number] = [
    view.getFloat64(GRID_OFF_VOXEL_SIZE, true),
    view.getFloat64(GRID_OFF_VOXEL_SIZE + 8, true),
    view.getFloat64(GRID_OFF_VOXEL_SIZE + 16, true),
  ];
  const vecBase = GRID_OFF_MAP + MAP_OFF_VECD;
  const translation: [number, number, number] = [
    view.getFloat64(vecBase, true),
    view.getFloat64(vecBase + 8, true),
    view.getFloat64(vecBase + 16, true),
  ];
  const gridClassId = view.getUint32(GRID_OFF_GRID_CLASS, true);
  const gridName = readCString(image, GRID_OFF_GRID_NAME);

  const treeOff = GRID_SIZE;
  const rootOff = treeOff + Number(view.getBigUint64(treeOff + TREE_OFF_NODE_OFFSET_ROOT, true));
  const background = effectiveBackground(view, rootOff);

  const tiles: TileInput[] = [];
  collectFloatTiles(view, rootOff, tiles);
  const leaves = traverseFloatLeaves(view, rootOff);
  return { voxelSize, translation, background, gridName, gridClassId, leaves, tiles };
}

/**
 * The value inactive regions near the active data read back as. Native grids
 * routinely store a far-field `mBackground` (e.g. a level-set narrow-band 3)
 * that differs from the value written into the *near-field* inactive internal
 * tiles surrounding the active voxels (typically 0). The re-layout emits no
 * inactive tiles, so it fills empty slots with a single background; using the
 * near-field inactive-tile value (the first inactive lower- then upper-tile
 * found) reproduces the source's readback around the data, falling back to the
 * root `mBackground` when the grid has no inactive internal tile at all.
 */
function effectiveBackground(view: DataView, rootOff: number): number {
  const rootBg = view.getFloat32(rootOff + FLOAT_LAYOUT.rootOffBackground, true);
  const L = FLOAT_LAYOUT;
  const nTiles = view.getUint32(rootOff + ROOT_OFF_TABLE_SIZE, true);
  let upperInactive: number | undefined;
  for (let t = 0; t < nTiles; t++) {
    const tileOff = rootOff + L.rootSize + t * L.rootTileSize;
    const child = readI64(view, tileOff + ROOT_TILE_OFF_CHILD);
    if (child === 0) continue;
    const upperOff = rootOff + child;
    for (let n = 0; n < UPPER_TABLE_COUNT; n++) {
      const slotOff = upperOff + L.upperOffTable + n * L.tableStride;
      if (maskBit(view, upperOff + UPPER_OFF_CHILD_MASK, n)) {
        // Descend one lower node: an inactive lower tile is the near-field value.
        const lowerOff = upperOff + readI64(view, slotOff);
        for (let m = 0; m < LOWER_TABLE_COUNT; m++) {
          const lslot = lowerOff + L.lowerOffTable + m * L.tableStride;
          if (
            !maskBit(view, lowerOff + LOWER_OFF_CHILD_MASK, m) &&
            !maskBit(view, lowerOff + NODE_OFF_VALUE_MASK, m)
          ) {
            return view.getFloat32(lslot, true);
          }
        }
      } else if (upperInactive === undefined && !maskBit(view, upperOff + NODE_OFF_VALUE_MASK, n)) {
        upperInactive = view.getFloat32(slotOff, true);
      }
    }
  }
  return upperInactive ?? rootBg;
}

/** Child-node offset (relative to `rootOff`) of each root-table entry that has a child. */
function* rootChildren(view: DataView, rootOff: number): Iterable<number> {
  const L = FLOAT_LAYOUT;
  const nTiles = view.getUint32(rootOff + ROOT_OFF_TABLE_SIZE, true);
  for (let t = 0; t < nTiles; t++) {
    const tileOff = rootOff + L.rootSize + t * L.rootTileSize;
    const child = readI64(view, tileOff + ROOT_TILE_OFF_CHILD);
    if (child !== 0) yield child;
  }
}

/** Index-space min corner of an internal node, from its bbox min. */
function nodeOrigin(view: DataView, nodeOff: number, dim: number): [number, number, number] {
  return [
    floorDiv(view.getInt32(nodeOff, true), dim),
    floorDiv(view.getInt32(nodeOff + 4, true), dim),
    floorDiv(view.getInt32(nodeOff + 8, true), dim),
  ];
}

/**
 * Collects every active tile of a FLOAT image: root tiles (state on, no
 * child), and upper/lower table slots whose value-mask bit is on and child bit
 * off. They are re-emitted as tiles, so the quantized grid keeps the source
 * topology (no expansion into constant leaves).
 */
function collectFloatTiles(view: DataView, rootOff: number, out: TileInput[]): void {
  const L = FLOAT_LAYOUT;
  const nTiles = view.getUint32(rootOff + ROOT_OFF_TABLE_SIZE, true);
  for (let t = 0; t < nTiles; t++) {
    const tileOff = rootOff + L.rootSize + t * L.rootTileSize;
    if (readI64(view, tileOff + ROOT_TILE_OFF_CHILD) !== 0) continue;
    if (view.getUint32(tileOff + ROOT_TILE_OFF_STATE, true) === 0) continue;
    const key = view.getBigUint64(tileOff + ROOT_TILE_OFF_KEY, true);
    out.push({ origin: keyToOrigin(key), dim: 4096, value: view.getFloat32(tileOff + L.rootTileOffValue, true) });
  }
  for (const child of rootChildren(view, rootOff)) {
    const upperOff = rootOff + child;
    const upperOrigin = nodeOrigin(view, upperOff, 4096);
    for (let n = 0; n < UPPER_TABLE_COUNT; n++) {
      const slotOff = upperOff + L.upperOffTable + n * L.tableStride;
      if (maskBit(view, upperOff + UPPER_OFF_CHILD_MASK, n)) {
        const lowerOff = upperOff + readI64(view, slotOff);
        const lowerOrigin = nodeOrigin(view, lowerOff, 128);
        for (let m = 0; m < LOWER_TABLE_COUNT; m++) {
          if (maskBit(view, lowerOff + LOWER_OFF_CHILD_MASK, m)) continue;
          if (!maskBit(view, lowerOff + NODE_OFF_VALUE_MASK, m)) continue;
          out.push({
            origin: [
              lowerOrigin[0] + (((m >> 8) & 15) << 3),
              lowerOrigin[1] + (((m >> 4) & 15) << 3),
              lowerOrigin[2] + ((m & 15) << 3),
            ],
            dim: 8,
            value: view.getFloat32(lowerOff + L.lowerOffTable + m * L.tableStride, true),
          });
        }
      } else if (maskBit(view, upperOff + NODE_OFF_VALUE_MASK, n)) {
        out.push({
          origin: [
            upperOrigin[0] + (((n >> 10) & 31) << 7),
            upperOrigin[1] + (((n >> 5) & 31) << 7),
            upperOrigin[2] + ((n & 31) << 7),
          ],
          dim: 128,
          value: view.getFloat32(slotOff, true),
        });
      }
    }
  }
}

/**
 * Inverse of `coordToKey`: each 21-bit field holds `u32(c) >>> 12`, so
 * shifting it back up by 12 in 32-bit arithmetic restores the signed,
 * 4096-aligned coordinate.
 */
function keyToOrigin(key: bigint): [number, number, number] {
  const axis = (shift: bigint): number => Number((key >> shift) & 0xfffffn) << 12;
  return [axis(42n), axis(21n), axis(0n)];
}

/** Walks root -> upper -> lower -> leaf, yielding one {@link LeafInput} per real leaf. */
function* traverseFloatLeaves(view: DataView, rootOff: number): Iterable<LeafInput> {
  for (const child of rootChildren(view, rootOff)) {
    const upperOff = rootOff + child;
    for (let n = 0; n < UPPER_TABLE_COUNT; n++) {
      if (!maskBit(view, upperOff + UPPER_OFF_CHILD_MASK, n)) continue;
      const slotOff = upperOff + FLOAT_LAYOUT.upperOffTable + n * FLOAT_LAYOUT.tableStride;
      yield* traverseLower(view, upperOff + readI64(view, slotOff));
    }
  }
}

function* traverseLower(view: DataView, lowerOff: number): Iterable<LeafInput> {
  const L = FLOAT_LAYOUT;
  const lowerOrigin = nodeOrigin(view, lowerOff, 128);
  for (let n = 0; n < LOWER_TABLE_COUNT; n++) {
    if (!maskBit(view, lowerOff + LOWER_OFF_CHILD_MASK, n)) continue;
    const slotOff = lowerOff + L.lowerOffTable + n * L.tableStride;
    const origin: [number, number, number] = [
      lowerOrigin[0] + (((n >> 8) & 15) << 3),
      lowerOrigin[1] + (((n >> 4) & 15) << 3),
      lowerOrigin[2] + ((n & 15) << 3),
    ];
    yield readLeaf(view, lowerOff + readI64(view, slotOff), origin);
  }
}

/** Reads one real FLOAT leaf (512 values + value mask) at `leafOff`. */
function readLeaf(view: DataView, leafOff: number, origin: [number, number, number]): LeafInput {
  const tableOff = FLOAT_LAYOUT.leafOffTable;
  const valueMask = new Uint32Array(16);
  for (let w = 0; w < 16; w++) valueMask[w] = view.getUint32(leafOff + LEAF_OFF_VALUE_MASK + w * 4, true);
  const values = new Float32Array(LEAF_TABLE_COUNT);
  for (let n = 0; n < LEAF_TABLE_COUNT; n++) values[n] = view.getFloat32(leafOff + tableOff + n * 4, true);
  return { origin, values, valueMask };
}

function readCString(image: Uint32Array, byteOffset: number): string {
  const bytes = new Uint8Array(image.buffer, image.byteOffset + byteOffset);
  let end = 0;
  while (end < 256 && bytes[end] !== 0) end++;
  return new TextDecoder().decode(bytes.subarray(0, end));
}
