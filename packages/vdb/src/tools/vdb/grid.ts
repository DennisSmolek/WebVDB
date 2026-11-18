/**
 * grid.ts — the parsed `VdbGrid`: a flat leaf table and a flat active-tile
 * table, each with O(1) lookup by (rounded-down) origin, backing
 * `readValue`/`iterLeaves`/`iterTiles`.
 *
 * `readValue` resolves a coordinate the way an OpenVDB accessor would for the
 * data this parser keeps: a stored leaf first, then an enclosing active tile
 * (lower 8^3, upper 128^3, root 4096^3), then the background. Inactive
 * internal-node tiles are not retained (see tree.ts), so a coordinate inside
 * one reads as background/inactive even when the file stored another inactive
 * value there (e.g. a level set's interior `-background`).
 */

import { LEAF_DIM, LOWER_TILE_DIM, ROOT_TILE_DIM, UPPER_TILE_DIM } from "./tree.js";
import type { LeafData, TileData } from "./tree.js";
import type { VdbBBox, VdbGrid, VdbLeaf, VdbTile, VdbTransformInfo } from "./types.js";

/** Rounds down to a multiple of `dim` (a power of two). `n & (dim - 1)` is the
 *  non-negative remainder for any 32-bit integer, negative ones included,
 *  since JS `&` works on the two's-complement bit pattern. */
function floorTo(n: number, dim: number): number {
  return n - (n & (dim - 1));
}

function leafKey(x: number, y: number, z: number): string {
  return `${x},${y},${z}`;
}

export interface GridInit {
  name: string;
  gridType: string;
  transform: VdbTransformInfo;
  metadata: Record<string, unknown>;
  indexBBox: VdbBBox | null;
  activeVoxelCount: number;
  background: number;
  leaves: LeafData[];
  /** Active tiles; omitted means none. */
  tiles?: TileData[];
}

export class VdbGridImpl implements VdbGrid {
  readonly name: string;
  readonly gridType: string;
  readonly transform: VdbTransformInfo;
  readonly metadata: Record<string, unknown>;
  readonly indexBBox: VdbBBox | null;
  readonly activeVoxelCount: bigint;
  readonly background: number;

  private readonly leaves: LeafData[];
  private readonly leafByOrigin: Map<string, LeafData>;
  private readonly tiles: TileData[];
  /** One origin -> tile map per tile size, smallest first (8, 128, 4096). */
  private readonly tileByOrigin: ReadonlyArray<{ dim: number; map: Map<string, TileData> }>;

  constructor(init: GridInit) {
    this.name = init.name;
    this.gridType = init.gridType;
    this.transform = init.transform;
    this.metadata = init.metadata;
    this.indexBBox = init.indexBBox;
    this.activeVoxelCount = BigInt(init.activeVoxelCount);
    this.background = init.background;
    this.leaves = init.leaves;
    this.leafByOrigin = new Map();
    for (const leaf of this.leaves) {
      this.leafByOrigin.set(leafKey(leaf.origin[0], leaf.origin[1], leaf.origin[2]), leaf);
    }
    this.tiles = init.tiles ?? [];
    const levels = [LOWER_TILE_DIM, UPPER_TILE_DIM, ROOT_TILE_DIM].map((dim) => ({
      dim,
      map: new Map<string, TileData>(),
    }));
    for (const tile of this.tiles) {
      const level = levels.find((l) => l.dim === tile.dim)!;
      level.map.set(leafKey(tile.origin[0], tile.origin[1], tile.origin[2]), tile);
    }
    this.tileByOrigin = levels;
  }

  readValue(ijk: [number, number, number]): { value: number; active: boolean } {
    const [x, y, z] = ijk;
    const origin = leafKey(floorTo(x, LEAF_DIM), floorTo(y, LEAF_DIM), floorTo(z, LEAF_DIM));
    const leaf = this.leafByOrigin.get(origin);
    if (!leaf) {
      for (const { dim, map } of this.tileByOrigin) {
        if (map.size === 0) continue;
        const tile = map.get(leafKey(floorTo(x, dim), floorTo(y, dim), floorTo(z, dim)));
        if (tile) return { value: tile.value, active: true };
      }
      return { value: this.background, active: false };
    }
    const lx = x & (LEAF_DIM - 1);
    const ly = y & (LEAF_DIM - 1);
    const lz = z & (LEAF_DIM - 1);
    const offset = (lx << 6) | (ly << 3) | lz;
    const wordIdx = offset >>> 5;
    const bit = offset & 31;
    const active = ((leaf.valueMask[wordIdx]! >>> bit) & 1) !== 0;
    return { value: leaf.values[offset]!, active };
  }

  *iterLeaves(): Iterable<VdbLeaf> {
    for (const leaf of this.leaves) {
      yield { origin: leaf.origin, values: leaf.values, valueMask: leaf.valueMask };
    }
  }

  *iterTiles(): Iterable<VdbTile> {
    for (const tile of this.tiles) {
      yield { origin: tile.origin, dim: tile.dim, value: tile.value };
    }
  }
}
