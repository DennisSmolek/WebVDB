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
 *
 * Values have `components` scalars (1, or 3 for vector grids, xyz
 * interleaved in the leaf arrays); vector values come back as `[x, y, z]`.
 */

import { decodePoints, type PointLeafData } from "./points.js";
import { LEAF_DIM, LOWER_TILE_DIM, ROOT_TILE_DIM, UPPER_TILE_DIM } from "./tree.js";
import type { LeafData, TileData } from "./tree.js";
import type {
  VdbBBox,
  VdbGrid,
  VdbLeaf,
  VdbPointAttributeInfo,
  VdbPointList,
  VdbTile,
  VdbTransformInfo,
  VdbValueType,
  Vec3Tuple,
} from "./types.js";

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
  valueType: VdbValueType;
  transform: VdbTransformInfo;
  metadata: Record<string, unknown>;
  indexBBox: VdbBBox | null;
  activeVoxelCount: number;
  /** `components` scalars. */
  background: number[];
  leaves: LeafData[];
  /** Active tiles; omitted means none. */
  tiles?: TileData[];
  /** Point data grids: per-leaf attribute data. */
  pointLeaves?: PointLeafData[];
  /** Point data grids: the most `points()` may allocate. Default: unlimited. */
  pointOutputBudget?: number;
}

type Value = number | Vec3Tuple;

class VdbGridImpl {
  readonly name: string;
  readonly gridType: string;
  readonly valueType: VdbValueType;
  readonly components: 1 | 3;
  readonly transform: VdbTransformInfo;
  readonly metadata: Record<string, unknown>;
  readonly indexBBox: VdbBBox | null;
  readonly activeVoxelCount: bigint;
  readonly background: Value;

  private readonly leaves: LeafData[];
  private readonly leafByOrigin: Map<string, LeafData>;
  private readonly tiles: TileData[];
  /** One origin -> tile map per tile size, smallest first (8, 128, 4096). */
  private readonly tileByOrigin: ReadonlyArray<{ dim: number; map: Map<string, TileData> }>;

  constructor(init: GridInit) {
    this.name = init.name;
    this.gridType = init.gridType;
    this.valueType = init.valueType;
    this.components = init.background.length === 3 ? 3 : 1;
    this.transform = init.transform;
    this.metadata = init.metadata;
    this.indexBBox = init.indexBBox;
    this.activeVoxelCount = BigInt(init.activeVoxelCount);
    this.background = this.toValue(init.background);
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

  private toValue(scalars: ArrayLike<number>, at = 0): Value {
    return this.components === 3
      ? [scalars[at]!, scalars[at + 1]!, scalars[at + 2]!]
      : scalars[at]!;
  }

  readValue(ijk: [number, number, number]): { value: Value; active: boolean } {
    const [x, y, z] = ijk;
    const origin = leafKey(floorTo(x, LEAF_DIM), floorTo(y, LEAF_DIM), floorTo(z, LEAF_DIM));
    const leaf = this.leafByOrigin.get(origin);
    if (!leaf) {
      for (const { dim, map } of this.tileByOrigin) {
        if (map.size === 0) continue;
        const tile = map.get(leafKey(floorTo(x, dim), floorTo(y, dim), floorTo(z, dim)));
        if (tile) return { value: this.toValue(tile.value), active: true };
      }
      return { value: this.components === 3 ? [...(this.background as Vec3Tuple)] : this.background, active: false };
    }
    const lx = x & (LEAF_DIM - 1);
    const ly = y & (LEAF_DIM - 1);
    const lz = z & (LEAF_DIM - 1);
    const offset = (lx << 6) | (ly << 3) | lz;
    const wordIdx = offset >>> 5;
    const bit = offset & 31;
    const active = ((leaf.valueMask[wordIdx]! >>> bit) & 1) !== 0;
    return { value: this.toValue(leaf.values, offset * this.components), active };
  }

  *iterLeaves(): Iterable<VdbLeaf> {
    for (const leaf of this.leaves) {
      yield { origin: leaf.origin, values: leaf.values, valueMask: leaf.valueMask };
    }
  }

  *iterTiles(): Iterable<VdbTile<Value>> {
    for (const tile of this.tiles) {
      yield { origin: tile.origin, dim: tile.dim, value: this.toValue(tile.value) };
    }
  }
}

class VdbPointDataGridImpl extends VdbGridImpl {
  readonly pointCount: number;
  readonly attributes: VdbPointAttributeInfo[];
  readonly groupNames: string[];
  private readonly pointLeaves: PointLeafData[];
  private readonly pointOutputBudget: number | undefined;
  private decoded: VdbPointList | null = null;

  constructor(init: GridInit) {
    super(init);
    this.pointLeaves = init.pointLeaves ?? [];
    this.pointOutputBudget = init.pointOutputBudget;
    let count = 0;
    for (const leaf of this.pointLeaves) count += leaf.offsets[leaf.offsets.length - 1]!;
    this.pointCount = count;
    const first = this.pointLeaves[0]?.descriptor;
    this.attributes = first ? first.attributes.map((a) => ({ ...a })) : [];
    this.groupNames = first ? first.groups.map((g) => g.name) : [];
  }

  points(): VdbPointList {
    this.decoded ??= decodePoints(this.pointLeaves, this.transform.matrix, {
      ...(this.pointOutputBudget !== undefined ? { maxOutputBytes: this.pointOutputBudget } : {}),
    });
    return this.decoded;
  }
}

/** Builds the public grid object for the parsed data. */
export function createVdbGrid(init: GridInit): VdbGrid {
  if (init.valueType === "ptdataidx32") return new VdbPointDataGridImpl(init) as unknown as VdbGrid;
  return new VdbGridImpl(init) as unknown as VdbGrid;
}
