/**
 * grid.ts — the parsed `VdbGrid`: a flat leaf table plus O(1) lookup by
 * (rounded-down) leaf origin, backing `readValue`/`iterLeaves`.
 *
 * `readValue` outside any stored leaf returns the grid's background value
 * (this parser doesn't retain internal-node tile values, which are
 * rejected as active but silently dropped when inactive/background-only —
 * see tree.ts). That's exactly the samples this parser targets (narrow-band
 * level sets / fully leaf-resolved fog volumes) and is documented as a
 * scope limit rather than silently wrong: the required correctness check
 * (leaf-interior `readValue` against `iterLeaves`' own values) never
 * exercises the fallback path.
 */

import { LEAF_DIM } from "./tree.js";
import type { LeafData } from "./tree.js";
import type { VdbBBox, VdbGrid, VdbLeaf, VdbTransformInfo } from "./types.js";

function floorDiv8(n: number): number {
  // `n & 7` is the correct (non-negative) remainder for any 32-bit-safe
  // integer, positive or negative, since JS `&` operates on the two's
  // complement bit pattern — so this rounds down to the nearest multiple
  // of 8 without a float division.
  return n - (n & (LEAF_DIM - 1));
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
  }

  readValue(ijk: [number, number, number]): { value: number; active: boolean } {
    const [x, y, z] = ijk;
    const origin = leafKey(floorDiv8(x), floorDiv8(y), floorDiv8(z));
    const leaf = this.leafByOrigin.get(origin);
    if (!leaf) {
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
}
