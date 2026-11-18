/**
 * stats.ts — per-node active-value statistics, computed the way NanoVDB's
 * `tools/GridStats.h` does: min/max are the extrema of *active* voxel values in
 * a node's subtree, `ave` is their arithmetic mean, and `stddev` is the
 * *population* standard deviation (divide by N, not N-1). A parent node's stats
 * are the exact aggregate of its children's plus its own active tiles, each
 * tile counted with GridStats' weight (see {@link StatsAccumulator.addTile}).
 *
 * The accumulator keeps N, sum and sum-of-squares in f64 so that ave/stddev are
 * computed at full precision before being narrowed to the f32 stored in the
 * grid. These stats feed the material's adaptive stepping (invariant D), so they
 * are computed for real, not stubbed.
 */

const INT_MIN = -0x80000000;
const INT_MAX = 0x7fffffff;

/** Running active-value statistics plus the tight index-space bbox of the
 *  active voxels they cover. */
export class StatsAccumulator {
  count = 0;
  private sum = 0;
  private sumSq = 0;
  min = Number.POSITIVE_INFINITY;
  max = Number.NEGATIVE_INFINITY;
  // Tight bbox of active voxels (inclusive). Empty when count === 0.
  bboxMin: [number, number, number] = [INT_MAX, INT_MAX, INT_MAX];
  bboxMax: [number, number, number] = [INT_MIN, INT_MIN, INT_MIN];

  /** Adds one active voxel at index coord (x,y,z) with the given value. */
  addVoxel(value: number, x: number, y: number, z: number): void {
    this.count += 1;
    this.sum += value;
    this.sumSq += value * value;
    if (value < this.min) this.min = value;
    if (value > this.max) this.max = value;
    if (x < this.bboxMin[0]) this.bboxMin[0] = x;
    if (y < this.bboxMin[1]) this.bboxMin[1] = y;
    if (z < this.bboxMin[2]) this.bboxMin[2] = z;
    if (x > this.bboxMax[0]) this.bboxMax[0] = x;
    if (y > this.bboxMax[1]) this.bboxMax[1] = y;
    if (z > this.bboxMax[2]) this.bboxMax[2] = z;
  }

  /**
   * Adds an active tile: `value` counted `weight` times, with the tile's
   * inclusive index bbox. NanoVDB GridStats passes the child node type's slot
   * count as the weight, so once tiles are involved `count` is a stats weight,
   * not a voxel count.
   */
  addTile(
    value: number,
    weight: number,
    bboxMin: readonly [number, number, number],
    bboxMax: readonly [number, number, number],
  ): void {
    this.count += weight;
    this.sum += value * weight;
    this.sumSq += value * value * weight;
    if (value < this.min) this.min = value;
    if (value > this.max) this.max = value;
    for (let a = 0; a < 3; a++) {
      if (bboxMin[a]! < this.bboxMin[a]!) this.bboxMin[a] = bboxMin[a]!;
      if (bboxMax[a]! > this.bboxMax[a]!) this.bboxMax[a] = bboxMax[a]!;
    }
  }

  /** Merges a child accumulator's totals into this one. */
  merge(child: StatsAccumulator): void {
    if (child.count === 0) return;
    this.count += child.count;
    this.sum += child.sum;
    this.sumSq += child.sumSq;
    if (child.min < this.min) this.min = child.min;
    if (child.max > this.max) this.max = child.max;
    for (let a = 0; a < 3; a++) {
      if (child.bboxMin[a]! < this.bboxMin[a]!) this.bboxMin[a] = child.bboxMin[a]!;
      if (child.bboxMax[a]! > this.bboxMax[a]!) this.bboxMax[a] = child.bboxMax[a]!;
    }
  }

  get average(): number {
    return this.count > 0 ? this.sum / this.count : 0;
  }

  /** Population standard deviation (GridStats convention). */
  get stdDev(): number {
    if (this.count === 0) return 0;
    const mean = this.sum / this.count;
    const variance = this.sumSq / this.count - mean * mean;
    return Math.sqrt(Math.max(0, variance));
  }

  get isEmpty(): boolean {
    return this.count === 0;
  }
}

const fround = Math.fround;

/** `Vec3<float>::lengthSqr()` evaluated in f32: `x*x + y*y + z*z`, left to right. */
export function lengthSqrF32(x: number, y: number, z: number): number {
  return fround(fround(fround(x * x) + fround(y * y)) + fround(z * z));
}

/**
 * Vector statistics, NanoVDB `tools/GridStats.h` `Stats<Vec3f, 1>` (rank 1):
 * every value is ranked by its f32 `lengthSqr()`. `min`/`max` (inherited) are
 * the extreme squared lengths and {@link minVec}/{@link maxVec} the vectors
 * that hold them (first one wins on ties, as `Pair::operator<` is strict);
 * `average` and `stdDev` are the mean and population standard deviation of the
 * squared lengths, not of the components.
 */
export class Vec3StatsAccumulator extends StatsAccumulator {
  minVec: [number, number, number] = [0, 0, 0];
  maxVec: [number, number, number] = [0, 0, 0];

  /** Adds one active voxel with vector value (vx, vy, vz) at index coord (x, y, z). */
  addVoxelVec(vx: number, vy: number, vz: number, x: number, y: number, z: number): void {
    const s = lengthSqrF32(vx, vy, vz);
    if (s < this.min) this.minVec = [vx, vy, vz];
    if (s > this.max) this.maxVec = [vx, vy, vz];
    this.addVoxel(s, x, y, z);
  }

  /** Adds an active vector tile with GridStats weight `weight` (see {@link StatsAccumulator.addTile}). */
  addTileVec(
    value: readonly [number, number, number],
    weight: number,
    bboxMin: readonly [number, number, number],
    bboxMax: readonly [number, number, number],
  ): void {
    const s = lengthSqrF32(value[0], value[1], value[2]);
    if (s < this.min) this.minVec = [value[0], value[1], value[2]];
    if (s > this.max) this.maxVec = [value[0], value[1], value[2]];
    this.addTile(s, weight, bboxMin, bboxMax);
  }

  override merge(child: StatsAccumulator): void {
    if (child.count === 0) return;
    if (child instanceof Vec3StatsAccumulator) {
      if (child.min < this.min) this.minVec = [...child.minVec];
      if (child.max > this.max) this.maxVec = [...child.maxVec];
    }
    super.merge(child);
  }
}
