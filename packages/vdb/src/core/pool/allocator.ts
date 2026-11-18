/**
 * First-fit range allocator in 32-byte units over `[start, end)` of the pool
 * (docs/design/K1-core.md §1.3–1.4). It only does bookkeeping; moving data is the
 * pool's job (growth and compaction in `update`).
 */

import { VDB_POOL_ALIGN, align32 } from "./layout.js";

/** One allocated range. */
export interface AllocatedRange {
  offset: number;
  size: number;
}

/** First-fit allocator. Sizes are rounded up to 32 bytes; offsets are 32-byte aligned. */
export class RangeAllocator {
  /** Allocated ranges sorted by offset. */
  #ranges: AllocatedRange[] = [];
  #start: number;
  #end: number;

  /** @param start first usable byte (32-aligned) @param end one past the last usable byte */
  constructor(start: number, end: number) {
    if (start % VDB_POOL_ALIGN !== 0) throw new Error(`RangeAllocator: start ${start} is not 32-byte aligned.`);
    this.#start = start;
    this.#end = end;
  }

  get start(): number {
    return this.#start;
  }

  get end(): number {
    return this.#end;
  }

  /** Bytes currently allocated (rounded sizes). */
  get allocatedBytes(): number {
    let n = 0;
    for (const r of this.#ranges) n += r.size;
    return n;
  }

  /** End of the highest allocated range, or `start` when empty. */
  get highWater(): number {
    const last = this.#ranges[this.#ranges.length - 1];
    return last ? last.offset + last.size : this.#start;
  }

  /** Allocated ranges in offset order (copies). */
  ranges(): AllocatedRange[] {
    return this.#ranges.map((r) => ({ ...r }));
  }

  /** Allocates `bytes` (rounded up to 32) at the lowest offset that fits; `null` when no gap does. */
  alloc(bytes: number): number | null {
    const size = align32(Math.max(bytes, 1));
    let cursor = this.#start;
    for (let i = 0; i <= this.#ranges.length; i++) {
      const next = this.#ranges[i];
      const gapEnd = next ? next.offset : this.#end;
      if (gapEnd - cursor >= size) {
        this.#ranges.splice(i, 0, { offset: cursor, size });
        return cursor;
      }
      if (next) cursor = next.offset + next.size;
    }
    return null;
  }

  /** Frees the range starting at `offset`. Throws if no range starts there (a pool bug). */
  free(offset: number): void {
    const i = this.#ranges.findIndex((r) => r.offset === offset);
    if (i < 0) throw new Error(`RangeAllocator: no range at offset ${offset}.`);
    this.#ranges.splice(i, 1);
  }

  /** Re-inserts a range that was just freed (undo of `free`). Throws if it overlaps an allocation (a pool bug). */
  restore(range: AllocatedRange): void {
    const i = this.#ranges.findIndex((r) => r.offset > range.offset);
    const at = i < 0 ? this.#ranges.length : i;
    const prev = this.#ranges[at - 1];
    const next = this.#ranges[at];
    if ((prev && prev.offset + prev.size > range.offset) || (next && range.offset + range.size > next.offset)) {
      throw new Error(`RangeAllocator: cannot restore [${range.offset}, +${range.size}); it overlaps an allocation.`);
    }
    this.#ranges.splice(at, 0, { ...range });
  }

  /** Size of the range starting at `offset`, or `undefined`. */
  sizeAt(offset: number): number | undefined {
    return this.#ranges.find((r) => r.offset === offset)?.size;
  }
}
