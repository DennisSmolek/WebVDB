/**
 * CPU mirror of `vendor/picovdb.wgsl`'s read accessor
 * (`picovdbReadAccessorGetLevelIndex` + `picovdbGetValue`), transliterated
 * function by function. Upstream's `ts/picovdb.ts` has no CPU value reader
 * (its `getGridFloat` is a commented-out TODO), so the spike needs its own.
 *
 * Mirrors the WGSL, not the Zig reader. The two differ in two places, and
 * both differences are kept:
 * - The root search's end bound for the last grid is
 *   `arrayLength(&picovdb_roots)`. That is the *padded* root count when the
 *   upstream loader's `rootsBuffer` (which includes the even-count padding
 *   root, key (0,0)) is bound. Zig uses the unpadded count.
 * - u8 dequantization is `unpack4x8unorm(b) * 6 - 3` (WGSL), not
 *   `(b / 127.5 - 1) * 3` (Zig).
 *
 * The accessor cache is mirrored too (the dirty-bit tests and the conditional
 * key update in the leaf step), so a sequence of probes takes the same cache
 * paths as one GPU invocation would.
 */

import {
  GRID_SIZE,
  GRID_TYPE_SDF_UINT8,
  HEADER_SIZE,
  LEAF_SIZE,
  LOWER_SIZE,
  PICOVDB_MAGIC,
  ROOT_SIZE,
  UPPER_SIZE,
} from "./convert.js";

const INVALID = 0xffffffff;

export interface LevelIndex {
  level: number; // 0 leaf, 1 lower, 2 upper, 3 root miss
  index: number;
  isSurface: boolean;
}

/** Word-offset views of a .pvdb file, sliced the way upstream's loader slices it. */
export class PicoVDBView {
  readonly u32: Uint32Array;
  readonly f32: Float32Array;
  readonly u8: Uint8Array;
  readonly gridCount: number;
  readonly upperCount: number;
  readonly lowerCount: number;
  readonly leafCount: number;
  readonly dataCount: number;
  /** u32 word offsets of each section. */
  readonly off: { grids: number; roots: number; uppers: number; lowers: number; leaves: number; data: number };
  /** Root entries the WGSL sees (arrayLength of the bound roots buffer = padded count). */
  readonly rootArrayLength: number;

  constructor(bytes: Uint8Array) {
    if (bytes.byteOffset % 4 !== 0) throw new Error("PicoVDBView: bytes must be 4-byte aligned");
    this.u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >>> 2);
    this.f32 = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >>> 2);
    this.u8 = bytes;
    if (this.u32[0] !== PICOVDB_MAGIC[0] || this.u32[1] !== PICOVDB_MAGIC[1]) throw new Error("PicoVDBView: bad magic");
    this.gridCount = this.u32[3]!;
    this.upperCount = this.u32[4]!;
    this.lowerCount = this.u32[5]!;
    this.leafCount = this.u32[6]!;
    this.dataCount = this.u32[7]!;
    this.rootArrayLength = Math.ceil(this.upperCount / 2) * 2; // ts/picovdb.ts getRootCountPadded
    let o = HEADER_SIZE / 4;
    const grids = o;
    o += (this.gridCount * GRID_SIZE) / 4;
    const roots = o;
    o += (this.rootArrayLength * ROOT_SIZE) / 4;
    const uppers = o;
    o += (this.upperCount * UPPER_SIZE) / 4;
    const lowers = o;
    o += (this.lowerCount * LOWER_SIZE) / 4;
    const leaves = o;
    o += (this.leafCount * LEAF_SIZE) / 4;
    this.off = { grids, roots, uppers, lowers, leaves, data: o };
    if ((o + this.dataCount * 4) * 4 > bytes.byteLength) throw new Error("PicoVDBView: truncated file");
  }

  grid(i: number): { upperStart: number; lowerStart: number; leafStart: number; dataStart: number; dataElemCount: number; gridType: number; bboxMin: [number, number, number]; bboxMax: [number, number, number] } {
    const g = this.off.grids + i * 16;
    const u = this.u32;
    const s = (k: number) => u[g + k]! | 0;
    return {
      upperStart: u[g + 1]!, lowerStart: u[g + 2]!, leafStart: u[g + 3]!, dataStart: u[g + 4]!,
      dataElemCount: u[g + 5]!, gridType: u[g + 6]!,
      bboxMin: [s(8), s(9), s(10)], bboxMax: [s(12), s(13), s(14)],
    };
  }

  /** Byte slices for the six WGSL bindings, exactly as upstream's PicoVDBFile exposes them. */
  bindingSlices(): { grids: Uint8Array; roots: Uint8Array; uppers: Uint8Array; lowers: Uint8Array; leaves: Uint8Array; data: Uint8Array } {
    const sl = (from: number, to: number) => this.u8.subarray(from * 4, to * 4);
    const { grids, roots, uppers, lowers, leaves, data } = this.off;
    return {
      grids: sl(grids, roots),
      roots: sl(roots, uppers),
      uppers: sl(uppers, lowers),
      lowers: sl(lowers, leaves),
      leaves: sl(leaves, data),
      data: sl(data, data + this.dataCount * 4),
    };
  }
}

function popcount(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

/** PicoVDBReadAccessor + the WGSL functions that operate on it. */
export class PicoVDBAccessor {
  key: [number, number, number] = [0x7fffffff, 0x7fffffff, 0x7fffffff];
  upper = INVALID;
  lower = INVALID;
  leaf = INVALID;
  private readonly g: ReturnType<PicoVDBView["grid"]>;

  constructor(readonly view: PicoVDBView, readonly gridIndex = 0) {
    this.g = view.grid(gridIndex);
  }

  private dirty(ijk: readonly number[]): number {
    return (ijk[0]! ^ this.key[0]) | (ijk[1]! ^ this.key[1]) | (ijk[2]! ^ this.key[2]);
  }

  private findUpperIndex(ijk: readonly number[]): number {
    const iu = (ijk[0]! >>> 0) >>> 12;
    const ju = (ijk[1]! >>> 0) >>> 12;
    const ku = (ijk[2]! >>> 0) >>> 12;
    const kx = (ku | (ju << 21)) >>> 0;
    const ky = ((iu << 10) | (ju >>> 11)) >>> 0;
    const v = this.view;
    const start = this.g.upperStart;
    const end = this.gridIndex === v.gridCount - 1 ? v.rootArrayLength : v.grid(this.gridIndex + 1).upperStart;
    for (let i = start; i < end; i++) {
      const r = v.off.roots + i * 2;
      if (v.u32[r] === kx && v.u32[r + 1] === ky) return i - start;
    }
    return -1;
  }

  private leafStep(ijk: readonly number[]): LevelIndex {
    const v = this.view;
    const n = ((ijk[0]! & 7) << 6) | ((ijk[1]! & 7) << 3) | (ijk[2]! & 7);
    const word = n >>> 5;
    const bitIndex = n & 31;
    const leaf = v.off.leaves + (this.g.leafStart + this.leaf) * (LEAF_SIZE / 4);
    const e = leaf + 4 + word * 3;
    const state = v.u32[e]!;
    const value = v.u32[e + 1]!;
    const packed = v.u32[e + 2]!;
    const bit = (1 << bitIndex) >>> 0;
    const isValue = (value & bit) !== 0;
    const isState = (state & bit) !== 0;
    const preceding = (value & (bit - 1)) >>> 0;
    const index = isValue ? (v.u32[leaf + 2]! + (packed & 0xffff) + popcount(preceding)) >>> 0 : isState ? 1 : 0;
    if (isValue) this.key = [ijk[0]!, ijk[1]!, ijk[2]!];
    return { level: 0, index, isSurface: isValue && isState };
  }

  private internalStep(ijk: readonly number[], level: 1 | 2): LevelIndex {
    const v = this.view;
    let n: number;
    let node: number;
    if (level === 1) {
      n = (((ijk[0]! & 0x7f) >>> 3) << 8) | (((ijk[1]! & 0x7f) >>> 3) << 4) | ((ijk[2]! & 0x7f) >>> 3);
      node = v.off.lowers + (this.g.lowerStart + this.lower) * (LOWER_SIZE / 4);
    } else {
      n = (((ijk[0]! & 0xfff) >>> 7) << 10) | (((ijk[1]! & 0xfff) >>> 7) << 5) | ((ijk[2]! & 0xfff) >>> 7);
      node = v.off.uppers + (this.g.upperStart + this.upper) * (UPPER_SIZE / 4);
    }
    const word = n >>> 5;
    const bitIndex = n & 31;
    const e = node + 4 + word * 3;
    const state = v.u32[e]!;
    const value = v.u32[e + 1]!;
    const packed = v.u32[e + 2]!;
    const bit = (1 << bitIndex) >>> 0;
    const isValue = (value & bit) !== 0;
    const isState = (state & bit) !== 0;
    if (!isValue) return { level, index: isState ? 1 : 0, isSurface: false };
    const precedingMask = (bit - 1) >>> 0;
    if (!isState) {
      const index = (v.u32[node + 2]! + (packed & 0xffff) + popcount((value & ~state & precedingMask) >>> 0)) >>> 0;
      return { level, index, isSurface: false };
    }
    const child = (v.u32[node]! + (packed >>> 16) + popcount((value & state & precedingMask) >>> 0)) >>> 0;
    this.key = [ijk[0]!, ijk[1]!, ijk[2]!];
    if (level === 2) {
      this.lower = child;
      return this.internalStep(ijk, 1);
    }
    this.leaf = child;
    return this.leafStep(ijk);
  }

  /** picovdbReadAccessorGetLevelIndex */
  getLevelIndex(ijk: readonly number[]): LevelIndex {
    const d = this.dirty(ijk);
    const cachedLeaf = this.leaf !== INVALID && (d & ~0x7) === 0;
    if (!cachedLeaf) this.leaf = INVALID;
    if (cachedLeaf) return this.leafStep(ijk);
    const cachedLower = this.lower !== INVALID && (d & ~0x7f) === 0;
    if (!cachedLower) this.lower = INVALID;
    if (cachedLower) return this.internalStep(ijk, 1);
    const cachedUpper = this.upper !== INVALID && (d & ~0xfff) === 0;
    if (!cachedUpper) this.upper = INVALID;
    if (cachedUpper) return this.internalStep(ijk, 2);
    const r = this.findUpperIndex(ijk);
    if (r === -1) return { level: 3, index: 0, isSurface: false };
    this.upper = r;
    this.key = [ijk[0]!, ijk[1]!, ijk[2]!];
    return this.internalStep(ijk, 2);
  }

  /** picovdbGetValue */
  getValue(index: number): number {
    const v = this.view;
    const dataStart = this.g.dataStart;
    if (this.g.gridType === GRID_TYPE_SDF_UINT8) {
      const byteOffset = dataStart * 16 + index;
      const b = v.u8[v.off.data * 4 + byteOffset]!;
      // unpack4x8unorm -> f32(b/255); fma(x, 6, -3) (x*6 is exact in f64, so one fround = fused).
      return Math.fround(Math.fround(b / 255) * 6 - 3);
    }
    return v.f32[v.off.data + dataStart * 4 + index]!;
  }

  /**
   * Value + derived "active" flag. PicoVDB has no separate active mask: a
   * value bit (leaf voxel, or internal tile with value & ~state) is exactly an
   * active NanoVDB voxel/tile, and every stored value has index >= 2 (0/1
   * are the implicit background/inside slots). So active <=> index >= 2.
   */
  read(ijk: readonly number[]): { value: number; active: boolean; level: number; index: number; isSurface: boolean } {
    const li = this.getLevelIndex(ijk);
    return { value: this.getValue(li.index), active: li.index >= 2, ...li };
  }
}
