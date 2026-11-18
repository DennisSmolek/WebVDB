/**
 * synthetic-vdb.ts — a tiny OpenVDB `.vdb` writer for tests.
 *
 * Writes one `Tree_<valueType>_5_4_3` grid with an explicit node layout (root
 * tiles, upper/lower nodes with tiles, leaves), so parser features such as
 * active tiles, vector values, half floats and compression are covered without
 * downloaded assets. Field order follows OpenVDB's `File`/`GridDescriptor`/
 * `RootNode`/`InternalNode`/`LeafNode` writers and `io/Compression.h`:
 *
 *   header, grid descriptor, [compression u32 if v>=222], metadata, transform,
 *   topology (background, root tiles, then each upper -> lowers -> leaf value
 *   masks), buffers (each leaf's value mask again, [v<222: origin, buffer
 *   count], then its values).
 *
 * Node value arrays (`writeCompressedValues`): from file version 222 a
 * metadata byte leads. Without active-mask compression it is
 * `NoMaskAndAllVals` (6) and every slot is written; with it, it is
 * `NoMaskOrInactiveVals` (0) and only active values are written (the writer
 * then requires a zero background and zero inactive values). Before 222 an
 * internal node writes one value per child-off slot, packed. The payload is
 * raw, zlib (`zipToStream`: i64 size, then the stream; i64 0 for an empty
 * array) or Blosc (i64 size, then a frame: LZ4 literal-only streams with byte
 * shuffle, split like c-blosc does). Half-float grids store values as halves
 * and, like `HalfWriter<true, T>`, write nothing at all for an empty array.
 */

import { zlibSync } from "fflate";

export type SynthValue = number | readonly number[];

export interface SynthTile {
  /** Index-space origin of the tile (8-, 128- or 4096-aligned for its level). */
  origin: [number, number, number];
  value: SynthValue;
  active: boolean;
}

export interface SynthLeaf {
  origin: [number, number, number];
  values: Float32Array; // 512 * components
  valueMask: Uint32Array; // 16 words
}

export interface SynthLower {
  origin: [number, number, number];
  leaves: SynthLeaf[];
  tiles: SynthTile[];
}

export interface SynthUpper {
  origin: [number, number, number];
  lowers: SynthLower[];
  tiles: SynthTile[];
}

export type SynthValueType = "float" | "vec3s" | "vec3d" | "vec3i";

export interface SynthSpec {
  fileVersion?: 221 | 222 | 223;
  name?: string;
  valueType?: SynthValueType;
  background: SynthValue;
  voxelSize?: number;
  /** Store values as half floats (real value types only). */
  half?: boolean;
  /** Per-grid compression (v>=222). Default: none. */
  compression?: { zip?: boolean; blosc?: boolean; activeMask?: boolean };
  rootTiles: SynthTile[];
  uppers: SynthUpper[];
}

const COMPONENTS: Record<SynthValueType, number> = { float: 1, vec3s: 3, vec3d: 3, vec3i: 3 };
const KIND: Record<SynthValueType, "f32" | "f64" | "i32"> = { float: "f32", vec3s: "f32", vec3d: "f64", vec3i: "i32" };
const KIND_BYTES = { f32: 4, f64: 8, i32: 4 };

export class Writer {
  private buf = new Uint8Array(1 << 16);
  private view = new DataView(this.buf.buffer);
  pos = 0;

  private ensure(n: number): void {
    if (this.pos + n <= this.buf.length) return;
    let size = this.buf.length;
    while (size < this.pos + n) size *= 2;
    const next = new Uint8Array(size);
    next.set(this.buf);
    this.buf = next;
    this.view = new DataView(next.buffer);
  }

  u8(v: number): void {
    this.ensure(1);
    this.view.setUint8(this.pos, v);
    this.pos += 1;
  }
  u16(v: number): void {
    this.ensure(2);
    this.view.setUint16(this.pos, v, true);
    this.pos += 2;
  }
  u32(v: number): void {
    this.ensure(4);
    this.view.setUint32(this.pos, v, true);
    this.pos += 4;
  }
  i32(v: number): void {
    this.ensure(4);
    this.view.setInt32(this.pos, v, true);
    this.pos += 4;
  }
  u64(v: number | bigint): void {
    this.ensure(8);
    this.view.setBigUint64(this.pos, BigInt(v), true);
    this.pos += 8;
  }
  i64(v: number): void {
    this.ensure(8);
    this.view.setBigInt64(this.pos, BigInt(v), true);
    this.pos += 8;
  }
  f32(v: number): void {
    this.ensure(4);
    this.view.setFloat32(this.pos, v, true);
    this.pos += 4;
  }
  f64(v: number): void {
    this.ensure(8);
    this.view.setFloat64(this.pos, v, true);
    this.pos += 8;
  }
  raw(bytes: Uint8Array): void {
    this.ensure(bytes.length);
    this.buf.set(bytes, this.pos);
    this.pos += bytes.length;
  }
  string(s: string): void {
    this.u32(s.length);
    for (let i = 0; i < s.length; i++) this.u8(s.charCodeAt(i));
  }
  vec3i(v: readonly number[]): void {
    for (const c of v) this.i32(c);
  }
  vec3d(v: readonly number[]): void {
    for (const c of v) this.f64(c);
  }
  words(w: Uint32Array): void {
    for (const x of w) this.u32(x);
  }
  patchU64(at: number, v: number): void {
    this.view.setBigUint64(at, BigInt(v), true);
  }
  bytes(): ArrayBuffer {
    return this.buf.slice(0, this.pos).buffer;
  }
}

// ---------------------------------------------------------------------------
// Half floats and Blosc frames (test-side encoders)
// ---------------------------------------------------------------------------

/** f32 -> IEEE binary16 bits, round to nearest even (normals, subnormals, zero, inf). */
export function floatToHalf(value: number): number {
  const f = new Float32Array([value]);
  const x = new Uint32Array(f.buffer)[0]!;
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
  return sign | h;
}

/** A literal-only LZ4 block (one final sequence holding every byte). */
function lz4Literals(src: Uint8Array): Uint8Array {
  const n = src.length;
  const out: number[] = [];
  if (n < 15) {
    out.push(n << 4);
  } else {
    out.push(0xf0);
    let rest = n - 15;
    while (rest >= 255) {
      out.push(255);
      rest -= 255;
    }
    out.push(rest);
  }
  const result = new Uint8Array(out.length + n);
  result.set(out);
  result.set(src, out.length);
  return result;
}

/**
 * A Blosc1 frame of `raw` (one block, byte shuffle, LZ4 codec) whose streams
 * are literal-only LZ4 blocks. Split into `typesize` streams under c-blosc's
 * rule (typesize <= 16, >= 128 elements, no leftover) unless `dontSplit`.
 */
export function bloscFrame(raw: Uint8Array, typesize: number, dontSplit = false): Uint8Array {
  const nbytes = raw.length;
  if (nbytes === 0) {
    // What OpenVDB stores for an empty full-width array: a bare 16-byte header.
    const frame = new Uint8Array(16);
    const v = new DataView(frame.buffer);
    frame[0] = 2;
    frame[1] = 1;
    frame[2] = 0x02;
    frame[3] = typesize;
    v.setUint32(12, 16, true);
    return frame;
  }
  const neblock = Math.floor(nbytes / typesize);
  const shuffled = new Uint8Array(nbytes);
  for (let i = 0; i < neblock; i++) {
    for (let j = 0; j < typesize; j++) shuffled[j * neblock + i] = raw[i * typesize + j]!;
  }
  shuffled.set(raw.subarray(neblock * typesize), neblock * typesize);
  const leftover = nbytes % typesize !== 0;
  const split = !dontSplit && typesize <= 16 && neblock >= 128 && !leftover;
  const nsplits = split ? typesize : 1;
  const streamLen = nbytes / nsplits;
  const streams: Uint8Array[] = [];
  for (let s = 0; s < nsplits; s++) streams.push(lz4Literals(shuffled.subarray(s * streamLen, (s + 1) * streamLen)));
  const body = streams.reduce((a, s) => a + 4 + s.length, 0);
  const cbytes = 16 + 4 + body;
  const frame = new Uint8Array(cbytes);
  const v = new DataView(frame.buffer);
  frame[0] = 2;
  frame[1] = 1;
  frame[2] = 0x01 | (1 << 5) | (dontSplit ? 0x10 : 0); // byte shuffle, LZ4
  frame[3] = typesize;
  v.setUint32(4, nbytes, true);
  v.setUint32(8, nbytes, true); // one block
  v.setUint32(12, cbytes, true);
  v.setInt32(16, 20, true); // block start
  let at = 20;
  for (const s of streams) {
    v.setInt32(at, s.length, true);
    frame.set(s, at + 4);
    at += 4 + s.length;
  }
  return frame;
}

/** A memcpy'd Blosc1 frame (flag 0x02): header plus the raw bytes. */
export function bloscMemcpyFrame(raw: Uint8Array, typesize: number): Uint8Array {
  const frame = new Uint8Array(16 + raw.length);
  const v = new DataView(frame.buffer);
  frame[0] = 2;
  frame[1] = 1;
  frame[2] = 0x02;
  frame[3] = typesize;
  v.setUint32(4, raw.length, true);
  v.setUint32(8, Math.max(raw.length, 1), true);
  v.setUint32(12, frame.length, true);
  frame.set(raw, 16);
  return frame;
}

// ---------------------------------------------------------------------------
// Grid writer
// ---------------------------------------------------------------------------

interface Ctx {
  w: Writer;
  version: number;
  comps: number;
  kind: "f32" | "f64" | "i32";
  half: boolean;
  zip: boolean;
  blosc: boolean;
  activeMask: boolean;
  background: number[];
}

const asArray = (v: SynthValue, comps: number): number[] => {
  const a = typeof v === "number" ? [v] : [...v];
  if (a.length !== comps) throw new Error(`synthetic-vdb: value ${JSON.stringify(v)} needs ${comps} component(s)`);
  return a;
};

/** Writes one full-width value (background, root tiles). */
function writeFull(ctx: Ctx, v: readonly number[]): void {
  for (const c of v) {
    if (ctx.kind === "f32") ctx.w.f32(c);
    else if (ctx.kind === "f64") ctx.w.f64(c);
    else ctx.w.i32(c);
  }
}

/** Encodes `count` values (`count * comps` scalars) as stored: halves or full width. */
function encodeScalars(ctx: Ctx, scalars: readonly number[]): Uint8Array {
  const bytes = ctx.half ? 2 : KIND_BYTES[ctx.kind];
  const out = new Uint8Array(scalars.length * bytes);
  const v = new DataView(out.buffer);
  scalars.forEach((s, i) => {
    if (ctx.half) v.setUint16(i * 2, floatToHalf(s), true);
    else if (ctx.kind === "f32") v.setFloat32(i * 4, s, true);
    else if (ctx.kind === "f64") v.setFloat64(i * 8, s, true);
    else v.setInt32(i * 4, s, true);
  });
  return out;
}

/** `writeData` / `HalfWriter::write` for `count` values. */
function writePayload(ctx: Ctx, scalars: readonly number[]): void {
  const count = scalars.length / ctx.comps;
  if (ctx.half && count === 0) return; // HalfWriter<true, T>: nothing for an empty array
  const raw = encodeScalars(ctx, scalars);
  if (ctx.blosc) {
    const frame = bloscFrame(raw, (ctx.half ? 2 : KIND_BYTES[ctx.kind]) * ctx.comps);
    ctx.w.i64(frame.length);
    ctx.w.raw(frame);
  } else if (ctx.zip) {
    if (raw.length === 0) {
      ctx.w.i64(0);
      return;
    }
    const z = zlibSync(raw, { level: 9 });
    ctx.w.i64(z.length);
    ctx.w.raw(z);
  } else {
    ctx.w.raw(raw);
  }
}

/** `writeCompressedValues` for a node table of `numValues` values. */
function writeNodeValues(ctx: Ctx, values: number[][], valueMask: Uint32Array, isChild: (n: number) => boolean): void {
  const numValues = values.length;
  const scalars: number[] = [];
  if (ctx.version >= 222) {
    if (ctx.activeMask) {
      ctx.w.u8(0); // NoMaskOrInactiveVals
      for (let n = 0; n < numValues; n++) {
        if (testBit(valueMask, n)) scalars.push(...values[n]!);
        else if (!isChild(n) && values[n]!.some((c) => c !== 0)) {
          throw new Error("synthetic-vdb: active-mask compression needs zero inactive values");
        }
      }
    } else {
      ctx.w.u8(6); // NoMaskAndAllVals
      for (const v of values) scalars.push(...v);
    }
  } else {
    for (let n = 0; n < numValues; n++) if (!isChild(n)) scalars.push(...values[n]!);
  }
  writePayload(ctx, scalars);
}

/** Slot of a child/tile at `origin` inside a node at `nodeOrigin` (log2dim, child extent). */
function slotOf(origin: readonly number[], nodeOrigin: readonly number[], log2dim: number, childTotal: number): number {
  const local = origin.map((c, a) => (c - nodeOrigin[a]!) >> childTotal);
  if (local.some((c) => c < 0 || c >= 1 << log2dim)) {
    throw new Error(`synthetic-vdb: [${origin.join(", ")}] is outside the node at [${nodeOrigin.join(", ")}]`);
  }
  return (local[0]! << (2 * log2dim)) | (local[1]! << log2dim) | local[2]!;
}

function setBit(words: Uint32Array, n: number): void {
  words[n >>> 5]! |= 1 << (n & 31);
}

function testBit(words: Uint32Array, n: number): boolean {
  return ((words[n >>> 5]! >>> (n & 31)) & 1) !== 0;
}

function writeInternal(
  ctx: Ctx,
  nodeOrigin: readonly number[],
  log2dim: number,
  childTotal: number,
  childOrigins: readonly (readonly number[])[],
  tiles: readonly SynthTile[],
): void {
  const numValues = 1 << (3 * log2dim);
  const childMask = new Uint32Array(numValues / 32);
  const valueMask = new Uint32Array(numValues / 32);
  const values: number[][] = Array.from({ length: numValues }, () => [...ctx.background]);
  const zero = new Array<number>(ctx.comps).fill(0);
  for (const o of childOrigins) {
    const n = slotOf(o, nodeOrigin, log2dim, childTotal);
    setBit(childMask, n);
    values[n] = [...zero]; // OpenVDB writes zero in child slots
  }
  for (const t of tiles) {
    const n = slotOf(t.origin, nodeOrigin, log2dim, childTotal);
    if (testBit(childMask, n)) throw new Error("synthetic-vdb: tile overlaps a child");
    values[n] = asArray(t.value, ctx.comps);
    if (t.active) setBit(valueMask, n);
  }
  ctx.w.words(childMask);
  ctx.w.words(valueMask);
  writeNodeValues(ctx, values, valueMask, (n) => testBit(childMask, n));
}

const byOrigin = (a: { origin: readonly number[] }, b: { origin: readonly number[] }): number =>
  a.origin[0]! - b.origin[0]! || a.origin[1]! - b.origin[1]! || a.origin[2]! - b.origin[2]!;

/** Header, grid descriptor start; returns the offsets to patch and the grid start. */
export function writeHeaderAndDescriptor(
  w: Writer,
  version: number,
  name: string,
  gridType: string,
  headerCompressed = false,
): { offsetsAt: number; gridPos: number } {
  w.u32(0x56444220);
  w.u32(0);
  w.u32(version);
  w.u32(12); // library major
  w.u32(1); // library minor
  w.u8(1); // has grid offsets
  if (version >= 220 && version < 222) w.u8(headerCompressed ? 1 : 0);
  for (let i = 0; i < 36; i++) w.u8("00000000-0000-0000-0000-000000000000".charCodeAt(i));
  w.u32(0); // file metadata: empty
  w.u32(1); // grid count
  w.string(name);
  w.string(gridType);
  w.string(""); // instance parent
  const offsetsAt = w.pos;
  w.u64(0);
  w.u64(0);
  w.u64(0);
  return { offsetsAt, gridPos: w.pos };
}

/** Grid metadata (`class`) and a uniform-scale transform. */
export function writeMetaAndTransform(w: Writer, gridClass: string, vs: number, translate?: readonly number[]): void {
  w.u32(1);
  w.string("class");
  w.string("string");
  w.u32(gridClass.length);
  for (const ch of gridClass) w.u8(ch.charCodeAt(0));
  if (translate) {
    w.string("UniformScaleTranslateMap");
    w.vec3d(translate);
  } else {
    w.string("UniformScaleMap");
  }
  w.vec3d([vs, vs, vs]); // scale
  w.vec3d([vs, vs, vs]); // voxel size
  w.vec3d([1 / vs, 1 / vs, 1 / vs]);
  w.vec3d([1 / (vs * vs), 1 / (vs * vs), 1 / (vs * vs)]);
  w.vec3d([1 / vs, 1 / vs, 1 / vs]);
}

/** Encodes `spec` as a complete single-grid `.vdb` file. */
export function writeSyntheticVdb(spec: SynthSpec): ArrayBuffer {
  const version = spec.fileVersion ?? 222;
  const valueType = spec.valueType ?? "float";
  const comps = COMPONENTS[valueType];
  const kind = KIND[valueType];
  const half = (spec.half ?? false) && kind !== "i32";
  const comp = spec.compression ?? {};
  if (version < 222 && (comp.blosc || comp.activeMask)) throw new Error("synthetic-vdb: blosc/active mask need v>=222");
  const w = new Writer();
  const ctx: Ctx = {
    w,
    version,
    comps,
    kind,
    half,
    zip: comp.zip ?? false,
    blosc: comp.blosc ?? false,
    activeMask: comp.activeMask ?? false,
    background: asArray(spec.background, comps),
  };
  if (ctx.activeMask && ctx.background.some((c) => c !== 0)) {
    throw new Error("synthetic-vdb: active-mask compression needs a zero background");
  }

  const gridType = `Tree_${valueType}_5_4_3${spec.half ? "_HalfFloat" : ""}`;
  const { offsetsAt, gridPos } = writeHeaderAndDescriptor(w, version, spec.name ?? "density", gridType, ctx.zip);
  if (version >= 222) w.u32((ctx.zip ? 1 : 0) | (ctx.activeMask ? 2 : 0) | (ctx.blosc ? 4 : 0));
  writeMetaAndTransform(w, "fog volume", spec.voxelSize ?? 1);

  const blockPos = w.pos;

  // Topology.
  w.u32(1); // buffer count
  writeFull(ctx, ctx.background);
  const uppers = [...spec.uppers].sort(byOrigin);
  w.u32(spec.rootTiles.length);
  w.u32(uppers.length);
  for (const t of spec.rootTiles) {
    w.vec3i(t.origin);
    writeFull(ctx, asArray(t.value, comps));
    w.u8(t.active ? 1 : 0);
  }
  const leafOrder: SynthLeaf[] = [];
  for (const upper of uppers) {
    w.vec3i(upper.origin);
    const lowers = [...upper.lowers].sort(
      (a, b) => slotOf(a.origin, upper.origin, 5, 7) - slotOf(b.origin, upper.origin, 5, 7),
    );
    writeInternal(ctx, upper.origin, 5, 7, lowers.map((l) => l.origin), upper.tiles);
    for (const lower of lowers) {
      const leaves = [...lower.leaves].sort(
        (a, b) => slotOf(a.origin, lower.origin, 4, 3) - slotOf(b.origin, lower.origin, 4, 3),
      );
      writeInternal(ctx, lower.origin, 4, 3, leaves.map((l) => l.origin), lower.tiles);
      for (const leaf of leaves) {
        w.words(leaf.valueMask);
        leafOrder.push(leaf);
      }
    }
  }

  // Buffers (LeafNode::writeBuffers; before v222 the origin and a buffer count follow the mask).
  for (const leaf of leafOrder) {
    if (leaf.values.length !== 512 * comps) throw new Error(`synthetic-vdb: leaf needs ${512 * comps} values`);
    w.words(leaf.valueMask);
    if (version < 222) {
      w.vec3i(leaf.origin);
      w.u8(1); // numBuffers
    }
    const values = Array.from({ length: 512 }, (_, n) => Array.from(leaf.values.subarray(n * comps, n * comps + comps)));
    writeNodeValues(ctx, values, leaf.valueMask, () => false);
  }

  const endPos = w.pos;
  w.patchU64(offsetsAt, gridPos);
  w.patchU64(offsetsAt + 8, blockPos);
  w.patchU64(offsetsAt + 16, endPos);
  return w.bytes();
}
