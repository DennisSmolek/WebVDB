/**
 * synthetic-vdb.ts — a tiny, uncompressed OpenVDB `.vdb` writer for tests.
 *
 * Writes one `Tree_float_5_4_3` grid with an explicit node layout (root tiles,
 * upper/lower nodes with tiles, leaves), so parser features such as active
 * tiles are covered without downloaded assets. Field order follows OpenVDB's
 * `File`/`GridDescriptor`/`RootNode`/`InternalNode`/`LeafNode` writers:
 *
 *   header, grid descriptor, [compression u32 if v>=222], metadata, transform,
 *   topology (root tiles, then each upper -> lowers -> leaf value masks),
 *   buffers (each leaf's value mask again, then its 512 values).
 *
 * Values are written uncompressed. From file version 222 every internal node
 * writes a value per slot behind a `NoMaskAndAllVals` (6) metadata byte;
 * before 222 it writes one value per child-off slot, packed, and each leaf
 * buffer starts with its origin and a buffer count.
 */

export interface SynthTile {
  /** Index-space origin of the tile (8-, 128- or 4096-aligned for its level). */
  origin: [number, number, number];
  value: number;
  active: boolean;
}

export interface SynthLeaf {
  origin: [number, number, number];
  values: Float32Array; // 512
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

export interface SynthSpec {
  fileVersion?: 221 | 222;
  name?: string;
  background: number;
  voxelSize?: number;
  rootTiles: SynthTile[];
  uppers: SynthUpper[];
}

class Writer {
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
  u64(v: number): void {
    this.ensure(8);
    this.view.setBigUint64(this.pos, BigInt(v), true);
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
  w: Writer,
  version: number,
  nodeOrigin: readonly number[],
  log2dim: number,
  childTotal: number,
  childOrigins: readonly (readonly number[])[],
  tiles: readonly SynthTile[],
  background: number,
): void {
  const numValues = 1 << (3 * log2dim);
  const childMask = new Uint32Array(numValues / 32);
  const valueMask = new Uint32Array(numValues / 32);
  const values = new Float32Array(numValues).fill(background);
  for (const o of childOrigins) {
    const n = slotOf(o, nodeOrigin, log2dim, childTotal);
    setBit(childMask, n);
    values[n] = 0; // OpenVDB writes zero in child slots
  }
  for (const t of tiles) {
    const n = slotOf(t.origin, nodeOrigin, log2dim, childTotal);
    if (testBit(childMask, n)) throw new Error("synthetic-vdb: tile overlaps a child");
    values[n] = t.value;
    if (t.active) setBit(valueMask, n);
  }
  w.words(childMask);
  w.words(valueMask);
  if (version >= 222) {
    w.u8(6); // NoMaskAndAllVals
    for (const v of values) w.f32(v);
  } else {
    for (let n = 0; n < numValues; n++) if (!testBit(childMask, n)) w.f32(values[n]!);
  }
}

const byOrigin = (a: { origin: readonly number[] }, b: { origin: readonly number[] }): number =>
  a.origin[0]! - b.origin[0]! || a.origin[1]! - b.origin[1]! || a.origin[2]! - b.origin[2]!;

/** Encodes `spec` as a complete single-grid `.vdb` file. */
export function writeSyntheticVdb(spec: SynthSpec): ArrayBuffer {
  const version = spec.fileVersion ?? 222;
  const vs = spec.voxelSize ?? 1;
  const w = new Writer();

  // Header.
  w.u32(0x56444220);
  w.u32(0);
  w.u32(version);
  w.u32(12); // library major
  w.u32(1); // library minor
  w.u8(1); // has grid offsets
  if (version >= 220 && version < 222) w.u8(0); // not compressed
  for (let i = 0; i < 36; i++) w.u8("00000000-0000-0000-0000-000000000000".charCodeAt(i));
  w.u32(0); // file metadata: empty
  w.u32(1); // grid count

  // Grid descriptor.
  w.string(spec.name ?? "density");
  w.string("Tree_float_5_4_3");
  w.string(""); // instance parent
  const offsetsAt = w.pos;
  w.u64(0);
  w.u64(0);
  w.u64(0);
  const gridPos = w.pos;
  if (version >= 222) w.u32(0); // per-grid compression: none

  // Grid metadata.
  w.u32(1);
  w.string("class");
  w.string("string");
  w.u32("fog volume".length);
  for (const ch of "fog volume") w.u8(ch.charCodeAt(0));

  // Transform.
  w.string("UniformScaleMap");
  w.vec3d([vs, vs, vs]); // scale
  w.vec3d([vs, vs, vs]); // voxel size
  w.vec3d([1 / vs, 1 / vs, 1 / vs]);
  w.vec3d([1 / (vs * vs), 1 / (vs * vs), 1 / (vs * vs)]);
  w.vec3d([1 / vs, 1 / vs, 1 / vs]);

  const blockPos = w.pos;

  // Topology.
  w.u32(1); // buffer count
  w.f32(spec.background);
  const uppers = [...spec.uppers].sort(byOrigin);
  w.u32(spec.rootTiles.length);
  w.u32(uppers.length);
  for (const t of spec.rootTiles) {
    w.vec3i(t.origin);
    w.f32(t.value);
    w.u8(t.active ? 1 : 0);
  }
  const leafOrder: SynthLeaf[] = [];
  for (const upper of uppers) {
    w.vec3i(upper.origin);
    const lowers = [...upper.lowers].sort(
      (a, b) => slotOf(a.origin, upper.origin, 5, 7) - slotOf(b.origin, upper.origin, 5, 7),
    );
    writeInternal(w, version, upper.origin, 5, 7, lowers.map((l) => l.origin), upper.tiles, spec.background);
    for (const lower of lowers) {
      const leaves = [...lower.leaves].sort(
        (a, b) => slotOf(a.origin, lower.origin, 4, 3) - slotOf(b.origin, lower.origin, 4, 3),
      );
      writeInternal(w, version, lower.origin, 4, 3, leaves.map((l) => l.origin), lower.tiles, spec.background);
      for (const leaf of leaves) {
        w.words(leaf.valueMask);
        leafOrder.push(leaf);
      }
    }
  }

  // Buffers.
  for (const leaf of leafOrder) {
    if (version < 222) {
      w.vec3i(leaf.origin);
      w.u8(1); // numBuffers
    }
    w.words(leaf.valueMask);
    if (version >= 222) w.u8(6); // NoMaskAndAllVals
    for (const v of leaf.values) w.f32(v);
  }

  const endPos = w.pos;
  w.patchU64(offsetsAt, gridPos);
  w.patchU64(offsetsAt + 8, blockPos);
  w.patchU64(offsetsAt + 16, endPos);
  return w.bytes();
}
