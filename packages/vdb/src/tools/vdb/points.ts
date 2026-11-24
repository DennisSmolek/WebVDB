/**
 * points.ts — OpenVDB point data grids (`points::PointDataGrid`, grid type
 * `Tree_ptdataidx32_5_4_3`): the multi-pass leaf buffers and the attribute
 * arrays they hold.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/openvdb/core/PointDataBuffers.js`, `src/openvdb/utils/points.js`): the
 * pass walk and the position decode follow the same OpenVDB sources his reader
 * cites. The format authority is OpenVDB itself (pinned a532de55):
 * `Grid::readBuffers` (multi-pass I/O), `points/PointDataGrid.h`
 * (`PointDataLeafNode::readBuffers`, the `PointDataIndex32`
 * `readCompressedValues` specialization), `points/AttributeSet.cc`
 * (`Descriptor::read`, `readMetadata`, `groupIndex`), `points/AttributeArray.h`
 * (`TypedAttributeArray::readMetadata` / `readBuffers` / `readPagedBuffers`,
 * the codecs) and `points/StreamCompression.cc` (`PagedInputStream`, `Page`).
 *
 * ## On-disk layout
 *
 * The tree topology is ordinary (leaves store only their value mask there).
 * The buffers start with a `u16 numPasses`; then the whole tree is walked
 * `numPasses` times, every leaf reading the part of its data that belongs to
 * the pass. With `n = (numPasses - 4) / 2` attributes:
 *
 *   pass 0          u16 compressed size of the leaf's voxel buffer
 *   pass 1          [u8 header, attribute descriptor, (header & 2) u64 skip]
 *                   unless a previous leaf shared its descriptor (header & 1);
 *                   then each attribute array's metadata
 *   pass 2..n+1     attribute (pass - 2): page headers of paged arrays
 *   pass n+2        the voxel buffer: value mask, then u16 size and a Blosc
 *                   frame (or 0xFFFF and 512 raw u32 end offsets)
 *   pass n+3..2n+2  attribute (pass - n - 3): page bodies of paged arrays, or
 *                   the whole buffer of a non-paged array
 *   pass 2n+3       cleanup (no I/O)
 *
 * Paged arrays (Blosc files) share pages: a page holds the arrays of many
 * consecutive leaves. A page header is an i32 compressed size and, when
 * positive, an i32 uncompressed size; a negative size means the page is
 * stored raw. Page bodies are read only when the next array starts a new
 * page, so the bytes stay views into the file until {@link decodePoints}.
 *
 * Each leaf voxel holds the *end* offset of its points in the leaf's arrays.
 * The position attribute `P` is a voxel-space offset from the voxel centre;
 * world position = `indexToWorld(ijk + P)`.
 *
 * ## Hostile input (K4.2 hardening)
 *
 * Every count, size and offset read from the file is bounded before it drives
 * an allocation or a loop, and every failure is a `VdbFormatError` or
 * `VDBUnsupportedError`:
 *
 * - The pass loops only visit the leaves that own an array for the pass, so
 *   their work is bounded by the array metadata actually stored (a large
 *   `numPasses` costs nothing). A leaf with more attributes than the passes
 *   cover is rejected.
 * - Page headers: a page body must fit in the file, and a Blosc page may claim
 *   at most {@link MAX_BLOSC_RATIO} times its compressed size. A page must be
 *   filled exactly by the arrays assigned to it.
 * - Point counts: the grid's total point count, times the 12 bytes of its
 *   positions, may not exceed {@link MAX_POINT_EXPANSION} times the file size.
 *   Uniform arrays store one value for any number of points, so nothing else
 *   bounds a count.
 * - Decoding: every output array and every decoded page is charged against a
 *   budget of {@link MAX_POINT_EXPANSION} times the file size before it is
 *   allocated. Strides and group offsets are range-checked (no int32
 *   wrap-around).
 *
 * ## OpenVDB v14.0.0 security checks (K7.C)
 *
 * OpenVDB v14 throws `IoError` on four PointDataGrid inconsistencies
 * (CHANGES "Security Fixes"; `TypedAttributeArray::readMetadata` /
 * `readBuffers` / `readPagedBuffers`, `PagedInputStream::createHandle` /
 * `read`, `Page::readHeader`, `codecs/PointDataCodec.h`
 * `validateReadOffsets` / `validatePointDataAttributeCount`). Each maps here:
 *
 * 1. Attribute byte count smaller than the array: a byte count below its 6
 *    header bytes, a size of 0, a constant stride of 0 or with `size * stride`
 *    past u32, and a total size below `size` fail in `readArrayMetadata`;
 *    the stored bytes must equal the array's exactly (stricter than v14's
 *    "at least") when {@link decodePoints} reads them.
 * 2. Page size that does not match its data: page headers of 0, i32 minimum or
 *    a non-positive uncompressed size, an array of 0 bytes or past its page
 *    (`createPageHandle`), and a page whose arrays don't fill it exactly
 *    (`checkStreamComplete`, stricter than v14's unread-page check); a
 *    decoded page must be exactly its header's size.
 * 3. Voxel offsets beyond the arrays: end offsets must not decrease, and every
 *    array's size must equal the leaf's last offset (v14: offsets at most the
 *    last, the last at most each size; ours is exact).
 * 4. Pass count too small: a leaf with more arrays than `(numPasses - 4) / 2`
 *    covers fails (and fewer than 4 passes with leaves).
 */

import type { ByteReader } from "./byte-reader.js";
import { decompressBlosc } from "./blosc.js";
import { VdbFormatError, VDBUnsupportedError } from "./errors.js";
import { halfToFloat } from "./half-float.js";
import { readMetadataMap } from "./metadata.js";
import type { VdbPointAttributeArray, VdbPointAttributeInfo, VdbPointList } from "./types.js";

const LEAF_NUM_VALUES = 512;
const LEAF_LOG2DIM = 3;

/** `PointDataLeafNode::readBuffers` descriptor header bits. */
const HEADER_SHARED_DESCRIPTOR = 0x1;
const HEADER_SKIP_BYTES = 0x2;

/** `AttributeArray::SerializationFlag`. */
const WRITE_STRIDED = 0x1;
const WRITE_UNIFORM = 0x2;
const WRITE_PAGED = 0x8;
/** `AttributeArray::Flag::CONSTANTSTRIDE`. */
const FLAG_CONSTANT_STRIDE = 0x8;

/** The `u16` voxel-buffer size that marks an uncompressed buffer. */
const VOXEL_BUFFER_UNCOMPRESSED = 0xffff;

/**
 * The most a Blosc1 frame from OpenVDB can expand: deflate's worst case is
 * about 1032:1 and LZ4's 255:1 (each length byte adds at most 255 output
 * bytes). A Blosc page header claiming more is not a real page.
 */
export const MAX_BLOSC_RATIO = 1032;

/**
 * Decoded point output allowed per byte of the source file. The real samples
 * need under 3 (waterfall_points: 156 MB in, 381 MB of positions and
 * attributes out); 256 matches LZ4's ceiling and leaves room for uniform
 * arrays, which store one value for a whole leaf.
 */
export const MAX_POINT_EXPANSION = 256;

/** Bytes of decoded position per point (3 x f32). */
const POSITION_BYTES = 12;

/** Descriptor limits: far above any real file, low enough to bound the loops. */
const MAX_ATTRIBUTES = 1 << 16;
const MAX_GROUPS = 1 << 16;

/** One attribute set descriptor (`AttributeSet::Descriptor`). */
export interface PointDescriptor {
  /** In attribute-set order (`index` = position). */
  attributes: VdbPointAttributeInfo[];
  /** Group name -> bit offset across the group attributes. */
  groups: { name: string; offset: number }[];
}

/** A page of a paged attribute stream (`compression::Page`). */
export interface PointPage {
  /** > 0: Blosc frame size; < 0: raw, `-compressedBytes` bytes. */
  compressedBytes: number;
  uncompressedBytes: number;
  /** The page body as stored (a view into the file), once read. */
  data: Uint8Array | null;
}

/** One leaf's attribute array (`TypedAttributeArray` metadata plus where its bytes are). */
export interface PointArrayData {
  flags: number;
  serialization: number;
  /** Points in the array (`mSize`). */
  size: number;
  /** Stride (constant-stride arrays) or total element count. */
  strideOrTotal: number;
  /** Uncompressed array bytes for paged arrays; stored bytes for non-paged ones. */
  bytes: number;
  isUniform: boolean;
  isPaged: boolean;
  /** Non-paged arrays: the stored bytes and whether they are a Blosc frame. */
  raw: Uint8Array | null;
  bloscCompressed: boolean;
  /** Paged arrays: the page and the array's byte offset inside it. */
  page: PointPage | null;
  pageOffset: number;
}

/** One point data leaf after the buffer passes. */
export interface PointLeafData {
  origin: [number, number, number];
  valueMask: Uint32Array;
  /** Per-voxel end offsets into the leaf's arrays. */
  offsets: Uint32Array;
  descriptor: PointDescriptor;
  arrays: PointArrayData[];
}

interface PagedStream {
  page: PointPage | null;
  byteIndex: number;
  uncompressedBytes: number;
}

export interface ReadPointBuffersOptions {
  fileVersion: number;
  /** Leaves in buffer order, with their topology origins. */
  leaves: readonly { origin: [number, number, number] }[];
}

/** `AttributeSet::Descriptor::read`. */
function readDescriptor(reader: ByteReader): PointDescriptor {
  const count = reader.u64();
  if (count > BigInt(MAX_ATTRIBUTES)) throw new VdbFormatError(`point attribute descriptor claims ${count} attributes`);
  const n = Number(count);
  const types: { type: string; codec: string }[] = [];
  for (let i = 0; i < n; i++) types.push({ type: reader.string(), codec: reader.string() });
  const names = new Array<string | undefined>(n);
  const seen = new Set<string>();
  for (let i = 0; i < n; i++) {
    const name = reader.string();
    const index = reader.u64();
    if (index >= count) throw new VdbFormatError(`point attribute "${name}" has index ${index} of ${n}`);
    // Descriptor::read inserts into a std::map, so a repeated name would leave an attribute unnamed.
    if (seen.has(name)) throw new VdbFormatError(`point attribute name "${name}" appears twice`);
    seen.add(name);
    names[Number(index)] = name;
  }
  const attributes = types.map(({ type, codec }, index) => {
    const name = names[index];
    if (name === undefined) throw new VdbFormatError(`point attribute ${index} has no name`);
    return { name, type, codec, index };
  });
  const groupCount = reader.u64();
  if (groupCount > BigInt(MAX_GROUPS)) throw new VdbFormatError(`point attribute descriptor claims ${groupCount} groups`);
  const groups: { name: string; offset: number }[] = [];
  const groupNames = new Set<string>();
  for (let i = 0; i < Number(groupCount); i++) {
    const name = reader.string();
    const offset = reader.u64();
    // The group map is a std::map too: the first entry for a name wins.
    if (groupNames.has(name)) continue;
    groupNames.add(name);
    // Range-checked against the group arrays when decoded (`Descriptor::groupIndex`);
    // anything past 2^32 bits is out of range for any descriptor.
    groups.push({ name, offset: offset > 0xffffffffn ? Number.MAX_SAFE_INTEGER : Number(offset) });
  }
  readMetadataMap(reader); // descriptor metadata (default values, string tables): not needed here
  return { attributes, groups };
}

/** `TypedAttributeArray::readMetadata`. */
function readArrayMetadata(reader: ByteReader): PointArrayData {
  const total = reader.u64();
  const flags = reader.u8();
  const serialization = reader.u8();
  const size = reader.u32();
  if (serialization >= 0x10) {
    throw new VDBUnsupportedError(`point attribute serialization flags 0x${serialization.toString(16)}`);
  }
  if (total < 6n || total > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VdbFormatError(`point attribute array byte count ${total} is out of range`);
  }
  const bytes = Number(total) - 2 - 4; // minus the two flag bytes and the u32 size
  const strideOrTotal = serialization & WRITE_STRIDED ? reader.u32() : 1;
  // OpenVDB v14 (`readMetadata`): size and stride come from the file, so a size of 0, a constant
  // stride of 0 or whose `size * stride` passes u32, or a total below the size is an IoError.
  if (size === 0) throw new VdbFormatError("point attribute array has a size of 0");
  const constantStride = !(serialization & WRITE_STRIDED) || (flags & FLAG_CONSTANT_STRIDE) !== 0;
  if (constantStride) {
    if (strideOrTotal === 0) throw new VdbFormatError("point attribute array has a constant stride of 0");
    if (size * strideOrTotal > 0xffffffff) {
      throw new VdbFormatError(`point attribute array of ${size} x stride ${strideOrTotal} values passes 2^32`);
    }
  } else if (strideOrTotal < size) {
    throw new VdbFormatError(`point attribute array has a total size of ${strideOrTotal}, below its ${size} points`);
  }
  return {
    flags,
    serialization,
    size,
    strideOrTotal,
    bytes,
    isUniform: (serialization & WRITE_UNIFORM) !== 0,
    isPaged: (serialization & WRITE_PAGED) !== 0,
    raw: null,
    bloscCompressed: false,
    page: null,
    pageOffset: 0,
  };
}

/** `PagedInputStream::createHandle`: assign the array its page, reading a page header when one starts. */
function createPageHandle(reader: ByteReader, stream: PagedStream, array: PointArrayData): void {
  if (stream.byteIndex === stream.uncompressedBytes) {
    const compressedBytes = reader.i32();
    const uncompressedBytes = compressedBytes > 0 ? reader.i32() : -compressedBytes;
    // OpenVDB v14 `Page::readHeader`: 0 and the i32 minimum (whose negation overflows) are invalid.
    if (compressedBytes === 0 || compressedBytes === -0x80000000 || uncompressedBytes <= 0) {
      throw new VdbFormatError(`point attribute page header ${compressedBytes}/${uncompressedBytes} is empty`);
    }
    // The body follows later in the file, so it can be no larger than the file.
    const storedBytes = Math.abs(compressedBytes);
    if (storedBytes > reader.length) {
      throw new VdbFormatError(
        `point attribute page header claims ${storedBytes} stored bytes in a ${reader.length}-byte file`,
      );
    }
    if (compressedBytes > 0 && uncompressedBytes > Math.max(compressedBytes * MAX_BLOSC_RATIO, BLOSC_PAD_BYTES)) {
      throw new VdbFormatError(
        `point attribute page header claims ${uncompressedBytes} bytes from a ${compressedBytes}-byte Blosc frame`,
      );
    }
    stream.page = { compressedBytes, uncompressedBytes, data: null };
    stream.uncompressedBytes = uncompressedBytes;
    stream.byteIndex = 0;
  }
  // OpenVDB v14 `PagedInputStream::createHandle`: a handle of n <= 0 bytes, or past the page, is an IoError.
  if (array.bytes <= 0 || stream.byteIndex + array.bytes > stream.uncompressedBytes) {
    throw new VdbFormatError(
      `point attribute array of ${array.bytes} bytes ${array.bytes <= 0 ? "is empty in" : "overruns"} its ${stream.uncompressedBytes}-byte page`,
    );
  }
  array.page = stream.page;
  array.pageOffset = stream.byteIndex;
  stream.byteIndex += array.bytes;
}

/** `PagedInputStream::read`: read the array's page body when it starts a new page. */
function readPage(reader: ByteReader, stream: PagedStream, array: PointArrayData): void {
  const page = array.page;
  if (!page) throw new VdbFormatError("paged point attribute array has no page");
  if (stream.byteIndex === stream.uncompressedBytes) {
    stream.uncompressedBytes = page.uncompressedBytes;
    page.data = reader.bytes_(Math.abs(page.compressedBytes));
    stream.byteIndex = 0;
  }
  stream.byteIndex += array.bytes;
}

/** `LeafNode::readBuffers` with the `PointDataIndex32` `readCompressedValues` specialization. */
function readVoxelBuffer(reader: ByteReader, fileVersion: number): { valueMask: Uint32Array; offsets: Uint32Array } {
  const valueMask = reader.maskWords(LEAF_NUM_VALUES);
  if (fileVersion < 222) {
    reader.vec3i(); // origin
    const numBuffers = reader.u8();
    if (numBuffers !== 1) throw new VdbFormatError(`point leaf numBuffers ${numBuffers} !== 1`);
  }
  const destBytes = LEAF_NUM_VALUES * 4;
  const size = reader.u16();
  const bytes =
    size === VOXEL_BUFFER_UNCOMPRESSED
      ? reader.bytes_(destBytes)
      : decompressBlosc(reader.bytes_(size), { maxBytes: destBytes, expectedCbytes: size });
  if (bytes.length !== destBytes) {
    throw new VdbFormatError(`point leaf voxel buffer decoded to ${bytes.length} bytes, expected ${destBytes}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const offsets = new Uint32Array(LEAF_NUM_VALUES);
  for (let i = 0; i < LEAF_NUM_VALUES; i++) offsets[i] = view.getUint32(i * 4, true);
  for (let i = 1; i < LEAF_NUM_VALUES; i++) {
    if (offsets[i]! < offsets[i - 1]!) throw new VdbFormatError("point leaf end offsets decrease");
  }
  return { valueMask, offsets };
}

/**
 * Reads a point data grid's buffers (after its topology): `u16 numPasses`,
 * then every pass over every leaf. Returns one entry per leaf, in order.
 */
export function readPointDataBuffers(reader: ByteReader, opts: ReadPointBuffersOptions): PointLeafData[] {
  const numPasses = reader.u16();
  const n = opts.leaves.length;
  if (n === 0) return [];
  if (numPasses < 4) throw new VdbFormatError(`point data grid has ${numPasses} buffer passes (expected >= 4)`);
  const attributeCount = Math.floor((numPasses - 4) / 2);

  const leaves: PointLeafData[] = opts.leaves.map((l) => ({
    origin: l.origin,
    valueMask: EMPTY_MASK,
    offsets: EMPTY_OFFSETS,
    descriptor: { attributes: [], groups: [] },
    arrays: [],
  }));
  let sharedDescriptor: PointDescriptor | null = null;

  // Pass 0: voxel buffer sizes (only used by OpenVDB for delayed loading).
  for (let i = 0; i < n; i++) reader.u16();

  // Pass 1: descriptors and attribute array metadata.
  let maxArrays = 0;
  for (const leaf of leaves) {
    if (sharedDescriptor) {
      leaf.descriptor = sharedDescriptor;
    } else {
      const header = reader.u8();
      leaf.descriptor = readDescriptor(reader);
      if (header & HEADER_SHARED_DESCRIPTOR) sharedDescriptor = leaf.descriptor;
      if (header & HEADER_SKIP_BYTES) {
        const skip = reader.u64();
        if (skip > BigInt(reader.length - reader.pos)) {
          throw new VdbFormatError(`point leaf descriptor asks to skip ${skip} bytes past the end of the file`);
        }
        reader.skip(Number(skip));
      }
      if (header > 3) throw new VDBUnsupportedError(`point leaf descriptor header flags 0x${header.toString(16)}`);
    }
    leaf.arrays = leaf.descriptor.attributes.map(() => readArrayMetadata(reader));
    maxArrays = Math.max(maxArrays, leaf.arrays.length);
  }
  // `PointDataLeafNode::buffers()` is 2 * attributes + 4, and the grid writes the
  // largest leaf's count: arrays past `attributeCount` would never be read.
  if (maxArrays > attributeCount) {
    throw new VdbFormatError(
      `point leaf has ${maxArrays} attribute arrays, but the grid writes ${numPasses} passes (room for ${attributeCount})`,
    );
  }
  // The leaves that own array i, so passes cost no more than the arrays stored.
  const owners: PointLeafData[][] = Array.from({ length: maxArrays }, () => []);
  for (const leaf of leaves) for (let i = 0; i < leaf.arrays.length; i++) owners[i]!.push(leaf);
  const streams: PagedStream[] = Array.from({ length: maxArrays }, () => ({
    page: null,
    byteIndex: 0,
    uncompressedBytes: 0,
  }));

  // Passes 2 .. n+1: page headers.
  for (let i = 0; i < maxArrays; i++) {
    for (const leaf of owners[i]!) {
      const array = leaf.arrays[i]!;
      if (array.isPaged) createPageHandle(reader, streams[i]!, array);
    }
    checkStreamComplete(streams[i]!, i, "page headers");
  }

  // Pass n+2: voxel buffers. Every attribute array holds one entry per point of the leaf.
  let totalPoints = 0;
  for (const leaf of leaves) {
    const { valueMask, offsets } = readVoxelBuffer(reader, opts.fileVersion);
    leaf.valueMask = valueMask;
    leaf.offsets = offsets;
    const points = offsets[LEAF_NUM_VALUES - 1]!;
    const bad = leaf.arrays.find((a) => a.size !== points);
    if (bad) {
      throw new VdbFormatError(`point leaf has ${points} points but an attribute array of size ${bad.size}`);
    }
    totalPoints += points;
  }
  if (totalPoints * POSITION_BYTES > MAX_POINT_EXPANSION * reader.length) {
    throw new VdbFormatError(
      `point data grid claims ${totalPoints} points, more than a ${reader.length}-byte file can hold`,
    );
  }

  // Passes n+3 .. 2n+2: page bodies, or the buffers of non-paged arrays.
  for (let i = 0; i < maxArrays; i++) {
    for (const leaf of owners[i]!) {
      const array = leaf.arrays[i]!;
      if (array.isPaged) {
        readPage(reader, streams[i]!, array);
      } else {
        // TypedAttributeArray::readBuffers.
        array.bloscCompressed = array.isUniform ? false : reader.u8() === 1;
        array.raw = reader.bytes_(array.bytes);
      }
    }
    checkStreamComplete(streams[i]!, i, "page bodies");
  }
  // Passes past 2n+2 are cleanup only (no I/O), however many the header claims.
  return leaves;
}

const EMPTY_MASK = new Uint32Array(LEAF_NUM_VALUES / 32);
const EMPTY_OFFSETS = new Uint32Array(LEAF_NUM_VALUES);

/** A paged stream must end on a page boundary, or the next pass reads out of step. */
function checkStreamComplete(stream: PagedStream, index: number, what: string): void {
  if (stream.byteIndex !== stream.uncompressedBytes) {
    throw new VdbFormatError(
      `point attribute ${index} ${what} end mid-page (${stream.byteIndex} of ${stream.uncompressedBytes} bytes)`,
    );
  }
}

// ---------------------------------------------------------------------------
// Attribute decoding
// ---------------------------------------------------------------------------

type ScalarKind = "bool" | "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "f16" | "f32" | "f64";

/** OpenVDB `typeNameAsString` -> element shape. */
const VALUE_TYPES: Record<string, { components: number; kind: ScalarKind }> = {
  bool: { components: 1, kind: "bool" },
  half: { components: 1, kind: "f16" },
  float: { components: 1, kind: "f32" },
  double: { components: 1, kind: "f64" },
  int8: { components: 1, kind: "i8" },
  uint8: { components: 1, kind: "u8" },
  int16: { components: 1, kind: "i16" },
  uint16: { components: 1, kind: "u16" },
  int32: { components: 1, kind: "i32" },
  uint32: { components: 1, kind: "u32" },
  int64: { components: 1, kind: "i64" },
  uint64: { components: 1, kind: "u64" },
  vec2i: { components: 2, kind: "i32" },
  vec2s: { components: 2, kind: "f32" },
  vec2d: { components: 2, kind: "f64" },
  vec3u8: { components: 3, kind: "u8" },
  vec3u16: { components: 3, kind: "u16" },
  vec3i: { components: 3, kind: "i32" },
  vec3s: { components: 3, kind: "f32" },
  vec3d: { components: 3, kind: "f64" },
  vec4i: { components: 4, kind: "i32" },
  vec4s: { components: 4, kind: "f32" },
  vec4d: { components: 4, kind: "f64" },
  quats: { components: 4, kind: "f32" },
  quatd: { components: 4, kind: "f64" },
  mat3s: { components: 9, kind: "f32" },
  mat3d: { components: 9, kind: "f64" },
  mat4s: { components: 16, kind: "f32" },
  mat4d: { components: 16, kind: "f64" },
};

const KIND_BYTES: Record<ScalarKind, number> = {
  bool: 1,
  i8: 1,
  u8: 1,
  i16: 2,
  u16: 2,
  i32: 4,
  u32: 4,
  i64: 8,
  u64: 8,
  f16: 2,
  f32: 4,
  f64: 8,
};

type ArrayCtor = { new (n: number): VdbPointAttributeArray; readonly BYTES_PER_ELEMENT: number };

const OUTPUT_CTOR: Record<ScalarKind, ArrayCtor> = {
  bool: Uint8Array,
  i8: Int8Array,
  u8: Uint8Array,
  i16: Int16Array,
  u16: Uint16Array,
  i32: Int32Array,
  u32: Uint32Array,
  i64: BigInt64Array,
  u64: BigUint64Array,
  f16: Float32Array,
  f32: Float32Array,
  f64: Float64Array,
};

function scalarReader(kind: ScalarKind): (view: DataView, at: number) => number | bigint {
  switch (kind) {
    case "bool":
    case "u8":
      return (v, at) => v.getUint8(at);
    case "i8":
      return (v, at) => v.getInt8(at);
    case "i16":
      return (v, at) => v.getInt16(at, true);
    case "u16":
      return (v, at) => v.getUint16(at, true);
    case "i32":
      return (v, at) => v.getInt32(at, true);
    case "u32":
      return (v, at) => v.getUint32(at, true);
    case "i64":
      return (v, at) => v.getBigInt64(at, true);
    case "u64":
      return (v, at) => v.getBigUint64(at, true);
    case "f16":
      return (v, at) => halfToFloat(v.getUint16(at, true));
    case "f32":
      return (v, at) => v.getFloat32(at, true);
    case "f64":
      return (v, at) => v.getFloat64(at, true);
  }
}

const fround = Math.fround;

let unitVecWeights: Float32Array | null = null;
/**
 * `math::QuantizedUnitVec::sNormalizationWeights`, generated: `1 / |(x, y, 126 - x - y)|`
 * per slot. Exported for the test against OpenVDB's literal table.
 */
export function unitVecWeightTable(): Float32Array {
  if (unitVecWeights) return unitVecWeights;
  const table = new Float32Array(8192);
  for (let slot = 0; slot < 8192; slot++) {
    let x = (slot & 0x1f80) >> 7;
    let y = slot & 0x007f;
    if (x + y > 126) {
      x = 127 - x;
      y = 127 - y;
    }
    const z = 126 - x - y;
    table[slot] = 1 / Math.sqrt(x * x + y * y + z * z);
  }
  unitVecWeights = table;
  return table;
}

/** `math::QuantizedUnitVec::unpack`. */
export function unpackUnitVec(data: number, out: Float32Array | Float64Array, at: number): void {
  const w = unitVecWeightTable()[data & 0x1fff]!;
  let x = (data & 0x1f80) >> 7;
  let y = data & 0x007f;
  if (x + y > 126) {
    x = 127 - x;
    y = 127 - y;
  }
  let vx = fround(x * w);
  let vy = fround(y * w);
  let vz = fround((126 - x - y) * w);
  if (data & 0x8000) vx = -vx;
  if (data & 0x4000) vy = -vy;
  if (data & 0x2000) vz = -vz;
  out[at] = vx;
  out[at + 1] = vy;
  out[at + 2] = vz;
}

/** How one stored element decodes. */
interface AttributeCodec {
  components: number;
  /** Bytes of one stored element (all components). */
  storageBytes: number;
  output: ArrayCtor;
  /** Decodes the element at byte `at` into `out[outAt .. outAt + components)`. */
  decode(view: DataView, at: number, out: VdbPointAttributeArray, outAt: number): void;
}

/** The codec for an attribute's (type, codec) pair, or null when unsupported. */
function attributeCodec(type: string, codec: string): AttributeCodec | null {
  const shape = VALUE_TYPES[type];
  if (!shape) return null;
  const { components, kind } = shape;

  const perComponent = (storage: ScalarKind, map: (x: number) => number, output: ArrayCtor): AttributeCodec => {
    const read = scalarReader(storage);
    const step = KIND_BYTES[storage];
    return {
      components,
      storageBytes: step * components,
      output,
      decode(view, at, out, outAt) {
        for (let c = 0; c < components; c++) (out as Float32Array)[outAt + c] = map(read(view, at + c * step) as number);
      },
    };
  };

  switch (codec) {
    case "null":
    case "str": {
      const read = scalarReader(kind);
      const step = KIND_BYTES[kind];
      return {
        components,
        storageBytes: step * components,
        output: OUTPUT_CTOR[kind],
        decode(view, at, out, outAt) {
          for (let c = 0; c < components; c++) (out as unknown as (number | bigint)[])[outAt + c] = read(view, at + c * step);
        },
      };
    }
    case "trnc":
      // TruncateTrait: float -> half, int -> short (and their Vec3s).
      if (kind === "f32") return perComponent("f16", (x) => x, Float32Array);
      if (kind === "i32") return perComponent("i16", (x) => x, Int32Array);
      return null;
    case "fxpt8":
    case "fxpt16":
    case "ufxpt8":
    case "ufxpt16": {
      if (kind !== "f32") return null;
      const oneByte = codec.endsWith("8");
      const max = oneByte ? 255 : 65535;
      // FixedPointCodec: float(s) / float(max), then PositionRange shifts by -0.5.
      const shift = codec.startsWith("u") ? 0 : 0.5;
      return perComponent(oneByte ? "u8" : "u16", (s) => fround(fround(s / max) - shift), Float32Array);
    }
    case "uvec":
      if (type !== "vec3s") return null;
      return {
        components: 3,
        storageBytes: 2,
        output: Float32Array,
        decode(view, at, out, outAt) {
          unpackUnitVec(view.getUint16(at, true), out as Float32Array, outAt);
        },
      };
    default:
      return null;
  }
}

/** Element stride of an array, or null for a variable-stride array. */
function arrayStride(array: PointArrayData): number | null {
  if (!(array.serialization & WRITE_STRIDED)) return 1;
  return array.flags & FLAG_CONSTANT_STRIDE ? array.strideOrTotal : null;
}

/** OpenVDB pads Blosc input shorter than this with zeros (`BLOSC_PAD_BYTES`). */
const BLOSC_PAD_BYTES = 48;

/**
 * Decompresses a Blosc frame that must hold `expectedBytes` bytes, accepting
 * the zero-padded 48-byte frame OpenVDB writes for shorter input (the rule in
 * `compression::bloscDecompress`).
 */
function decompressExact(frame: Uint8Array, expectedBytes: number, what: string): Uint8Array {
  const bytes = decompressBlosc(frame, {
    maxBytes: Math.max(expectedBytes, BLOSC_PAD_BYTES),
    expectedCbytes: frame.length,
  });
  const padded = bytes.length === BLOSC_PAD_BYTES && expectedBytes <= BLOSC_PAD_BYTES;
  if (bytes.length !== expectedBytes && !padded) {
    throw new VdbFormatError(`${what} decoded to ${bytes.length} bytes, expected ${expectedBytes}`);
  }
  return bytes;
}

/**
 * The decoded-output budget of one {@link decodePoints} call. Output arrays
 * and decoded pages are charged before they are allocated; a short-lived
 * buffer is only checked against what is left.
 */
class DecodeBudget {
  private used = 0;

  constructor(private readonly limit: number) {}

  private fail(bytes: number, what: string): never {
    throw new VdbFormatError(
      `decoding ${what} needs ${bytes} bytes, past the point grid's ${this.limit}-byte output budget ` +
        `(${MAX_POINT_EXPANSION} x the file size; ${this.used} bytes already used)`,
    );
  }

  /** Reserves `bytes` for an allocation that lives until the decode ends. */
  charge(bytes: number, what: string): void {
    if (bytes > this.limit - this.used) this.fail(bytes, what);
    this.used += bytes;
  }

  /** Checks a short-lived allocation against the budget left. */
  check(bytes: number, what: string): void {
    if (bytes > this.limit - this.used) this.fail(bytes, what);
  }
}

interface DecodeContext {
  budget: DecodeBudget;
  /** Decoded Blosc pages, shared by the arrays they hold; dropped when the decode ends. */
  pages: Map<PointPage, Uint8Array>;
}

/** The array's stored bytes (decompressed), checked to be exactly `expectedBytes`. */
function arrayBytes(
  array: PointArrayData,
  expectedBytes: number,
  ctx: DecodeContext,
): { bytes: Uint8Array; offset: number } {
  if (array.isPaged) {
    if (array.bytes !== expectedBytes) {
      throw new VdbFormatError(`paged point attribute holds ${array.bytes} bytes, expected ${expectedBytes}`);
    }
    const page = array.page;
    if (!page?.data) throw new VdbFormatError("paged point attribute has no page body");
    let decoded = page.compressedBytes > 0 ? ctx.pages.get(page) : page.data;
    if (!decoded) {
      ctx.budget.charge(Math.max(page.uncompressedBytes, BLOSC_PAD_BYTES), "a point attribute page");
      decoded = decompressExact(page.data, page.uncompressedBytes, "point attribute page");
      ctx.pages.set(page, decoded);
    }
    if (array.pageOffset + expectedBytes > decoded.length) {
      throw new VdbFormatError(`paged point attribute runs past its ${decoded.length}-byte page`);
    }
    return { bytes: decoded, offset: array.pageOffset };
  }
  if (!array.raw) throw new VdbFormatError("point attribute has no buffer");
  if (array.bloscCompressed) {
    ctx.budget.check(expectedBytes, "a point attribute buffer");
    return { bytes: decompressExact(array.raw, expectedBytes, "point attribute"), offset: 0 };
  }
  if (array.raw.length !== expectedBytes) {
    throw new VdbFormatError(`point attribute holds ${array.raw.length} bytes, expected ${expectedBytes}`);
  }
  return { bytes: array.raw, offset: 0 };
}

/** Sorted attribute indices of the group arrays (`Descriptor::groupIndex`). */
function groupArrayIndices(descriptor: PointDescriptor): number[] {
  return descriptor.attributes
    .filter((a) => a.type === "uint8" && a.codec === "grp")
    .map((a) => a.index)
    .sort((a, b) => a - b);
}

export interface DecodePointsOptions {
  /**
   * Upper bound on the bytes the decode may allocate (outputs plus decoded
   * pages). The parser passes {@link MAX_POINT_EXPANSION} times the file size.
   * Default: unlimited.
   */
  maxOutputBytes?: number;
}

/**
 * Decodes the points of a parsed point data grid: world-space positions from
 * `P` (through `matrix`, a row-major 4x4 index->world transform) and every
 * other attribute the codec table covers.
 */
export function decodePoints(
  leaves: readonly PointLeafData[],
  matrix: readonly number[],
  options: DecodePointsOptions = {},
): VdbPointList {
  const ctx: DecodeContext = {
    budget: new DecodeBudget(options.maxOutputBytes ?? Number.POSITIVE_INFINITY),
    pages: new Map(),
  };
  let count = 0;
  for (const leaf of leaves) count += leaf.offsets[LEAF_NUM_VALUES - 1]!;

  ctx.budget.charge(count * POSITION_BYTES, "point positions");
  const positions = new Float32Array(count * 3);
  const attributes: Record<string, VdbPointAttributeArray> = {};
  const attributeShapes: Record<string, { components: number; stride: number }> = {};
  const groups: Record<string, Uint8Array> = {};
  const skipped: VdbPointAttributeInfo[] = [];
  const skippedNames = new Set<string>();
  const skip = (info: VdbPointAttributeInfo): void => {
    if (skippedNames.has(info.name)) return;
    skippedNames.add(info.name);
    skipped.push(info);
  };

  const m = matrix;
  let base = 0;
  for (const leaf of leaves) {
    const leafCount = leaf.offsets[LEAF_NUM_VALUES - 1]!;
    const { descriptor, arrays } = leaf;

    // Positions: voxel ijk + P offset, then index -> world.
    const pInfo = descriptor.attributes.find((a) => a.name === "P");
    // f32 offsets (every OpenVDB P codec but vec3d/null) decode in place into
    // `positions`, then each point is transformed over its own offset.
    let p: ArrayLike<number> | null = null;
    let pAt = 0;
    if (pInfo) {
      const codec = attributeCodec(pInfo.type, pInfo.codec);
      if (!codec || codec.components !== 3) {
        throw new VDBUnsupportedError(`point position attribute P is ${pInfo.type}/${pInfo.codec}`);
      }
      if (codec.output === Float32Array) {
        decodeArrayInto(arrays[pInfo.index]!, codec, leafCount, pInfo, ctx, positions, base * 3);
        p = positions;
        pAt = base * 3;
      } else {
        ctx.budget.check(leafCount * 3 * codec.output.BYTES_PER_ELEMENT, "a leaf's point offsets");
        const offsets = new codec.output(leafCount * 3);
        decodeArrayInto(arrays[pInfo.index]!, codec, leafCount, pInfo, ctx, offsets, 0);
        p = offsets as ArrayLike<number>;
      }
    }
    let voxel = 0;
    for (let i = 0; i < leafCount; i++) {
      while (leaf.offsets[voxel]! <= i) voxel++;
      const ix = leaf.origin[0] + (voxel >> (2 * LEAF_LOG2DIM));
      const iy = leaf.origin[1] + ((voxel >> LEAF_LOG2DIM) & 7);
      const iz = leaf.origin[2] + (voxel & 7);
      const x = ix + (p ? p[pAt + i * 3]! : 0);
      const y = iy + (p ? p[pAt + i * 3 + 1]! : 0);
      const z = iz + (p ? p[pAt + i * 3 + 2]! : 0);
      const out = (base + i) * 3;
      positions[out] = m[0]! * x + m[1]! * y + m[2]! * z + m[3]!;
      positions[out + 1] = m[4]! * x + m[5]! * y + m[6]! * z + m[7]!;
      positions[out + 2] = m[8]! * x + m[9]! * y + m[10]! * z + m[11]!;
    }

    // Other attributes, decoded straight into the grid-wide output.
    for (const info of descriptor.attributes) {
      if (info.name === "P" || (info.type === "uint8" && info.codec === "grp")) continue;
      const codec = attributeCodec(info.type, info.codec);
      const array = arrays[info.index]!;
      const stride = arrayStride(array);
      if (!codec || stride === null) {
        skip(info);
        continue;
      }
      let target = attributes[info.name];
      const shape = attributeShapes[info.name];
      if (!target) {
        const length = count * stride * codec.components;
        ctx.budget.charge(length * codec.output.BYTES_PER_ELEMENT, `point attribute "${info.name}"`);
        target = new codec.output(length);
        attributes[info.name] = target;
        attributeShapes[info.name] = { components: codec.components, stride };
      } else if (
        !shape ||
        shape.components !== codec.components ||
        shape.stride !== stride ||
        target.constructor !== codec.output
      ) {
        throw new VDBUnsupportedError(`point attribute "${info.name}" changes type between leaves`);
      }
      decodeArrayInto(array, codec, leafCount * stride, info, ctx, target, base * stride * codec.components);
    }

    // Groups: one bit per point in the uint8 group arrays (`Descriptor::groupIndex`).
    if (descriptor.groups.length > 0) {
      const groupArrays = groupArrayIndices(descriptor);
      const decodedGroups = new Map<number, Uint8Array>();
      const uint8 = attributeCodec("uint8", "null")!;
      for (const { name, offset } of descriptor.groups) {
        if (offset >= groupArrays.length * 8) {
          throw new VdbFormatError(`point group "${name}" offset ${offset} is out of range`);
        }
        const arrayIndex = groupArrays[Math.floor(offset / 8)]!;
        let bits = decodedGroups.get(arrayIndex);
        if (!bits) {
          ctx.budget.check(leafCount, "a leaf's group bits");
          bits = new Uint8Array(leafCount);
          decodeArrayInto(arrays[arrayIndex]!, uint8, leafCount, descriptor.attributes[arrayIndex]!, ctx, bits, 0);
          decodedGroups.set(arrayIndex, bits);
        }
        let target = groups[name];
        if (!target) {
          ctx.budget.charge(count, `point group "${name}"`);
          target = groups[name] = new Uint8Array(count);
        }
        const bit = offset % 8;
        for (let i = 0; i < leafCount; i++) target[base + i] = (bits[i]! >> bit) & 1;
      }
    }

    base += leafCount;
  }

  return { count, positions, attributes, attributeShapes, groups, skipped };
}

/**
 * Decodes `elements` elements of one leaf array into `out` from `outAt` (a
 * uniform array repeats its single value). The array's size and stride are
 * checked before any byte is read.
 */
function decodeArrayInto(
  array: PointArrayData,
  codec: AttributeCodec,
  elements: number,
  info: VdbPointAttributeInfo,
  ctx: DecodeContext,
  out: VdbPointAttributeArray,
  outAt: number,
): void {
  const stride = arrayStride(array) ?? 1;
  if (array.size * stride !== elements) {
    throw new VdbFormatError(
      `point attribute "${info.name}" holds ${array.size} x ${stride} values, but the leaf has ${elements / stride} points`,
    );
  }
  const stored = array.isUniform ? 1 : elements;
  const storedBytes = stored * codec.storageBytes;
  const { bytes, offset } = arrayBytes(array, storedBytes, ctx);
  const view = new DataView(bytes.buffer, bytes.byteOffset + offset, storedBytes);
  for (let i = 0; i < elements; i++) {
    codec.decode(view, array.isUniform ? 0 : i * codec.storageBytes, out, outAt + i * codec.components);
  }
}
