/**
 * synthetic-points.ts — a tiny OpenVDB point data grid (`Tree_ptdataidx32_5_4_3`)
 * writer for tests. Not a test file.
 *
 * Follows OpenVDB's writers: `Grid::writeBuffers` (u16 pass count, then the
 * tree once per pass), `PointDataLeafNode::writeBuffers` (the pass layout),
 * `AttributeSet::Descriptor::write`, `TypedAttributeArray::writeMetadata` /
 * `writeBuffers` / `writePagedBuffers` and `PagedOutputStream` (page headers
 * in the size pass, page bodies in the data pass; a page is flushed when the
 * next array would overflow it). All leaves sit in one lower node in one upper
 * node at the origin.
 *
 * `mode: "plain"` writes an uncompressed file (arrays not paged, each buffer
 * behind a "not Blosc" byte unless uniform). `mode: "paged"` writes a Blosc
 * file: every array is paged; pages are raw (negative size) or Blosc frames,
 * and the voxel buffers raw (0xFFFF) or Blosc frames.
 */

import { bloscFrame, Writer, writeHeaderAndDescriptor, writeMetaAndTransform } from "./synthetic-vdb.js";

export interface SynthPointAttr {
  name: string;
  type: string;
  codec: string;
  /** Uniform array: one stored element for the whole leaf. */
  uniform?: boolean;
  /** Elements per point (strided array). Default 1. */
  stride?: number;
}

export interface SynthPointLeaf {
  origin: [number, number, number];
  /** Per-voxel end offsets (512). */
  offsets: Uint32Array;
  /** Stored (uncompressed) bytes of each attribute array, in attribute order. */
  arrays: Uint8Array[];
}

export interface SynthPointSpec {
  fileVersion?: 222 | 223;
  voxelSize: number;
  translate?: [number, number, number];
  attributes: SynthPointAttr[];
  groups?: { name: string; offset: number }[];
  leaves: SynthPointLeaf[];
  mode: "plain" | "paged";
  /** Paged mode: page capacity in bytes (OpenVDB uses 1 MiB). */
  pageSize?: number;
  /** Paged mode: store pages as Blosc frames (else raw). */
  pageBlosc?: boolean;
  /** Paged mode: store voxel buffers as Blosc frames (else raw, 0xFFFF). */
  voxelBlosc?: boolean;
  /** Write the descriptor once, flagged shared (header bit 1). Default true. */
  sharedDescriptor?: boolean;
  /** Hooks that corrupt fields as they are written (points-hostile.test.ts). */
  tamper?: SynthPointTamper;
}

/** One attribute array's metadata, as `TypedAttributeArray::writeMetadata` writes it. */
export interface SynthArrayMeta {
  /** The u64 byte count: 2 flag bytes + the u32 size + the stored bytes. */
  total: bigint;
  flags: number;
  serialization: number;
  size: number;
  /** Written when `serialization` has the strided bit. */
  stride: number;
}

export interface SynthPointTamper {
  /** The u16 pass count (default `4 + 2 * attributes`). */
  numPasses?: number;
  /** Rewrites one leaf's array metadata. */
  arrayMeta?: (leaf: number, attr: number, meta: SynthArrayMeta) => SynthArrayMeta;
  /** Rewrites one attribute's page headers (`[-raw]` or `[compressed, uncompressed]` per page). */
  pageHeaders?: (attr: number, headers: number[][]) => number[][];
  /** Writes leaf `leaf`'s descriptor block (header byte included) instead of the real one. */
  descriptor?: (w: Writer, leaf: number) => void;
  /** Rewrites a leaf's end offsets as stored in its voxel buffer (the mask keeps the real ones). */
  storedOffsets?: (leaf: number, offsets: Uint32Array) => Uint32Array;
}

const FLAG_CONSTANT_STRIDE = 0x8;
const WRITE_STRIDED = 0x1;
const WRITE_UNIFORM = 0x2;
const WRITE_PAGED = 0x8;

export function writeDescriptor(w: Writer, attrs: SynthPointAttr[], groups: { name: string; offset: number }[]): void {
  w.u64(attrs.length);
  for (const a of attrs) {
    w.string(a.type);
    w.string(a.codec);
  }
  // The name map is a std::map: sorted by name.
  const names = attrs.map((a, index) => ({ name: a.name, index })).sort((a, b) => (a.name < b.name ? -1 : 1));
  for (const { name, index } of names) {
    w.string(name);
    w.u64(index);
  }
  const sortedGroups = [...groups].sort((a, b) => (a.name < b.name ? -1 : 1));
  w.u64(sortedGroups.length);
  for (const g of sortedGroups) {
    w.string(g.name);
    w.u64(g.offset);
  }
  w.u32(0); // descriptor metadata: empty
}

interface Page {
  bytes: Uint8Array;
}

/** PagedOutputStream: groups consecutive array buffers into pages. */
function paginate(buffers: Uint8Array[], pageSize: number): Page[] {
  const pages: Page[] = [];
  let current: Uint8Array[] = [];
  let size = 0;
  const flush = (): void => {
    if (size === 0) return;
    const bytes = new Uint8Array(size);
    let at = 0;
    for (const b of current) {
      bytes.set(b, at);
      at += b.length;
    }
    pages.push({ bytes });
    current = [];
    size = 0;
  };
  for (const b of buffers) {
    if (b.length > pageSize) {
      flush();
      pages.push({ bytes: b.slice() });
    } else {
      if (b.length + size > pageSize) flush();
      current.push(b);
      size += b.length;
    }
  }
  flush();
  return pages;
}

/** A page body as stored: a Blosc frame (input padded to 48 bytes, typesize 4) or raw. */
function pageBody(page: Page, blosc: boolean): { header: number[]; body: Uint8Array } {
  if (!blosc || page.bytes.length <= 16) return { header: [-page.bytes.length], body: page.bytes };
  let input = page.bytes;
  if (input.length < 48) {
    input = new Uint8Array(48);
    input.set(page.bytes);
  }
  const frame = bloscFrame(input, 4);
  return { header: [frame.length, page.bytes.length], body: frame };
}

function setBit(words: Uint32Array, n: number): void {
  words[n >>> 5]! |= 1 << (n & 31);
}

/** Active voxels: those with at least one point. */
export function pointMask(offsets: Uint32Array): Uint32Array {
  const mask = new Uint32Array(16);
  let prev = 0;
  for (let n = 0; n < 512; n++) {
    if (offsets[n]! > prev) setBit(mask, n);
    prev = offsets[n]!;
  }
  return mask;
}

export function writeSyntheticPoints(spec: SynthPointSpec): ArrayBuffer {
  const version = spec.fileVersion ?? 223;
  const paged = spec.mode === "paged";
  const shared = spec.sharedDescriptor ?? true;
  const w = new Writer();
  const { offsetsAt, gridPos } = writeHeaderAndDescriptor(w, version, "points", "Tree_ptdataidx32_5_4_3");
  w.u32(paged ? 0x6 : 0x2); // blosc + active mask, or active mask only
  writeMetaAndTransform(w, "unknown", spec.voxelSize, spec.translate);
  const blockPos = w.pos;

  // Internal nodes store no active values: metadata 0 and an empty payload
  // (a bare 16-byte Blosc frame behind its i64 size in Blosc files).
  const emptyNodeValues = (): void => {
    w.u8(0);
    if (paged) {
      w.i64(16);
      w.raw(bloscFrame(new Uint8Array(0), 4));
    }
  };

  // Topology.
  w.u32(1); // buffer count
  w.u32(0); // background (PointDataIndex32)
  w.u32(0); // root tiles
  w.u32(1); // root children
  w.vec3i([0, 0, 0]);
  const upperChild = new Uint32Array(1024);
  setBit(upperChild, 0);
  w.words(upperChild);
  w.words(new Uint32Array(1024));
  emptyNodeValues();
  const lowerChild = new Uint32Array(128);
  const leaves = [...spec.leaves].sort((a, b) => a.origin[0] - b.origin[0] || a.origin[1] - b.origin[1] || a.origin[2] - b.origin[2]);
  for (const leaf of leaves) {
    if (leaf.origin.some((c) => c < 0 || c >= 128 || c % 8 !== 0)) throw new Error("synthetic-points: leaf outside the lower node");
    setBit(lowerChild, ((leaf.origin[0] >> 3) << 8) | ((leaf.origin[1] >> 3) << 4) | (leaf.origin[2] >> 3));
  }
  w.words(lowerChild);
  w.words(new Uint32Array(128));
  emptyNodeValues();
  const masks = leaves.map((l) => pointMask(l.offsets));
  for (const m of masks) w.words(m);

  // Voxel buffers as stored.
  const tamper = spec.tamper ?? {};
  const voxelBuffers = leaves.map((l, i) => {
    const offsets = tamper.storedOffsets ? tamper.storedOffsets(i, l.offsets.slice()) : l.offsets;
    const raw = new Uint8Array(offsets.buffer.slice(offsets.byteOffset, offsets.byteOffset + 2048));
    if (paged && spec.voxelBlosc) {
      const frame = bloscFrame(raw, 4);
      return { size: frame.length, bytes: frame };
    }
    return { size: 0xffff, bytes: raw };
  });

  // Pages per attribute (paged mode).
  const pageSize = spec.pageSize ?? 1 << 20;
  const pages = spec.attributes.map((_, a) =>
    paginate(
      leaves.map((l) => l.arrays[a]!),
      pageSize,
    ).map((p) => pageBody(p, spec.pageBlosc ?? false)),
  );

  const n = spec.attributes.length;
  const numPasses = 4 + 2 * n;
  w.u16(tamper.numPasses ?? numPasses);
  for (let pass = 0; pass < numPasses; pass++) {
    if (pass >= 2 && pass < n + 2) {
      // Page headers of attribute (pass - 2).
      const a = pass - 2;
      let headers = pages[a]!.map((p) => p.header);
      if (tamper.pageHeaders) headers = tamper.pageHeaders(a, headers);
      if (paged) for (const h of headers.flat()) w.i32(h);
      continue;
    }
    if (pass >= n + 3 && pass < 2 * n + 3) {
      const a = pass - n - 3;
      if (paged) {
        for (const p of pages[a]!) w.raw(p.body);
      } else {
        for (const leaf of leaves) {
          if (!spec.attributes[a]!.uniform) w.u8(0); // not Blosc-compressed
          w.raw(leaf.arrays[a]!);
        }
      }
      continue;
    }
    leaves.forEach((leaf, i) => {
      if (pass === 0) {
        w.u16(voxelBuffers[i]!.size);
      } else if (pass === 1) {
        if (i === 0 || !shared) {
          if (tamper.descriptor) {
            tamper.descriptor(w, i);
          } else {
            w.u8(shared ? 1 : 0);
            writeDescriptor(w, spec.attributes, spec.groups ?? []);
          }
        }
        const count = leaf.offsets[511]!;
        spec.attributes.forEach((attr, a) => {
          const stride = attr.stride ?? 1;
          const bytes = leaf.arrays[a]!.length;
          let meta: SynthArrayMeta = {
            total: BigInt(2 + 4 + bytes),
            flags: FLAG_CONSTANT_STRIDE,
            serialization: (stride !== 1 ? WRITE_STRIDED : 0) | (attr.uniform ? WRITE_UNIFORM : 0) | (paged ? WRITE_PAGED : 0),
            size: count,
            stride,
          };
          if (tamper.arrayMeta) meta = tamper.arrayMeta(i, a, meta);
          w.u64(meta.total);
          w.u8(meta.flags);
          w.u8(meta.serialization);
          w.u32(meta.size);
          if (meta.serialization & WRITE_STRIDED) w.u32(meta.stride);
        });
      } else if (pass === n + 2) {
        w.words(masks[i]!);
        w.u16(voxelBuffers[i]!.size);
        w.raw(voxelBuffers[i]!.bytes);
      }
    });
  }

  const endPos = w.pos;
  w.patchU64(offsetsAt, gridPos);
  w.patchU64(offsetsAt + 8, blockPos);
  w.patchU64(offsetsAt + 16, endPos);
  return w.bytes();
}
