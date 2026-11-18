/**
 * Minimal NanoVDB grid images for pool tests: a valid GridData / TreeData /
 * RootData header (ABI 32.9.1 offsets, cross-checked in image-header.test.ts)
 * followed by a deterministic byte pattern, at any size ≥ 800 bytes. The tree
 * itself is not walkable; the pool never walks it.
 */

export interface SyntheticImageOptions {
  bytes?: number;
  gridType?: number;
  gridClass?: number;
  flags?: number;
  name?: string;
  /** 3x3 row-major index -> world linear part (default: uniform `voxelSize`). */
  matD?: number[];
  voxelSize?: number;
  translation?: [number, number, number];
  bboxMin?: [number, number, number];
  bboxMax?: [number, number, number];
  background?: number;
  rootMax?: number;
  /** Seeds the payload pattern so two images differ. */
  seed?: number;
}

function invert3(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m as [number, number, number, number, number, number, number, number, number];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map(
    (v) => v / det,
  );
}

export function makeImage(o: SyntheticImageOptions = {}): Uint32Array {
  const bytes = o.bytes ?? 1024;
  if (bytes < 800 || bytes % 4) throw new Error("makeImage: bytes must be ≥ 800 and a multiple of 4");
  const out = new Uint32Array(bytes / 4);
  const v = new DataView(out.buffer);
  const vs = o.voxelSize ?? 1;
  const mat = o.matD ?? [vs, 0, 0, 0, vs, 0, 0, 0, vs];
  const inv = invert3(mat);
  const t = o.translation ?? [0, 0, 0];

  // Payload pattern first; the header overwrites the front.
  const seed = (o.seed ?? 1) >>> 0;
  for (let k = 0; k < out.length; k++) out[k] = (Math.imul(k + 1, 2654435761) ^ seed) >>> 0;

  v.setUint32(0, 0x6f6e614e, true); // "Nano"
  v.setUint32(4, 0x31424456, true); // "VDB1"
  v.setUint32(8, 0xffffffff, true); // checksum disabled
  v.setUint32(12, 0xffffffff, true);
  v.setUint32(16, (32 << 21) | (9 << 10) | 1, true);
  v.setUint32(20, o.flags ?? 0x3e, true);
  v.setUint32(24, 0, true); // grid index
  v.setUint32(28, 1, true); // grid count
  v.setUint32(32, bytes, true); // mGridSize lo
  v.setUint32(36, 0, true); // hi
  const name = new TextEncoder().encode(o.name ?? "synthetic");
  for (let k = 0; k < 256; k++) v.setUint8(40 + k, name[k] ?? 0);
  const map = 296;
  for (let k = 0; k < 9; k++) {
    v.setFloat32(map + 4 * k, mat[k]!, true);
    v.setFloat32(map + 36 + 4 * k, inv[k]!, true);
    v.setFloat64(map + 88 + 8 * k, mat[k]!, true);
    v.setFloat64(map + 160 + 8 * k, inv[k]!, true);
  }
  for (let k = 0; k < 3; k++) {
    v.setFloat32(map + 72 + 4 * k, t[k]!, true);
    v.setFloat64(map + 232 + 8 * k, t[k]!, true);
    v.setFloat64(608 + 8 * k, vs, true);
  }
  v.setUint32(632, o.gridClass ?? 2, true);
  v.setUint32(636, o.gridType ?? 1, true);
  // TreeData at 672: root right after it.
  v.setBigUint64(672 + 24, 64n, true);
  const root = 736;
  const lo = o.bboxMin ?? [-4, -4, -4];
  const hi = o.bboxMax ?? [3, 3, 3];
  for (let k = 0; k < 3; k++) {
    v.setInt32(root + 4 * k, lo[k]!, true);
    v.setInt32(root + 12 + 4 * k, hi[k]!, true);
  }
  v.setUint32(root + 24, 0, true); // tile count
  v.setFloat32(root + 28, o.background ?? 0, true);
  v.setFloat32(root + 32, 0, true);
  v.setFloat32(root + 36, o.rootMax ?? 1, true);
  return out;
}
