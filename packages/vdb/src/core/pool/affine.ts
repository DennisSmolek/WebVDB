/**
 * f64 affine composition for grid records (docs/design/K1-core.md §1.6).
 * Affines here are 3x4 row-major: row r is `[m_r0, m_r1, m_r2, t_r]`, so
 * `out_r = m_r0 x + m_r1 y + m_r2 z + t_r`. The record stores them rounded to f32.
 * Negated translations are written `0 - x` so a zero stays +0 (stable record bits).
 */

import { VDBError } from "../errors.js";
import type { VDBImageHeader } from "./image-header.js";
import type { VDBVec3 } from "./layout.js";

/** A 3x4 row-major affine in f64. */
export type Affine3x4 = Float64Array;

/** World <-> index affines of one grid. */
export interface GridAffines {
  worldToIndex: Affine3x4;
  indexToWorld: Affine3x4;
}

function invert3x3(m: ArrayLike<number>, what: string): Float64Array {
  const [a, b, c, d, e, f, g, h, i] = [m[0]!, m[1]!, m[2]!, m[3]!, m[4]!, m[5]!, m[6]!, m[7]!, m[8]!];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!Number.isFinite(det) || det === 0) {
    throw new VDBError("VDB_BAD_OPTION", `${what} is singular (determinant ${det}); pass an invertible matrix.`);
  }
  const s = 1 / det;
  return new Float64Array([
    A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
    B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
    C * s, -(a * h - b * g) * s, (a * e - b * d) * s,
  ]);
}

/** Builds a 3x4 affine from a 3x3 row-major linear part and a translation. */
function affine(lin: ArrayLike<number>, t: ArrayLike<number>): Affine3x4 {
  const out = new Float64Array(12);
  for (let r = 0; r < 3; r++) {
    out[r * 4 + 0] = lin[r * 3 + 0]!;
    out[r * 4 + 1] = lin[r * 3 + 1]!;
    out[r * 4 + 2] = lin[r * 3 + 2]!;
    out[r * 4 + 3] = t[r]!;
  }
  return out;
}

/** `a ∘ b` (apply b, then a). */
export function composeAffine(a: Affine3x4, b: Affine3x4): Affine3x4 {
  const out = new Float64Array(12);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 4; c++) {
      let v = a[r * 4 + 0]! * b[0 * 4 + c]! + a[r * 4 + 1]! * b[1 * 4 + c]! + a[r * 4 + 2]! * b[2 * 4 + c]!;
      if (c === 3) v += a[r * 4 + 3]!;
      out[r * 4 + c] = v;
    }
  }
  return out;
}

/** Applies `m` to the point `p`. */
export function applyAffine(m: Affine3x4, p: ArrayLike<number>): VDBVec3 {
  const out: VDBVec3 = [0, 0, 0];
  for (let r = 0; r < 3; r++) {
    out[r] = m[r * 4]! * p[0]! + m[r * 4 + 1]! * p[1]! + m[r * 4 + 2]! * p[2]! + m[r * 4 + 3]!;
  }
  return out;
}

/** The file's own map: index -> (object) world and its inverse, from `matD`/`invMatD`/`vecD`. */
export function fileAffines(header: VDBImageHeader): GridAffines {
  const { matD, invMatD, vecD } = header;
  const t = [0, 1, 2].map((r) => 0 - (invMatD[r * 3]! * vecD[0]! + invMatD[r * 3 + 1]! * vecD[1]! + invMatD[r * 3 + 2]! * vecD[2]!));
  return { indexToWorld: affine(matD, vecD), worldToIndex: affine(invMatD, t) };
}

/**
 * Validates a 4x4 column-major `objectToWorld` (WebGPU / three convention) and
 * returns it with its inverse as 3x4 affines. The bottom row must be `0 0 0 1`.
 */
export function objectToWorldAffines(m: ArrayLike<number>): { forward: Affine3x4; inverse: Affine3x4 } {
  if (m.length !== 16) {
    throw new VDBError("VDB_BAD_OPTION", `objectToWorld must have 16 entries (4x4 column-major), got ${m.length}.`);
  }
  for (let k = 0; k < 16; k++) {
    if (!Number.isFinite(m[k])) {
      throw new VDBError("VDB_BAD_OPTION", `objectToWorld[${k}] is ${m[k]}; every entry must be finite.`);
    }
  }
  if (m[3] !== 0 || m[7] !== 0 || m[11] !== 0 || m[15] !== 1) {
    throw new VDBError(
      "VDB_BAD_OPTION",
      `objectToWorld must be affine (bottom row 0 0 0 1, i.e. entries 3, 7, 11, 15); got ${m[3]} ${m[7]} ${m[11]} ${m[15]}. ` +
        `Pass a column-major matrix such as three's Matrix4.elements.`,
    );
  }
  // Column-major: element (row r, column c) is m[c * 4 + r].
  const lin = [m[0]!, m[4]!, m[8]!, m[1]!, m[5]!, m[9]!, m[2]!, m[6]!, m[10]!];
  const t = [m[12]!, m[13]!, m[14]!];
  const inv = invert3x3(lin, "objectToWorld");
  const invT = [0, 1, 2].map((r) => 0 - (inv[r * 3]! * t[0]! + inv[r * 3 + 1]! * t[1]! + inv[r * 3 + 2]! * t[2]!));
  return { forward: affine(lin, t), inverse: affine(inv, invT) };
}

/**
 * The record's affines in f64: the file map, optionally followed by an instance
 * `objectToWorld`. `worldToIndex = map⁻¹ ∘ objectToWorld⁻¹`, `indexToWorld = objectToWorld ∘ map`.
 */
export function composeGridAffines(header: VDBImageHeader, objectToWorld: ArrayLike<number> | null): GridAffines {
  const file = fileAffines(header);
  if (objectToWorld === null) return file;
  const o = objectToWorldAffines(objectToWorld);
  return {
    indexToWorld: composeAffine(o.forward, file.indexToWorld),
    worldToIndex: composeAffine(file.worldToIndex, o.inverse),
  };
}
