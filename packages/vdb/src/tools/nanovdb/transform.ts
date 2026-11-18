/**
 * transform.ts — apply an affine transform to a NanoVDB grid as a metadata-only
 * Map edit (SPEC decision D6). An axis-aligned uniform-scale + translate does not
 * move any voxel between nodes, so the entire tree/leaf region is untouched: we
 * copy the image and rewrite only GridData's `Map` (mat/invMat/vec, both the f32
 * and f64 copies), the `voxelSize` triple and the world-space bbox.
 *
 * v1 handles uniform scale + translate only; a rotation, shear or non-uniform
 * scale throws (those need GPU resampling — the roadmap). The voxel bytes are
 * provably preserved (the tree region is bit-for-bit identical to the source),
 * which the test asserts.
 */

import {
  GRID_OFF_MAP,
  GRID_OFF_VOXEL_SIZE,
  GRID_OFF_WORLD_BBOX,
  GRID_SIZE,
  MAP_OFF_INVMATD,
  MAP_OFF_INVMATF,
  MAP_OFF_MATD,
  MAP_OFF_MATF,
  MAP_OFF_TAPERD,
  MAP_OFF_TAPERF,
  MAP_OFF_VECD,
  MAP_OFF_VECF,
  ROOT_OFF_BBOX_MAX,
  ROOT_OFF_BBOX_MIN,
  TREE_OFF_NODE_OFFSET_ROOT,
} from "./bytes.js";

/** Uniform scale + translate specified directly. */
export interface TransformSpec {
  /** New uniform voxel size (world units per voxel). */
  voxelSize: number;
  /** New world-space position of index origin (the Map's translation). */
  worldOrigin: [number, number, number];
}

export type TransformInput = Float32Array | number[] | TransformSpec;

/**
 * Returns a copy of `image` with its transform replaced by an affine uniform
 * scale + translate. Accepts either a row-major 4x4 matrix (`Float32Array` /
 * `number[16]`) or an explicit `{ voxelSize, worldOrigin }`.
 */
export function transform(image: Uint32Array, input: TransformInput): Uint32Array {
  const { voxelSize, translation } = resolve(input);
  if (!(voxelSize > 0)) {
    throw new Error(`transform: voxelSize must be positive, got ${voxelSize}`);
  }

  const out = image.slice(); // deep copy — source is never mutated
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);

  writeMap(view, voxelSize, translation);

  // voxelSize triple (f64[3]).
  view.setFloat64(GRID_OFF_VOXEL_SIZE, voxelSize, true);
  view.setFloat64(GRID_OFF_VOXEL_SIZE + 8, voxelSize, true);
  view.setFloat64(GRID_OFF_VOXEL_SIZE + 16, voxelSize, true);

  // World bbox = index bbox mapped through the new transform.
  const rootOff = GRID_SIZE + Number(view.getBigUint64(GRID_SIZE + TREE_OFF_NODE_OFFSET_ROOT, true));
  const imin: [number, number, number] = [
    view.getInt32(rootOff + ROOT_OFF_BBOX_MIN, true),
    view.getInt32(rootOff + ROOT_OFF_BBOX_MIN + 4, true),
    view.getInt32(rootOff + ROOT_OFF_BBOX_MIN + 8, true),
  ];
  const imax: [number, number, number] = [
    view.getInt32(rootOff + ROOT_OFF_BBOX_MAX, true),
    view.getInt32(rootOff + ROOT_OFF_BBOX_MAX + 4, true),
    view.getInt32(rootOff + ROOT_OFF_BBOX_MAX + 8, true),
  ];
  for (let a = 0; a < 3; a++) {
    view.setFloat64(GRID_OFF_WORLD_BBOX + a * 8, imin[a]! * voxelSize + translation[a]!, true);
    // "+ one voxel" convention: the world max corner is (indexMax + 1) * voxelSize.
    view.setFloat64(GRID_OFF_WORLD_BBOX + 24 + a * 8, (imax[a]! + 1) * voxelSize + translation[a]!, true);
  }

  return out;
}

function resolve(input: TransformInput): { voxelSize: number; translation: [number, number, number] } {
  if (input instanceof Float32Array || Array.isArray(input)) {
    const m = input;
    if (m.length !== 16) {
      throw new Error(`transform: expected a 4x4 (16-element) matrix, got ${m.length}`);
    }
    const sx = m[0]!;
    const sy = m[5]!;
    const sz = m[10]!;
    const offDiagonal = [m[1]!, m[2]!, m[4]!, m[6]!, m[8]!, m[9]!];
    const eps = 1e-6 * Math.max(1, Math.abs(sx), Math.abs(sy), Math.abs(sz));
    const rotated = offDiagonal.some((v) => Math.abs(v) > eps);
    const nonUniform = Math.abs(sx - sy) > eps || Math.abs(sx - sz) > eps;
    if (rotated || nonUniform) {
      throw new Error(
        `transform: matrix is ${rotated ? "rotated/sheared" : "non-uniformly scaled"} ` +
          `(scale [${sx}, ${sy}, ${sz}]). v1 supports uniform scale + translate only; ` +
          `rotation/shear/anisotropy need GPU resampling (roadmap).`,
      );
    }
    return { voxelSize: sx, translation: [m[3]!, m[7]!, m[11]!] };
  }
  return { voxelSize: input.voxelSize, translation: [...input.worldOrigin] };
}

function writeMap(view: DataView, s: number, t: [number, number, number]): void {
  const inv = 1 / s;
  const mat = [s, 0, 0, 0, s, 0, 0, 0, s];
  const invMat = [inv, 0, 0, 0, inv, 0, 0, 0, inv];
  const base = GRID_OFF_MAP;
  for (let i = 0; i < 9; i++) {
    view.setFloat32(base + MAP_OFF_MATF + i * 4, mat[i]!, true);
    view.setFloat32(base + MAP_OFF_INVMATF + i * 4, invMat[i]!, true);
    view.setFloat64(base + MAP_OFF_MATD + i * 8, mat[i]!, true);
    view.setFloat64(base + MAP_OFF_INVMATD + i * 8, invMat[i]!, true);
  }
  for (let i = 0; i < 3; i++) {
    view.setFloat32(base + MAP_OFF_VECF + i * 4, t[i]!, true);
    view.setFloat64(base + MAP_OFF_VECD + i * 8, t[i]!, true);
  }
  view.setFloat32(base + MAP_OFF_TAPERF, 1, true);
  view.setFloat64(base + MAP_OFF_TAPERD, 1, true);
}
