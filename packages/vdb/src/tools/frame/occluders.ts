/**
 * K6.5b: mesh → volume shadows, the CPU twin of the core's occluder lookup (`vdbFrameOccluderVisibility`,
 * core/frame/wgsl/occluders.ts). A host depth shadow map (three's light shadow map, or any depth / r32float texture
 * drawn with a known view-projection per layer) answers "is this world position lit by the light": the first layer
 * whose clip volume holds the position is tested, with the host's bias, and filtered over its 2 × 2 nearest texels
 * (bilinear percentage-closer filtering, the hardware comparison filter) or one texel. Positions outside every layer
 * are lit (three's frustum rule). This is what volumetric fog in Unreal and Frostbite does per froxel (Hillaire 2015:
 * the light's shadow map sampled at each froxel), here also per shadow-grid voxel and per 'volume' lit position.
 */

import type { Vec3 } from "./index.js";

/** One host depth map as the twin reads it. */
export interface VDBFrameOccluderCpu {
  /** Per layer (cascade or cube face; ≤ 6): the view-projection the layer was drawn with (column-major, WebGPU depth [0, 1]). */
  readonly viewProjection: readonly ArrayLike<number>[];
  /** Added to the position's depth before the test (three's `shadow.bias` sign); subtracted with reversed depth. */
  readonly bias: number;
  /** The map stores 1 at the near plane (the test is `depth − bias ≥ stored`). */
  readonly reversedDepth: boolean;
  /** `'bilinear'`: 2 × 2 PCF with bilinear weights; `'nearest'`: one texel. */
  readonly filter: "bilinear" | "nearest";
  /** Texels across and up of every layer. */
  readonly size: readonly [number, number];
  /** The stored depth of texel (x, y) of `layer` (x right, y down from the top row, as `textureLoad`). */
  depthAt(layer: number, x: number, y: number): number;
}

/** Most layers of one occluder map (a cube's 6 faces). */
export const VDB_FRAME_OCCLUDER_MAX_LAYERS = 6;
/** Most occluder maps one kernel binds (fixed slots). */
export const VDB_FRAME_MAX_OCCLUDERS = 4;

/** Clip coordinates of a world position (column-major 4 × 4). */
function clipOf(m: ArrayLike<number>, p: Vec3): [number, number, number, number] {
  return [0, 1, 2, 3].map((r) => m[r]! * p[0] + m[4 + r]! * p[1] + m[8 + r]! * p[2] + m[12 + r]!) as [number, number, number, number];
}

/**
 * The layer that holds `world` and its map coordinates: the first layer (in order) whose clip volume contains the
 * position (|x|, |y| ≤ 1, 0 ≤ z ≤ 1, w > 0), with uv (0 at the left / top) and the NDC depth. Null: outside every layer.
 */
export function vdbFrameOccluderLayer(viewProjection: readonly ArrayLike<number>[], world: Vec3): { layer: number; uv: [number, number]; depth: number } | null {
  for (let layer = 0; layer < viewProjection.length; layer++) {
    const c = clipOf(viewProjection[layer]!, world);
    if (!(c[3] > 0)) continue;
    const x = c[0] / c[3];
    const y = c[1] / c[3];
    const z = c[2] / c[3];
    if (Math.abs(x) > 1 || Math.abs(y) > 1 || z < 0 || z > 1) continue;
    return { layer, uv: [x * 0.5 + 0.5, 0.5 - y * 0.5], depth: z };
  }
  return null;
}

/**
 * Visibility of `world` from the occluder's light, in [0, 1]: 1 lit, 0 behind an occluder (the twin of
 * `vdbFrameOccluderVisibility`). A march from the position toward the light stops at the first occluder, so this is
 * the march's transmittance factor.
 */
export function vdbFrameOccluderVisibility(o: VDBFrameOccluderCpu, world: Vec3): number {
  const at = vdbFrameOccluderLayer(o.viewProjection, world);
  if (!at) return 1;
  const reference = o.reversedDepth ? at.depth - o.bias : at.depth + o.bias;
  const [w, h] = o.size;
  const lit = (x: number, y: number) => {
    const d = o.depthAt(at.layer, Math.min(Math.max(x, 0), w - 1), Math.min(Math.max(y, 0), h - 1));
    return (o.reversedDepth ? reference >= d : reference <= d) ? 1 : 0;
  };
  if (o.filter === "nearest") return lit(Math.floor(at.uv[0] * w), Math.floor(at.uv[1] * h));
  const px = at.uv[0] * w - 0.5;
  const py = at.uv[1] * h - 0.5;
  const bx = Math.floor(px);
  const by = Math.floor(py);
  const fx = px - bx;
  const fy = py - by;
  const top = lit(bx, by) + (lit(bx + 1, by) - lit(bx, by)) * fx;
  const bottom = lit(bx, by + 1) + (lit(bx + 1, by + 1) - lit(bx, by + 1)) * fx;
  return top + (bottom - top) * fy;
}

/**
 * The transmittance target's occluded total of one light slot (core transmittance kernel): the direct part stays
 * unoccluded (the gather multiplies it by the visibility at its own, finer position) and the scattered part
 * (total − direct) is multiplied by the voxel's visibility, so light diffusing into a mesh shadow is smooth at grid
 * resolution.
 */
export function vdbFrameOccludedTotal(direct: number, total: number, visibility: number): number {
  return visibility < 1 ? direct + (total - direct) * visibility : total;
}
