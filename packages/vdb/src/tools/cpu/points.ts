/**
 * CPU twins (D11) of the point-sprite WGSL (K6.6, `core/surface/points.ts`): `vdbPointSpriteSize` and
 * `vdbPointSpriteVertex`, f32 arithmetic in the WGSL's order (`Math.fround` per operation; GPUs may fuse
 * multiply-adds, so compare with a few-ulp tolerance), and {@link pointSpriteClip}, the raw pass's vertex.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (`src/three-openvdb/points/webgpu/VDBPoints.js`:
 * the attenuated size with a 1-pixel minimum) and three.js r187 (MIT) `PointsNodeMaterial.setupVertexSprite`
 * (the clip-space offset). Copyright (c) mjurczyk/openvdb contributors, MIT (see THIRD_PARTY_NOTICES.md).
 */

const f = Math.fround;

/** Twin of `vdbPointSpriteSize(pixelScale, viewDepth, minSize)`: `max(pixelScale / viewDepth, minSize)` pixels. */
export function pointSpriteSize(pixelScale: number, viewDepth: number, minSize: number): number {
  return Math.max(f(f(pixelScale) / f(viewDepth)), f(minSize));
}

/** Twin of `vdbPointSpriteVertex(clipCentre, corner, sizePx, viewportSize)`. */
export function pointSpriteVertex(
  clipCentre: readonly number[],
  corner: readonly [number, number],
  sizePx: number,
  viewportSize: readonly [number, number],
): [number, number, number, number] {
  const off = [0, 1].map((k) => f(f(f(f(corner[k]!) * f(sizePx)) / f(f(viewportSize[k]!) * 0.5)) * f(clipCentre[3]!)));
  return [f(f(clipCentre[0]!) + off[0]!), f(f(clipCentre[1]!) + off[1]!), f(clipCentre[2]!), f(clipCentre[3]!)];
}

const mat4Vec4 = (m: ArrayLike<number>, v: readonly number[]): number[] =>
  [0, 1, 2, 3].map((r) => f(f(f(f(m[r]!) * f(v[0]!)) + f(f(m[4 + r]!) * f(v[1]!))) + f(f(f(m[8 + r]!) * f(v[2]!)) + f(f(m[12 + r]!) * f(v[3]!)))));

/**
 * Twin of the raw point pass's vertex: `view = modelView · (local, 1)`, size from the view depth, then the
 * corner's clip position. Returns the clip position and the sprite width in pixels.
 *
 * @param local The point in local space.
 * @param modelView Column-major local → view.
 * @param projection Column-major view → clip.
 * @param corner The quad corner in `[−0.5, 0.5]²`.
 * @param pixelScale Pixels at view depth 1.
 * @param minSize Smallest sprite in pixels.
 * @param viewportSize Viewport width and height in pixels.
 */
export function pointSpriteClip(
  local: readonly number[],
  modelView: ArrayLike<number>,
  projection: ArrayLike<number>,
  corner: readonly [number, number],
  pixelScale: number,
  minSize: number,
  viewportSize: readonly [number, number],
): { clip: [number, number, number, number]; sizePx: number } {
  const view = mat4Vec4(modelView, [local[0]!, local[1]!, local[2]!, 1]);
  const sizePx = pointSpriteSize(pixelScale, -view[2]!, minSize);
  return { clip: pointSpriteVertex(mat4Vec4(projection, view), corner, sizePx, viewportSize), sizePx };
}
