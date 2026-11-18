/**
 * `VDBLevelSetMaterial`: a NanoVDB level set's zero surface, traced on the GPU
 * on the sparse pool (K3.7, R1 §6 Step 7), as a three r187
 * `MeshStandardNodeMaterial` (lights, shadows, depth, fog: three's own).
 *
 *   const grids = new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['levelSet'] });
 *   const material = new VDBLevelSetMaterial(renderer, grids, id, { color: 0xffffff, roughness: 0.5 });
 *   scene.add(new Mesh(vdbProxyGeometry(pool.info(id), { pad: 1 }), material));
 *   // per frame: pool.update(encoder) (or pool.flush()), then grids.update(), then render.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/levelset/webgpu/LevelSetSurfaceMaterial.js`,
 * `LevelSetSurfaceCore.js`): the back-face proxy box (the camera can be
 * inside), perspective and orthographic rays (so shadow passes work), the mask,
 * depth, received-shadow position and the `setupLighting` override that moves
 * `positionView` to the hit. Copyright (c) mjurczyk/openvdb contributors, MIT
 * (see THIRD_PARTY.md). Changes: the trace is the core's `vdbLevelSetHit` on
 * the sparse tree (HDDA skip, full-resolution trilinear SDF) instead of a
 * sphere trace of a dense 8-bit texture; the ray comes from the pixel through
 * the inverse projection, not from interpolated box positions; the normal is
 * `vdbLevelSetNormal` (central differences at `normalStep` voxels through the
 * grid affine); the proxy is in world space and the grid is placed with
 * `pool.setTransform` (DIVERGES.md).
 *
 * The trace runs in the grid's index space: `o = worldToIndex(ray origin)`,
 * `d = worldToIndexDir(unit world direction)`, so `t` is in world units. The
 * ray is the pixel's (`viewportUV` through `cameraProjectionMatrixInverse`):
 * perspective from `cameraPosition` over `t >= 0`; orthographic from the pixel
 * on the camera plane along the view axis over `t` in `[-1e5, 1e5]`. The mesh
 * must have an identity transform (the pool record holds the grid's
 * placement); give it `vdbProxyGeometry(pool.info(id), { pad: 1 })`, which
 * covers the trilinear support the trace clips to.
 */

import * as THREE from "three";
import { MeshStandardNodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBError } from "../core/errors.js";
import { getDevice, type ThreeRendererLike } from "./device.js";
import { VDBGridNode, type VDBTslNode } from "./grid-node.js";
import type { VDBUintUniform } from "./material.js";

const { Fn, float, uniform, vec2, vec3, vec4, wgslFn, positionLocal, positionView, viewportUV } = TSL as unknown as Record<string, VDBTslNode>;
const { cameraFar, cameraNear, cameraPosition, cameraProjectionMatrix, cameraProjectionMatrixInverse, cameraViewMatrix, cameraWorldMatrix, modelViewMatrix } =
  TSL as unknown as Record<string, VDBTslNode>;
const { viewZToOrthographicDepth, viewZToPerspectiveDepth } = TSL as unknown as Record<string, VDBTslNode>;

/** Default compiled cap on the samples one fragment's trace takes (the live `maxSamples` clamps under it). */
export const VDB_LEVEL_SET_SAMPLES_CAP = 1024;

/** Parameters of {@link VDBLevelSetMaterial}: three's standard material parameters plus the trace's. */
export interface VDBLevelSetMaterialParameters extends THREE.MeshStandardMaterialParameters {
  /**
   * Index voxels per SDF unit. Default: `1 / voxelSize.x` from `grids.pool.info(id)` when the pool
   * has `info` and the id is a number, else 1. A level set stores distances in its own world units,
   * so this is 1 / its NanoVDB voxel size.
   */
  sdfToIndex?: number;
  /** Trilinear samples per fragment trace (live), clamped to `[1, maxSamplesCap]`. Default 512. */
  maxSamples?: number;
  /** Compiled cap on {@link maxSamples}. Default {@link VDB_LEVEL_SET_SAMPLES_CAP}. */
  maxSamplesCap?: number;
  /** Central-difference step of the normal, in index voxels (live). Default 1. */
  normalStep?: number;
}

type ScalarUniform = ReturnType<typeof TSL.uniform> & { value: number };

const TRACE = "vdbThreeLevelSetTrace";
const NORMAL = "vdbThreeLevelSetNormal";

function traceWgsl(cap: number): string {
  return /* wgsl */ `fn ${TRACE}(grid_id: u32, origin_w: vec3f, dir_w: vec3f, t_range: vec2f, sdf_to_index: f32, max_samples: f32) -> vec4f {
    let o = vdbWorldToIndex(grid_id, origin_w);
    let d = vdbWorldToIndexDir(grid_id, dir_w);
    var span = t_range;
    let n = u32(clamp(max_samples, 1.0, ${cap}.0));
    if !vdbLevelSetHit(grid_id, o, d, &span, sdf_to_index, n) {
        return vec4f(0.0);
    }
    return vec4f(origin_w + dir_w * span.x, 1.0);
}`;
}

const NORMAL_WGSL = /* wgsl */ `fn ${NORMAL}(grid_id: u32, p_w: vec3f, h: f32) -> vec3f {
    return vdbLevelSetNormal(grid_id, vdbWorldToIndex(grid_id, p_w), h);
}`;

function bad(message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", `VDBLevelSetMaterial: ${message}`);
}

/**
 * A level set's surface on a world-space proxy box (see the file comment).
 * Shades as a `MeshStandardNodeMaterial`: `color`, `roughness`, `metalness`,
 * maps-free; lights and shadows read the hit point.
 *
 * - Exposes the trace as TSL nodes for custom shading or debugging:
 *   {@link hitWorldNode}, {@link hitViewNode}, {@link hitMaskNode},
 *   {@link normalWorldNode}, {@link normalViewNode} (e.g. `material.colorNode =
 *   material.normalViewNode.mul(0.5).add(0.5)` with no lights shows the normals).
 * - Perspective and orthographic cameras (directional-light shadow maps).
 *
 * @typeParam TGridId What the constructor got as the id: a number (default)
 *   makes {@link gridId} a `uint` uniform; a node is kept as given.
 */
export class VDBLevelSetMaterial<TGridId extends number | VDBTslNode = number> extends MeshStandardNodeMaterial {
  readonly isVDBLevelSetMaterial = true;
  /** The pool node this material reads. */
  readonly grids: VDBGridNode;
  /** The grid id: a `uint` uniform for a numeric id (`material.gridId.value = otherId`), else the caller's node. */
  readonly gridId: TGridId extends number ? VDBUintUniform : VDBTslNode;
  /** Index voxels per SDF unit (live). */
  readonly sdfToIndex: ScalarUniform;
  /** Trilinear samples per trace (live, clamped to the compiled cap). */
  readonly maxSamples: ScalarUniform;
  /** Central-difference step of the normal in index voxels (live). */
  readonly normalStep: ScalarUniform;
  /** The trace entry's WGSL (for inspection and tests). */
  readonly traceSource: string;
  /** `vec4(hit world position, 1)` on a hit, `vec4(0)` on a miss (fragment stage). */
  readonly traceNode: VDBTslNode;
  /** The hit in world space. */
  readonly hitWorldNode: VDBTslNode;
  /** The hit in view space. */
  readonly hitViewNode: VDBTslNode;
  /** True where the ray hit the surface (the material's `maskNode`). */
  readonly hitMaskNode: VDBTslNode;
  /** World-space unit normal at the hit. */
  readonly normalWorldNode: VDBTslNode;
  /** View-space unit normal at the hit (the material's `normalNode`). */
  readonly normalViewNode: VDBTslNode;

  /**
   * @param renderer The initialized `WebGPURenderer` the pool and `grids` live on.
   * @param grids A `VDBGridNode` on this renderer with profile `'march'` and helpers including `'levelSet'`.
   * @param gridId The level set's pool id (a number, made a `uint` uniform) or a `uint` node.
   * @param params three's `MeshStandardMaterialParameters` plus the trace's ({@link VDBLevelSetMaterialParameters}).
   * @throws VDBError `VDB_BAD_OPTION` for a `grids` this material can't use, a bad id or cap;
   *   `VDB_UNSUPPORTED_DEVICE` for an uninitialized renderer.
   */
  constructor(renderer: ThreeRendererLike, grids: VDBGridNode, gridId: TGridId, params: VDBLevelSetMaterialParameters = {}) {
    const { sdfToIndex, maxSamples, maxSamplesCap, normalStep, ...standard } = params;
    super(standard);
    this.name = "VDBLevelSetMaterial";
    if (!(grids instanceof VDBGridNode)) throw bad("grids must be a VDBGridNode.");
    getDevice(renderer);
    if (grids.renderer !== renderer) {
      throw bad("grids was made for another renderer: create one VDBGridNode per renderer.");
    }
    if (grids.options.profile !== "march" || !grids.options.helpers?.includes("levelSet")) {
      throw bad("grids needs profile 'march' and the 'levelSet' helper: new VDBGridNode(renderer, pool, { profile: 'march', helpers: ['levelSet'] }).");
    }
    if (typeof gridId === "number") {
      if (!Number.isInteger(gridId) || gridId < 0) throw bad(`gridId must be a non-negative integer, got ${gridId}.`);
    } else if (gridId === null || typeof gridId !== "object") {
      throw bad("gridId must be a number or a uint TSL node.");
    }
    const cap = maxSamplesCap ?? VDB_LEVEL_SET_SAMPLES_CAP;
    if (!Number.isInteger(cap) || cap < 1) throw bad(`maxSamplesCap must be an integer >= 1, got ${JSON.stringify(maxSamplesCap)}.`);
    let toIndex = sdfToIndex;
    if (toIndex === undefined) {
      const info = typeof gridId === "number" ? (grids.pool as { info?: (id: number) => { voxelSize: readonly number[] } }).info?.(gridId) : undefined;
      toIndex = info ? 1 / info.voxelSize[0]! : 1;
    }
    if (!(Number.isFinite(toIndex) && toIndex > 0)) throw bad(`sdfToIndex must be a positive finite number, got ${toIndex}.`);

    this.grids = grids;
    this.gridId = (typeof gridId === "number" ? uniform(gridId, "uint") : gridId) as this["gridId"];
    this.sdfToIndex = uniform(toIndex) as ScalarUniform;
    this.maxSamples = uniform(maxSamples ?? 512) as ScalarUniform;
    this.normalStep = uniform(normalStep ?? 1) as ScalarUniform;
    this.traceSource = traceWgsl(cap);

    const traceFn = wgslFn(this.traceSource, [grids.module] as never);
    const normalFn = wgslFn(NORMAL_WGSL, [grids.module] as never);

    // The pixel's view ray, from the fragment's viewport position through the inverse projection (not from
    // interpolated box positions, whose varying precision moves the ray by ~0.02 px on large faces). Perspective:
    // from the camera, t >= 0. Orthographic (directional-light shadow maps): from the pixel on the camera plane
    // along the view axis, t in [-1e5, 1e5].
    const ray = (builder: { camera?: { isOrthographicCamera?: boolean } }) => {
      const uv = viewportUV;
      const ndc = vec2(uv.x.mul(2.0).sub(1.0), float(1.0).sub(uv.y.mul(2.0)));
      const clip = cameraProjectionMatrixInverse.mul(vec4(ndc, 0.5, 1.0));
      const view = clip.xyz.div(clip.w);
      const ortho = builder.camera?.isOrthographicCamera === true;
      const dirView = ortho ? vec3(0.0, 0.0, -1.0) : view.normalize();
      const origin = ortho ? cameraWorldMatrix.mul(vec4(view.x, view.y, 0.0, 1.0)).xyz : cameraPosition;
      const dir = cameraWorldMatrix.mul(vec4(dirView, 0.0)).xyz.normalize();
      return { origin, dir, range: ortho ? vec2(-1e5, 1e5) : vec2(0.0, 1e30) };
    };
    this.traceNode = Fn((builder: { camera?: { isOrthographicCamera?: boolean } }) => {
      const r = ray(builder);
      return traceFn({
        grid_id: this.gridId,
        origin_w: r.origin,
        dir_w: r.dir,
        t_range: r.range,
        sdf_to_index: this.sdfToIndex,
        max_samples: this.maxSamples,
      });
    })
      .once()()
      .toVar("vdbLevelSetTrace");
    this.hitWorldNode = this.traceNode.xyz;
    this.hitMaskNode = this.traceNode.w.greaterThan(0.5);
    this.hitViewNode = cameraViewMatrix.mul(vec4(this.hitWorldNode, 1.0)).xyz.toVar("vdbLevelSetHitView");
    this.normalWorldNode = Fn(() => normalFn({ grid_id: this.gridId, p_w: this.hitWorldNode, h: this.normalStep }))
      .once()()
      .toVar("vdbLevelSetNormalWorld");
    this.normalViewNode = cameraViewMatrix.mul(vec4(this.normalWorldNode, 0.0)).xyz.normalize();

    this.side = THREE.BackSide;
    this.shadowSide = THREE.BackSide;
    this.maskNode = this.hitMaskNode;
    // A vertexNode makes the fragment's positionView a variable, which setupLighting moves to the hit.
    this.vertexNode = cameraProjectionMatrix.mul(modelViewMatrix.mul(vec4(positionLocal, 1.0)));
    const hitView = this.hitViewNode;
    this.depthNode = Fn((builder: { camera?: { isOrthographicCamera?: boolean } }) =>
      builder.camera?.isOrthographicCamera
        ? viewZToOrthographicDepth(hitView.z, cameraNear, cameraFar)
        : viewZToPerspectiveDepth(hitView.z, cameraNear, cameraFar),
    )();
    this.receivedShadowPositionNode = this.hitWorldNode;
    this.normalNode = this.normalViewNode;
  }

  /** Lights read `positionView`: the shading point is the hit, not the box face. */
  override setupLighting(builder: Parameters<MeshStandardNodeMaterial["setupLighting"]>[0]): ReturnType<MeshStandardNodeMaterial["setupLighting"]> {
    positionView.assign(this.hitViewNode).toStack();
    return super.setupLighting(builder);
  }
}
