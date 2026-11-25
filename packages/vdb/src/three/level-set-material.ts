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
 * The trace, normal and pixel ray are the core's `vdbLevelSetSurface` / `vdbLevelSetViewRay` WGSL
 * (`core/surface/level-set-wgsl.ts`, K6.6), the same functions the raw pass `vdbCreateLevelSetPassGpu`
 * calls; this material adds three's lighting, shadows and depth conventions around them.
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/levelset/webgpu/LevelSetSurfaceMaterial.js`,
 * `LevelSetSurfaceCore.js`): the back-face proxy box (the camera can be
 * inside), perspective and orthographic rays (so shadow passes work), the mask,
 * depth, received-shadow position and the `setupLighting` override that moves
 * `positionView` to the hit. Copyright (c) mjurczyk/openvdb contributors, MIT
 * (see THIRD_PARTY_NOTICES.md). Changes: the trace is the core's `vdbLevelSetHit` on
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
import { vdbLevelSetSurfaceWgsl } from "../core/surface/level-set-wgsl.js";
import { getDevice, type VDBThreeRendererLike } from "./device.js";
import { VDBGridNode, type VDBTslNode } from "./grid-node.js";
import type { VDBUintUniform } from "./material.js";

const { Fn, code, uint, uniform, vec4, wgslFn, positionLocal, positionView, viewportUV } = TSL as unknown as Record<string, VDBTslNode>;
const { cameraFar, cameraNear, cameraProjectionMatrix, cameraProjectionMatrixInverse, cameraViewMatrix, cameraWorldMatrix, modelViewMatrix } =
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

const SURFACE = "vdbThreeLevelSetSurface";

/**
 * The material's entry: the core's pixel ray and surface (one call site: compile time, K3.7), packed into a
 * `mat4x4f` for TSL (three's `wgslFn` takes built-in types only): column 0 `(world hit, hit ? 1 : 0)`,
 * column 1 `(world normal, t)`, column 2 `(ndc depth, 0, 0, 0)`.
 */
function surfaceWgsl(cap: number): string {
  return /* wgsl */ `fn ${SURFACE}(grid_id: u32, uv: vec2f, projection_inverse: mat4x4f, camera_world: mat4x4f, view_projection: mat4x4f, orthographic: u32, sdf_to_index: f32, max_samples: f32, normal_step: f32) -> mat4x4f {
    let ray = vdbLevelSetViewRay(uv, projection_inverse, camera_world, orthographic != 0u);
    let s = vdbLevelSetSurface(grid_id, ray, view_projection, sdf_to_index, u32(clamp(max_samples, 1.0, ${cap}.0)), normal_step);
    return mat4x4f(vec4f(s.world, select(0.0, 1.0, s.hit)), vec4f(s.normal, s.t), vec4f(s.ndcDepth, 0.0, 0.0, 0.0), vec4f(0.0));
}`;
}

/** One surface `CodeNode` per `VDBGridNode` (included after its module; deduplicated within a shader). */
const surfaceCode = new WeakMap<VDBGridNode, VDBTslNode>();
function surfaceCodeOf(grids: VDBGridNode): VDBTslNode {
  let c = surfaceCode.get(grids);
  if (!c) surfaceCode.set(grids, (c = code(vdbLevelSetSurfaceWgsl(), [grids.module])));
  return c;
}

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
  /** Type flag: true. */
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
  /** The entry's WGSL (for inspection and tests): calls the core's `vdbLevelSetViewRay` and `vdbLevelSetSurface`. */
  readonly traceSource: string;
  /** The packed surface (`mat4`, fragment stage): see {@link traceNode}, {@link normalWorldNode}; column 2 `.x` is the NDC depth. */
  readonly surfaceNode: VDBTslNode;
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
  constructor(renderer: VDBThreeRendererLike, grids: VDBGridNode, gridId: TGridId, params: VDBLevelSetMaterialParameters = {}) {
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
    this.traceSource = surfaceWgsl(cap);

    const surfaceFn = wgslFn(this.traceSource, [surfaceCodeOf(grids)] as never);

    // The pixel's view ray (core vdbLevelSetViewRay): from the fragment's viewport uv through the inverse
    // projection, not from interpolated box positions, whose varying precision moves the ray by ~0.02 px on
    // large faces (V57). Perspective from the camera; orthographic (directional-light shadow maps) from the
    // pixel on the camera plane along the view axis.
    this.surfaceNode = Fn((builder: { camera?: { isOrthographicCamera?: boolean } }) =>
      surfaceFn({
        grid_id: this.gridId,
        uv: viewportUV,
        projection_inverse: cameraProjectionMatrixInverse,
        camera_world: cameraWorldMatrix,
        view_projection: cameraProjectionMatrix.mul(cameraViewMatrix),
        orthographic: uint(builder.camera?.isOrthographicCamera === true ? 1 : 0),
        sdf_to_index: this.sdfToIndex,
        max_samples: this.maxSamples,
        normal_step: this.normalStep,
      }),
    )
      .once()()
      .toVar("vdbThreeLevelSetSurfaceOut");
    this.traceNode = this.surfaceNode.element(0);
    this.hitWorldNode = this.traceNode.xyz;
    this.hitMaskNode = this.traceNode.w.greaterThan(0.5);
    this.hitViewNode = cameraViewMatrix.mul(vec4(this.hitWorldNode, 1.0)).xyz.toVar("vdbLevelSetHitView");
    this.normalWorldNode = this.surfaceNode.element(1).xyz;
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
