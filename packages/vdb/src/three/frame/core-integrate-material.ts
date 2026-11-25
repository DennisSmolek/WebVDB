// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameIntegrateMaterial.js L33–138 (the proxy quad over the froxel area,
//   sRGB blending and its clamp rule, the scene depth with the empty-depth rule).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: the composite reads the core renderer's integrated prefix (two taps) through the core WGSL
// (`vdbFrameSliceCoord`, `vdbFrameIntegrateTaps`, `vdbFrameCompositeColor` by `wgslFn`). Divergences:
// DIVERGES.md (K6.1 rows).

/**
 * The proxy material of `VDBFrameRenderer` with `backend: 'core'` (internal, K6.1 A/B): every volume proxy
 * draws it; its fragment is the core composite (`vdbFrameIntegrate`'s maths) over three's
 * `viewportDepthTexture` and `viewportTexture`, sampling the core prefix through a stand-in texture
 * (`core-textures.ts`).
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import {
  VDB_FRAME_COMPOSITE_COLOR_FN_WGSL,
  VDB_FRAME_INTEGRATE_TAPS_WGSL,
  VDB_FRAME_SLICE_COORD_WGSL,
  VDB_FRAME_SRGB_EOTF_WGSL,
  VDB_FRAME_SRGB_OETF_WGSL,
} from "../../core/frame/wgsl/integrate.js";
import type { VDBFroxelLayout } from "../../core/frame/types.js";

const { Discard, Fn, If, clamp, float, mix, orthographicDepthToViewZ, perspectiveDepthToViewZ, positionGeometry, screenUV, select, texture3D, uint, uniform, vec2, vec3, vec4, viewportDepthTexture, viewportTexture, wgslFn } = TSL as unknown as Record<string, VDBTslNode>;

const sliceCoordFn = wgslFn(VDB_FRAME_SLICE_COORD_WGSL);
const tapsFn = wgslFn(VDB_FRAME_INTEGRATE_TAPS_WGSL);
const oetfFn = wgslFn(VDB_FRAME_SRGB_OETF_WGSL);
const eotfFn = wgslFn(VDB_FRAME_SRGB_EOTF_WGSL);
const compositeFn = wgslFn(VDB_FRAME_COMPOSITE_COLOR_FN_WGSL, [oetfFn, eotfFn]);

/** A 1³ zero placeholder until the first core update binds the prefix. */
function placeholder(): THREE.Data3DTexture {
  const t = new THREE.Data3DTexture(new Uint16Array(4), 1, 1, 1);
  t.type = THREE.HalfFloatType;
  t.format = THREE.RGBAFormat;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  return t;
}

export class VDBFrameCoreIntegrateMaterial extends NodeMaterial {
  /** The prefix texture node: its `value` is the stand-in of `outputs.froxels.integrated`. */
  readonly prefix: VDBTslNode = texture3D(placeholder());
  readonly frameGrid = uniform(new THREE.Vector3(1, 1, 1));
  readonly frameRange = uniform(new THREE.Vector2(0.1, 1.0));
  readonly frameScreenRect = uniform(new THREE.Vector4(-1, -1, 1, 1));
  readonly frameCameraNear = uniform(0.1);
  readonly frameCameraFar = uniform(1000.0);
  readonly frameCameraOrthographic = uniform(0, "int");
  srgbBlending: boolean | null = null;
  srgbClamp: boolean | null = null;

  constructor() {
    super();
    this.name = "VDBFrameCoreIntegrateMaterial";
    this.vertexNode = vec4(mix(this.frameScreenRect.xy, this.frameScreenRect.zw, positionGeometry.xy.mul(0.5).add(0.5)), 0.0, 1.0);
    this.setSrgbBlending(true, true);
    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.transparent = true;
    this.depthTest = false;
    this.depthWrite = false;
  }

  /** The froxel layout of the core outputs (each update). */
  setLayout(layout: VDBFroxelLayout, camera: { near: number; far: number; isOrthographicCamera?: boolean }): void {
    this.frameGrid.value.set(layout.width, layout.height, layout.depth);
    this.frameRange.value.set(layout.range[0], layout.range[1]);
    this.frameScreenRect.value.set(...layout.screenRect);
    this.frameCameraNear.value = camera.near;
    this.frameCameraFar.value = camera.far;
    this.frameCameraOrthographic.value = camera.isOrthographicCamera ? 1 : 0;
  }

  /** As `VDBFrameIntegrateMaterial.setSrgbBlending` (his rules). */
  setSrgbBlending(srgbBlending: boolean, srgbClamp: boolean): void {
    if (srgbBlending === this.srgbBlending && srgbClamp === this.srgbClamp) return;
    this.srgbBlending = srgbBlending;
    this.srgbClamp = srgbClamp;
    this.fragmentNode = this.createFragmentNode();
    this.needsUpdate = true;
    if (srgbBlending) {
      this.blending = THREE.NoBlending;
    } else {
      this.blending = THREE.CustomBlending;
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.OneFactor;
      this.blendDst = THREE.OneMinusSrcAlphaFactor;
      this.blendSrcAlpha = THREE.OneFactor;
      this.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
    }
  }

  // NOTE Empty depth (no geometry) gives an infinite scene depth (as the TSL integrate pass)
  private createSceneDepth(): VDBTslNode {
    return Fn((_: unknown, builder: { renderer: { reversedDepthBuffer?: boolean } }) => {
      const depth: VDBTslNode = (viewportDepthTexture() as VDBTslNode).x;
      const near = this.frameCameraNear;
      const far = this.frameCameraFar;
      const isEmpty = builder.renderer.reversedDepthBuffer ? depth.lessThanEqual(0.0) : depth.greaterThanEqual(1.0);
      const viewZ = select(this.frameCameraOrthographic.equal(1), orthographicDepthToViewZ(depth, near, far), perspectiveDepthToViewZ(depth, near, far));
      return select(isEmpty, float(1e20), viewZ.negate());
    });
  }

  private createFragmentNode(): VDBTslNode {
    const srgb = !!this.srgbBlending;
    const clampColors = !!this.srgbClamp;
    const sceneDepth = this.createSceneDepth();
    return Fn(() => {
      const grid = this.frameGrid;
      const depth = sceneDepth().toVar();
      const uv = vec2(screenUV.x, float(1.0).sub(screenUV.y));
      const kc = sliceCoordFn({ viewDepth: depth, range: this.frameRange, depth: grid.z }).toVar();
      const xy = clamp(uv.mul(grid.xy), vec2(0.5), grid.xy.sub(0.5)).div(grid.xy).toVar();
      const layers = grid.z.add(1.0);
      const a = this.prefix.sample(vec3(xy, kc.x.add(0.5).div(layers))).level(float(0.0));
      const b = this.prefix.sample(vec3(xy, kc.x.add(1.5).div(layers))).level(float(0.0));
      const r = tapsFn({ a, b, c: kc.y }).toVar();
      If(r.w.greaterThanEqual(1.0), () => {
        Discard();
      });
      const background: VDBTslNode = srgb ? (viewportTexture() as VDBTslNode) : vec4(0.0);
      return compositeFn({ r, sceneColor: background, srgb: uint(srgb ? 1 : 0), clampColors: uint(clampColors ? 1 : 0) });
    })();
  }
}
