// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameIntegrateMaterial.js L33–138 (constructor, sRGB blending,
//   scene depth), L219–294 (slice march and composite); K3.5: L45–47 (the alpha atlas), L80–106
//   (setSpectral, updateNodes), L140–217 (createSpectralFragmentNode).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript.
// Divergences: DIVERGES.md (K3.1 rows).

/**
 * Marches the froxel slices of each screen pixel and stops at the scene depth
 * (`viewportDepthTexture` of the current render), then composites over the scene colour, in sRGB
 * like WebGLRenderer (`srgbBlending`) or premultiplied in linear space. It is the material of every
 * `VDBFrameVolume` proxy; the first proxy drawn in a frame renders the froxels. Spectral (K3.5): the
 * transmittance of each channel from the second atlas target; hardware blending has one alpha, so
 * the pass reads the scene colour and composites itself.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { frameSample, frameSliceDepth, maxComponent, type FroxelUniforms } from "./chunks.js";

// NOTE TSL's r187 typings narrow node types; the ported shader code is written against untyped nodes,
// as volume-source.ts is (VDBTslNode).
const { Break, Discard, Fn, If, Loop, clamp, float, floor, max, min, mix, mod, orthographicDepthToViewZ, perspectiveDepthToViewZ, positionGeometry, pow, sRGBTransferEOTF, sRGBTransferOETF, screenUV, select, texture, uniform, vec2, vec3, vec4, viewportDepthTexture, viewportTexture } = TSL as unknown as Record<string, VDBTslNode>;

const maxSlices = 1024;

export class VDBFrameIntegrateMaterial extends NodeMaterial {
  readonly froxelAtlas: VDBTslNode = texture(new THREE.Texture());
  // NOTE Renderer settings.spectral: second atlas target (red alpha, blue alpha, 0, green alpha)
  readonly froxelAlphaAtlas: VDBTslNode = texture(new THREE.Texture());
  spectral = false;
  readonly frameCameraNear = uniform(0.1);
  readonly frameCameraFar = uniform(1000.0);
  readonly frameCameraOrthographic = uniform(0, "int");
  // NOTE Screen rectangle (clip x0, y0, x1, y1) and slices (first, end) of the froxel area. Other froxels are empty.
  readonly frameScreenRect = uniform(new THREE.Vector4(-1.0, -1.0, 1.0, 1.0));
  readonly frameSliceRange = uniform(new THREE.Vector2(0.0, 1.0));
  readonly froxelUniforms: FroxelUniforms;
  srgbBlending: boolean | null = null;
  srgbClamp: boolean | null = null;

  constructor({ froxelUniforms }: { froxelUniforms: FroxelUniforms }) {
    super();

    this.name = "VDBFrameIntegrateMaterial";
    this.froxelUniforms = froxelUniforms;

    // NOTE A quad over the screen rectangle of the froxel area (frame volume proxy geometry)
    this.vertexNode = vec4(mix(this.frameScreenRect.xy, this.frameScreenRect.zw, positionGeometry.xy.mul(0.5).add(0.5)), 0.0, 1.0);
    this.setSrgbBlending(true, true);

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.transparent = true;
    this.depthTest = false;
    this.depthWrite = false;
  }

  // NOTE WebGLRenderer blended on the sRGB canvas. WebGPURenderer blends in linear space.
  // NOTE sRGB blending reads the scene color and blends in sRGB, the same as WebGLRenderer.
  // NOTE srgbClamp: clamp the colors to 0.0 - 1.0. Tone mapping comes after this pass, so clamp only without it.
  setSrgbBlending(srgbBlending: boolean, srgbClamp: boolean): void {
    if (srgbBlending === this.srgbBlending && srgbClamp === this.srgbClamp) {
      return;
    }

    this.srgbBlending = srgbBlending;
    this.srgbClamp = srgbClamp;
    this.updateNodes();
  }

  setSpectral(spectral: boolean): void {
    if (spectral === this.spectral) {
      return;
    }

    this.spectral = spectral;
    this.updateNodes();
  }

  // NOTE Spectral: transmittance of each channel. Hardware blending has one alpha, so the pass reads the scene color.
  updateNodes(): void {
    this.fragmentNode = this.spectral ? this.createSpectralFragmentNode() : this.createFragmentNode();
    this.needsUpdate = true;

    if (this.srgbBlending || this.spectral) {
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

  // NOTE The 8-bit sRGB canvas of WebGLRenderer clamped both colors to 0.0 - 1.0 after tone mapping.
  // NOTE With tone mapping, the colors stay HDR. A clamp would cut bright scene color at the volume edges.
  createSrgbColor(): (color: VDBTslNode) => VDBTslNode {
    const { srgbClamp } = this;

    return (color) => (srgbClamp ? clamp(sRGBTransferOETF(color), 0.0, 1.0) : sRGBTransferOETF(color));
  }

  // NOTE Empty depth (no geometry) gives an infinite scene depth
  createSceneDepth(): VDBTslNode {
    return Fn((_: unknown, builder: { renderer: { reversedDepthBuffer?: boolean } }) => {
      const depth: VDBTslNode = (viewportDepthTexture() as VDBTslNode).x;
      const near = this.frameCameraNear;
      const far = this.frameCameraFar;
      const isEmpty = builder.renderer.reversedDepthBuffer ? depth.lessThanEqual(0.0) : depth.greaterThanEqual(1.0);
      const viewZ = select(
        this.frameCameraOrthographic.equal(1),
        orthographicDepthToViewZ(depth, near, far),
        perspectiveDepthToViewZ(depth, near, far),
      );

      return select(isEmpty, float(1e20), viewZ.negate());
    });
  }

  // NOTE The same march as createFragmentNode, with the alpha of each channel.
  // NOTE Composite: scene color * transmittance + volume color, in sRGB (srgbBlending) or in linear space.
  createSpectralFragmentNode(): VDBTslNode {
    const { froxelUniforms, srgbBlending } = this;
    const srgbColor = this.createSrgbColor();
    const { froxelGrid, froxelColumns, froxelAtlasSize } = froxelUniforms;
    const sceneDepth = this.createSceneDepth();

    return Fn(() => {
      const depth = sceneDepth().toVar();
      const uv = vec2(screenUV.x, float(1.0).sub(screenUV.y));
      const froxelUv = clamp(uv.mul(froxelGrid.xy), vec2(0.5), froxelGrid.xy.sub(0.5)).toVar();
      const sliceCount = froxelGrid.z;

      const radiance = vec3(0.0).toVar();
      const transmittance = vec3(1.0).toVar();
      const sliceStart = frameSliceDepth(froxelUniforms, this.frameSliceRange.x.div(sliceCount)).toVar();

      Loop(maxSlices, ({ i }: { i: VDBTslNode }) => {
        const slice = this.frameSliceRange.x.add(float(i));

        If(slice.greaterThanEqual(this.frameSliceRange.y).or(sliceStart.greaterThanEqual(depth)).or(maxComponent(transmittance).lessThan(0.001)), () => {
          Break();
        });

        const sliceEnd = frameSliceDepth(froxelUniforms, slice.add(1.0).div(sliceCount)).toVar();
        const coverage = min(sliceEnd, depth).sub(sliceStart).div(sliceEnd.sub(sliceStart)).toVar();

        const tile = vec2(mod(slice, froxelColumns), floor(slice.div(froxelColumns)));
        const atlasUv = tile.mul(froxelGrid.xy).add(froxelUv).div(froxelAtlasSize).toVar();
        const froxel = frameSample(this.froxelAtlas, atlasUv).toVar();
        const froxelAlphas = frameSample(this.froxelAlphaAtlas, atlasUv).toVar();
        const froxelAlpha = clamp(vec3(froxelAlphas.x, froxel.a, froxelAlphas.y), 0.0, 1.0).toVar();

        // NOTE Froxels store premultiplied slice color and alpha of each channel. Scene depth can cut the last slice.
        If(maxComponent(froxelAlpha).greaterThan(0.0), () => {
          const sliceAlpha = select(coverage.lessThan(1.0), vec3(1.0).sub(pow(vec3(1.0).sub(froxelAlpha), vec3(coverage))), froxelAlpha).toVar();

          radiance.addAssign(transmittance.mul(froxel.rgb).mul(sliceAlpha.div(max(froxelAlpha, vec3(1e-6)))));
          transmittance.mulAssign(vec3(1.0).sub(sliceAlpha));
        });

        sliceStart.assign(sliceEnd);
      });

      const alpha = vec3(1.0).sub(transmittance).toVar();

      If(maxComponent(alpha).lessThanEqual(0.0), () => {
        Discard();
      });

      const background: VDBTslNode = (viewportTexture() as VDBTslNode).toVar();

      // NOTE Output alpha uses the green channel
      const outputAlpha = alpha.y.add(background.a.mul(float(1.0).sub(alpha.y)));

      if (srgbBlending) {
        const volumeColor = srgbColor(radiance.div(max(alpha, vec3(1e-6))));
        const backgroundColor = srgbColor(background.rgb);

        return vec4(sRGBTransferEOTF(mix(backgroundColor, volumeColor, alpha)), outputAlpha);
      }

      return vec4(radiance.add(background.rgb.mul(transmittance)), outputAlpha);
    })();
  }

  createFragmentNode(): VDBTslNode {
    const { froxelUniforms, srgbBlending } = this;
    const { froxelGrid, froxelColumns, froxelAtlasSize } = froxelUniforms;
    const srgbColor = this.createSrgbColor();

    const sceneDepth = this.createSceneDepth();

    return Fn(() => {
      const depth = sceneDepth().toVar();
      const uv = vec2(screenUV.x, float(1.0).sub(screenUV.y));
      const froxelUv = clamp(uv.mul(froxelGrid.xy), vec2(0.5), froxelGrid.xy.sub(0.5)).toVar();
      const sliceCount = froxelGrid.z;

      const radiance = vec3(0.0).toVar();
      const transmittance = float(1.0).toVar();
      const sliceStart = frameSliceDepth(froxelUniforms, this.frameSliceRange.x.div(sliceCount)).toVar();

      Loop(maxSlices, ({ i }: { i: VDBTslNode }) => {
        const slice = this.frameSliceRange.x.add(float(i));

        If(slice.greaterThanEqual(this.frameSliceRange.y).or(sliceStart.greaterThanEqual(depth)).or(transmittance.lessThan(0.001)), () => {
          Break();
        });

        const sliceEnd = frameSliceDepth(froxelUniforms, slice.add(1.0).div(sliceCount)).toVar();
        const coverage = min(sliceEnd, depth).sub(sliceStart).div(sliceEnd.sub(sliceStart));

        const tile = vec2(mod(slice, froxelColumns), floor(slice.div(froxelColumns)));
        const froxel = frameSample(this.froxelAtlas, tile.mul(froxelGrid.xy).add(froxelUv).div(froxelAtlasSize)).toVar();
        const froxelAlpha = clamp(froxel.a, 0.0, 1.0).toVar();

        // NOTE Froxels store premultiplied slice color and alpha. Scene depth can cut the last slice.
        If(froxelAlpha.greaterThan(0.0), () => {
          const sliceAlpha = select(coverage.lessThan(1.0), float(1.0).sub(pow(float(1.0).sub(froxelAlpha), coverage)), froxelAlpha).toVar();

          radiance.addAssign(transmittance.mul(froxel.rgb).mul(sliceAlpha.div(froxelAlpha)));
          transmittance.mulAssign(float(1.0).sub(sliceAlpha));
        });

        sliceStart.assign(sliceEnd);
      });

      const alpha = float(1.0).sub(transmittance).toVar();

      If(alpha.lessThanEqual(0.0), () => {
        Discard();
      });

      if (srgbBlending) {
        const background: VDBTslNode = (viewportTexture() as VDBTslNode).toVar();
        const volumeColor = srgbColor(radiance.div(alpha));
        const backgroundColor = srgbColor(background.rgb);

        return vec4(sRGBTransferEOTF(mix(backgroundColor, volumeColor, alpha)), alpha.add(background.a.mul(float(1.0).sub(alpha))));
      }

      // NOTE Premultiplied output. The renderer output pass applies tone mapping and color space.
      return vec4(radiance, alpha);
    })();
  }
}
