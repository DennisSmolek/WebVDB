// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameInjectMaterial.js L59–101 (constructor, blending),
//   L142–284 (froxel pre-integration, shadow march), L395–398 (ambient), L453–532 (scene lights,
//   radiance, premultiplied output); K3.4: L73–76 (octave and two-sided uniforms), L270–277 (octave
//   march depth and transmittance), L488–512 (the two-sided back march, frameShadowedLight with octaves).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; reads density only through `vdbVolumeNodes` (the VDBVolumeSource seam), so
// one material is built per volume source; grey only (no spectral outputs); no hemisphere,
// light-probe or environment terms yet (they read the 'frame' lighting uniforms; a later step);
// with a sparse source the shadow march skips empty space through the source's HDDA query (K3.2,
// same lattice, same values). Divergences: DIVERGES.md (K3.1, K3.2 and K3.4 rows), V23–V25.

/**
 * `'volume'` lighting (R1 §2): lights one volume in each froxel with a shadow march through that
 * volume's own source. Other volumes do not cast shadows on it. The renderer draws it once per
 * volume, one instance per slice of the volume's froxel rectangle.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import type { VDBThreeVolumeSource, VDBVolumeNodes, VDBVolumeNodesOptions, VDBVolumeUniforms } from "../volume-source.js";
import { vdbVolumeNodes } from "../volume-source.js";
import {
  createFrameVolumeUniforms,
  frameDistanceFade,
  frameFragCoord,
  frameGetFroxel,
  frameNoise,
  frameShadowAbsorption,
  frameShadowMarchDepth,
  frameShadowStep,
  frameShadowTransmittance,
  frameShadowedLight,
  frameTileVertex,
  frameTwoSidedDiffusion,
  getBoxSteps,
  getFroxelLod,
  getSampledLod,
  type FrameVolumeUniforms,
  type FroxelUniforms,
  type LightUniforms,
  type TileUniforms,
} from "./chunks.js";
import { vdbFrameLights } from "./lights.js";

// NOTE TSL's r187 typings narrow node types; the ported shader code is written against untyped nodes,
// as volume-source.ts is (VDBTslNode).
const { Break, Continue, Discard, Fn, If, Loop, ceil, dot, exp, float, int, length, mat3, max, min, mix, normalize, select, smoothstep, uniform, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

export interface VDBFrameInjectMaterialOptions {
  froxelUniforms: FroxelUniforms;
  froxelTileUniforms: TileUniforms;
  lightUniforms: LightUniforms;
  /** The volume source (dense, or sparse since K3.2: the same node shape). */
  source: VDBThreeVolumeSource;
  /** The source's live uniforms (`vdbVolumeUniforms`). */
  sourceUniforms: VDBVolumeUniforms;
  /** Sample the source's mip levels (his shaders built with mipmaps). */
  mipmaps: boolean;
  /** Emission colour hook of `vdbVolumeNodes` (DIVERGES V24). */
  emissionColor?: VDBVolumeNodesOptions["emissionColor"];
  /** Skip empty space in the shadow march when the source can (`skipEmpty`; sparse). Default false (K3.2 A/B). */
  hdda?: boolean;
}

// NOTE Lights one volume in each froxel, with a shadow march in this volume. Other volumes do not cast shadows.
// NOTE The renderer draws it one time for each volume, with the uniforms of that volume.
export class VDBFrameInjectMaterial extends NodeMaterial {
  readonly froxelUniforms: FroxelUniforms;
  readonly lightUniforms: LightUniforms;
  readonly volumeUniforms: FrameVolumeUniforms;
  readonly sourceUniforms: VDBVolumeUniforms;
  readonly source: VDBThreeVolumeSource;
  readonly frameLights = uniform(0, "int");
  readonly frameShadowSteps = uniform(32, "int");
  // NOTE Multiple scattering of this volume (phase.ts). 0.0 is single scattering.
  readonly frameMultipleScattering = uniform(0.0);
  // NOTE Scattering octaves of the renderer settings (phase.ts). Count 0 is off.
  readonly frameScatteringOctaves = uniform(new THREE.Vector4());
  // NOTE multipleScattering 'two-sided' (phase.ts): scene light shadows also march away from the light
  readonly frameTwoSided = uniform(0.0);
  readonly frameCameraMatrixWorld = uniform(new THREE.Matrix4());
  // NOTE The froxel rectangle and the slices of the volume of this draw. One instance for each slice.
  readonly frameRect = uniform(new THREE.Vector4());
  readonly frameSliceStart = uniform(0.0);
  mipmaps: boolean;
  hdda: boolean;
  readonly emissionColor: VDBVolumeNodesOptions["emissionColor"];

  constructor(o: VDBFrameInjectMaterialOptions) {
    super();

    this.name = "VDBFrameInjectMaterial";
    this.froxelUniforms = o.froxelUniforms;
    this.lightUniforms = o.lightUniforms;
    this.volumeUniforms = createFrameVolumeUniforms();
    this.source = o.source;
    this.sourceUniforms = o.sourceUniforms;
    this.mipmaps = o.mipmaps;
    this.hdda = o.hdda ?? false;
    this.emissionColor = o.emissionColor;

    this.vertexNode = frameTileVertex(o.froxelTileUniforms, this.frameRect, this.frameSliceStart);
    this.fragmentNode = this.createFragmentNode();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.transparent = true;
    this.depthTest = false;
    this.depthWrite = false;

    // NOTE Premultiplied "over" blend. Overlapping volumes in one froxel multiply their transmittance.
    this.blending = THREE.CustomBlending;
    this.blendEquation = THREE.AddEquation;
    this.blendSrc = THREE.OneFactor;
    this.blendDst = THREE.OneMinusSrcAlphaFactor;
    this.blendSrcAlpha = THREE.OneFactor;
    this.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
  }

  // NOTE Mip sampling is in the shader only when the volume uses mipmaps
  setMipmaps(mipmaps: boolean): void {
    if (mipmaps !== this.mipmaps) {
      this.mipmaps = mipmaps;
      this.fragmentNode = this.createFragmentNode();
      this.needsUpdate = true;
    }
  }

  // NOTE HDDA skipping is compiled in only when on (K3.2 A/B)
  setHdda(hdda: boolean): void {
    if (hdda !== this.hdda) {
      this.hdda = hdda;
      this.fragmentNode = this.createFragmentNode();
      this.needsUpdate = true;
    }
  }

  createFragmentNode(): VDBTslNode {
    const { froxelUniforms, lightUniforms, volumeUniforms, sourceUniforms, source } = this;
    const { froxelGrid, froxelSubSteps, froxelAtlasSize, frameJitter } = froxelUniforms;
    const { frameAmbientColor, frameLightCount, frameLightPositions, frameLightColors, frameLightSpots } = lightUniforms;
    const { frameViewToLocal, frameLocalToView, frameWorldToLocal, frameOpacity, frameAbsorbance, frameScatterColor } = volumeUniforms;
    const frameWorldScale = sourceUniforms.worldScale;
    const volume: VDBVolumeNodes = vdbVolumeNodes(source, sourceUniforms, { useMipmaps: this.mipmaps, ...(this.emissionColor ? { emissionColor: this.emissionColor } : {}) });
    const skipEmpty = this.hdda && volume.skipEmpty ? volume.skipEmpty.bind(volume) : null;

    const useLights = (flag: number) => this.frameLights.bitAnd(int(flag)).notEqual(0);

    return Fn(() => {
      const fragCoord = frameFragCoord(froxelAtlasSize).toVar();
      const froxel = frameGetFroxel(froxelUniforms, fragCoord);

      If(froxel.slice.greaterThanEqual(froxelGrid.z), () => {
        Discard();
      });

      const viewNear = froxel.viewNear.toVar();
      const viewFar = froxel.viewFar.toVar();
      const localNear = frameViewToLocal.mul(vec4(viewNear, 1.0)).xyz.toVar();
      const localFar = frameViewToLocal.mul(vec4(viewFar, 1.0)).xyz.toVar();
      const boxSegment = volume.boxSegment(localNear, localFar).toVar();

      If(boxSegment.x.greaterThanEqual(boxSegment.y), () => {
        Discard();
      });

      const segment = boxSegment.y.sub(boxSegment.x).toVar();
      const widthLod = getFroxelLod(froxelUniforms, froxel, volume, float(0.0), frameWorldScale);
      const sampledLod = getSampledLod(sourceUniforms, source.channels.mask, widthLod, this.mipmaps);
      const steps = getBoxSteps(volume, localNear, localFar, segment, sampledLod, froxelSubSteps);
      const subStepCount = float(steps);
      const subStepLength = length(viewFar.sub(viewNear)).mul(segment).div(subStepCount).toVar();
      const lod = max(widthLod, volume.getLod(subStepLength.div(frameWorldScale))).toVar();

      // NOTE Pre-integrate the slice along the view ray. Light the visibility-weighted mean position, not the froxel center.
      const transmittance = float(1.0).toVar();
      const visibleWeight = float(0.0).toVar();
      const visibleTSum = float(0.0).toVar();
      const jitter = frameNoise(frameJitter, fragCoord).toVar();

      Loop({ start: int(0), end: steps, type: "int", condition: "<" }, ({ i }: { i: VDBTslNode }) => {
        const t = boxSegment.x.add(segment.mul(float(i).add(jitter).div(subStepCount))).toVar();
        const opticalDepth = volume
          .sampleExtinction(mix(localNear, localFar, t), lod)
          .mul(frameOpacity)
          .mul(frameDistanceFade(froxelUniforms, mix(viewNear, viewFar, t)))
          .mul(subStepLength);
        const stepTransmittance = exp(opticalDepth.negate()).toVar();
        const weight = transmittance.mul(float(1.0).sub(stepTransmittance)).toVar();

        visibleWeight.addAssign(weight);
        visibleTSum.addAssign(weight.mul(t));
        transmittance.mulAssign(stepTransmittance);
      });

      const alpha = float(1.0).sub(transmittance).toVar();

      If(visibleWeight.lessThanEqual(0.0).or(alpha.lessThanEqual(1e-5)), () => {
        Discard();
      });

      const localPosition = mix(localNear, localFar, visibleTSum.div(visibleWeight)).toVar();
      const viewPosition = frameLocalToView.mul(vec4(localPosition, 1.0)).xyz;
      const worldPosition = this.frameCameraMatrixWorld.mul(vec4(viewPosition, 1.0)).xyz.toVar();
      const worldToLocalDirection = mat3(frameWorldToLocal);
      const rayDirection = normalize(mat3(this.frameCameraMatrixWorld).mul(viewFar.sub(viewNear))).toVar();
      const shadowNoise = frameNoise(frameJitter, fragCoord.add(vec2(17.0, 31.0))).toVar();
      const albedo = volume.sampleAlbedo(localPosition, lod).toVar();

      // NOTE Shadow march inside this volume only. Call in the Fn body.
      // NOTE Returns (direct, total, reduced depth, effective depth). The absorption uses the albedo at the lit position.
      const volumeTransmittance = (localDirection: VDBTslNode, maxLocalDistance: VDBTslNode) => {
        const result = vec4(1.0, 1.0, 0.0, 0.0).toVar();

        If(frameAbsorbance.greaterThan(0.0).and(this.frameShadowSteps.greaterThan(0)), () => {
          const direction = normalize(localDirection).toVar();
          const marchDistance = min(volume.boxExit(localPosition, direction), maxLocalDistance).toVar();

          If(marchDistance.greaterThan(0.0), () => {
            const localStep = marchDistance.div(float(this.frameShadowSteps)).toVar();
            const stepLod = max(lod, volume.getLod(localStep)).toVar();
            const worldStep = localStep.mul(length(mat3(frameLocalToView).mul(direction)));
            const stepScale = worldStep.mul(frameAbsorbance).toVar();
            const depths = vec3(0.0).toVar();

            if (skipEmpty) {
              // K3.2: the same sample lattice t_k = localStep (k + noise), but runs of samples the source proves
              // empty (exactly 0 extinction, so 0 depth) are jumped over. Positions are computed, not accumulated.
              // The source is asked only at the start and after a zero sample, so dense runs pay nothing for it.
              const k = int(0).toVar();
              const cellEnd = float(-1.0).toVar();
              const lastZero = float(1.0).toVar();
              const tEnd = marchDistance.add(localStep).toVar();

              Loop(k.lessThan(this.frameShadowSteps), () => {
                const t = localStep.mul(float(k).add(shadowNoise)).toVar();

                If(t.greaterThanEqual(cellEnd).and(lastZero.greaterThan(0.5)), () => {
                  const span = skipEmpty(localPosition, direction, t, tEnd, stepLod).toVar();
                  cellEnd.assign(span.y);

                  If(span.x.greaterThan(t), () => {
                    k.assign(max(k.add(1), int(ceil(span.x.div(localStep).sub(shadowNoise)))));
                    Continue();
                  });
                });

                const extinction = volume.sampleExtinction(localPosition.add(direction.mul(t)), stepLod).mul(stepScale).toVar();
                lastZero.assign(select(extinction.lessThanEqual(0.0), float(1.0), float(0.0)));

                depths.addAssign(frameShadowStep(extinction, frameShadowAbsorption(extinction, albedo), this.frameMultipleScattering));

                If(frameShadowMarchDepth(depths, this.frameMultipleScattering, this.frameScatteringOctaves).greaterThan(8.0), () => {
                  Break();
                });

                k.addAssign(1);
              });
            } else {
              const probe = localPosition.add(direction.mul(localStep).mul(shadowNoise)).toVar();

              Loop(this.frameShadowSteps, () => {
                const extinction = volume.sampleExtinction(probe, stepLod).mul(stepScale).toVar();

                depths.addAssign(frameShadowStep(extinction, frameShadowAbsorption(extinction, albedo), this.frameMultipleScattering));

                If(frameShadowMarchDepth(depths, this.frameMultipleScattering, this.frameScatteringOctaves).greaterThan(8.0), () => {
                  Break();
                });

                probe.addAssign(direction.mul(localStep));
              });
            }

            const shadow = frameShadowTransmittance(depths, this.frameMultipleScattering, this.frameScatteringOctaves);

            result.assign(vec4(shadow.x, shadow.y, depths.z, depths.y));
          });
        });

        return result;
      };

      const irradiance = vec3(0.0).toVar();

      // NOTE Ambient light. The light probe SH, hemisphere and environment terms come with K3.3/K3.4.
      If(useLights(vdbFrameLights.useAmbientLights), () => {
        irradiance.addAssign(frameAmbientColor);
      });

      // NOTE Point and spot lights have no distance falloff
      Loop(frameLightCount, ({ i }: { i: VDBTslNode }) => {
        const position = frameLightPositions.element(i).toVar();
        const color = frameLightColors.element(i).toVar();
        const spot = frameLightSpots.element(i).toVar();
        const isDirectional = position.w.lessThan(1.5);
        const isSpot = position.w.greaterThan(2.5);
        const enabled = select(
          isDirectional,
          useLights(vdbFrameLights.useDirectionalLights),
          select(isSpot, useLights(vdbFrameLights.useSpotLights), useLights(vdbFrameLights.usePointLights)),
        );
        const spotAttenuation = select(
          isSpot,
          smoothstep(spot.w, color.w, dot(normalize(position.xyz.sub(worldPosition)), spot.xyz)),
          float(1.0),
        ).toVar();

        If(enabled.and(spotAttenuation.greaterThan(0.0)), () => {
          const toLight = frameWorldToLocal.mul(vec4(position.xyz, 1.0)).xyz.sub(localPosition).toVar();
          const localDirection = select(isDirectional, worldToLocalDirection.mul(position.xyz), toLight);
          const maxDistance = select(isDirectional, float(1e6), length(toLight));
          const lightDirection = select(isDirectional, position.xyz, normalize(position.xyz.sub(worldPosition)));
          const shadow = volumeTransmittance(localDirection, maxDistance);

          // NOTE Two-sided: diffusion of the slab between the light side and the far side (frameTwoSidedDiffusion)
          If(this.frameTwoSided.greaterThan(0.5), () => {
            const back = volumeTransmittance(localDirection.negate(), float(1e6));

            shadow.y.assign(frameTwoSidedDiffusion(shadow.x, shadow.z, shadow.w, back.z));
          });

          const light = frameShadowedLight(volumeUniforms.framePhase, volumeUniforms.framePhaseSpectrum, this.frameScatteringOctaves, rayDirection, lightDirection, shadow.xy.toVar());

          irradiance.addAssign(color.rgb.mul(light).mul(spotAttenuation));
        });
      });

      const radiance = max(frameScatterColor, albedo.mul(irradiance).mul(1.0 / Math.PI)).add(volume.sampleEmission(localPosition, lod));

      // NOTE Premultiplied slice color and alpha. Linear filtering of these values gives transparent edges.
      return vec4(radiance.mul(alpha), alpha);
    })();
  }
}
