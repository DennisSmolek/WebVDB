// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameInjectMaterial.js L59–101 (constructor, blending),
//   L142–284 (froxel pre-integration, shadow march), L395–398 (ambient), L453–532 (scene lights,
//   radiance, premultiplied output); K3.4: L73–76 (octave and two-sided uniforms), L270–277 (octave
//   march depth and transmittance), L488–512 (the two-sided back march, frameShadowedLight with octaves);
//   K3.5: L80–81 and L103–130 (spectral flag, setSpectral, rebuildNodes and the two spectral outputs),
//   L200–229 (the spectral pre-integration), L286–348 (the spectral shadow march), L350–383
//   (totalTransmittance, the diffusion pair, isotropicTransmittance), L385–451 (the light-probe,
//   environment and hemisphere terms), L499–532 (frameSpectralShadowedLight, the spectral outputs).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; reads density only through `vdbVolumeNodes` (the VDBVolumeSource seam), so
// one material is built per volume source; with a sparse source the shadow march skips empty space
// through the source's HDDA query (K3.2, same lattice, same values; also for the spectral march); the
// hemisphere and environment-light loops are compiled in only when the scene has such lights, and isotropic
// light marches each direction once (his code marches the front direction again when diffusion is off):
// the same values, a smaller shader (the HDDA march is large on SwiftShader).
// Divergences: DIVERGES.md (K3.1, K3.2, K3.4 and K3.5 rows), V23–V25.

/**
 * `'volume'` lighting (R1 §2): lights one volume in each froxel with a shadow march through that
 * volume's own source. Other volumes do not cast shadows on it. The renderer draws it once per
 * volume, one instance per slice of the volume's froxel rectangle. Lights: ambient and the light
 * probe (along the outward normal), the scene environment (environment lights with one march each,
 * or the PMREM irradiance along the outward direction), hemisphere lights (sky and ground marches),
 * and the directional, point and spot lights, each gated by the volume's `lights` bitmask. Diffusion
 * multiple scattering gives isotropic light the two-sided total of a march pair (white furnace).
 * Spectral (K3.5): per-channel extinction (the volume's spectrum), albedo and multiple scattering,
 * two atlas targets.
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import type { VDBThreeVolumeSource, VDBVolumeNodes, VDBVolumeNodesOptions, VDBVolumeUniforms } from "../volume-source.js";
import { vdbVolumeNodes } from "../volume-source.js";
import {
  createFrameVolumeUniforms,
  frameDistanceFade,
  frameFragCoord,
  frameGetFroxel,
  frameIsotropicDiffusion,
  frameMeanComponent,
  frameNoise,
  frameShadowAbsorption,
  frameShadowMarchDepth,
  frameShadowStep,
  frameShadowTotal,
  frameShadowTransmittance,
  frameShadowedLight,
  frameSpectralShadowDepths,
  frameSpectralShadowTotal,
  frameSpectralShadowTransmittance,
  frameSpectralShadowUnit,
  frameSpectralShadowedLight,
  frameTileVertex,
  frameTwoSidedDiffusion,
  getBoxSteps,
  getFroxelLod,
  getSampledLod,
  maxComponent,
  minComponent,
  type FrameSpectralDepths,
  type FrameVolumeUniforms,
  type FroxelUniforms,
  type LightUniforms,
  type TileUniforms,
} from "./chunks.js";
import { VDBFrameEnvironmentMaterial } from "./environment-material.js";
import { vdbFrameLights } from "./lights.js";

// NOTE TSL's r187 typings narrow node types; the ported shader code is written against untyped nodes,
// as volume-source.ts is (VDBTslNode).
const { Break, Continue, Discard, Fn, If, Loop, ceil, dot, exp, float, getShIrradianceAt, int, length, mat3, max, min, mix, normalize, outputStruct, property, select, smoothstep, uniform, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

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
  /** Renderer settings.spectral (K3.5). Default false. */
  spectral?: boolean;
  /** Compile the hemisphere-light loop (the scene has hemisphere lights). Default false. */
  hemisphereLights?: boolean;
  /** Compile the environment-light loop (the renderer has 'directional' / SH environment lights). Default false. */
  environmentLights?: boolean;
}

/** A shadow march result: (direct, total, reduced depth, effective depth), floats or (spectral) vec3. */
interface InjectShadow {
  x: VDBTslNode;
  y: VDBTslNode;
  z: VDBTslNode;
  w: VDBTslNode;
}

// NOTE Lights one volume in each froxel, with a shadow march in this volume. Other volumes do not cast shadows.
// NOTE The renderer draws it one time for each volume, with the uniforms of that volume.
export class VDBFrameInjectMaterial extends VDBFrameEnvironmentMaterial {
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
  // NOTE Renderer settings.spectral: extinction and multiple scattering of each channel, two atlas targets
  spectral: boolean;
  // NOTE Light loops compiled in (setLightTerms). Without them the loop counts are 0, so the result is the same.
  hemisphereLights: boolean;
  environmentLights: boolean;
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
    this.spectral = o.spectral ?? false;
    this.hemisphereLights = o.hemisphereLights ?? false;
    this.environmentLights = o.environmentLights ?? false;
    this.emissionColor = o.emissionColor;

    this.vertexNode = frameTileVertex(o.froxelTileUniforms, this.frameRect, this.frameSliceStart);
    this.rebuildNodes();

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
      this.environmentNodes = [];
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  // NOTE HDDA skipping is compiled in only when on (K3.2 A/B)
  setHdda(hdda: boolean): void {
    if (hdda !== this.hdda) {
      this.hdda = hdda;
      this.environmentNodes = [];
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  setSpectral(spectral: boolean): void {
    if (spectral !== this.spectral) {
      this.spectral = spectral;
      this.environmentNodes = [];
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  // NOTE Compile the hemisphere and environment-light loops only for scenes that have those lights
  setLightTerms(hemisphereLights: boolean, environmentLights: boolean): void {
    if (hemisphereLights !== this.hemisphereLights || environmentLights !== this.environmentLights) {
      this.hemisphereLights = hemisphereLights;
      this.environmentLights = environmentLights;
      this.environmentNodes = [];
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  // NOTE Spectral: atlas target 0 is (color, green alpha), target 1 is (red alpha, blue alpha, 0, green alpha).
  // NOTE Both targets blend with the green alpha (w of each output).
  override rebuildNodes(): void {
    if (!this.spectral) {
      this.colorNode = null;
      this.outputNode = null;
      this.fragmentNode = this.createFragmentNode();

      return;
    }

    const injectColor = property("vec4");
    const injectAlpha = property("vec4");

    this.fragmentNode = null;
    this.colorNode = this.createFragmentNode({ color: injectColor, alpha: injectAlpha });
    this.outputNode = outputStruct(injectColor, injectAlpha);
  }

  // NOTE spectralOutputs: { color, alpha } property nodes of the spectral targets. null: one target.
  createFragmentNode(spectralOutputs: { color: VDBTslNode; alpha: VDBTslNode } | null = null): VDBTslNode {
    const spectral = spectralOutputs !== null;
    const { froxelUniforms, lightUniforms, volumeUniforms, sourceUniforms, source } = this;
    const { froxelGrid, froxelSubSteps, froxelAtlasSize, frameJitter } = froxelUniforms;
    const { frameAmbientColor, frameLightProbe, frameHemiCount, frameHemiSkyColors, frameHemiGroundColors, frameHemiDirections, frameLightCount, frameLightPositions, frameLightColors, frameLightSpots, frameEnvironmentCount, frameEnvironmentDirections, frameEnvironmentColors } = lightUniforms;
    const { frameViewToLocal, frameLocalToView, frameWorldToLocal, frameOpacity, frameAbsorbance, frameScatterColor, frameExtinctionSpectrum } = volumeUniforms;
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
      // NOTE Spectral: transmittance of each channel. The visible position uses the mean of the channels.
      const transmittance = (spectral ? vec3(1.0) : float(1.0)).toVar();
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
        const stepTransmittance = exp((spectral ? frameExtinctionSpectrum.mul(opticalDepth) : opticalDepth).negate()).toVar();
        const stepWeight = transmittance.mul(float(1.0).sub(stepTransmittance));
        const weight = (spectral ? frameMeanComponent(stepWeight) : stepWeight).toVar();

        visibleWeight.addAssign(weight);
        visibleTSum.addAssign(weight.mul(t));
        transmittance.mulAssign(stepTransmittance);
      });

      const alpha = (spectral ? vec3(1.0) : float(1.0)).sub(transmittance).toVar();

      If(visibleWeight.lessThanEqual(0.0).or((spectral ? maxComponent(alpha) : alpha).lessThanEqual(1e-5)), () => {
        Discard();
      });

      const localPosition = mix(localNear, localFar, visibleTSum.div(visibleWeight)).toVar();
      const viewPosition = frameLocalToView.mul(vec4(localPosition, 1.0)).xyz;
      const worldPosition = this.frameCameraMatrixWorld.mul(vec4(viewPosition, 1.0)).xyz.toVar();
      const worldToLocalDirection = mat3(frameWorldToLocal);
      const rayDirection = normalize(mat3(this.frameCameraMatrixWorld).mul(viewFar.sub(viewNear))).toVar();
      const shadowNoise = frameNoise(frameJitter, fragCoord.add(vec2(17.0, 31.0))).toVar();
      const albedo = volume.sampleAlbedo(localPosition, lod).toVar();

      // NOTE The shadow sample loop of both marches. sample(extinction) adds one sample (extinction × world step × absorbance);
      // stop() is the march depth test after it. The HDDA variant (K3.2) jumps over the runs of samples the source proves empty.
      const marchSamples = (direction: VDBTslNode, marchDistance: VDBTslNode, localStep: VDBTslNode, stepLod: VDBTslNode, stepScale: VDBTslNode, sample: (extinction: VDBTslNode) => void, stop: () => VDBTslNode) => {
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

            sample(extinction);

            If(stop(), () => {
              Break();
            });

            k.addAssign(1);
          });
        } else {
          const probe = localPosition.add(direction.mul(localStep).mul(shadowNoise)).toVar();

          Loop(this.frameShadowSteps, () => {
            const extinction = volume.sampleExtinction(probe, stepLod).mul(stepScale).toVar();

            sample(extinction);

            If(stop(), () => {
              Break();
            });

            probe.addAssign(direction.mul(localStep));
          });
        }
      };

      // NOTE Shadow march inside this volume only. Call in the Fn body.
      // NOTE Returns (direct, total, reduced depth, effective depth). The absorption uses the albedo at the lit position.
      const volumeTransmittance = (localDirection: VDBTslNode, maxLocalDistance: VDBTslNode): InjectShadow => {
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

            marchSamples(
              direction,
              marchDistance,
              localStep,
              stepLod,
              stepScale,
              (extinction) => {
                depths.addAssign(frameShadowStep(extinction, frameShadowAbsorption(extinction, albedo), this.frameMultipleScattering));
              },
              () => frameShadowMarchDepth(depths, this.frameMultipleScattering, this.frameScatteringOctaves).greaterThan(8.0),
            );

            const shadow = frameShadowTransmittance(depths, this.frameMultipleScattering, this.frameScatteringOctaves);

            result.assign(vec4(shadow.x, shadow.y, depths.z, depths.y));
          });
        });

        return result;
      };

      // NOTE Spectral shadow march. One volume has one spectrum, so the march sums one optical depth.
      // Then the depths of each channel are linear in it (frameSpectralShadowDepths).
      // Returns { x: direct, y: total, z: reduced depth, w: effective depth }, vec3 values.
      const shadowUnit: FrameSpectralDepths | null = spectral ? frameSpectralShadowUnit(frameExtinctionSpectrum, albedo, this.frameMultipleScattering) : null;

      const spectralVolumeTransmittance = (localDirection: VDBTslNode, maxLocalDistance: VDBTslNode): InjectShadow => {
        const direct = vec3(1.0).toVar();
        const total = vec3(1.0).toVar();
        const reduced = vec3(0.0).toVar();
        const effective = vec3(0.0).toVar();

        If(frameAbsorbance.greaterThan(0.0).and(this.frameShadowSteps.greaterThan(0)), () => {
          const direction = normalize(localDirection).toVar();
          const marchDistance = min(volume.boxExit(localPosition, direction), maxLocalDistance).toVar();

          If(marchDistance.greaterThan(0.0), () => {
            const localStep = marchDistance.div(float(this.frameShadowSteps)).toVar();
            const stepLod = max(lod, volume.getLod(localStep)).toVar();
            const worldStep = localStep.mul(length(mat3(frameLocalToView).mul(direction)));
            const stepScale = worldStep.mul(frameAbsorbance).toVar();
            const opticalDepth = float(0.0).toVar();

            marchSamples(
              direction,
              marchDistance,
              localStep,
              stepLod,
              stepScale,
              (extinction) => {
                opticalDepth.addAssign(extinction);
              },
              // NOTE Stop when all channels are dark
              () => minComponent(frameShadowMarchDepth(frameSpectralShadowDepths(shadowUnit!, opticalDepth), this.frameMultipleScattering, this.frameScatteringOctaves)).greaterThan(8.0),
            );

            const depths = frameSpectralShadowDepths(shadowUnit!, opticalDepth);
            const shadow = frameSpectralShadowTransmittance(depths, this.frameMultipleScattering, this.frameScatteringOctaves);

            direct.assign(shadow.x);
            total.assign(shadow.y);
            reduced.assign(depths.z);
            effective.assign(depths.y);
          });
        });

        return { x: direct, y: total, z: reduced, w: effective };
      };

      const march = spectral ? spectralVolumeTransmittance : volumeTransmittance;

      // NOTE Isotropic light (no phase): total transmittance of a march's total (y) value
      const shadowTotal = (total: VDBTslNode) => (spectral ? frameSpectralShadowTotal(total, this.frameScatteringOctaves, albedo) : frameShadowTotal(total, this.frameScatteringOctaves));

      // NOTE Diffusion: two-sided totals of isotropic light (frameIsotropicDiffusion) of a direction and the opposite one.
      // Spectral: vec3 values, one for each channel. Call in the Fn body.
      const diffusion = this.frameMultipleScattering.greaterThan(0.0);
      const isotropicValue = () => (spectral ? vec3(1.0) : float(1.0)).toVar();

      // NOTE Total transmittance of isotropic light from one direction ({ x }) and the opposite one ({ y }), each marched once.
      // Diffusion: the two-sided totals of the pair. Single scattering or octaves: the total of each march. Call in the Fn body.
      const isotropicPair = (localDirection: VDBTslNode, withOpposite: boolean) => {
        const front = march(localDirection, float(1e6));
        const x = isotropicValue();
        const y = isotropicValue();

        if (withOpposite) {
          const back = march(localDirection.negate(), float(1e6));

          If(diffusion, () => {
            x.assign(frameIsotropicDiffusion(front.z, back.z, front.w));
            y.assign(frameIsotropicDiffusion(back.z, front.z, back.w));
          }).Else(() => {
            x.assign(shadowTotal(front.y));
            y.assign(shadowTotal(back.y));
          });
        } else {
          If(diffusion, () => {
            const back = march(localDirection.negate(), float(1e6));

            x.assign(frameIsotropicDiffusion(front.z, back.z, front.w));
          }).Else(() => {
            x.assign(shadowTotal(front.y));
          });
        }

        return { x, y };
      };

      const outwardNormal = select(
        length(localPosition).greaterThan(1e-4),
        normalize(mat3(this.frameCameraMatrixWorld).mul(mat3(frameLocalToView).mul(localPosition))),
        vec3(0.0, 1.0, 0.0),
      ).toVar();

      const irradiance = vec3(0.0).toVar();

      If(useLights(vdbFrameLights.useAmbientLights), () => {
        irradiance.addAssign(frameAmbientColor);
        irradiance.addAssign(getShIrradianceAt(outwardNormal, frameLightProbe));
      });

      // NOTE Outward direction of the scene environment (no environment lights). Diffusion: the outward and the inward
      // irradiance, each with its two-sided total. Uniform light then gives E.
      const outwardEnvironment = () => {
        if (this.environment) {
          const outwardDirection = worldToLocalDirection.mul(outwardNormal).toVar();
          const front = march(outwardDirection, float(1e6));

          If(diffusion, () => {
            const back = march(outwardDirection.negate(), float(1e6));

            irradiance.addAssign(this.environmentIrradiance(outwardNormal).mul(frameIsotropicDiffusion(front.z, back.z, front.w).mul(0.5)));
            irradiance.addAssign(this.environmentIrradiance(outwardNormal.negate()).mul(frameIsotropicDiffusion(back.z, front.z, back.w).mul(0.5)));
          }).Else(() => {
            irradiance.addAssign(this.environmentIrradiance(outwardNormal).mul(shadowTotal(front.y)));
          });
        }
      };

      // NOTE Environment lights replace the outward direction: one shadow march for each light, no phase
      If(useLights(vdbFrameLights.useEnvironment), () => {
        if (this.environmentLights) {
          If(frameEnvironmentCount.greaterThan(0), () => {
            const environment = vec3(0.0).toVar();

            Loop(frameEnvironmentCount, ({ i }: { i: VDBTslNode }) => {
              const direction = worldToLocalDirection.mul(frameEnvironmentDirections.element(i)).toVar();

              environment.addAssign(frameEnvironmentColors.element(i).mul(isotropicPair(direction, false).x));
            });

            irradiance.addAssign(max(environment, vec3(0.0)));
          }).Else(outwardEnvironment);
        } else {
          outwardEnvironment();
        }
      });

      if (this.hemisphereLights) {
        If(useLights(vdbFrameLights.useHemisphereLights), () => {
          Loop(frameHemiCount, ({ i }: { i: VDBTslNode }) => {
            const upDirection = worldToLocalDirection.mul(frameHemiDirections.element(i)).toVar();
            // NOTE Diffusion: sky and ground are an opposite pair, two-sided totals
            const pair = isotropicPair(upDirection, true);

            irradiance.addAssign(frameHemiSkyColors.element(i).mul(0.5).mul(pair.x));
            irradiance.addAssign(frameHemiGroundColors.element(i).mul(0.5).mul(pair.y));
          });
        });
      }

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
          const shadow = march(localDirection, maxDistance);

          // NOTE Two-sided: diffusion of the slab between the light side and the far side (frameTwoSidedDiffusion)
          If(this.frameTwoSided.greaterThan(0.5), () => {
            const back = march(localDirection.negate(), float(1e6));

            shadow.y.assign(frameTwoSidedDiffusion(shadow.x, shadow.z, shadow.w, back.z));
          });

          const light = spectral
            ? frameSpectralShadowedLight(volumeUniforms.framePhase, volumeUniforms.framePhaseSpectrum, this.frameScatteringOctaves, rayDirection, lightDirection, shadow, albedo)
            : frameShadowedLight(volumeUniforms.framePhase, volumeUniforms.framePhaseSpectrum, this.frameScatteringOctaves, rayDirection, lightDirection, (shadow as VDBTslNode).xy.toVar());

          irradiance.addAssign(color.rgb.mul(light).mul(spotAttenuation));
        });
      });

      const radiance = max(frameScatterColor, albedo.mul(irradiance).mul(1.0 / Math.PI)).add(volume.sampleEmission(localPosition, lod));

      // NOTE Premultiplied slice color and alpha. Linear filtering of these values gives transparent edges.
      if (spectralOutputs) {
        spectralOutputs.color.assign(vec4(radiance.mul(alpha), alpha.y));
        spectralOutputs.alpha.assign(vec4(alpha.x, alpha.z, 0.0, alpha.y));

        return vec4(0.0);
      }

      return vec4(radiance.mul(alpha), alpha);
    })();
  }
}
