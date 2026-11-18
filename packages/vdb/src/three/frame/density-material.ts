// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameDensityMaterial.js L45–361 (density shadow uniforms,
//   environment modes, constructor, the grey `buildNodes`).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; reads density only through `vdbVolumeNodes` (the VDBVolumeSource seam), so
// one material is built per volume source (DIVERGES V30); grey only (the spectral variant is K3.5);
// octaves off (K3.4). Divergences: DIVERGES.md (K3.3 rows), V23–V25, V30.

/**
 * The density pass of `'frame'` lighting (R1 §2 step 3): per froxel per volume (one instance per
 * slice of the volume's froxel rectangle), up to `subSteps` extinction samples across the slice;
 * albedo and emission at the visibility-weighted position, plus the shadow grid's transmittance
 * there (the scene light slots get the phase here, at each volume's own visible position). The
 * environment is added along the outward direction of the volume (or from the environment grid).
 * Four RGBA16F targets, extinction-weighted and additive, so the lighting pass reads
 * extinction-weighted means of all volumes in a froxel.
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import type { VDBThreeVolumeSource, VDBVolumeNodes, VDBVolumeNodesOptions, VDBVolumeUniforms } from "../volume-source.js";
import { vdbVolumeNodes } from "../volume-source.js";
import {
  createFrameVolumeUniforms,
  createGridNodes,
  frameDistanceFade,
  frameFragCoord,
  frameGetFroxel,
  frameNoise,
  frameShadowTotal,
  frameShadowedLight,
  frameTileVertex,
  getBoxSteps,
  getFroxelLod,
  getSampledLod,
  type EnvironmentGridUniforms,
  type FrameVolumeUniforms,
  type FroxelUniforms,
  type GridUniforms,
  type LightUniforms,
  type TileUniforms,
} from "./chunks.js";
import { VDBFrameEnvironmentMaterial } from "./environment-material.js";

const { Discard, Fn, If, Loop, dot, exp, float, int, length, mat3, max, min, mix, normalize, outputStruct, property, select, texture, uniform, vec2, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

// NOTE Shadow grid data for the density pass. Each volume samples its own visible position.
export const createDensityShadowUniforms = () => ({
  frameTransmittanceMap0: texture(new THREE.Texture()),
  frameTransmittanceMap1: texture(new THREE.Texture()),
  // NOTE Direct transmittance of the 6 light slots
  frameTransmittanceMap2: texture(new THREE.Texture()),
  frameTransmittanceMap3: texture(new THREE.Texture()),
  // NOTE Scene lights with a shadow slot. Their slots get the phase here, at the visible position of each volume.
  frameShadowSceneLightCount: uniform(0, "int"),
  // NOTE One phase for all volumes, from the renderer settings
  framePhase: uniform(new THREE.Vector4()),
  framePhaseSpectrum: uniform(new THREE.Vector3()),
  // NOTE 1 with diffusion multiple scattering. Then sky up and down have two-sided totals (frameIsotropicDiffusion).
  frameDiffusion: uniform(0, "int"),
  frameUseShadows: uniform(0, "int"),
  frameCameraMatrixWorld: uniform(new THREE.Matrix4()),
  // NOTE Environment light of the last cascade voxels (VDBFrameEnvironmentGridMaterial)
  frameEnvironmentGridMap: texture(new THREE.Texture()),
  // NOTE frameDensityEnvironment values
  frameEnvironmentMode: uniform(0, "int"),
});

export type DensityShadowUniforms = ReturnType<typeof createDensityShadowUniforms>;

// NOTE Environment in the density pass: outward direction, none (only the slot light, in the lighting pass),
// or the environment grid
export const frameDensityEnvironment = { outward: 0, none: 1, grid: 2 } as const;

export interface VDBFrameDensityMaterialOptions {
  froxelUniforms: FroxelUniforms;
  froxelTileUniforms: TileUniforms;
  gridUniforms: GridUniforms;
  environmentGridUniforms: EnvironmentGridUniforms;
  densityShadowUniforms: DensityShadowUniforms;
  lightUniforms: LightUniforms;
  source: VDBThreeVolumeSource;
  sourceUniforms: VDBVolumeUniforms;
  mipmaps: boolean;
  emissionColor?: VDBVolumeNodesOptions["emissionColor"];
}

// NOTE Writes the extinction and material of one volume into the froxel grid. Lights are added in the lighting pass.
// NOTE The environment is added here, along the outward direction of each volume.
// NOTE One material per volume source. Pool meshes set the volume uniforms before each draw.
export class VDBFrameDensityMaterial extends VDBFrameEnvironmentMaterial {
  readonly volumeUniforms: FrameVolumeUniforms = createFrameVolumeUniforms();
  readonly frameRect = uniform(new THREE.Vector4());
  readonly frameSliceStart = uniform(0.0);
  readonly frameUniforms: VDBFrameDensityMaterialOptions;
  readonly sourceUniforms: VDBVolumeUniforms;
  readonly source: VDBThreeVolumeSource;
  mipmaps: boolean;

  constructor(o: VDBFrameDensityMaterialOptions) {
    super();

    this.name = "VDBFrameDensityMaterial";
    this.frameUniforms = o;
    this.source = o.source;
    this.sourceUniforms = o.sourceUniforms;
    this.mipmaps = o.mipmaps;
    this.buildNodes();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.transparent = true;
    this.blending = THREE.CustomBlending;
    this.blendEquation = THREE.AddEquation;
    this.blendSrc = THREE.OneFactor;
    this.blendDst = THREE.OneFactor;
    this.blendSrcAlpha = THREE.OneFactor;
    this.blendDstAlpha = THREE.OneFactor;
  }

  // NOTE Mip sampling is in the shader only when the volume uses mipmaps
  setMipmaps(mipmaps: boolean): void {
    if (mipmaps !== this.mipmaps) {
      this.mipmaps = mipmaps;
      this.environmentNodes = [];
      this.buildNodes();
      this.needsUpdate = true;
    }
  }

  override rebuildNodes(): void {
    this.buildNodes();
  }

  buildNodes(): void {
    const { froxelUniforms, froxelTileUniforms, gridUniforms, environmentGridUniforms, densityShadowUniforms, lightUniforms, emissionColor } = this.frameUniforms;
    const volume: VDBVolumeNodes = vdbVolumeNodes(this.source, this.sourceUniforms, { useMipmaps: this.mipmaps, ...(emissionColor ? { emissionColor } : {}) });
    const grid = createGridNodes(gridUniforms);
    const { frameViewToLocal, frameLocalToView, frameOpacity, frameScatterColor } = this.volumeUniforms;
    const frameWorldScale = this.sourceUniforms.worldScale;
    const { frameCameraMatrixWorld } = densityShadowUniforms;
    const { froxelSubSteps, frameJitter } = froxelUniforms;

    const densityAlbedo = property("vec4");
    const densityEmission = property("vec4");
    const densityShadow0 = property("vec4");
    const densityShadow1 = property("vec4");

    this.vertexNode = frameTileVertex(froxelTileUniforms, this.frameRect, this.frameSliceStart);

    this.colorNode = Fn(() => {
      const fragCoord = frameFragCoord(froxelTileUniforms.frameTargetSize).toVar();
      const froxel = frameGetFroxel(froxelUniforms, fragCoord);
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
      const sampledLod = getSampledLod(this.sourceUniforms, this.source.channels.mask, widthLod, this.mipmaps);
      const steps = getBoxSteps(volume, localNear, localFar, segment, sampledLod, froxelSubSteps);
      const subStepCount = float(steps);
      const subStepLength = length(viewFar.sub(viewNear)).mul(segment).div(subStepCount).toVar();
      const lod = max(widthLod, volume.getLod(subStepLength.div(frameWorldScale))).toVar();

      // NOTE Mean extinction of the slice. The visibility-weighted mean position is the light position in the lighting pass.
      const extinctionSum = float(0.0).toVar();
      const transmittance = float(1.0).toVar();
      const visibleWeight = float(0.0).toVar();
      const visibleTSum = float(0.0).toVar();
      const jitter = frameNoise(frameJitter, fragCoord).toVar();

      Loop({ start: int(0), end: steps, type: "int", condition: "<" }, ({ i }: { i: VDBTslNode }) => {
        const t = boxSegment.x.add(segment.mul(float(i).add(jitter).div(subStepCount))).toVar();
        const extinction = volume
          .sampleExtinction(mix(localNear, localFar, t), lod)
          .mul(frameOpacity)
          .mul(frameDistanceFade(froxelUniforms, mix(viewNear, viewFar, t)))
          .toVar();
        const stepTransmittance = exp(extinction.mul(subStepLength).negate()).toVar();
        const weight = transmittance.mul(float(1.0).sub(stepTransmittance)).toVar();

        visibleWeight.addAssign(weight);
        visibleTSum.addAssign(weight.mul(t));
        extinctionSum.addAssign(extinction);
        transmittance.mulAssign(stepTransmittance);
      });

      // NOTE The mean over the whole slice: the box part's mean, weighted by its share of the slice
      const extinction = extinctionSum.div(subStepCount).mul(segment).toVar();

      If(extinction.lessThanEqual(0.0), () => {
        Discard();
      });

      const visibleT = select(visibleWeight.greaterThan(0.0), visibleTSum.div(visibleWeight), float(0.5)).toVar();
      const visiblePosition = mix(localNear, localFar, visibleT).toVar();
      const albedo = volume.sampleAlbedo(visiblePosition, lod).toVar();
      const emission = volume.sampleEmission(visiblePosition, lod).add(frameScatterColor).toVar();

      // NOTE Shadows at the visible position of this volume, not of other volumes in the froxel
      // NOTE shadow0, shadow1: total transmittance. direct0, direct1: direct transmittance of the 6 light slots.
      const shadow0 = vec4(1.0).toVar();
      const shadow1 = vec4(1.0).toVar();
      const direct0 = vec4(1.0).toVar();
      const direct1 = vec4(1.0).toVar();
      const visibleWorldPosition = frameCameraMatrixWorld.mul(vec4(mix(viewNear, viewFar, visibleT), 1.0)).xyz.toVar();

      If(densityShadowUniforms.frameUseShadows.equal(1), () => {
        const shadows = grid.sampleShadows(
          [densityShadowUniforms.frameTransmittanceMap0, densityShadowUniforms.frameTransmittanceMap1, densityShadowUniforms.frameTransmittanceMap2, densityShadowUniforms.frameTransmittanceMap3],
          visibleWorldPosition,
        );

        shadow0.assign(shadows[0]);
        shadow1.assign(shadows[1]);
        direct0.assign(shadows[2]);
        direct1.assign(shadows[3]);
      });

      // NOTE Scene light slots store phase * direct + scattered (frameShadowedLight), green channel.
      // Phases of each channel: the lighting pass adds the red and blue phase.
      // NOTE Other slots (sky up and down, the environment light) store the isotropic total transmittance.
      const rayDirection = normalize(mat3(frameCameraMatrixWorld).mul(viewFar.sub(viewNear))).toVar();
      const lightSlots: [VDBTslNode, "x" | "y" | "z" | "w", VDBTslNode][] = [
        [shadow0, "z", direct0.x],
        [shadow0, "w", direct0.y],
        [shadow1, "x", direct0.z],
        [shadow1, "y", direct0.w],
        [shadow1, "z", direct1.x],
        [shadow1, "w", direct1.y],
      ];

      lightSlots.forEach(([shadow, channel, direct], index) => {
        If(int(index).lessThan(densityShadowUniforms.frameShadowSceneLightCount), () => {
          const light = lightUniforms.frameLightPositions.element(index).toVar();

          shadow[channel].assign(
            frameShadowedLight(
              densityShadowUniforms.framePhase,
              densityShadowUniforms.framePhaseSpectrum,
              rayDirection,
              select(light.w.lessThan(1.5), light.xyz, normalize(light.xyz.sub(visibleWorldPosition))),
              vec2(direct, shadow[channel]),
            ).g,
          );
        }).Else(() => {
          shadow[channel].assign(frameShadowTotal(shadow[channel]));
        });
      });

      shadow0.x.assign(frameShadowTotal(shadow0.x));
      shadow0.y.assign(frameShadowTotal(shadow0.y));

      const { frameEnvironmentMode } = densityShadowUniforms;

      // NOTE Environment along the outward direction from the volume center, the same as per-volume lighting.
      // NOTE Its shadow is between the sky down and sky up transmittance of the shadow grid.
      if (this.environment) {
        If(frameEnvironmentMode.equal(frameDensityEnvironment.outward), () => {
          const skyDirection = normalize(lightUniforms.frameSkyDirection).toVar();
          const outwardDirection = select(
            length(visiblePosition).greaterThan(1e-4),
            normalize(mat3(frameCameraMatrixWorld).mul(mat3(frameLocalToView).mul(visiblePosition))),
            skyDirection,
          ).toVar();
          const skyBlend = dot(outwardDirection, skyDirection).mul(0.5).add(0.5).toVar();
          const environmentShadow = mix(shadow0.y, shadow0.x, skyBlend);

          // NOTE Diffusion: the outward and the inward irradiance, each with its two-sided total. Uniform light then gives E.
          If(densityShadowUniforms.frameDiffusion.equal(1), () => {
            const inwardShadow = mix(shadow0.x, shadow0.y, skyBlend);

            emission.addAssign(
              albedo
                .mul(this.environmentIrradiance(outwardDirection).mul(environmentShadow).add(this.environmentIrradiance(outwardDirection.negate()).mul(inwardShadow)).mul(0.5))
                .mul(1.0 / Math.PI),
            );
          }).Else(() => {
            emission.addAssign(albedo.mul(this.environmentIrradiance(outwardDirection)).mul(environmentShadow).mul(1.0 / Math.PI));
          });
        });
      }

      // NOTE Environment light of the environment grid at the visible position
      If(frameEnvironmentMode.equal(frameDensityEnvironment.grid), () => {
        const viewPosition = mix(viewNear, viewFar, visibleT);
        const worldPosition = frameCameraMatrixWorld.mul(vec4(viewPosition, 1.0)).xyz.toVar();
        const environment = grid.sampleEnvironmentGrid(
          densityShadowUniforms.frameEnvironmentGridMap,
          environmentGridUniforms,
          worldPosition,
          vec4(lightUniforms.frameEnvironmentUnshadowed.mul(frameShadowTotal(float(1.0))), 1.0),
        );

        emission.addAssign(albedo.mul(environment.rgb).mul(1.0 / Math.PI));
      });

      // NOTE Values are weighted by extinction. Additive blending then gives extinction-weighted means.
      densityAlbedo.assign(vec4(albedo.mul(extinction), extinction));
      densityEmission.assign(vec4(emission.mul(extinction), visibleT.mul(extinction)));
      // NOTE Light slots can be more than 1 (phase). The limit keeps them below the half float maximum.
      densityShadow0.assign(min(shadow0.mul(extinction), 6.0e4));
      densityShadow1.assign(min(shadow1.mul(extinction), 6.0e4));

      return vec4(0.0);
    })();

    this.outputNode = outputStruct(densityAlbedo, densityEmission, densityShadow0, densityShadow1);
  }
}
