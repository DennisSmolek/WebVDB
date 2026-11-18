// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameLightingMaterial.js L43–119 (constructor, uniforms),
//   L277–437 (the grey `createFragmentNode`); K3.5: L55–62 (the spectral maps and groups), L86–112
//   (setSpectral, rebuildNodes), L114–275 (createSpectralFragmentNode).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; source-free. K3.4: the octave uniform (L71–72) and the octave light of lights
// without a shadow slot (L386–397).
// Divergences: DIVERGES.md (K3.3 rows).

/**
 * The lighting pass of `'frame'` lighting (R1 §2 step 4): each froxel of the volumes' froxel area is
 * lit once from the combined density targets: ambient, light probe SH and hemisphere light with the
 * sky up / down transmittance, the scene lights (with a shadow slot: the density pass's
 * phase × direct + scattered; else unshadowed), and the dominant directional environment light in
 * its slot. Writes premultiplied (radiance × alpha, alpha) into the froxel atlas, as the inject pass.
 * Spectral (K3.5): the transmittance of each channel is sampled from the slot groups' targets at the
 * froxel's visible position (the density pass has no slots then); two atlas targets, (radiance × alpha
 * of each channel, green alpha) and (red alpha, blue alpha, 0, green alpha).
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { createGridNodes, frameFragCoord, frameGetFroxel, frameLoad, framePhaseColor, frameShadowedLight, frameSpectralShadowTotal, frameSpectralShadowedLight, frameTileVertex, frameUnpackSpectralGroup, type FroxelUniforms, type GridUniforms, type LightUniforms, type TileUniforms } from "./chunks.js";
import type { DensityShadowUniforms } from "./density-material.js";
import { frameSpectralGroupCount } from "./spectrum.js";

const { Discard, Fn, If, Loop, clamp, dot, exp, float, getDistanceAttenuation, getShIrradianceAt, int, ivec2, length, mat3, max, mix, normalize, outputStruct, property, select, smoothstep, texture, uniform, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

export interface VDBFrameLightingMaterialOptions {
  froxelUniforms: FroxelUniforms;
  froxelTileUniforms: TileUniforms;
  froxelAreaUniforms: { frameAreaRect: VDBTslNode; frameAreaSliceStart: VDBTslNode };
  lightUniforms: LightUniforms;
  gridUniforms: GridUniforms;
  densityShadowUniforms: DensityShadowUniforms;
}

// NOTE Lights each froxel one time from the combined data of all volumes.
// NOTE The environment is added in the density pass, along the outward direction of each volume.
export class VDBFrameLightingMaterial extends NodeMaterial {
  readonly frameDensityAlbedoMap = texture(new THREE.Texture());
  readonly frameDensityEmissionMap = texture(new THREE.Texture());
  readonly frameDensityShadowMap0 = texture(new THREE.Texture());
  readonly frameDensityShadowMap1 = texture(new THREE.Texture());
  readonly frameShadowLightCount = uniform(0, "int");
  readonly frameCameraMatrixWorld = uniform(new THREE.Matrix4());
  // NOTE One phase for all volumes, from the renderer settings
  readonly framePhase = uniform(new THREE.Vector4());
  readonly framePhaseSpectrum = uniform(new THREE.Vector3());
  // NOTE Scattering octaves of the renderer settings (phase.ts). Count 0 is off.
  readonly frameScatteringOctaves = uniform(new THREE.Vector4());
  // NOTE Environment lights with a shadow slot (0 or 1). Light i has slot frameEnvironmentShadowSlot + i.
  readonly frameEnvironmentLightCount = uniform(0, "int");
  readonly frameEnvironmentShadowSlot = uniform(0, "int");
  readonly frameUniforms: VDBFrameLightingMaterialOptions;
  // NOTE Renderer settings.spectral: extinction of each channel (density target 2), and the 3 transmittance
  // targets of each slot group (spectrum.ts), in group order. Groups from frameSpectralGroups on are not read.
  readonly frameDensityExtinctionMap = texture(new THREE.Texture());
  readonly frameSpectralTransmittanceMaps = Array.from({ length: frameSpectralGroupCount * 3 }, () => texture(new THREE.Texture()));
  readonly frameSpectralGroups = uniform(0, "int");
  // NOTE Renderer settings.spectral: two atlas targets
  spectral = false;

  constructor(o: VDBFrameLightingMaterialOptions) {
    super();

    this.name = "VDBFrameLightingMaterial";
    this.frameUniforms = o;

    // NOTE Only the froxel area of the volumes. Other froxels have no density.
    this.vertexNode = frameTileVertex(o.froxelTileUniforms, o.froxelAreaUniforms.frameAreaRect, o.froxelAreaUniforms.frameAreaSliceStart);
    this.rebuildNodes();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }

  setSpectral(spectral: boolean): void {
    if (spectral !== this.spectral) {
      this.spectral = spectral;
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  // NOTE Spectral: atlas target 0 is (color, green alpha), target 1 is (red alpha, blue alpha, 0, green alpha)
  rebuildNodes(): void {
    if (!this.spectral) {
      this.colorNode = null;
      this.outputNode = null;
      this.fragmentNode = this.createFragmentNode();

      return;
    }

    const lightingColor = property("vec4");
    const lightingAlpha = property("vec4");

    this.fragmentNode = null;
    this.colorNode = this.createSpectralFragmentNode({ color: lightingColor, alpha: lightingAlpha });
    this.outputNode = outputStruct(lightingColor, lightingAlpha);
  }

  // NOTE Spectral lighting. The density pass has no light slots, so the slots are sampled here,
  // at the extinction-weighted visible position of all volumes in the froxel. Albedo and extinction are of each channel.
  createSpectralFragmentNode(outputs: { color: VDBTslNode; alpha: VDBTslNode }): VDBTslNode {
    const { froxelUniforms, lightUniforms, gridUniforms } = this.frameUniforms;
    const { frameAmbientColor, frameLightProbe, frameHemiSkyColor, frameHemiGroundColor, frameSkyDirection, frameLightCount, frameLightPositions, frameLightColors, frameLightSpots, frameLightFalloffs, frameEnvironmentColors } = lightUniforms;
    const grid = createGridNodes(gridUniforms);
    const atlasSize = froxelUniforms.froxelAtlasSize;
    const octaves = this.frameScatteringOctaves;

    return Fn(() => {
      const fragCoord = frameFragCoord(atlasSize).toVar();
      const texel = ivec2(fragCoord).toVar();
      const densityAlbedo = frameLoad(this.frameDensityAlbedoMap, texel, atlasSize).toVar();
      const weight = densityAlbedo.a.toVar();

      If(weight.lessThanEqual(0.0), () => {
        Discard();
      });

      const densityEmission = frameLoad(this.frameDensityEmissionMap, texel, atlasSize).toVar();
      const extinction = frameLoad(this.frameDensityExtinctionMap, texel, atlasSize).rgb.toVar();
      const froxel = frameGetFroxel(froxelUniforms, fragCoord);
      const viewNear = froxel.viewNear.toVar();
      const viewFar = froxel.viewFar.toVar();

      const visibleT = clamp(densityEmission.a.div(weight), 0.0, 1.0);
      const worldPosition = this.frameCameraMatrixWorld.mul(vec4(mix(viewNear, viewFar, visibleT), 1.0)).xyz.toVar();
      const rayDirection = normalize(mat3(this.frameCameraMatrixWorld).mul(viewFar.sub(viewNear))).toVar();
      const albedo = densityAlbedo.rgb.div(max(extinction, vec3(1e-8))).toVar();
      const emission = densityEmission.rgb.div(weight);

      // NOTE Transmittance targets of the slot groups in use. 1.0 outside the grid, with no shadows and for other groups.
      const maps = this.frameSpectralTransmittanceMaps.map(() => vec4(1.0).toVar());

      for (let group = 0; group < frameSpectralGroupCount; group++) {
        const groupMaps = this.frameSpectralTransmittanceMaps.slice(group * 3, group * 3 + 3);

        If(int(group).lessThan(this.frameSpectralGroups), () => {
          const shadows = grid.sampleShadows(groupMaps, worldPosition);

          shadows.forEach((shadow, index) => maps[group * 3 + index]!.assign(shadow));
        });
      }

      // NOTE Group 0: sky (up, down) and slot 0. Group g: slots 2g - 1 and 2g. Slot entries are (total, direct).
      const groups = Array.from({ length: frameSpectralGroupCount }, (_, group) => frameUnpackSpectralGroup(maps[group * 3]!, maps[group * 3 + 1]!, maps[group * 3 + 2]!));
      const slotEntry = (slot: number) => (slot === 0 ? groups[0]!.b : slot % 2 === 1 ? groups[(slot + 1) / 2]!.a : groups[slot / 2]!.b);

      const skyDirection = normalize(frameSkyDirection).toVar();
      const skyUp = frameSpectralShadowTotal(groups[0]!.a.x, octaves, albedo).mul(0.5).toVar();
      const skyDown = frameSpectralShadowTotal(groups[0]!.a.y, octaves, albedo).mul(0.5).toVar();

      const irradiance = vec3(frameAmbientColor).toVar();

      irradiance.addAssign(getShIrradianceAt(skyDirection, frameLightProbe).mul(skyUp));
      irradiance.addAssign(getShIrradianceAt(skyDirection.negate(), frameLightProbe).mul(skyDown));
      irradiance.addAssign(frameHemiSkyColor.mul(skyUp));
      irradiance.addAssign(frameHemiGroundColor.mul(skyDown));

      const slots = Array.from({ length: 6 }, (_, slot) => ({
        total: slotEntry(slot).x.toVar(),
        direct: slotEntry(slot).y.toVar(),
      }));

      Loop(frameLightCount, ({ i }: { i: VDBTslNode }) => {
        const position = frameLightPositions.element(i).toVar();
        const color = frameLightColors.element(i).toVar();
        const spot = frameLightSpots.element(i).toVar();
        const falloff = frameLightFalloffs.element(i).toVar();
        const lightVector = position.xyz.sub(worldPosition).toVar();

        // NOTE A light with no shadow slot: no shadow (1.0), then octaves still add their light
        let total: VDBTslNode = vec3(1.0);
        let direct: VDBTslNode = vec3(1.0);

        slots.forEach((slot, index) => {
          const useSlot = i.equal(index).and(i.lessThan(this.frameShadowLightCount));

          total = select(useSlot, slot.total, total);
          direct = select(useSlot, slot.direct, direct);
        });

        const spotAttenuation = select(position.w.greaterThan(2.5), smoothstep(spot.w, color.w, dot(normalize(lightVector), spot.xyz)), float(1.0));

        const distanceAttenuation = select(
          position.w.greaterThan(1.5),
          getDistanceAttenuation({ lightDistance: length(lightVector), cutoffDistance: falloff.x, decayExponent: falloff.y }),
          float(1.0),
        );

        // NOTE Directional position xyz is the direction to the light
        const light = frameSpectralShadowedLight(
          this.framePhase,
          this.framePhaseSpectrum,
          octaves,
          rayDirection,
          select(position.w.lessThan(1.5), position.xyz, normalize(lightVector)),
          { x: direct.toVar(), y: total.toVar() },
          albedo,
        );

        irradiance.addAssign(color.rgb.mul(light).mul(spotAttenuation).mul(distanceAttenuation));
      });

      // NOTE Dominant directional environment light with a shadow slot. Isotropic, no phase.
      Loop(this.frameEnvironmentLightCount, ({ i }: { i: VDBTslNode }) => {
        const slot = this.frameEnvironmentShadowSlot.add(i).toVar();
        let total: VDBTslNode = vec3(1.0);

        slots.forEach((value, index) => {
          total = select(slot.equal(index), value.total, total);
        });

        irradiance.addAssign(frameEnvironmentColors.element(i).mul(frameSpectralShadowTotal(total.toVar(), octaves, albedo)));
      });

      const radiance = albedo.mul(irradiance).mul(1.0 / Math.PI).add(emission);
      const alpha = vec3(1.0).sub(exp(extinction.negate().mul(length(viewFar.sub(viewNear))))).toVar();

      outputs.color.assign(vec4(radiance.mul(alpha), alpha.y));
      outputs.alpha.assign(vec4(alpha.x, alpha.z, 0.0, alpha.y));

      return vec4(0.0);
    })();
  }

  createFragmentNode(): VDBTslNode {
    const { froxelUniforms, lightUniforms, densityShadowUniforms, gridUniforms } = this.frameUniforms;
    const { frameAmbientColor, frameLightProbe, frameHemiSkyColor, frameHemiGroundColor, frameSkyDirection, frameLightCount, frameLightPositions, frameLightColors, frameLightSpots, frameLightFalloffs, frameEnvironmentColors } = lightUniforms;
    const grid = createGridNodes(gridUniforms);
    const atlasSize = froxelUniforms.froxelAtlasSize;

    return Fn(() => {
      const fragCoord = frameFragCoord(atlasSize).toVar();
      const texel = ivec2(fragCoord).toVar();
      const densityAlbedo = frameLoad(this.frameDensityAlbedoMap, texel, atlasSize).toVar();
      const extinction = densityAlbedo.a.toVar();

      If(extinction.lessThanEqual(0.0), () => {
        Discard();
      });

      const densityEmission = frameLoad(this.frameDensityEmissionMap, texel, atlasSize).toVar();
      const froxel = frameGetFroxel(froxelUniforms, fragCoord);
      const viewNear = froxel.viewNear.toVar();
      const viewFar = froxel.viewFar.toVar();

      // NOTE Spot light angles and light falloff use the extinction-weighted visible position of all volumes
      const visibleT = clamp(densityEmission.a.div(extinction), 0.0, 1.0);
      const worldPosition = this.frameCameraMatrixWorld.mul(vec4(mix(viewNear, viewFar, visibleT), 1.0)).xyz.toVar();
      const rayDirection = normalize(mat3(this.frameCameraMatrixWorld).mul(viewFar.sub(viewNear))).toVar();

      // NOTE Extinction-weighted mean transmittance. Each volume sampled its own visible position.
      const shadow0 = frameLoad(this.frameDensityShadowMap0, texel, atlasSize).div(extinction).toVar();
      const shadow1 = frameLoad(this.frameDensityShadowMap1, texel, atlasSize).div(extinction).toVar();
      const shadowSlots = [shadow0.z, shadow0.w, shadow1.x, shadow1.y, shadow1.z, shadow1.w];

      // NOTE Phases of each channel: slots hold the green phase. Direct transmittance from the shadow grid adds the red and blue difference.
      const colorPhase = this.framePhaseSpectrum.z.greaterThan(0.5).toVar();
      const direct0 = vec4(1.0).toVar();
      const direct1 = vec4(1.0).toVar();

      If(colorPhase.and(densityShadowUniforms.frameUseShadows.equal(1)), () => {
        const directs = grid.sampleShadows([densityShadowUniforms.frameTransmittanceMap2, densityShadowUniforms.frameTransmittanceMap3], worldPosition);

        direct0.assign(directs[0]);
        direct1.assign(directs[1]);
      });

      const directSlots = [direct0.x, direct0.y, direct0.z, direct0.w, direct1.x, direct1.y];

      const skyDirection = normalize(frameSkyDirection).toVar();
      const skyUp = shadow0.x.mul(0.5);
      const skyDown = shadow0.y.mul(0.5);

      const irradiance = vec3(frameAmbientColor).toVar();

      irradiance.addAssign(getShIrradianceAt(skyDirection, frameLightProbe).mul(skyUp));
      irradiance.addAssign(getShIrradianceAt(skyDirection.negate(), frameLightProbe).mul(skyDown));
      irradiance.addAssign(frameHemiSkyColor.mul(skyUp));
      irradiance.addAssign(frameHemiGroundColor.mul(skyDown));

      Loop(frameLightCount, ({ i }: { i: VDBTslNode }) => {
        const position = frameLightPositions.element(i).toVar();
        const color = frameLightColors.element(i).toVar();
        const spot = frameLightSpots.element(i).toVar();
        const falloff = frameLightFalloffs.element(i).toVar();
        const lightVector = position.xyz.sub(worldPosition).toVar();

        let shadow: VDBTslNode = float(1.0);
        let direct: VDBTslNode = float(1.0);

        shadowSlots.forEach((slot, index) => {
          shadow = select(i.equal(index), slot, shadow);
          direct = select(i.equal(index), directSlots[index], direct);
        });

        const spotAttenuation = select(position.w.greaterThan(2.5), smoothstep(spot.w, color.w, dot(normalize(lightVector), spot.xyz)), float(1.0));

        const distanceAttenuation = select(
          position.w.greaterThan(1.5),
          getDistanceAttenuation({ lightDistance: length(lightVector), cutoffDistance: falloff.x, decayExponent: falloff.y }),
          float(1.0),
        );

        // NOTE Directional position xyz is the direction to the light
        const lightDirection = select(position.w.lessThan(1.5), position.xyz, normalize(lightVector)).toVar();
        const unshadowed = frameShadowedLight(this.framePhase, this.framePhaseSpectrum, this.frameScatteringOctaves, rayDirection, lightDirection, vec2(1.0));
        const hasSlot = i.lessThan(this.frameShadowLightCount).toVar();

        // NOTE A light with a shadow slot: the density pass stored phase * direct + scattered light.
        // A light with no shadow slot: no shadow, then octaves still add their light.
        const lightFactor = select(hasSlot, vec3(shadow), unshadowed).toVar();

        If(colorPhase.and(hasSlot), () => {
          const phase = framePhaseColor(this.framePhase, this.framePhaseSpectrum, rayDirection, lightDirection).toVar();

          lightFactor.addAssign(phase.sub(phase.g).mul(direct));
        });

        irradiance.addAssign(color.rgb.mul(lightFactor).mul(spotAttenuation).mul(distanceAttenuation));
      });

      // NOTE Dominant directional environment light with a shadow slot. Isotropic, no phase.
      Loop(this.frameEnvironmentLightCount, ({ i }: { i: VDBTslNode }) => {
        const slot = this.frameEnvironmentShadowSlot.add(i).toVar();
        let shadow: VDBTslNode = float(1.0);

        shadowSlots.forEach((value, index) => {
          shadow = select(slot.equal(index), value, shadow);
        });

        irradiance.addAssign(frameEnvironmentColors.element(i).mul(shadow));
      });

      const albedo = densityAlbedo.rgb.div(extinction);
      const emission = densityEmission.rgb.div(extinction);
      const radiance = albedo.mul(irradiance).mul(1.0 / Math.PI).add(emission);
      const alpha = float(1.0).sub(exp(extinction.negate().mul(length(viewFar.sub(viewNear)))));

      return vec4(radiance.mul(alpha), alpha);
    })();
  }
}
