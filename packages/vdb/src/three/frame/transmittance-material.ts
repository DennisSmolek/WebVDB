// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameTransmittanceMaterial.js L43–276 (light slots, the grey
//   transmittance pass; K3.4: the octave uniform, the two-sided back march and setNeural / the neural
//   slot light).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; grey only (the spectral slot groups are K3.5); the network is a parameter of
// setNeural (VDBFrameNeuralNetwork). Source-free: the pass reads only the shadow grid (and its mips).
// Divergences: DIVERGES.md (K3.3, K3.4 rows).

/**
 * The transmittance pass of `'frame'` lighting (R1 §2 step 2): from each voxel of every shadow grid
 * cascade, a march through the combined extinction of all volumes toward each light slot (≤ 6) and
 * the sky up and down, one finest-cascade voxel per step. Output: 4 RGBA16F targets, (sky up total,
 * sky down total, slot 0 total, slot 1 total), (slots 2–5 total), (slots 0–3 direct), (slots 4–5
 * direct, 1, 1). Multiple scattering: diffusion (1D), octaves, 'two-sided' (a back march per slot) or
 * 'neural' (the network's scattered light per slot). CPU twin: `@pmndrs/vdb/tools`
 * `vdbFrameTransmittanceVoxel`.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import {
  createGridNodes,
  frameFragCoord,
  frameFullscreenVertex,
  frameIsotropicDiffusion,
  frameNoise,
  frameShadowMarchDepth,
  frameShadowStep,
  frameShadowTransmittance,
  frameTwoSidedDiffusion,
  type GridUniforms,
  type LightUniforms,
} from "./chunks.js";
import { createNeuralNodes } from "./neural.js";
import type { VDBFrameNeuralNetwork } from "../../tools/frame/neural.js";

const { Break, Fn, If, Loop, abs, float, int, length, max, min, normalize, outputStruct, property, select, texture, uniform, uniformArray, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

// NOTE Light slots: sky up, sky down, then 6 lights. Targets 0 and 1: total transmittance of all slots.
// NOTE Targets 2 and 3: direct transmittance of the 6 lights. Then the lighting gives the phase only to the direct light.
// NOTE Scene lights take the first slots. The dominant directional environment light takes the next free slot.
export const frameShadowLightSlots = 6;

const maxShadowSteps = 1024;

// NOTE Mip levels 1 and up of the neural inputs (his frameNeuralLevels − 1)
export const frameNeuralMipLevels = 5;

export interface VDBFrameTransmittanceMaterialOptions {
  gridUniforms: GridUniforms;
  lightUniforms: LightUniforms;
}

// NOTE Transmittance from each grid voxel to each light, through the combined extinction of all volumes
export class VDBFrameTransmittanceMaterial extends NodeMaterial {
  readonly frameShadowDensityMap = texture(new THREE.Texture());
  readonly frameShadowLightCount = uniform(0, "int");
  readonly frameShadowLights = uniformArray(
    Array.from({ length: frameShadowLightSlots }, () => new THREE.Vector4()),
    "vec4",
  );
  // NOTE Multiple scattering of the renderer settings (phase.ts). 0.0 is single scattering.
  readonly frameMultipleScattering = uniform(0.0);
  // NOTE Scattering octaves of the renderer settings (phase.ts). Count 0 is off.
  readonly frameScatteringOctaves = uniform(new THREE.Vector4());
  // NOTE multipleScattering 'two-sided' (phase.ts): light slots also march away from the light
  readonly frameTwoSided = uniform(0.0);
  // NOTE multipleScattering 'neural': mips of the last cascade (VDBFrameGridMipMaterial), levels 1 and up,
  // and their atlas layouts (columns, width, height). Five levels for his six-level network.
  readonly frameNeuralMipMaps = Array.from({ length: frameNeuralMipLevels }, () => texture(new THREE.Texture()));
  readonly frameNeuralMipLayouts = Array.from({ length: frameNeuralMipLevels }, () => uniform(new THREE.Vector3(1.0, 1.0, 1.0)));
  readonly gridUniforms: GridUniforms;
  readonly lightUniforms: LightUniforms;
  /** The network compiled in (multipleScattering 'neural'), or null. */
  neural: VDBFrameNeuralNetwork | null = null;

  constructor(o: VDBFrameTransmittanceMaterialOptions) {
    super();

    this.name = "VDBFrameTransmittanceMaterial";
    this.gridUniforms = o.gridUniforms;
    this.lightUniforms = o.lightUniforms;
    this.buildNodes();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }

  // NOTE multipleScattering 'neural' (not spectral): scattered light of the light slots from the network
  setNeural(neural: VDBFrameNeuralNetwork | null): void {
    if (neural !== this.neural) {
      this.neural = neural;
      this.buildNodes();
      this.needsUpdate = true;
    }
  }

  buildNodes(): void {
    const { gridUniforms, lightUniforms } = this;
    const grid = createGridNodes(gridUniforms);
    const { frameGridMin, frameGridSize, frameGridVoxel, frameGridCount, frameGridAtlasSize } = gridUniforms;
    const { frameSkyDirection } = lightUniforms;
    const ms = this.frameMultipleScattering;
    const octaves = this.frameScatteringOctaves;
    const neural = this.neural ? createNeuralNodes(this.neural, gridUniforms, grid, this.frameShadowDensityMap, this.frameNeuralMipMaps, this.frameNeuralMipLayouts) : null;

    const transmittance0 = property("vec4");
    const transmittance1 = property("vec4");
    const transmittance2 = property("vec4");
    const transmittance3 = property("vec4");

    // NOTE (extinction, absorption)
    const gridExtinction = (worldPosition: VDBTslNode) => {
      const found = grid.findCascade(worldPosition, int(0));
      const cascade = max(found, int(0));

      return select(found.greaterThanEqual(0), grid.sampleGrid(this.frameShadowDensityMap, cascade, grid.gridUvw(cascade, worldPosition)).rg, vec2(0.0));
    };

    // NOTE Each step moves one voxel of the finest cascade at the probe, on the fastest axis of the direction.
    // The march goes to coarser cascades when it leaves a fine cascade.
    // NOTE opticalDepth (optional float variable) gets the direct optical depth
    const march = (origin: VDBTslNode, direction: VDBTslNode, maxDistance: VDBTslNode, jitter: VDBTslNode, opticalDepth: VDBTslNode | null = null) => {
      const depths = vec3(0.0).toVar();
      const traveled = float(0.0).toVar();
      const probe = vec3(origin).toVar();

      Loop(maxShadowSteps, () => {
        const cascade = grid.findCascade(probe, int(0)).toVar();

        If(cascade.lessThan(0).or(traveled.greaterThanEqual(maxDistance)).or(frameShadowMarchDepth(depths, ms, octaves).greaterThan(6.0)), () => {
          Break();
        });

        const voxelsPerUnit = abs(direction).div(frameGridVoxel.element(cascade));
        const stepLength = min(float(1.0).div(max(voxelsPerUnit.x, max(voxelsPerUnit.y, voxelsPerUnit.z))), maxDistance.sub(traveled)).toVar();

        const sample = gridExtinction(probe.add(direction.mul(stepLength).mul(jitter))).toVar();

        depths.addAssign(frameShadowStep(sample.x, sample.y, ms).mul(stepLength));
        probe.addAssign(direction.mul(stepLength));
        traveled.addAssign(stepLength);
      });

      const transmittance = frameShadowTransmittance(depths, ms, octaves);

      if (opticalDepth) {
        opticalDepth.assign(depths.x);
      }

      // NOTE (direct, total, reduced depth, effective depth). Sky up and down use z and w for the diffusion total.
      return vec4(transmittance.x, transmittance.y, depths.z, depths.y);
    };

    // NOTE Sky up and down are isotropic, and opposite. Diffusion: two-sided total (frameIsotropicDiffusion).
    const skyTotal = (front: VDBTslNode, back: VDBTslNode) => select(ms.greaterThan(0.0), frameIsotropicDiffusion(front.z, back.z, front.w), front.y);

    // NOTE Light w is 1.0 for a light direction, 2.0 or more for a light position
    // NOTE Neural: total is direct + the network light. local: the light-independent network inputs of the voxel.
    const lightTransmittance = (slot: number, worldPosition: VDBTslNode, jitter: VDBTslNode, local: [VDBTslNode, VDBTslNode] | null) => {
      const result = vec4(1.0, 1.0, 0.0, 0.0).toVar();

      If(int(slot).lessThan(this.frameShadowLightCount), () => {
        const light = this.frameShadowLights.element(slot).toVar();
        const isDirection = light.w.lessThan(1.5);
        const toLight = light.xyz.sub(worldPosition).toVar();
        const direction = select(isDirection, normalize(light.xyz), normalize(toLight)).toVar();
        const opticalDepth = neural ? float(0.0).toVar() : null;

        result.assign(march(worldPosition, direction, select(isDirection, float(1e20), length(toLight)), jitter, opticalDepth));

        // NOTE Two-sided: diffusion of the slab between the light side and the far side (frameTwoSidedDiffusion)
        If(this.frameTwoSided.greaterThan(0.5), () => {
          const back = march(worldPosition, direction.negate(), float(1e20), jitter).toVar();

          result.y.assign(frameTwoSidedDiffusion(result.x, result.z, result.w, back.z));
        });

        if (neural && local) {
          result.y.assign(result.x.add(neural.neuralLight(worldPosition, direction, opticalDepth, local[0], local[1])));
        }
      });

      return result;
    };

    this.vertexNode = frameFullscreenVertex;

    this.colorNode = Fn(() => {
      const fragCoord = frameFragCoord(frameGridAtlasSize).toVar();
      const cell = grid.getGridCell(fragCoord);
      const cascade = cell.cascade.toVar();

      transmittance0.assign(vec4(1.0));
      transmittance1.assign(vec4(1.0));
      transmittance2.assign(vec4(1.0));
      transmittance3.assign(vec4(1.0));

      If(cascade.lessThan(frameGridCount), () => {
        const voxel = frameGridVoxel.element(cascade).toVar();
        const worldPosition = frameGridMin.element(cascade).add(cell.gridUvw.mul(frameGridSize.element(cascade))).toVar();

        // NOTE Skip voxels with no extinction near them. The lighting pass does not read them.
        let nearExtinction: VDBTslNode = float(0.0);

        for (let i = 0; i < 8; i++) {
          const corner = vec3(i & 1, (i >> 1) & 1, (i >> 2) & 1).sub(0.5);

          nearExtinction = nearExtinction.add(gridExtinction(worldPosition.add(corner.mul(voxel))).x);
        }

        If(nearExtinction.greaterThan(0.0), () => {
          const jitter = frameNoise(gridUniforms.frameJitter, fragCoord).toVar();
          const skyDirection = normalize(frameSkyDirection).toVar();
          const local = neural ? neural.getLocalInputs(worldPosition) : null;

          const lights = Array.from({ length: frameShadowLightSlots }, (_, slot) => lightTransmittance(slot, worldPosition, jitter, local).toVar());

          const skyUp = march(worldPosition, skyDirection, float(1e20), jitter).toVar();
          const skyDown = march(worldPosition, skyDirection.negate(), float(1e20), jitter).toVar();

          transmittance0.assign(vec4(skyTotal(skyUp, skyDown), skyTotal(skyDown, skyUp), lights[0]!.y, lights[1]!.y));
          transmittance1.assign(vec4(lights[2]!.y, lights[3]!.y, lights[4]!.y, lights[5]!.y));
          transmittance2.assign(vec4(lights[0]!.x, lights[1]!.x, lights[2]!.x, lights[3]!.x));
          transmittance3.assign(vec4(lights[4]!.x, lights[5]!.x, 1.0, 1.0));
        });
      });

      return vec4(0.0);
    })();

    this.outputNode = outputStruct(transmittance0, transmittance1, transmittance2, transmittance3);
  }
}
