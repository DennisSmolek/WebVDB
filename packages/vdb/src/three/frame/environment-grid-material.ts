// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameEnvironmentGridMaterial.js L35–283 (the grey pass).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; grey only (the spectral branch is K3.5). K3.4: the octave uniform and terms.
// Divergences: DIVERGES.md (K3.3 rows).

/**
 * The environment grid of `'frame'` lighting (K3.3): the environment light of each voxel of the last
 * shadow grid cascade, from the environment lights with no shadow slot. Sum of colour × transmittance,
 * with a one-voxel-step march that stays in the last cascade (which fits every caster and does not
 * follow the camera). Diffusion multiple scattering uses the two-sided isotropic total, which needs
 * the opposite march too. The renderer draws it only when its signature changes.
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
  frameMaxEnvironmentLights,
  frameShadowMarchDepth,
  frameShadowStep,
  frameShadowTotal,
  frameShadowTransmittance,
  type EnvironmentGridUniforms,
  type GridUniforms,
  type LightUniforms,
} from "./chunks.js";

const { Break, Fn, If, Loop, abs, float, floor, max, select, texture, uniform, uniformArray, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

const maxSteps = 1024;

export interface VDBFrameEnvironmentGridMaterialOptions {
  gridUniforms: GridUniforms;
  environmentGridUniforms: EnvironmentGridUniforms;
  lightUniforms: LightUniforms;
}

export class VDBFrameEnvironmentGridMaterial extends NodeMaterial {
  readonly frameShadowDensityMap = texture(new THREE.Texture());
  readonly frameLightCount = uniform(0, "int");
  readonly frameLightDirections = uniformArray(
    Array.from({ length: frameMaxEnvironmentLights }, () => new THREE.Vector3(0.0, 1.0, 0.0)),
    "vec3",
  );
  readonly frameLightColors = uniformArray(
    Array.from({ length: frameMaxEnvironmentLights }, () => new THREE.Vector3()),
    "vec3",
  );
  // NOTE Multiple scattering of the renderer settings (phase.ts). 0.0 is single scattering.
  readonly frameMultipleScattering = uniform(0.0);
  // NOTE Scattering octaves of the renderer settings (phase.ts). Count 0 is off.
  readonly frameScatteringOctaves = uniform(new THREE.Vector4());

  constructor(readonly frameUniforms: VDBFrameEnvironmentGridMaterialOptions) {
    super();

    this.name = "VDBFrameEnvironmentGridMaterial";
    this.vertexNode = frameFullscreenVertex;
    this.buildNodes();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }

  buildNodes(): void {
    const { gridUniforms, environmentGridUniforms, lightUniforms } = this.frameUniforms;
    const grid = createGridNodes(gridUniforms);
    const { frameGridMin, frameGridSize, frameGridVoxel, frameGridResolution } = gridUniforms;
    const { frameEnvironmentGridCascade, frameEnvironmentGridColumns, frameEnvironmentGridAtlasSize } = environmentGridUniforms;
    const { frameEnvironmentUnshadowed } = lightUniforms;
    const ms = this.frameMultipleScattering;
    const octaves = this.frameScatteringOctaves;

    this.colorNode = Fn(() => {
      const cascade = frameEnvironmentGridCascade;
      const resolution = frameGridResolution;
      const fragCoord = frameFragCoord(frameEnvironmentGridAtlasSize).toVar();
      const tile = floor(fragCoord.div(resolution)).toVar();
      const z = tile.y.mul(frameEnvironmentGridColumns).add(tile.x);
      const gridUvw = vec3(fragCoord.sub(tile.mul(resolution)), z.add(0.5)).div(resolution);
      const voxel = frameGridVoxel.element(cascade).toVar();
      const gridMin = frameGridMin.element(cascade).toVar();
      const gridSize = frameGridSize.element(cascade).toVar();
      const worldPosition = gridMin.add(gridUvw.mul(gridSize)).toVar();
      // NOTE Octaves: unshadowed light also gets the octaves
      const result = vec4(frameEnvironmentUnshadowed.mul(frameShadowTotal(float(1.0), octaves)), 1.0).toVar();

      const extinctionAt = (position: VDBTslNode) => grid.sampleGrid(this.frameShadowDensityMap, cascade, position.sub(gridMin).div(gridSize)).r;

      // NOTE Skip voxels with no extinction near them. The density pass does not read them.
      let nearExtinction: VDBTslNode = float(0.0);

      for (let i = 0; i < 8; i++) {
        const corner = vec3(i & 1, (i >> 1) & 1, (i >> 2) & 1).sub(0.5);

        nearExtinction = nearExtinction.add(extinctionAt(worldPosition.add(corner.mul(voxel))));
      }

      If(z.lessThan(resolution).and(nearExtinction.greaterThan(0.0)), () => {
        const irradiance = vec3(0.0).toVar();

        // NOTE (extinction, effective, reduced) depths of the march from worldPosition along direction
        const marchDepths = (direction: VDBTslNode) => {
          const voxelsPerUnit = abs(direction).div(voxel);
          const stepLength = float(1.0).div(max(voxelsPerUnit.x, max(voxelsPerUnit.y, voxelsPerUnit.z))).toVar();
          const step = direction.mul(stepLength).toVar();
          const probe = worldPosition.add(step.mul(0.5)).toVar();
          const depths = vec3(0.0).toVar();

          Loop(maxSteps, () => {
            const uvw = probe.sub(gridMin).div(gridSize);

            If(
              uvw.x
                .lessThan(0.0)
                .or(uvw.y.lessThan(0.0))
                .or(uvw.z.lessThan(0.0))
                .or(uvw.x.greaterThan(1.0))
                .or(uvw.y.greaterThan(1.0))
                .or(uvw.z.greaterThan(1.0))
                .or(frameShadowMarchDepth(depths, ms, octaves).greaterThan(6.0)),
              () => {
                Break();
              },
            );

            const sample = grid.sampleGrid(this.frameShadowDensityMap, cascade, uvw).toVar();

            depths.addAssign(frameShadowStep(sample.r, sample.g, ms).mul(stepLength));
            probe.addAssign(step);
          });

          return depths;
        };

        Loop(this.frameLightCount, ({ i }: { i: VDBTslNode }) => {
          const direction = this.frameLightDirections.element(i).toVar();
          const depths = marchDepths(direction);

          // NOTE Environment lights are isotropic: total transmittance, no phase
          const shadow = frameShadowTransmittance(depths, ms, octaves);

          // NOTE Diffusion: two-sided total (frameIsotropicDiffusion). It needs the march in the opposite direction too.
          const diffusion = ms.greaterThan(0.0);
          const backDepths = vec3(0.0).toVar();

          If(diffusion, () => {
            backDepths.assign(marchDepths(direction.negate()));
          });

          const total = select(diffusion, frameIsotropicDiffusion(depths.z, backDepths.z, depths.y), frameShadowTotal(shadow.y, octaves));

          irradiance.addAssign(this.frameLightColors.element(i).mul(total));
        });

        result.assign(vec4(max(irradiance, vec3(0.0)), 1.0));
      });

      return result;
    })();
  }
}
