// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadowMaterial.js L27–173 (constants, createShadowRayUniforms,
//   frameShadowDepth, the Beer shadow map pass with its spectral outputs).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; reads the world shadow grid through our grid chunks (`createGridNodes`); source-free, so one
// material serves every volume source. CPU twin: `@pmndrs/vdb/tools` `vdbFrameShadowMapTexel`.
// Divergences: none.

/**
 * K3.6: the Beer shadow map of all volumes for one light (his `FrameVolumeShadowMaterial`), from the world shadow
 * grid. A fullscreen pass in the light's shadow camera: each texel marches its light ray through the last grid
 * cascade (all volumes; it does not move with the camera, so the cost is fixed) with `frameShadowSteps` samples.
 * Texel: (τ·first depth, τ·last depth, 0, τ), depths ÷ far (half-float precision); empty texels are 0, so the
 * linear filter of the map is right. Spectral: two draws; output 0 has the channel mean, output 1 (τ rgb, 0).
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { createGridNodes, type GridUniforms } from "./chunks.js";
import { frameMaxCastShadowSteps, frameShadowMaxOpticalDepth } from "../../tools/frame/shadows.js";

const { Break, Fn, If, Loop, cameraPosition, cameraProjectionMatrix, cameraProjectionMatrixInverse, cameraWorldMatrix, dot, float, length, max, min, normalize, positionGeometry, select, texture, uniform, varying, vec3, vec4 } =
  TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;

export { frameMaxCastShadowSteps };

/** Ray uniforms of one light shadow camera; the pass and the receivers measure depth with them. */
export const createShadowRayUniforms = () => ({
  frameShadowOrigin: uniform(new THREE.Vector3()),
  frameShadowForward: uniform(new THREE.Vector3(0.0, 0.0, -1.0)),
  frameShadowPerspective: uniform(0, "int"),
  frameShadowFar: uniform(1.0),
});

export type ShadowRayUniforms = { frameShadowOrigin: N; frameShadowForward: N; frameShadowPerspective: N; frameShadowFar: N };

/** Depth of a world position: the distance from the light (perspective) or from the camera plane (orthographic). */
export const frameShadowDepth = (rayUniforms: ShadowRayUniforms, worldPosition: N): N => {
  const { frameShadowOrigin, frameShadowForward, frameShadowPerspective } = rayUniforms;
  const offset = worldPosition.sub(frameShadowOrigin);

  return select(frameShadowPerspective.equal(1), length(offset), dot(offset, frameShadowForward));
};

export class VDBFrameVolumeShadowMaterial extends NodeMaterial {
  readonly frameShadowDensityMap = texture(new THREE.Texture());
  readonly frameShadowSteps = uniform(32, "int");
  readonly frameShadowOutput = uniform(0, "int");
  readonly frameUniforms: { gridUniforms: GridUniforms; rayUniforms: ShadowRayUniforms };
  spectral = false;

  constructor(o: { gridUniforms: GridUniforms }, rayUniforms: ShadowRayUniforms) {
    super();

    this.name = "VDBFrameVolumeShadowMaterial";
    this.frameUniforms = { gridUniforms: o.gridUniforms, rayUniforms };
    this.vertexNode = vec4(positionGeometry.xy, 0.0, 1.0);
    this.fragmentNode = this.createFragmentNode();

    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }

  setSpectral(spectral: boolean): void {
    if (spectral !== this.spectral) {
      this.spectral = spectral;
      this.fragmentNode = this.createFragmentNode();
      this.needsUpdate = true;
    }
  }

  createFragmentNode(): N {
    const { spectral } = this;
    const { gridUniforms, rayUniforms } = this.frameUniforms;
    const grid = createGridNodes(gridUniforms);
    const { frameGridMin, frameGridSize, frameGridCount } = gridUniforms;
    const { frameShadowFar } = rayUniforms;
    const ndc = varying(positionGeometry.xy);

    // NOTE World position of a point on the pixel ray of the current camera
    const unproject = (z: N): N => {
      const view = cameraProjectionMatrixInverse.mul(vec4(ndc, z, 1.0));
      const world = cameraWorldMatrix.mul(vec4(view.xyz.div(view.w), 1.0));

      return world.xyz;
    };

    return Fn(() => {
      // NOTE Two points in the depth range of all backends give the ray direction
      const near = unproject(float(0.25)).toVar();
      const direction = normalize(unproject(float(0.75)).sub(near)).toVar();
      const perspective = cameraProjectionMatrix.element(3).w.lessThan(0.5);

      // NOTE Perspective rays start at the light. Orthographic rays start on the camera plane.
      const origin = select(perspective, cameraPosition, near.sub(direction.mul(dot(near.sub(cameraPosition), direction)))).toVar();

      // NOTE Spectral: optical depth of each channel
      const opticalDepth = (spectral ? vec3(0.0) : float(0.0)).toVar();
      const first = float(0.0).toVar();
      const last = float(0.0).toVar();

      If(frameGridCount.greaterThan(0), () => {
        const cascade = frameGridCount.sub(1).toVar();
        const boxMin = frameGridMin.element(cascade);
        const boxMax = boxMin.add(frameGridSize.element(cascade));
        const inverseDirection = vec3(1.0).div(direction);
        const tA = boxMin.sub(origin).mul(inverseDirection);
        const tB = boxMax.sub(origin).mul(inverseDirection);
        const tMin = min(tA, tB);
        const tMax = max(tA, tB);
        const tStart = max(0.0, max(tMin.x, max(tMin.y, tMin.z))).toVar();
        const tEnd = min(tMax.x, min(tMax.y, tMax.z)).toVar();

        If(tEnd.greaterThan(tStart), () => {
          const steps = max(this.frameShadowSteps, 1);
          const stepLength = tEnd.sub(tStart).div(float(steps)).toVar();
          const firstT = float(-1.0).toVar();
          const lastT = float(-1.0).toVar();

          Loop(steps, ({ i }: { i: N }) => {
            If((spectral ? min(opticalDepth.x, min(opticalDepth.y, opticalDepth.z)) : opticalDepth).greaterThan(frameShadowMaxOpticalDepth), () => {
              Break();
            });

            const t = tStart.add(float(i).mul(stepLength)).toVar();
            const sample = origin.add(direction.mul(t.add(stepLength.mul(0.5))));
            const gridSample = grid.sampleGrid(this.frameShadowDensityMap, cascade, grid.gridUvw(cascade, sample));
            const extinction = (spectral ? gridSample.rgb : gridSample.r).toVar();

            If((spectral ? dot(extinction, vec3(1.0)) : extinction).greaterThan(0.0), () => {
              If(firstT.lessThan(0.0), () => {
                firstT.assign(t);
              });

              lastT.assign(t.add(stepLength));
              opticalDepth.addAssign(extinction.mul(stepLength));
            });
          });

          If((spectral ? dot(opticalDepth, vec3(1.0)) : opticalDepth).greaterThan(0.0), () => {
            first.assign(frameShadowDepth(rayUniforms, origin.add(direction.mul(firstT))).div(frameShadowFar));
            last.assign(frameShadowDepth(rayUniforms, origin.add(direction.mul(lastT))).div(frameShadowFar));
          });
        });
      });

      if (spectral) {
        const meanDepth = dot(opticalDepth, vec3(1.0 / 3.0)).toVar();

        return select(this.frameShadowOutput.equal(1), vec4(opticalDepth, 0.0), vec4(first.mul(meanDepth), last.mul(meanDepth), 0.0, meanDepth));
      }

      return vec4(first.mul(opticalDepth), last.mul(opticalDepth), 0.0, opticalDepth);
    })();
  }
}
