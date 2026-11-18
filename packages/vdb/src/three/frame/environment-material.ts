// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameEnvironmentMaterial.js (all, 56 lines),
//   src/three-openvdb/frame/webgpu/FrameEnvironmentMapMaterial.js (all, 44 lines).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript. Divergences: none.

/**
 * Passes that read the scene environment (K3.3): the base class (PMREM radiance and three's IBL
 * irradiance at a direction; a new or removed environment rebuilds the shader) and the 64 × 32
 * lat-long map the CPU turns into environment lights (`@pmndrs/vdb/tools` `VDBFrameEnvironmentLights`).
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { frameEnvironmentMapHeight, frameEnvironmentMapRoughness, frameEnvironmentMapWidth } from "../../tools/frame/environment.js";
import { frameFragCoord, frameFullscreenVertex } from "./chunks.js";

const { Fn, cos, float, materialEnvIntensity, pmremTexture, sin, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

// NOTE Base of the passes that read the scene environment.
// The PMREM node needs a texture, so a new or removed environment builds the shader again.
export class VDBFrameEnvironmentMaterial extends NodeMaterial {
  environment: THREE.Texture | null = null;
  environmentNodes: VDBTslNode[] = [];

  constructor() {
    super();

    // NOTE No own envMap. Then the PMREM rotation and intensity uniforms read the pass scene values.
    (this as unknown as { envMap: THREE.Texture | null }).envMap = null;
  }

  // NOTE Subclasses build their nodes again here; they call environmentIrradiance() in them.
  rebuildNodes(): void {}

  setEnvironment(environment: THREE.Texture | null | undefined): void {
    const next = environment ?? null;
    const hasEnvironment = !!next;

    if (hasEnvironment !== !!this.environment) {
      this.environment = next;
      this.environmentNodes = [];
      this.rebuildNodes();
      this.needsUpdate = true;
    } else if (hasEnvironment && this.environment !== next) {
      this.environment = next;
      this.environmentNodes.forEach((node) => {
        node.value = next;
      });
    }
  }

  // NOTE Radiance of the PMREM environment at the roughness. Rotation is from the pass scene, intensity is not applied.
  environmentRadiance(direction: VDBTslNode, roughness: number): VDBTslNode {
    const node = pmremTexture(this.environment, direction, float(roughness));

    this.environmentNodes.push(node);

    return node;
  }

  // NOTE Diffuse irradiance of the PMREM environment, the same as three.js IBL irradiance
  environmentIrradiance(direction: VDBTslNode): VDBTslNode {
    return this.environmentRadiance(direction, 1.0).mul(Math.PI).mul(materialEnvIntensity);
  }
}

// NOTE Small lat-long map of the scene environment, for the environment lights (tools/frame/environment.ts).
// NOTE RGB is radiance, A is the v coordinate. The direction is the same as vdbFrameEnvironmentMapDirection().
export class VDBFrameEnvironmentMapMaterial extends VDBFrameEnvironmentMaterial {
  constructor() {
    super();

    this.name = "VDBFrameEnvironmentMapMaterial";
    this.vertexNode = frameFullscreenVertex;
    this.fragmentNode = this.createFragmentNode();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }

  override rebuildNodes(): void {
    this.fragmentNode = this.createFragmentNode();
  }

  createFragmentNode(): VDBTslNode {
    const mapSize = vec2(frameEnvironmentMapWidth, frameEnvironmentMapHeight);

    return Fn(() => {
      const uv = frameFragCoord(mapSize).div(mapSize).toVar();
      const phi = uv.x.mul(2.0 * Math.PI);
      const theta = uv.y.mul(Math.PI);
      const direction = vec3(sin(theta).mul(cos(phi)), cos(theta), sin(theta).mul(sin(phi)));

      if (!this.environment) {
        return vec4(0.0, 0.0, 0.0, uv.y);
      }

      return vec4(this.environmentRadiance(direction, frameEnvironmentMapRoughness), uv.y);
    })();
  }
}
