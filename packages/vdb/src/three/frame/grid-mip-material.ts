// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameGridMipMaterial.js L1–69 (all).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript. CPU twin: `@pmndrs/vdb/tools` `vdbFrameGridMipLevel`. Divergences: none.

/**
 * Box filter mips of the last world shadow grid cascade, for multipleScattering `'neural'` (K3.4).
 * Level k (1 to levels − 1) has resolution / 2^k voxels on each side; its z slices are tiles in an
 * atlas (getAtlasLayout), the same as the grid atlas. Each texel is the mean of 2 × 2 × 2 texels of
 * the level before it (edge voxels clamped). The source is the grid atlas (level 1) or the level before.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { frameFragCoord, frameFullscreenVertex, frameLoad } from "./chunks.js";

const { Fn, float, floor, ivec2, min, mod, texture, uniform, vec2, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

export class VDBFrameGridMipMaterial extends NodeMaterial {
  readonly frameMipSource = texture(new THREE.Texture());
  // NOTE Source: voxels on each side, tile columns, first tile, texture size
  readonly frameMipSourceResolution = uniform(1.0);
  readonly frameMipSourceColumns = uniform(1.0);
  readonly frameMipSourceTile = uniform(0.0);
  readonly frameMipSourceSize = uniform(new THREE.Vector2(1.0, 1.0));
  // NOTE Target: voxels on each side, tile columns, texture size
  readonly frameMipResolution = uniform(1.0);
  readonly frameMipColumns = uniform(1.0);
  readonly frameMipSize = uniform(new THREE.Vector2(1.0, 1.0));

  constructor() {
    super();

    this.name = "VDBFrameGridMipMaterial";

    const sourceResolution = this.frameMipSourceResolution;
    const columns = this.frameMipSourceColumns;

    // NOTE Source texel of voxel (x, y, z), edge voxels clamped
    const loadSource = (x: VDBTslNode, y: VDBTslNode, z: VDBTslNode): VDBTslNode => {
      const last = sourceResolution.sub(1.0);
      const tile = this.frameMipSourceTile.add(min(z, last));
      const tileXY = vec2(mod(tile, columns), floor(tile.div(columns)));
      const texel = tileXY.mul(sourceResolution).add(vec2(min(x, last), min(y, last)));

      return frameLoad(this.frameMipSource, ivec2(texel), this.frameMipSourceSize).rg;
    };

    this.vertexNode = frameFullscreenVertex;

    this.colorNode = Fn(() => {
      const resolution = this.frameMipResolution;
      const fragCoord = floor(frameFragCoord(this.frameMipSize)).toVar();
      const tile = floor(fragCoord.div(resolution)).toVar();
      const z = tile.y.mul(this.frameMipColumns).add(tile.x).toVar();
      const x = fragCoord.x.sub(tile.x.mul(resolution)).toVar();
      const y = fragCoord.y.sub(tile.y.mul(resolution)).toVar();
      let sum: VDBTslNode = vec2(0.0);

      for (let i = 0; i < 8; i++) {
        sum = sum.add(loadSource(x.mul(2.0).add(float(i & 1)), y.mul(2.0).add(float((i >> 1) & 1)), z.mul(2.0).add(float((i >> 2) & 1))));
      }

      return vec4(sum.div(8.0), 0.0, 1.0);
    })();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }
}
