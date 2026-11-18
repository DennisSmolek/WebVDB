// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameTemporalMaterial.js L31–217 (one atlas target).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; one atlas target (the spectral second target is K3.5). Divergences: none.

/**
 * The temporal pass (R1 §2 step 5): blends this frame's froxels with the previous frames', each
 * froxel centre reprojected into the previous froxel grid (bilinear in the slice, linear between
 * slices), the history clamped to the current froxel's 6 neighbours (less ghosting), and the history
 * weight dropping to 0 for a quarter cell of motion or more. CPU twin of the reprojection:
 * `@pmndrs/vdb/tools` `vdbFrameReproject`.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { frameFragCoord, frameGetViewPosition, frameLoad, frameSample, frameSliceDepth, frameTileVertex, type FroxelUniforms, type TileUniforms } from "./chunks.js";

const { Fn, If, clamp, floor, ivec2, length, log, max, min, mix, mod, texture, uniform, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

export interface VDBFrameTemporalMaterialOptions {
  froxelUniforms: FroxelUniforms;
  froxelTileUniforms: TileUniforms;
  froxelAreaUniforms: { frameAreaRect: VDBTslNode; frameAreaSliceStart: VDBTslNode };
}

// NOTE Blends the froxels of this frame with the reprojected froxels of the previous frames
export class VDBFrameTemporalMaterial extends NodeMaterial {
  readonly frameCurrentMap = texture(new THREE.Texture());
  readonly frameHistoryMap = texture(new THREE.Texture());
  readonly frameUseHistory = uniform(0, "int");
  readonly frameTemporalBlend = uniform(0.1);
  readonly frameCameraMatrixWorld = uniform(new THREE.Matrix4());
  readonly framePreviousViewMatrix = uniform(new THREE.Matrix4());
  readonly framePreviousProjectionMatrix = uniform(new THREE.Matrix4());
  readonly framePreviousRange = uniform(new THREE.Vector2(0.1, 1.0));
  readonly froxelUniforms: FroxelUniforms;

  constructor(o: VDBFrameTemporalMaterialOptions) {
    super();

    this.name = "VDBFrameTemporalMaterial";
    this.froxelUniforms = o.froxelUniforms;

    // NOTE Only the froxel area of the volumes. The renderer clears the target first.
    this.vertexNode = frameTileVertex(o.froxelTileUniforms, o.froxelAreaUniforms.frameAreaRect, o.froxelAreaUniforms.frameAreaSliceStart);
    this.fragmentNode = this.createFragmentNode();

    this.side = THREE.DoubleSide;
    this.forceSinglePass = true;
    this.depthTest = false;
    this.depthWrite = false;
    this.blending = THREE.NoBlending;
  }

  createFragmentNode(): VDBTslNode {
    const { froxelUniforms } = this;
    const { froxelGrid, froxelColumns, froxelAtlasSize } = froxelUniforms;
    const currentMap = this.frameCurrentMap;
    const historyMap = this.frameHistoryMap;

    const getTileOrigin = (slice: VDBTslNode) => vec2(mod(slice, froxelColumns), floor(slice.div(froxelColumns))).mul(froxelGrid.xy);

    // NOTE Bilinear filter in the slice. Half a cell inside the tile, then the filter does not read other tiles.
    const sampleHistory = (slice: VDBTslNode, uv: VDBTslNode) => {
      const cell = clamp(uv.mul(froxelGrid.xy), vec2(0.5), froxelGrid.xy.sub(0.5));

      return frameSample(historyMap, getTileOrigin(slice).add(cell).div(froxelAtlasSize));
    };

    const loadCurrent = (texel: VDBTslNode) => frameLoad(currentMap, ivec2(texel), froxelAtlasSize);

    return Fn(() => {
      const fragCoord = frameFragCoord(froxelAtlasSize).toVar();
      const current = loadCurrent(fragCoord).toVar();
      const resolved = vec4(current).toVar();

      const tile = floor(fragCoord.div(froxelGrid.xy)).toVar();
      const slice = tile.y.mul(froxelColumns).add(tile.x).toVar();

      If(this.frameUseHistory.equal(1).and(slice.lessThan(froxelGrid.z)), () => {
        // NOTE Reproject the froxel center into the froxel grid of the previous frame
        const cell = fragCoord.sub(tile.mul(froxelGrid.xy)).toVar();
        const uv = cell.div(froxelGrid.xy);
        const viewDepth = frameSliceDepth(froxelUniforms, slice.add(0.5).div(froxelGrid.z));
        const worldPosition = this.frameCameraMatrixWorld.mul(vec4(frameGetViewPosition(froxelUniforms, uv, viewDepth), 1.0));
        const previousView = this.framePreviousViewMatrix.mul(worldPosition).toVar();
        const previousClip = this.framePreviousProjectionMatrix.mul(previousView).toVar();

        If(previousClip.w.greaterThan(0.0), () => {
          const previousUv = previousClip.xy.div(previousClip.w).mul(0.5).add(0.5).toVar();
          const previousRange = this.framePreviousRange;
          const previousT = log(max(previousView.z.negate(), 1e-6).div(previousRange.x)).div(log(previousRange.y.div(previousRange.x)));
          const previousSlice = previousT.mul(froxelGrid.z).sub(0.5).toVar();

          const inside = min(previousUv.x, previousUv.y)
            .greaterThanEqual(0.0)
            .and(max(previousUv.x, previousUv.y).lessThanEqual(1.0))
            .and(previousSlice.greaterThanEqual(-0.5))
            .and(previousSlice.lessThanEqual(froxelGrid.z.sub(0.5)));

          If(inside, () => {
            const slice0 = clamp(floor(previousSlice), 0.0, froxelGrid.z.sub(1.0)).toVar();
            const slice1 = min(slice0.add(1.0), froxelGrid.z.sub(1.0));
            const sliceBlend = clamp(previousSlice.sub(slice0), 0.0, 1.0).toVar();

            // NOTE The history counts less as the froxel moves: a quarter of a cell of motion in a frame uses only the
            // current froxel. A still camera moves no froxel, so the history still gathers the jittered samples.
            const motion = length(vec3(previousUv.sub(uv).mul(froxelGrid.xy), previousSlice.sub(slice)));
            const blend = mix(this.frameTemporalBlend, 1.0, clamp(motion.mul(4.0), 0.0, 1.0)).toVar();
            const history = mix(sampleHistory(slice0, previousUv), sampleHistory(slice1, previousUv), sliceBlend);

            // NOTE Clamp the history to the current neighbors. This limits ghosting of moving volumes.
            const tileOrigin = tile.mul(froxelGrid.xy);
            const cellMax = froxelGrid.xy.sub(1.0);
            const neighborMin = vec4(current).toVar();
            const neighborMax = vec4(current).toVar();

            const addNeighbor = (texel: VDBTslNode) => {
              const neighbor = loadCurrent(texel).toVar();

              neighborMin.assign(min(neighborMin, neighbor));
              neighborMax.assign(max(neighborMax, neighbor));
            };

            [
              vec2(max(cell.x.sub(1.0), 0.5), cell.y),
              vec2(min(cell.x.add(1.0), cellMax.x.add(0.5)), cell.y),
              vec2(cell.x, max(cell.y.sub(1.0), 0.5)),
              vec2(cell.x, min(cell.y.add(1.0), cellMax.y.add(0.5))),
            ].forEach((neighborCell) => {
              addNeighbor(tileOrigin.add(neighborCell));
            });

            [-1.0, 1.0].forEach((offset) => {
              const neighborSlice = slice.add(offset);

              If(neighborSlice.greaterThanEqual(0.0).and(neighborSlice.lessThan(froxelGrid.z)), () => {
                addNeighbor(getTileOrigin(neighborSlice).add(cell));
              });
            });

            resolved.assign(mix(clamp(history, neighborMin, neighborMax), current, blend));
          });
        });
      });

      return resolved;
    })();
  }
}
