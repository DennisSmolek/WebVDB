// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameShadowDensityMaterial.js L12–131 (K3.5: with the spectral
//   outputs, setSpectral / rebuildNodes and the spectral branch L111–121).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; reads density only through `vdbVolumeNodes` (the VDBVolumeSource seam), so
// one material is built per volume source (as the inject material, DIVERGES V30); with the sparse
// source the grid voxel lod reads its coarse LOD (`'coarse-pyramid'`, D28). Divergences: DIVERGES.md (K3.3 rows), V23–V25, V30.

/**
 * The shadow density pass of `'frame'` lighting (R1 §2 step 1): writes the extinction and the
 * absorption of one volume into one cascade of the world shadow grid (RG half-float slice atlas,
 * additive, so overlapping volumes sum; spectral: two RGBA targets, (extinction rgb, 0) and
 * (absorption rgb, 0), with the volume's extinction spectrum and per-channel albedo). The source is read at the grid voxel's lod. Absorption
 * limits multiple scattering in the shadow marches: dark volumes keep sharp shadows.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import type { VDBThreeVolumeSource, VDBVolumeNodes, VDBVolumeUniforms } from "../volume-source.js";
import { vdbVolumeNodes } from "../volume-source.js";
import { createFrameVolumeUniforms, createGridNodes, frameFragCoord, frameShadowAbsorption, frameTileVertex, type FrameVolumeUniforms, type GridUniforms, type TileUniforms } from "./chunks.js";

const { Discard, Fn, If, clamp, float, max, outputStruct, property, uniform, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

export interface VDBFrameShadowDensityMaterialOptions {
  gridUniforms: GridUniforms;
  gridTileUniforms: TileUniforms;
  source: VDBThreeVolumeSource;
  sourceUniforms: VDBVolumeUniforms;
  mipmaps: boolean;
}

// NOTE Writes the extinction and absorption of one volume into one cascade of the world-space shadow grid.
// NOTE One material per volume source. Pool meshes (one per cascade) set the rect and slices before each draw.
export class VDBFrameShadowDensityMaterial extends NodeMaterial {
  readonly volumeUniforms: FrameVolumeUniforms;
  readonly sourceUniforms: VDBVolumeUniforms;
  readonly source: VDBThreeVolumeSource;
  readonly gridUniforms: GridUniforms;
  readonly frameRect = uniform(new THREE.Vector4());
  readonly frameSliceStart = uniform(0.0);
  mipmaps: boolean;
  // NOTE Renderer settings.spectral: two targets, (extinction rgb, 0) and (absorption rgb, 0)
  spectral = false;

  constructor(o: VDBFrameShadowDensityMaterialOptions) {
    super();

    this.name = "VDBFrameShadowDensityMaterial";
    this.volumeUniforms = createFrameVolumeUniforms();
    this.sourceUniforms = o.sourceUniforms;
    this.source = o.source;
    this.gridUniforms = o.gridUniforms;
    this.mipmaps = o.mipmaps;
    this.vertexNode = frameTileVertex(o.gridTileUniforms, this.frameRect, this.frameSliceStart);
    this.gridTileUniforms = o.gridTileUniforms;
    this.rebuildNodes();

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

  private readonly gridTileUniforms: TileUniforms;

  // NOTE Mip sampling is in the shader only when the volume uses mipmaps
  setMipmaps(mipmaps: boolean): void {
    if (mipmaps !== this.mipmaps) {
      this.mipmaps = mipmaps;
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  setSpectral(spectral: boolean): void {
    if (spectral !== this.spectral) {
      this.spectral = spectral;
      this.rebuildNodes();
      this.needsUpdate = true;
    }
  }

  rebuildNodes(): void {
    if (!this.spectral) {
      this.colorNode = null;
      this.outputNode = null;
      this.fragmentNode = this.createFragmentNode(this.gridTileUniforms);

      return;
    }

    const extinction = property("vec4");
    const absorption = property("vec4");

    this.fragmentNode = null;
    this.colorNode = this.createFragmentNode(this.gridTileUniforms, { extinction, absorption });
    this.outputNode = outputStruct(extinction, absorption);
  }

  // NOTE spectralOutputs: { extinction, absorption } property nodes of the spectral targets. null: one target.
  createFragmentNode(gridTileUniforms: TileUniforms, spectralOutputs: { extinction: VDBTslNode; absorption: VDBTslNode } | null = null): VDBTslNode {
    const { gridUniforms, sourceUniforms } = this;
    const volume: VDBVolumeNodes = vdbVolumeNodes(this.source, sourceUniforms, { useMipmaps: this.mipmaps });
    const grid = createGridNodes(gridUniforms);
    const { frameWorldToLocal, frameAbsorbance, frameExtinctionSpectrum } = this.volumeUniforms;
    const frameWorldScale = sourceUniforms.worldScale;
    const { frameGridMin, frameGridSize, frameGridVoxel, frameGridCount } = gridUniforms;

    return Fn(() => {
      const cell = grid.getGridCell(frameFragCoord(gridTileUniforms.frameTargetSize));
      const cascade = cell.cascade.toVar();

      If(cascade.greaterThanEqual(frameGridCount), () => {
        Discard();
      });

      const worldPosition = frameGridMin.element(cascade).add(cell.gridUvw.mul(frameGridSize.element(cascade)));
      const localPosition = frameWorldToLocal.mul(vec4(worldPosition, 1.0)).xyz.toVar();

      If(volume.insideBox(localPosition).not(), () => {
        Discard();
      });

      // NOTE Absorbance scales the shadow of this volume, not its look from the camera
      const voxel = frameGridVoxel.element(cascade);
      const lod = volume.getLod(max(voxel.x, max(voxel.y, voxel.z)).div(frameWorldScale));
      const extinction = volume.sampleExtinction(localPosition, lod).mul(frameAbsorbance).toVar();

      If(extinction.lessThanEqual(0.0), () => {
        Discard();
      });

      if (spectralOutputs) {
        // NOTE Extinction and absorption of each channel. Absorption is the part that the albedo of the channel does not scatter.
        const spectralExtinction = frameExtinctionSpectrum.mul(extinction).toVar();
        const albedo = clamp(volume.sampleAlbedo(localPosition, lod), 0.0, 1.0);

        spectralOutputs.extinction.assign(vec4(spectralExtinction, 0.0));
        spectralOutputs.absorption.assign(vec4(spectralExtinction.mul(float(1.0).sub(albedo)), 0.0));

        return vec4(0.0);
      }

      return vec4(extinction, frameShadowAbsorption(extinction, volume.sampleAlbedo(localPosition, lod)), 0.0, 0.0);
    })();
  }
}
