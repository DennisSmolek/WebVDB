// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameVolume.js L8–11 (render order, all lights), L152–270 (props,
//   proxy), L339–353 (box matrix), L554–568 (getDensityScale), L624–650 (dispose).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; takes a VDBVolumeSource (dense `.wvdb`, or the sparse pool since K3.2) instead of resampling grids
// on the CPU (his L279–552: conversion, progressive upload, level sets, points, instances); one
// volume per frame volume; the mean density is the source's. Divergences: DIVERGES.md (K3.1 rows),
// V23–V25.

/**
 * A volume in the froxel renderer (his `FrameVolume`, R1 §2): a `Group` whose proxy mesh draws the
 * frame. Its props are his (same names and defaults) for the ported subset; density is read only
 * through the source's nodes (`vdbVolumeNodes`), so Step 2 swaps in the sparse source.
 *
 *   const source = vdbDenseTextures(wvdbDenseVolume(readWvdb(bytes)));
 *   const volume = new VDBFrameVolume(source, { bbox: source.volume.bbox, extinction: 100 });
 *   scene.add(volume);                       // then renderer.render(scene, camera) as usual
 *   volume.frameRenderer.settings.multipleScattering = false;
 */

import * as THREE from "three";
import { VDBError } from "../../core/errors.js";
import type { VDBThreeVolumeSource, VDBVolumeNodesOptions, VDBVolumeUniforms } from "../volume-source.js";
import { vdbVolumeUniforms } from "../volume-source.js";
import type { VDBFrameDensityMaterial } from "./density-material.js";
import type { VDBFrameInjectMaterial } from "./inject-material.js";
import type { VDBFrameShadowDensityMaterial } from "./shadow-density-material.js";
import { vdbFrameAllLights } from "./lights.js";
import type { VDBFramePhase } from "./phase.js";
import { vdbFrameRenderer, type VDBFrameRenderer } from "./renderer.js";
import type { FrameVolumeRecord } from "./view.js";

// NOTE Draw the frame after all other transparent objects
const frameRenderOrder = 1000000;

/** Props of {@link VDBFrameVolume}: his `FrameVolume` props (names and defaults) for the ported subset. */
export interface VDBFrameVolumeProps {
  /** The froxel renderer (default: the shared {@link vdbFrameRenderer}). */
  frameRenderer?: VDBFrameRenderer;
  /** World box of the volume (default: the source's box: the `.wvdb` bbox or the sparse grid's world box; else the unit box at the origin). */
  bbox?: THREE.Box3 | [[number, number, number], [number, number, number]];
  baseColor?: THREE.ColorRepresentation;
  scatterColor?: THREE.ColorRepresentation;
  /** With normalizeDensity: optical depth of the longest box side at the mean density (default 100). */
  extinction?: number;
  /** Extinction scale of each channel (renderer `settings.spectral`; with `angstromExponent`, spectrum.ts). */
  extinctionScale?: [number, number, number] | THREE.Vector3Like;
  /** Ångström exponent of the extinction spectrum: a number or a name of `frameAngstromExponents` (default 'cloud': 0, grey). */
  angstromExponent?: string | number;
  phase?: VDBFramePhase;
  phaseG?: number;
  phaseBackG?: number;
  phaseBackWeight?: number;
  phaseGExponent?: number;
  densityScale?: number;
  normalizeDensity?: boolean;
  densityCutoff?: number;
  absorbance?: number;
  opacity?: number;
  emissiveIntensity?: number;
  /** Kelvin at emission values 0 and 1. */
  temperatureRange?: [number, number];
  offset3D?: THREE.Vector3Like;
  wrap3D?: THREE.Wrapping;
  /** Light types bitmask ({@link vdbFrameLights}); default all. */
  lights?: number;
  shadowSteps?: number;
  /** Sample the source's mip levels (default true; needs a source with a mip atlas). */
  mipmaps?: boolean;
  /** Emission colour hook (DIVERGES V24); default white. */
  emissionColor?: VDBVolumeNodesOptions["emissionColor"];
  /**
   * HDDA empty-space skipping in the shadow march (sparse sources; K3.2). Skipped samples are exactly
   * 0, so the image does not change (within float noise); dense sources ignore it. Default false (D30: on the Apple GPU it
   * slowed the shadow march 1.07–3.31× in all 12 paired runs, +7 to +26 ms per frame; owner rule D29 was "on
   * unless the cost is large"). Timings in docs/handoffs/K3-2a.md.
   */
  hdda?: boolean;
}

const toBox = (bbox: VDBFrameVolumeProps["bbox"], source: VDBThreeVolumeSource): THREE.Box3 => {
  if (bbox instanceof THREE.Box3) return bbox.clone();
  const b = bbox ?? (source.kind === "dense" ? source.volume.bbox : source.bbox);
  if (!b) return new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
  return new THREE.Box3(new THREE.Vector3(...b[0]), new THREE.Vector3(...b[1]));
};

export class VDBFrameVolume extends THREE.Group {
  readonly isVDBFrameVolume = true as const;
  readonly source: VDBThreeVolumeSource;
  readonly sourceUniforms: VDBVolumeUniforms;
  readonly frameRenderer: VDBFrameRenderer;
  readonly proxy: THREE.Mesh<THREE.InstancedBufferGeometry>;
  readonly volumes: FrameVolumeRecord[];
  /** Built by the renderer on first use (the source is compiled into it). */
  injectMaterial: VDBFrameInjectMaterial | null = null;
  /** 'frame' lighting: the density and shadow density materials, built by the renderer on first use (K3.3). */
  densityMaterial: VDBFrameDensityMaterial | null = null;
  shadowDensityMaterial: VDBFrameShadowDensityMaterial | null = null;
  readonly emissionColor: VDBVolumeNodesOptions["emissionColor"];

  baseColor: THREE.Color;
  scatterColor: THREE.Color;
  extinction: number;
  extinctionScale: THREE.Vector3;
  angstromExponent: string | number;
  phase: VDBFramePhase;
  phaseG: number;
  phaseBackG: number;
  phaseBackWeight: number;
  phaseGExponent: number;
  densityScale: number;
  normalizeDensity: boolean;
  densityCutoff: number;
  absorbance: number;
  opacity: number;
  emissiveIntensity: number;
  temperatureRange: THREE.Vector2;
  offset3D: THREE.Vector3;
  wrap3D: THREE.Wrapping;
  lights: number;
  shadowSteps: number;
  mipmaps: boolean;
  hdda: boolean;

  constructor(source: VDBThreeVolumeSource, props: VDBFrameVolumeProps = {}) {
    super();

    const ok = source && ((source.kind === "dense" && !!source.textures?.density) || (source.kind === "sparse" && typeof source.createNodes === "function"));
    if (!ok) {
      throw new VDBError("VDB_BAD_OPTION", "VDBFrameVolume: pass a volume source from vdbDenseTextures() or vdbSparseSource() (@pmndrs/vdb/three).");
    }

    const {
      frameRenderer = vdbFrameRenderer(),
      bbox,
      baseColor = 0xffffff,
      scatterColor = 0x000000,
      extinction = 100.0,
      extinctionScale = [1.0, 1.0, 1.0],
      angstromExponent = "cloud",
      phase = "henyey-greenstein",
      phaseG = 0.8,
      phaseBackG = -0.3,
      phaseBackWeight = 0.2,
      phaseGExponent = 0.0,
      densityScale = 1.0,
      normalizeDensity = true,
      densityCutoff = 0.0,
      absorbance = 1.0,
      opacity = 1.0,
      emissiveIntensity = 1.0,
      temperatureRange = [0.0, 6500.0],
      offset3D = new THREE.Vector3(),
      wrap3D = THREE.ClampToEdgeWrapping,
      lights = vdbFrameAllLights,
      shadowSteps = 32,
      mipmaps = true,
      emissionColor,
      hdda = false,
    } = props;

    this.source = source;
    this.frameRenderer = frameRenderer;
    this.emissionColor = emissionColor;
    this.baseColor = new THREE.Color(baseColor);
    this.scatterColor = new THREE.Color(scatterColor);
    this.extinction = extinction;
    this.extinctionScale = Array.isArray(extinctionScale) ? new THREE.Vector3().fromArray(extinctionScale) : new THREE.Vector3().copy(extinctionScale);
    this.angstromExponent = angstromExponent;
    this.phase = phase;
    this.phaseG = phaseG;
    this.phaseBackG = phaseBackG;
    this.phaseBackWeight = phaseBackWeight;
    this.phaseGExponent = phaseGExponent;
    this.densityScale = densityScale;
    this.normalizeDensity = normalizeDensity;
    this.densityCutoff = densityCutoff;
    this.absorbance = absorbance;
    this.opacity = opacity;
    this.emissiveIntensity = emissiveIntensity;
    this.temperatureRange = new THREE.Vector2().fromArray(temperatureRange);
    this.offset3D = new THREE.Vector3().copy(offset3D);
    this.wrap3D = wrap3D;
    this.lights = lights;
    this.shadowSteps = shadowSteps;
    this.mipmaps = mipmaps;
    this.hdda = hdda;

    this.sourceUniforms = vdbVolumeUniforms(source, { extinction, densityScale, normalizeDensity });

    this.proxy = new THREE.Mesh(frameRenderer.createProxyGeometry(), frameRenderer.integrateMaterial);
    this.proxy.frustumCulled = false;
    this.proxy.renderOrder = frameRenderOrder;
    this.proxy.onBeforeRender = (renderer, scene, camera) => {
      frameRenderer.render(renderer as never, scene, camera, this.proxy);
    };
    this.add(this.proxy);

    // FrameVolume.js L339–353: the volume box as a matrix of the unit box.
    const box = toBox(bbox, source);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).max(new THREE.Vector3(1e-6, 1e-6, 1e-6));
    const matrix = new THREE.Matrix4().compose(center, new THREE.Quaternion(), size);

    this.volumes = [
      {
        matrix,
        matrixWorld: new THREE.Matrix4(),
        localToView: new THREE.Matrix4(),
        viewToLocal: new THREE.Matrix4(),
        worldToLocal: new THREE.Matrix4(),
        worldBox: new THREE.Box3(),
        viewNear: 0.0,
        viewFar: 0.0,
        ready: true,
      },
    ];
  }

  /** Mean of the non-zero density texels (0–1), the source's (his `meanDensity`). */
  get meanDensity(): number {
    return this.source.stats.meanNonZero;
  }

  /** densityScale for the shaders. normalizeDensity divides it by the mean density. */
  getDensityScale(): number {
    const meanDensity = this.meanDensity;

    return this.normalizeDensity && meanDensity > 0.0 ? this.densityScale / meanDensity : this.densityScale;
  }

  /** Re-read the source after `source.replace()` (LOD swap): textures, sizes and the mean density. */
  updateSource(): void {
    this.sourceUniforms.update(this.source, { densityScale: this.densityScale, normalizeDensity: this.normalizeDensity });
  }

  /** Frees the proxy geometry and the pass materials. The source belongs to the caller. */
  dispose(): void {
    this.injectMaterial?.dispose();
    this.injectMaterial = null;
    this.densityMaterial?.dispose();
    this.densityMaterial = null;
    this.shadowDensityMaterial?.dispose();
    this.shadowDensityMaterial = null;
    this.proxy.geometry.dispose();
  }
}
