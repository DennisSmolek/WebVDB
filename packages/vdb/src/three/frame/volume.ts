// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameVolume.js L8–11 (render order, all lights), L152–270 (props,
//   proxy), L339–353 (box matrix), L554–568 (getDensityScale), L624–650 (dispose).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
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
 *   const volume = new VDBFrameVolume(source, { extinction: 100 });  // box: the source's (volume.bbox)
 *   scene.add(volume);                       // then renderer.render(scene, camera) as usual
 *   volume.frameRenderer.settings.multipleScattering = false;
 *   volume.setMedium("fog");                 // a measured medium: albedo, spectrum, the renderer's phase
 *
 * The phase (K5.9): the renderer's `settings.phase*` is the phase of every volume it draws. `'frame'`
 * lighting lights all volumes in one pass with it; in `'volume'` lighting (each volume marches its own
 * light) a volume's own `phase*` props, when set, override it for that volume. The source (dense or
 * sparse), its kind and its LOD are fixed at construction: make a new source and a new volume to change
 * them (`source.replace` swaps a dense frame of the same channels; the volume follows it by itself). A sparse
 * source made with `followImage` follows a `pool.replace` of its grid (a coarse-first swap), box included (G16a).
 */

import * as THREE from "three";
import { VDBError } from "../../core/errors.js";
import type { VDBThreeVolumeSource, VDBVolumeNodesOptions, VDBVolumeUniforms } from "../volume-source.js";
import { vdbVolumeUniforms } from "../volume-source.js";
import type { VDBFrameDensityMaterial } from "./density-material.js";
import type { VDBFrameInjectMaterial } from "./inject-material.js";
import type { VDBFrameShadowDensityMaterial } from "./shadow-density-material.js";
import type { VDBFrameEmissionDirectionMaterial } from "./emission-light.js";
import { vdbFrameAllLights } from "./lights.js";
import type { VDBFramePhase, VDBFramePhaseSource } from "./phase.js";
import { getFrameMediumProps, type VDBFrameMediumVolumeProps } from "./medium.js";
import { vdbFrameRenderer, type VDBFrameRenderer } from "./renderer.js";
import type { FrameVolumeRecord } from "./view.js";

// NOTE Draw the frame after all other transparent objects
const frameRenderOrder = 1000000;

/** Props of {@link VDBFrameVolume}: his `FrameVolume` props (names and defaults) for the ported subset. */
export interface VDBFrameVolumeProps {
  /**
   * The froxel renderer (default: the shared {@link vdbFrameRenderer}). Its `settings` apply to every
   * volume that uses it; pass your own `new VDBFrameRenderer()` (or `vdbFrameRenderer(threeRenderer)`)
   * for separate settings.
   */
  frameRenderer?: VDBFrameRenderer;
  /**
   * Box of the volume in its local space (default: the source's box: the `.wvdb` bbox or the sparse grid's world
   * box; else the unit box at the origin). Read back as {@link VDBFrameVolume.bbox}.
   */
  bbox?: THREE.Box3 | [[number, number, number], [number, number, number]];
  baseColor?: THREE.ColorRepresentation;
  scatterColor?: THREE.ColorRepresentation;
  /** With normalizeDensity: optical depth of the longest box side at the mean density (default 100). */
  extinction?: number;
  /** Extinction scale of each channel (renderer `settings.spectral`; with `angstromExponent`, spectrum.ts). */
  extinctionScale?: [number, number, number] | THREE.Vector3Like;
  /** Ångström exponent of the extinction spectrum: a number or a name of `frameAngstromExponents` (default 'cloud': 0, grey). */
  angstromExponent?: string | number;
  /**
   * `'volume'` lighting only: this volume's phase, overriding the renderer's `settings.phase` (default: unset, the
   * renderer's). `'frame'` lighting always uses the renderer's (one lighting pass for all volumes). The same for
   * `phaseG`, `phaseBackG`, `phaseBackWeight` and `phaseGExponent`, field by field.
   */
  phase?: VDBFramePhase | null;
  phaseG?: number | null;
  phaseBackG?: number | null;
  phaseBackWeight?: number | null;
  phaseGExponent?: number | null;
  densityScale?: number;
  normalizeDensity?: boolean;
  /**
   * The mean density `normalizeDensity` divides by (default: unset, the source's `stats.meanNonZero`, which follows
   * `source.replace`). Set one value for a sequence so every frame has the same extinction scale: his sequences divide
   * by the mean of all frames' non-zero texels (`FrameVolume.getDensityScale`).
   */
  densityMean?: number | null;
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
   * unless the cost is large"). Timings in docs/archive/handoffs/K3-2a.md.
   */
  hdda?: boolean;
}

const toBox = (bbox: VDBFrameVolumeProps["bbox"], source: VDBThreeVolumeSource): THREE.Box3 => {
  if (bbox instanceof THREE.Box3) return bbox.clone();
  const b = bbox ?? (source.kind === "dense" ? source.volume.bbox : source.bbox);
  if (!b) return new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
  return new THREE.Box3(new THREE.Vector3(...b[0]), new THREE.Vector3(...b[1]));
};

/** FrameVolume.js L339–353: the volume box as a matrix of the unit box (into `target`). */
const boxMatrix = (box: THREE.Box3, target: THREE.Matrix4): THREE.Matrix4 => {
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).max(new THREE.Vector3(1e-6, 1e-6, 1e-6));
  return target.compose(center, new THREE.Quaternion(), size);
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
  /** K3.6: the emission light's direction-pass material of this volume (made by the light on first use). */
  emissionDirectionMaterial: VDBFrameEmissionDirectionMaterial | null = null;
  readonly emissionColor: VDBVolumeNodesOptions["emissionColor"];

  baseColor: THREE.Color;
  scatterColor: THREE.Color;
  extinction: number;
  extinctionScale: THREE.Vector3;
  angstromExponent: string | number;
  /** `'volume'` lighting: phase overrides (null: the renderer's `settings`); see {@link VDBFrameVolumeProps.phase}. */
  phase: VDBFramePhase | null;
  phaseG: number | null;
  phaseBackG: number | null;
  phaseBackWeight: number | null;
  phaseGExponent: number | null;
  densityScale: number;
  normalizeDensity: boolean;
  /** See {@link VDBFrameVolumeProps.densityMean}; null: the source's mean. */
  densityMean: number | null;
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
  /**
   * The volume's box in its local space: the `bbox` prop, else the source's box. A sparse source's box is re-read
   * when the source's `version` moves (G16a: one made with `followImage` follows its grid's image, e.g. a coarse-first
   * swap), so this object is updated in place then. Do not mutate.
   */
  readonly bbox: THREE.Box3;

  /** The medium fields at construction, for `setMedium(null)`. */
  private readonly initialMedium: { baseColor: THREE.Color; extinctionScale: THREE.Vector3; angstromExponent: string | number };
  /** `source.version` the uniforms were last updated for. */
  private sourceVersion: number;
  /** G16a: the box is a sparse source's (no `bbox` prop), so it follows the source. */
  private readonly boxFromSource: boolean;
  private readonly resolvedPhase: VDBFramePhaseSource = { phase: "henyey-greenstein", phaseG: 0.8, phaseBackG: -0.3, phaseBackWeight: 0.2, phaseGExponent: 0.0 };

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
      phase = null,
      phaseG = null,
      phaseBackG = null,
      phaseBackWeight = null,
      phaseGExponent = null,
      densityScale = 1.0,
      normalizeDensity = true,
      densityMean = null,
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
    this.densityMean = densityMean;
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
    this.initialMedium = { baseColor: this.baseColor.clone(), extinctionScale: this.extinctionScale.clone(), angstromExponent };

    this.sourceUniforms = vdbVolumeUniforms(source, { extinction, densityScale, normalizeDensity });
    this.sourceVersion = source.version;

    this.proxy = new THREE.Mesh(frameRenderer.createProxyGeometry(), frameRenderer.integrateMaterial);
    this.proxy.frustumCulled = false;
    this.proxy.renderOrder = frameRenderOrder;
    this.proxy.onBeforeRender = (renderer, scene, camera) => {
      frameRenderer.render(renderer as never, scene, camera, this.proxy);
    };
    this.add(this.proxy);

    // FrameVolume.js L339–353: the volume box as a matrix of the unit box.
    const box = toBox(bbox, source);
    this.bbox = box;
    this.boxFromSource = bbox === undefined && source.kind === "sparse";

    this.volumes = [
      {
        matrix: boxMatrix(box, new THREE.Matrix4()),
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

  /** The mean density `normalizeDensity` divides by: `densityMean` when set, else the source's mean of the non-zero texels (his `meanDensity`). */
  get meanDensity(): number {
    return this.densityMean ?? this.source.stats.meanNonZero;
  }

  /** The phase this volume is drawn with in `'volume'` lighting: its own `phase*` fields where set, else the renderer's `settings` (a reused object). */
  getPhase(): Readonly<VDBFramePhaseSource> {
    const s = this.frameRenderer.settings;
    const p = this.resolvedPhase;
    p.phase = this.phase ?? s.phase;
    p.phaseG = this.phaseG ?? s.phaseG;
    p.phaseBackG = this.phaseBackG ?? s.phaseBackG;
    p.phaseBackWeight = this.phaseBackWeight ?? s.phaseBackWeight;
    p.phaseGExponent = this.phaseGExponent ?? s.phaseGExponent;
    return p;
  }

  /**
   * A measured medium in one call (`'cloud'`, `'fog'`, `'ice'`, `'soot'`, `'white-smoke'`, `'dark-smoke'`, `'haze'`,
   * `'dust'`, or {@link VDBFrameMediumVolumeProps} from `vdbFrameMedium`), as his GUI's 'Medium' field: the volume's
   * `baseColor`, `extinctionScale` and `angstromExponent`; the phase on the renderer (`frameRenderer.settings.phase`,
   * `phaseG`, `phaseGExponent`: the phase of 'frame' lighting and the default of 'volume' lighting; the volume's own
   * phase overrides are cleared) and `settings.spectral` on. The renderer's settings are shared by every volume it
   * draws. `null` restores the volume's constructor colour, extinction scale and Ångström exponent (the renderer is left
   * as it is). Returns the props applied (null for `null`); an unknown name throws `VDB_BAD_OPTION`.
   */
  setMedium(medium: string | VDBFrameMediumVolumeProps | null): VDBFrameMediumVolumeProps | null {
    if (medium === null) {
      this.baseColor.copy(this.initialMedium.baseColor);
      this.extinctionScale.copy(this.initialMedium.extinctionScale);
      this.angstromExponent = this.initialMedium.angstromExponent;
      return null;
    }
    const props = typeof medium === "string" ? getFrameMediumProps(medium) : medium;
    if (!props) throw new VDBError("VDB_BAD_OPTION", `VDBFrameVolume.setMedium: unknown medium ${JSON.stringify(medium)}; use one of vdbFrameMediumNames (@pmndrs/vdb/tools).`);
    this.baseColor.copy(props.baseColor);
    this.extinctionScale.copy(props.extinctionScale);
    this.angstromExponent = props.angstromExponent;
    this.phase = this.phaseG = this.phaseBackG = this.phaseBackWeight = this.phaseGExponent = null;
    const s = this.frameRenderer.settings;
    s.phase = props.phase;
    s.phaseG = props.phaseG;
    s.phaseGExponent = props.phaseGExponent;
    s.spectral = true;
    return props;
  }

  /** densityScale for the shaders. normalizeDensity divides it by the mean density. */
  getDensityScale(): number {
    const meanDensity = this.meanDensity;

    return this.normalizeDensity && meanDensity > 0.0 ? this.densityScale / meanDensity : this.densityScale;
  }

  /**
   * Re-read the source (textures, sizes, LOD levels). The renderer calls {@link VDBFrameVolume.syncSource} before
   * every draw, so a `source.replace()` needs no call; this forces it.
   */
  updateSource(): void {
    this.sourceVersion = this.source.version;
    this.sourceUniforms.update(this.source, { densityScale: this.densityScale, normalizeDensity: this.normalizeDensity });
    // G16a: a sparse source's box (a followImage source re-reads it after an image change)
    if (this.boxFromSource && this.source.kind === "sparse") {
      const [min, max] = this.source.bbox;
      const b = this.bbox;
      if (b.min.x !== min[0] || b.min.y !== min[1] || b.min.z !== min[2] || b.max.x !== max[0] || b.max.y !== max[1] || b.max.z !== max[2]) {
        b.min.set(...min);
        b.max.set(...max);
        for (const volume of this.volumes) boxMatrix(b, volume.matrix);
      }
    }
  }

  /** {@link VDBFrameVolume.updateSource} if the source changed since (its `version`: a resized dense frame, a replaced pool image). Cheap; called per draw. */
  syncSource(): void {
    if (this.source.version !== this.sourceVersion) this.updateSource();
  }

  /** Frees the proxy geometry and the pass materials. The source belongs to the caller. */
  dispose(): void {
    this.injectMaterial?.dispose();
    this.injectMaterial = null;
    this.densityMaterial?.dispose();
    this.densityMaterial = null;
    this.shadowDensityMaterial?.dispose();
    this.shadowDensityMaterial = null;
    this.emissionDirectionMaterial?.dispose();
    this.emissionDirectionMaterial = null;
    this.proxy.geometry.dispose();
  }
}
