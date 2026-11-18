// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L97–142 (geometry, targets, scenes), L144–160
//   (getClipRect), L162–174 (getMaxTextureSize), L191–246 (settings, the 'volume' subset), L393–470
//   (constructor, render), L474–481 (createProxyGeometry), L599–631 (collectVolumes,
//   updateMipmapFlags), L716–720 (getSliceRange), L722–776 (updateTargets, 'volume' part), L897–970
//   (renderFrame, updateFroxelArea), L1014–1061 (prepareLightScene, renderInject), L1430–1434
//   (getFroxelRect), L1476–1652 (light records, visitLight, updateLights: scene lights), L1718–1784
//   (dispose);
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L800–830 (setVolumeUniforms).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; `'volume'` lighting only (the 'frame' lighting grid, transmittance, density,
// lighting, temporal, environment, volume-shadow and neural passes are K3.3–K3.6); one inject
// material per volume source (the source is compiled into it); grey only; per-pass timing hooks.
// Divergences: DIVERGES.md (K3.1 rows), V23–V25.

/**
 * The froxel renderer (K3.1, R1 §6 Step 1): his `FrameRenderer` with `lighting: 'volume'`. Each
 * `VDBFrameVolume` proxy calls {@link VDBFrameRenderer.render} from `onBeforeRender`; the first
 * proxy in a render draws the froxels (one inject pass per volume, one instance per slice) into a
 * slice atlas, and every proxy then draws the integrate material, which composites the atlas over
 * the scene up to the scene depth.
 */

import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { createFroxelUniforms, createLightUniforms, createTileUniforms, frameMaxLights } from "./chunks.js";
import { VDBFrameInjectMaterial } from "./inject-material.js";
import { VDBFrameIntegrateMaterial } from "./integrate-material.js";
import { getAtlasLayout, getFroxelSize } from "./layout.js";
import { getFramePhase, getFramePhaseSpectrum, getFrameMultipleScattering, type VDBFrameMultipleScattering } from "./phase.js";
import { getFrameExtinctionSpectrum } from "./spectrum.js";
import { FrameVolumeCollector, getFrameFadeRadius, getFroxelArea, getFroxelRect, getFroxelSliceRange, type FrameVolumeRecord } from "./view.js";
import type { VDBFrameVolume } from "./volume.js";

/** Renderer settings: his `frameRenderer.settings`, the subset ported in Step 1 (same names and meanings). */
export interface VDBFrameRendererSettings {
  /** Froxel grid width (default 512). */
  width: number;
  /** Froxel grid height (default 512). */
  height: number;
  /** N > 0: at most one froxel per N x N canvas (CSS) pixels. 0: no cap. */
  pixelsPerFroxel: number;
  /** Slices (default 24). */
  depth: number;
  /** Most sub-steps across one slice (default 32). */
  subSteps: number;
  near: number;
  far: number;
  /** View distance where the density fades to 0 (FogExp2-like). Infinity: no fade. */
  fadeRadius: number;
  /** Golden-ratio jitter of the sub-steps and shadow start (default false). */
  jitter: boolean;
  /** His froxel reprojection. Not ported yet (K3.3): must stay false. */
  temporal: boolean;
  /** Only `'volume'` is ported (K3.3 ports `'frame'`). */
  lighting: "volume" | "frame";
  /** Blend the volumes over the scene in sRGB, as WebGLRenderer. false blends in linear space. */
  srgbBlending: boolean;
  /**
   * Multiple scattering in the shadow march: false (single scattering) or true (1D diffusion).
   * `'neural'` is the same as true with `'volume'` lighting (his network runs only in `'frame'`
   * lighting). `'two-sided'` and `'octaves'` are K3.4: they warn and run as true and false.
   */
  multipleScattering: VDBFrameMultipleScattering;
  /** Grey only in Step 1 (K3.5). */
  spectral: boolean;
}

/** Per-pass hook (D25: timings are provisional). `pass` is `'inject'` (per volume) or `'froxels'` (all). */
export interface VDBFramePassHook {
  begin?(pass: "froxels" | "inject", detail?: { volume: VDBFrameVolume }): void;
  end?(pass: "froxels" | "inject", detail?: { volume: VDBFrameVolume }): void;
}

const clearColor = new THREE.Color();
const froxelSize = { width: 1, height: 1 };
const lightPosition = new THREE.Vector3();
const lightTarget = new THREE.Vector3();
const lightColor = new THREE.Color();

const tileQuadPosition = new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0], 3);

const createRenderTarget = (width: number, height: number, options: THREE.RenderTargetOptions = {}) =>
  new THREE.RenderTarget(width, height, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: false,
    depthBuffer: false,
    ...options,
  });

const createScene = () => {
  const scene = new THREE.Scene();
  scene.matrixWorldAutoUpdate = false;

  return scene;
};

// NOTE Instanced tile quads: one instance for each slice
const createTileGeometry = () => {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = 0;
  geometry.setAttribute("position", tileQuadPosition);

  return geometry;
};

// NOTE Clip rectangle of a froxel cell rectangle, padding cells larger on each side. An empty rectangle stays empty.
const clipRect = { x0: 0, y0: 0, x1: 0, y1: 0 };

const getClipRect = (rect: THREE.Vector4, width: number, height: number, padding: number) => {
  if (rect.z <= rect.x || rect.w <= rect.y) {
    clipRect.x0 = clipRect.y0 = clipRect.x1 = clipRect.y1 = 0.0;

    return clipRect;
  }

  clipRect.x0 = (Math.max(0, rect.x - padding) / width) * 2.0 - 1.0;
  clipRect.y0 = (Math.max(0, rect.y - padding) / height) * 2.0 - 1.0;
  clipRect.x1 = (Math.min(width, rect.z + padding) / width) * 2.0 - 1.0;
  clipRect.y1 = (Math.min(height, rect.w + padding) / height) * 2.0 - 1.0;

  return clipRect;
};

const getMaxTextureSize = (renderer: WebGPURenderer) => {
  const backend = renderer.backend as unknown as { device?: GPUDevice };

  return backend.device ? backend.device.limits.maxTextureDimension2D : Infinity;
};

interface LightRecord {
  position: THREE.Vector4;
  color: THREE.Vector4;
  spot: THREE.Vector4;
  falloff: THREE.Vector2;
}

interface LightList {
  length: number;
  entries: LightRecord[];
}

type RendererLike = WebGPURenderer & { isWebGPURenderer?: boolean };

/** His `FrameRenderer` with `'volume'` lighting. One instance can serve many volumes and cameras. */
export class VDBFrameRenderer {
  settings: VDBFrameRendererSettings = {
    width: 512,
    height: 512,
    pixelsPerFroxel: 0,
    depth: 24,
    subSteps: 32,
    near: 0.1,
    far: Infinity,
    fadeRadius: Infinity,
    jitter: false,
    temporal: false,
    lighting: "volume",
    srgbBlending: true,
    multipleScattering: true,
    spectral: false,
  };

  /** Per-pass hooks (timing, debugging). */
  passHook: VDBFramePassHook | null = null;

  /**
   * Timing/debug hook: proxies draw the integrate pass over the last froxel atlas without rendering
   * the froxels again (the per-pass timing harness times integrate alone with it). Not his.
   */
  holdFroxels = false;

  // NOTE TSL uniform nodes. Passes share them, the CPU writes .value.
  readonly froxelUniforms = createFroxelUniforms();
  // NOTE Canvas and drawing buffer sizes that settings.pixelsPerFroxel holds (layout.ts getFroxelSize)
  readonly froxelHeldSize = { canvasX: 0, canvasY: 0, bufferX: 0, bufferY: 0 };
  readonly froxelTileUniforms = createTileUniforms();
  readonly froxelArea = { rect: new THREE.Vector4(), first: 0, count: 0 };
  readonly lightUniforms = createLightUniforms();

  readonly integrateMaterial: VDBFrameIntegrateMaterial;

  // NOTE The light scene draws the volume inject pass. The mesh takes the inject material of each volume.
  readonly injectScene = createScene();
  readonly injectMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.Material> = new THREE.Mesh(createTileGeometry(), new THREE.MeshBasicMaterial());

  atlasTarget: THREE.RenderTarget | null = null;
  lightLimitWarned = false;
  private readonly warned = new Set<string>();

  // NOTE collectVolumes result of the current render
  readonly collected: { renderId: number; camera: THREE.Camera | null; scene: THREE.Object3D | null; result: FrameVolumeCollector<VDBFrameVolume> | null } = {
    renderId: -1,
    camera: null,
    scene: null,
    result: null,
  };
  readonly volumeCollector = new FrameVolumeCollector<VDBFrameVolume>();
  readonly sliceRange = { first: 0, count: 0 };
  readonly lightState = {
    camera: null as THREE.Camera | null,
    directions: { length: 0, entries: [] } as LightList,
    points: { length: 0, entries: [] } as LightList,
    spots: { length: 0, entries: [] } as LightList,
  };

  cachedFrame = -1;
  cachedCamera: THREE.Camera | null = null;
  rendererWarned = false;

  constructor() {
    this.integrateMaterial = new VDBFrameIntegrateMaterial({ froxelUniforms: this.froxelUniforms });
    this.injectMesh.frustumCulled = false;
    this.injectScene.add(this.injectMesh);
  }

  private warnOnce(key: string, message: string): void {
    if (!this.warned.has(key)) {
      this.warned.add(key);
      console.warn(`VDBFrameRenderer: ${message}`);
    }
  }

  // NOTE Each frame volume proxy calls this. The first proxy in a frame renders the frame. Other proxies draw nothing.
  render(renderer: RendererLike, scene: THREE.Object3D & { overrideMaterial?: THREE.Material | null }, camera: THREE.Camera, proxy: THREE.Mesh<THREE.InstancedBufferGeometry>): void {
    if (!renderer.isWebGPURenderer) {
      if (!this.rendererWarned) {
        console.error("VDBFrameVolume: needs WebGPURenderer (three/webgpu).");
        this.rendererWarned = true;
      }

      return;
    }

    const frame = renderer.info.render.calls;

    // NOTE Override passes (depth, normals) of other effects must not draw or start the frame
    if (scene.overrideMaterial) {
      proxy.geometry.instanceCount = 0;

      return;
    }

    if (frame === this.cachedFrame && camera === this.cachedCamera) {
      proxy.geometry.instanceCount = 0;

      return;
    }

    if (this.holdFroxels && this.atlasTarget) {
      this.cachedFrame = frame;
      this.cachedCamera = camera;
      proxy.geometry.instanceCount = 1;

      return;
    }

    const { volumes } = this.collectVolumes(renderer, scene, camera);

    if (volumes.length > 0) {
      this.renderFrame(renderer, scene, camera, volumes);
    }

    // NOTE Nested renders increase info.render.calls. Keep the new value. Then other proxies in this frame skip.
    this.cachedFrame = renderer.info.render.calls;
    this.cachedCamera = camera;

    proxy.geometry.instanceCount = volumes.length > 0 ? 1 : 0;
  }

  /**
   * Draws only the froxel atlas for this scene and camera (no integrate): the per-pass timing
   * harness renders this alone. Returns the number of volumes in view.
   */
  renderFroxels(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera): number {
    this.collected.renderId = -1;
    const { volumes } = this.collectVolumes(renderer, scene, camera);

    if (volumes.length > 0) {
      this.renderFrame(renderer, scene, camera, volumes);
    }

    return volumes.length;
  }

  // NOTE A quad. The integrate material puts it on the screen rectangle of the froxel area.
  createProxyGeometry(): THREE.InstancedBufferGeometry {
    const geometry = createTileGeometry();

    geometry.instanceCount = 1;

    return geometry;
  }

  // NOTE Visible volumes, one time in a render (the render id of three.js NodeFrame, internal)
  collectVolumes(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera): FrameVolumeCollector<VDBFrameVolume> {
    const renderId = (renderer as unknown as { _nodes?: { nodeFrame?: { renderId?: number } } })._nodes?.nodeFrame?.renderId ?? -1;
    const { collected } = this;

    if (renderId !== -1 && collected.renderId === renderId && collected.camera === camera && collected.scene === scene && collected.result) {
      return collected.result;
    }

    collected.renderId = renderId;
    collected.camera = camera;
    collected.scene = scene;
    collected.result = this.volumeCollector.collect(scene, camera as THREE.PerspectiveCamera, this.settings, false, null);

    if (collected.result.volumes.length > 0) {
      this.froxelUniforms.froxelRange.value.set(collected.result.near, collected.result.far);
    }

    return collected.result;
  }

  // NOTE Slices of a volume. Returns this.sliceRange ({ first, count }), used again in the next call.
  getSliceRange(volume: FrameVolumeRecord) {
    const { froxelRange, froxelGrid } = this.froxelUniforms;

    return getFroxelSliceRange(volume, froxelRange.value, froxelGrid.value.z, this.sliceRange);
  }

  private checkSettings(): void {
    const s = this.settings;
    if (s.lighting !== "volume") this.warnOnce("lighting", `lighting '${s.lighting}' is not ported yet (K3.3); rendering 'volume' lighting.`);
    if (s.temporal) this.warnOnce("temporal", "temporal is not ported yet (K3.3); rendering without it.");
    if (s.spectral) this.warnOnce("spectral", "spectral is not ported yet (K3.5); rendering grey.");
    if (s.multipleScattering === "two-sided" || s.multipleScattering === "octaves") {
      this.warnOnce("ms", `multipleScattering '${s.multipleScattering}' is not ported yet (K3.4); running it as ${s.multipleScattering === "two-sided" ? "true" : "false"}.`);
    }
  }

  /** The multiple scattering mode actually run (see settings.multipleScattering). */
  private multipleScatteringMode(): VDBFrameMultipleScattering {
    const ms = this.settings.multipleScattering;
    if (ms === "two-sided") return true;
    if (ms === "octaves") return false;
    return ms;
  }

  updateTargets(renderer: RendererLike): void {
    const { depth, subSteps } = this.settings;
    const { width, height } = getFroxelSize(this.settings, renderer, froxelSize, this.froxelHeldSize);
    const froxelAtlas = getAtlasLayout(width, height, depth);

    if (!this.atlasTarget || this.atlasTarget.width !== froxelAtlas.width || this.atlasTarget.height !== froxelAtlas.height) {
      const maxTextureSize = getMaxTextureSize(renderer);

      if (froxelAtlas.width > maxTextureSize || froxelAtlas.height > maxTextureSize) {
        console.warn("VDBFrameRenderer: froxel atlas is larger than maxTextureSize. Decrease width, height or depth.");
      }

      if (this.atlasTarget) {
        this.atlasTarget.dispose();
      }

      this.atlasTarget = createRenderTarget(froxelAtlas.width, froxelAtlas.height);
    }

    this.froxelUniforms.froxelGrid.value.set(width, height, depth);
    this.froxelUniforms.froxelColumns.value = froxelAtlas.columns;
    this.froxelUniforms.froxelSubSteps.value = Math.max(1, Math.round(subSteps));
    this.froxelTileUniforms.frameTileSize.value.set(width, height);
    this.froxelTileUniforms.frameTileColumns.value = froxelAtlas.columns;
    this.froxelTileUniforms.frameTargetSize.value.set(froxelAtlas.width, froxelAtlas.height);
    this.froxelUniforms.froxelAtlasSize.value.set(froxelAtlas.width, froxelAtlas.height);
    this.integrateMaterial.froxelAtlas.value = this.atlasTarget.texture;
  }

  renderFrame(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera, volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"]): void {
    this.checkSettings();

    const currentRenderTarget = renderer.getRenderTarget();
    const currentActiveCubeFace = renderer.getActiveCubeFace();
    const currentActiveMipmapLevel = renderer.getActiveMipmapLevel();
    const currentAutoClear = renderer.autoClear;
    const currentClearAlpha = renderer.getClearAlpha();
    const currentScissorTest = renderer.getScissorTest();

    renderer.getClearColor(clearColor);

    this.updateTargets(renderer);
    this.updateFroxelArea(volumes, camera);

    const cam = camera as THREE.PerspectiveCamera & { isOrthographicCamera?: boolean };
    this.froxelUniforms.froxelProjectionInverse.value.copy(camera.projectionMatrixInverse);
    this.froxelUniforms.frameFadeRadius.value = getFrameFadeRadius(this.settings);
    this.integrateMaterial.frameCameraNear.value = cam.near;
    this.integrateMaterial.frameCameraFar.value = cam.far;
    this.integrateMaterial.frameCameraOrthographic.value = cam.isOrthographicCamera ? 1 : 0;
    this.integrateMaterial.setSrgbBlending(
      this.settings.srgbBlending && renderer.outputColorSpace === THREE.SRGBColorSpace,
      renderer.toneMapping === THREE.NoToneMapping,
    );

    // NOTE Pass materials are unlit. Nested renders do not update shadow maps.
    renderer.autoClear = false;
    renderer.setScissorTest(false);
    renderer.setClearColor(0x000000, 0.0);

    // NOTE Golden ratio steps give a new, evenly spread jitter value for each frame
    this.froxelUniforms.frameJitter.value = this.settings.jitter ? (this.jitterIndex * 0.618034) % 1.0 : -1.0;
    this.jitterIndex = (this.jitterIndex + 1) % 1024;

    this.passHook?.begin?.("froxels");
    this.renderInject(renderer, scene, camera, volumes);
    this.passHook?.end?.("froxels");

    renderer.setScissorTest(currentScissorTest);
    renderer.autoClear = currentAutoClear;
    renderer.setClearColor(clearColor, currentClearAlpha);
    renderer.setRenderTarget(currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel);
  }

  jitterIndex = 0;

  // NOTE Froxel area of the volumes, for the integrate pass
  updateFroxelArea(volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"], camera: THREE.Camera): void {
    const { froxelGrid, froxelRange } = this.froxelUniforms;
    const { x: width, y: height, z: depth } = froxelGrid.value;
    const area = getFroxelArea(volumes, camera, width, height, froxelRange.value, depth, this.froxelArea);

    // NOTE One more cell on each side: the linear filter of the integrate pass reads one cell away
    const { x0, y0, x1, y1 } = getClipRect(area.rect, width, height, area.count > 0 ? 1 : 0);

    this.integrateMaterial.frameScreenRect.value.set(x0, y0, x1, y1);
    this.integrateMaterial.frameSliceRange.value.set(area.first, area.first + area.count);
  }

  renderInject(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera, volumes: FrameVolumeCollector<VDBFrameVolume>["volumes"]): void {
    const { atlasTarget, injectScene, injectMesh } = this;

    this.injectMesh.layers.mask = camera.layers.mask;
    this.updateLights(scene, camera);

    renderer.setRenderTarget(atlasTarget);
    renderer.clear(true, false, false);

    for (let i = 0; i < volumes.length; i++) {
      const { frameVolume, volume } = volumes[i]!;
      const injectMaterial = this.getInjectMaterial(frameVolume);

      if (frameVolume.phase === "lorenz-mie" || frameVolume.phase === "lorenz-mie-rgb") {
        this.warnOnce("mie", "Lorenz-Mie phases are not ported yet (K3.4); they render as isotropic.");
      }

      // NOTE Draw only the froxel rectangle and the slices of this volume. One instance for each slice.
      if (!this.getFroxelRect(volume, camera, injectMaterial.frameRect.value)) {
        continue;
      }

      const { first, count } = this.getSliceRange(volume);

      injectMaterial.frameSliceStart.value = first;
      injectMesh.geometry.instanceCount = count;
      injectMesh.material = injectMaterial;

      injectMaterial.frameCameraMatrixWorld.value.copy(camera.matrixWorld);
      setVolumeUniforms(injectMaterial, frameVolume, volume);
      injectMaterial.frameLights.value = frameVolume.lights;
      injectMaterial.frameShadowSteps.value = frameVolume.shadowSteps;
      injectMaterial.frameMultipleScattering.value = getFrameMultipleScattering(this.multipleScatteringMode(), frameVolume);

      this.passHook?.begin?.("inject", { volume: frameVolume });
      renderer.setRenderTarget(atlasTarget);
      renderer.render(injectScene, camera);
      this.passHook?.end?.("inject", { volume: frameVolume });
    }
  }

  /** The inject material of a volume: its source is compiled in (DIVERGES: one material per source). */
  getInjectMaterial(frameVolume: VDBFrameVolume): VDBFrameInjectMaterial {
    const mipmaps = frameVolume.mipmaps && frameVolume.source.lodLevels > 0;
    let material = frameVolume.injectMaterial;

    if (!material || material.froxelUniforms !== this.froxelUniforms) {
      material?.dispose();
      material = new VDBFrameInjectMaterial({
        froxelUniforms: this.froxelUniforms,
        froxelTileUniforms: this.froxelTileUniforms,
        lightUniforms: this.lightUniforms,
        source: frameVolume.source,
        sourceUniforms: frameVolume.sourceUniforms,
        mipmaps,
        ...(frameVolume.emissionColor ? { emissionColor: frameVolume.emissionColor } : {}),
      });
      frameVolume.injectMaterial = material;
    }

    material.setMipmaps(mipmaps);

    return material;
  }

  // NOTE Screen rectangle of the volume in froxel cells. Writes target. Returns false when it covers no cell.
  getFroxelRect(volume: FrameVolumeRecord, camera: THREE.Camera, target: THREE.Vector4): boolean {
    const { x: width, y: height } = this.froxelUniforms.froxelGrid.value;

    return getFroxelRect(volume, camera, width, height, target);
  }

  // NOTE A pooled light record of updateLights: (position, w type), (color, w spot cos), (spot direction, cos), falloff
  getLightRecord(list: LightList): LightRecord {
    if (list.length === list.entries.length) {
      list.entries.push({ position: new THREE.Vector4(), color: new THREE.Vector4(), spot: new THREE.Vector4(), falloff: new THREE.Vector2() });
    }

    const record = list.entries[list.length++]!;

    record.spot.set(0.0, 0.0, 0.0, 0.0);
    record.falloff.set(0.0, 0.0);

    return record;
  }

  // NOTE Light i of updateLights in slot order: directional, point, spot
  getSortedLight(index: number): LightRecord {
    const { directions, points, spots } = this.lightState;

    if (index < directions.length) {
      return directions.entries[index]!;
    }

    if (index < directions.length + points.length) {
      return points.entries[index - directions.length]!;
    }

    return spots.entries[index - directions.length - points.length]!;
  }

  // NOTE traverseVisible callback of updateLights. this.lightState holds the state of the current call.
  readonly visitLight = (object: THREE.Object3D): void => {
    const state = this.lightState;
    const light = object as THREE.Light & {
      isAmbientLight?: boolean;
      isHemisphereLight?: boolean;
      isLightProbe?: boolean;
      isDirectionalLight?: boolean;
      isSpotLight?: boolean;
      isPointLight?: boolean;
      target?: THREE.Object3D;
      angle?: number;
      penumbra?: number;
      distance?: number;
      decay?: number;
    };

    if (!light.isLight || !light.layers.test(state.camera!.layers)) {
      return;
    }

    lightColor.copy(light.color).multiplyScalar(light.intensity);
    lightPosition.setFromMatrixPosition(light.matrixWorld);

    if (light.isAmbientLight) {
      this.lightUniforms.frameAmbientColor.value.add(lightColor);
    } else if (light.isHemisphereLight || light.isLightProbe) {
      this.warnOnce("light-type", "hemisphere lights and light probes are not ported yet (K3.3); they are ignored.");
    } else if (light.isDirectionalLight) {
      const record = this.getLightRecord(state.directions);

      lightTarget.setFromMatrixPosition(light.target!.matrixWorld);
      lightPosition.sub(lightTarget).normalize();
      record.position.set(lightPosition.x, lightPosition.y, lightPosition.z, 1.0);
      record.color.set(lightColor.r, lightColor.g, lightColor.b, 0.0);
    } else if (light.isSpotLight) {
      const record = this.getLightRecord(state.spots);

      lightTarget.setFromMatrixPosition(light.target!.matrixWorld);
      record.position.set(lightPosition.x, lightPosition.y, lightPosition.z, 3.0);
      record.color.set(lightColor.r, lightColor.g, lightColor.b, Math.cos(light.angle! * (1.0 - light.penumbra!)));
      lightTarget.subVectors(lightPosition, lightTarget).normalize();
      record.spot.set(lightTarget.x, lightTarget.y, lightTarget.z, Math.cos(light.angle!));
      record.falloff.set(light.distance!, light.decay!);
    } else if (light.isPointLight) {
      const record = this.getLightRecord(state.points);

      record.position.set(lightPosition.x, lightPosition.y, lightPosition.z, 2.0);
      record.color.set(lightColor.r, lightColor.g, lightColor.b, 0.0);
      record.falloff.set(light.distance!, light.decay!);
    }
  };

  // NOTE Scene lights into the light uniforms. Node materials in custom passes have no three.js light uniforms.
  // NOTE Step 1: ambient, directional, point and spot lights; the scene environment is not used (K3.4).
  updateLights(scene: THREE.Object3D & { environment?: THREE.Texture | null }, camera: THREE.Camera): void {
    const { frameAmbientColor, frameLightCount, frameLightPositions, frameLightColors, frameLightSpots, frameLightFalloffs } = this.lightUniforms;
    const state = this.lightState;
    const { directions, points, spots } = state;

    state.camera = camera;
    directions.length = 0;
    points.length = 0;
    spots.length = 0;

    frameAmbientColor.value.setRGB(0.0, 0.0, 0.0);

    if (scene.environment) {
      this.warnOnce("environment", "the scene environment does not light volumes yet (K3.4).");
    }

    scene.traverseVisible(this.visitLight);
    state.camera = null;

    const lightTotal = directions.length + points.length + spots.length;

    if (lightTotal > frameMaxLights && !this.lightLimitWarned) {
      console.warn(`VDBFrameRenderer: too many lights. Only ${frameMaxLights} lights are used.`);
      this.lightLimitWarned = true;
    }

    const count = Math.min(lightTotal, frameMaxLights);
    const positions = frameLightPositions.array as THREE.Vector4[];
    const colors = frameLightColors.array as THREE.Vector4[];
    const spotArray = frameLightSpots.array as THREE.Vector4[];
    const falloffs = frameLightFalloffs.array as THREE.Vector2[];

    for (let i = 0; i < count; i++) {
      const light = this.getSortedLight(i);

      positions[i]!.copy(light.position);
      colors[i]!.copy(light.color);
      spotArray[i]!.copy(light.spot);
      falloffs[i]!.copy(light.falloff);
    }

    frameLightCount.value = count;
  }

  dispose(): void {
    this.atlasTarget?.dispose();
    this.atlasTarget = null;
    this.integrateMaterial.dispose();
    this.injectMesh.geometry.dispose();
    (this.injectMesh.material as THREE.Material).dispose();
    this.cachedFrame = -1;
    this.cachedCamera = null;
  }
}

/** frameShaderChunks.js L800–830 `setVolumeUniforms`: the values of one volume before its inject draw. */
export function setVolumeUniforms(material: VDBFrameInjectMaterial, frameVolume: VDBFrameVolume, volume: FrameVolumeRecord): void {
  const u = material.volumeUniforms;
  const s = material.sourceUniforms;

  u.frameViewToLocal.value.copy(volume.viewToLocal);
  u.frameLocalToView.value.copy(volume.localToView);
  u.frameWorldToLocal.value.copy(volume.worldToLocal);
  s.worldScale.value = volume.matrixWorld.getMaxScaleOnAxis();
  s.baseColor.value.copy(frameVolume.baseColor);
  u.frameScatterColor.value.copy(frameVolume.scatterColor);
  s.extinction.value = frameVolume.extinction;
  getFrameExtinctionSpectrum(frameVolume, u.frameExtinctionSpectrum.value);
  s.densityScale.value = frameVolume.getDensityScale();
  s.densityCutoff.value = frameVolume.densityCutoff;
  u.frameAbsorbance.value = frameVolume.absorbance;
  u.frameOpacity.value = frameVolume.opacity;
  s.emissiveIntensity.value = frameVolume.emissiveIntensity;
  s.temperatureRange.value.copy(frameVolume.temperatureRange);
  s.offset3D.value.copy(frameVolume.offset3D);
  s.wrap3D.value = frameVolume.wrap3D;
  getFramePhase(frameVolume, u.framePhase.value);
  getFramePhaseSpectrum(frameVolume, u.framePhaseSpectrum.value);
}

/** The shared renderer of every `VDBFrameVolume` made without one (his `frameRenderer` singleton). */
let shared: VDBFrameRenderer | null = null;

export function vdbFrameRenderer(): VDBFrameRenderer {
  shared ??= new VDBFrameRenderer();
  return shared;
}
