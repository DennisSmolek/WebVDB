// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameVolumeShadows.js L34–1027 (targets, the light shadow reset, receiver
//   uniforms, cascade lookup, FrameVolumeShadowNode, FrameVolumeShadows: entries, install, cameras, cascades,
//   the budgeted map updates, dispose).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; every three r187 internal goes through `three-internals.ts`, which self-tests it and throws
// VDB_ENGINE_MISMATCH (his: a warning, then no volume shadow); the receiver depth and cascade maths have CPU twins
// (`@pmndrs/vdb/tools` `vdbFrameShadow*`). Divergences: DIVERGES.md (K3.6 rows).

/**
 * K3.6: shadows of `VDBFrameVolume`s on meshes (`frameRenderer.settings.castShadows`, his `FrameVolumeShadows`).
 * Each shadow-casting directional, spot or point light gets a Beer shadow map made from the world shadow grid's
 * last cascade ({@link VDBFrameVolumeShadowMaterial}); the light's `shadow.filterNode` multiplies three's shadow
 * by the volume transmittance at the receiver. Directional lights: cascades of the view (three's CSMShadowNode
 * split), tiles of one atlas. Spot: the light's shadow camera. Point: a cube map. A map piece is drawn again only
 * when its camera or the grid changes, at most `castShadowUpdatesPerFrame` pieces per render (oldest first).
 *
 * three: receivers need `receiveShadow`, the light `castShadow`, and `renderer.shadowMap.enabled`. The maps are
 * drawn before the first receiver of each render (the node's `updateBefore`), so the grid is ready before the
 * volumes draw.
 */

import * as THREE from "three";
import { CubeRenderTarget, Node } from "three/webgpu";
import * as TSL from "three/tsl";
import type { WebGPURenderer } from "three/webgpu";
import type { VDBTslNode } from "../grid-node.js";
import { frameMaxShadowMapCascades, frameShadowCascadeMargin, frameShadowCoarseMapSize, frameShadowCoarseSteps, frameShadowMinOpticalDepth, vdbFrameShadowCascadeBreaks, vdbFrameShadowCascadeFar } from "../../tools/frame/shadows.js";
import { VDBFrameVolumeShadowMaterial, createShadowRayUniforms, frameMaxCastShadowSteps, frameShadowDepth, type ShadowRayUniforms } from "./volume-shadow-material.js";
import { vdbFrameCheckFilterInputs, vdbFrameCheckLightReset, vdbFrameRenderUpdateType, vdbFrameShadowFilters, vdbFrameSharedLightNode } from "./three-internals.js";
import type { VDBFrameRenderer } from "./renderer.js";

const { Fn, If, clamp, cubeTexture, dot, exp, float, int, max, positionView, positionWorld, renderGroup, select, texture, uniform, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;
type RendererLike = WebGPURenderer & { isWebGPURenderer?: boolean; coordinateSystem: number };

export { frameMaxShadowMapCascades };

const clearColor = new THREE.Color();
const lightDirection = new THREE.Vector3();
const lightUp = new THREE.Vector3();
const point = new THREE.Vector3();
const lightView = new THREE.Matrix4();
const lightViewInverse = new THREE.Matrix4();
const casterBox = new THREE.Box3();
const casterWorldBox = new THREE.Box3();
const sliceBox = new THREE.Box3();
const tileViewport = new THREE.Vector4();
const nearCorners = [0, 1, 2, 3].map(() => new THREE.Vector3());
const farCorners = [0, 1, 2, 3].map(() => new THREE.Vector3());

// NOTE Values for change checks. One array, reused for each check.
const signature: number[] = [];

// NOTE Fullscreen triangle. Positions are clip coordinates.
const createFullscreenGeometry = () => new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));

const targetOptions: THREE.RenderTargetOptions = {
  type: THREE.HalfFloatType,
  format: THREE.RGBAFormat,
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  generateMipmaps: false,
  depthBuffer: false,
};

type ShadowLight = THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight;

// NOTE Directional lights: an atlas with one tile for each cascade, in a row
const createTarget = (size: number, light: ShadowLight): THREE.RenderTarget => {
  if ((light as THREE.PointLight).isPointLight) {
    return new CubeRenderTarget(size, targetOptions) as unknown as THREE.RenderTarget;
  }

  const tiles = (light as THREE.DirectionalLight).isDirectionalLight ? frameMaxShadowMapCascades : 1;

  return new THREE.RenderTarget(size * tiles, size, targetOptions);
};

// NOTE Clears a new map one time. Then three makes its GPU texture as a render target texture now.
// Else the first render makes a new GPU texture, and receivers that sampled the map before keep the destroyed one.
const initTarget = (renderer: RendererLike, target: THREE.RenderTarget, cube: boolean): THREE.RenderTarget => {
  const currentRenderTarget = renderer.getRenderTarget();
  const currentActiveCubeFace = renderer.getActiveCubeFace();
  const currentActiveMipmapLevel = renderer.getActiveMipmapLevel();
  const currentClearAlpha = renderer.getClearAlpha();

  renderer.getClearColor(clearColor);
  renderer.setClearColor(0x000000, 0.0);

  for (let face = 0; face < (cube ? 6 : 1); face++) {
    renderer.setRenderTarget(target, face);
    renderer.clear(true, false, false);
  }

  renderer.setClearColor(clearColor, currentClearAlpha);
  renderer.setRenderTarget(currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel);

  return target;
};

const isShadowLight = (light: THREE.Object3D): light is ShadowLight => {
  const l = light as THREE.Light & { isDirectionalLight?: boolean; isSpotLight?: boolean; isPointLight?: boolean };

  return !!(l.isDirectionalLight || l.isSpotLight || l.isPointLight) && l.castShadow;
};

// NOTE Corners of the view between two depths, in world space. Corners 0 and 2 are diagonal.
const getSliceCorners = (camera: THREE.Camera, near: number, far: number): void => {
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ] as const;
  const perspective = (camera as THREE.PerspectiveCamera).isPerspectiveCamera;

  corners.forEach(([x, y], index) => {
    for (const [target, depth] of [
      [nearCorners[index]!, near],
      [farCorners[index]!, far],
    ] as const) {
      // NOTE A point on the corner ray, then moved to the depth (perspective) or set to it (orthographic)
      target.set(x, y, 0.5).applyMatrix4(camera.projectionMatrixInverse);

      if (perspective) {
        target.multiplyScalar(depth / -target.z);
      } else {
        target.z = -depth;
      }

      target.applyMatrix4(camera.matrixWorld);
    }
  });
};

// NOTE A light shadow node is built one time; a new filterNode has no effect until it is built again.
// disposeShadow of the shared light node makes the receivers build a new shadow node (three-internals.ts, internal 1).
// Returns the old shadow node, or null when there is none. Receivers can use it until the end of this render.
const resetLightShadow = (light: ShadowLight): { dispose(): void } | null => {
  vdbFrameCheckLightReset();

  const lightNode = vdbFrameSharedLightNode(light) as { shadowNode: { dispose(): void } | null; disposeShadow(): void } | null;

  if (!lightNode || !lightNode.shadowNode) {
    return null;
  }

  const { shadowNode } = lightNode;

  // NOTE disposeShadow without the GPU dispose
  lightNode.shadowNode = { dispose() {} };
  lightNode.disposeShadow();

  return shadowNode;
};

// NOTE A uniform that copies a value of the entry one time in each render (the render group).
// A light shadow node is built one time; its object-group uniforms would keep the value of the first build.
const renderUniform = (value: unknown, type: string, read: () => unknown): N => uniform(value, type).setGroup(renderGroup).onRenderUpdate(read);

interface Cascade {
  camera: THREE.OrthographicCamera;
  viewProjection: THREE.Matrix4;
  origin: THREE.Vector3;
  far: number;
  signature: number[];
  lastDraw: number;
  drawn: { viewProjection: THREE.Matrix4; origin: THREE.Vector3; far: number };
}

interface Receiver {
  enabled: N;
  rayUniforms: ShadowRayUniforms;
  cascades: { count: N; breaks: N; items: { viewProjection: N; origin: N; far: N }[] } | null;
}

/** One shadow light's maps and state. */
export interface VDBFrameShadowEntry {
  light: ShadowLight;
  isPoint: boolean;
  target: THREE.RenderTarget;
  size: number;
  spectral: boolean;
  spectralTarget: THREE.RenderTarget | null;
  spectralMap: N | null;
  renderId: number;
  cubeCamera: THREE.CubeCamera | null;
  map: N;
  enabled: N;
  rayUniforms: ShadowRayUniforms;
  drawn: { origin: THREE.Vector3; forward: THREE.Vector3; perspective: number; far: number; cascadeCount: number; cascadeBreaks: THREE.Vector4 };
  item: { signature: number[]; lastDraw: number };
  cascadeCount: number;
  cascadeBreaks: THREE.Vector4;
  cascades: Cascade[] | null;
  receiver: Receiver;
  userFilter: ((inputs: unknown) => N) | null;
}

// NOTE Receivers read the drawn state: the values of the camera that drew each map or cascade tile.
// A map that waits for the frame budget is older, but its values match, so its shadow stays in place.
const createReceiverUniforms = (entry: VDBFrameShadowEntry): Receiver => {
  const { drawn, cascades } = entry;

  return {
    enabled: renderUniform(0, "int", () => entry.enabled.value),
    rayUniforms: {
      frameShadowOrigin: renderUniform(new THREE.Vector3(), "vec3", () => drawn.origin),
      frameShadowForward: renderUniform(new THREE.Vector3(), "vec3", () => drawn.forward),
      frameShadowPerspective: renderUniform(0, "int", () => drawn.perspective),
      frameShadowFar: renderUniform(1.0, "float", () => drawn.far),
    },
    cascades: cascades
      ? {
          count: renderUniform(0, "int", () => drawn.cascadeCount),
          breaks: renderUniform(new THREE.Vector4(), "vec4", () => drawn.cascadeBreaks),
          items: cascades.map((cascade) => ({
            viewProjection: renderUniform(new THREE.Matrix4(), "mat4", () => cascade.drawn.viewProjection),
            origin: renderUniform(new THREE.Vector3(), "vec3", () => cascade.drawn.origin),
            far: renderUniform(1.0, "float", () => cascade.drawn.far),
          })),
        }
      : null,
  };
};

// NOTE Picks the cascade of the receiver by its view depth, as three's CSMShadowNode (twin: vdbFrameShadowCascadeIndex / Uv).
// Returns the atlas uv, the ray uniforms of the cascade, and whether the receiver is in a cascade.
const getCascadeLookup = (receiver: Receiver, size: number): { uv: N; inside: N; rayUniforms: ShadowRayUniforms } => {
  const { count, breaks, items } = receiver.cascades!;
  const { frameShadowForward } = receiver.rayUniforms;
  const depth = positionView.z.negate();
  const breakList = [breaks.x, breaks.y, breaks.z, breaks.w];

  // NOTE Cascade index: the number of cascade far distances in front of the receiver
  let index: N = int(0);

  for (let i = 0; i < frameMaxShadowMapCascades - 1; i++) {
    index = index.add(select(depth.greaterThan(breakList[i]).and(int(i + 1).lessThan(count)), int(1), int(0)));
  }

  const pick = (key: "viewProjection" | "origin" | "far"): N => items.reduce<N>((result, item, i) => (i === 0 ? item[key] : select(index.equal(i), item[key], result)), null);

  const viewProjection = pick("viewProjection");
  const clip = viewProjection.mul(vec4(positionWorld, 1.0));
  const tileUv = vec2(clip.x.mul(0.5).add(0.5), float(0.5).sub(clip.y.mul(0.5)));
  const lastBreak = breakList.reduce<N>((result, value, i) => (i === 0 ? value : select(count.equal(i + 1), value, result)), null);

  // NOTE Half a texel in from the tile sides. Then the linear filter does not read the next tile.
  const inset = float(0.5 / size);
  const clampedUv = clamp(tileUv, vec2(inset), vec2(float(1.0).sub(inset)));

  return {
    uv: vec2(clampedUv.x.add(float(index)).div(frameMaxShadowMapCascades), clampedUv.y),
    inside: depth.lessThanEqual(lastBreak).and(tileUv.x.greaterThanEqual(0.0)).and(tileUv.x.lessThanEqual(1.0)).and(tileUv.y.greaterThanEqual(0.0)).and(tileUv.y.lessThanEqual(1.0)),
    rayUniforms: {
      frameShadowOrigin: pick("origin"),
      frameShadowForward,
      frameShadowPerspective: int(0),
      frameShadowFar: pick("far"),
    },
  };
};

/**
 * Volume transmittance at a receiver from the Beer shadow map of the light (part of the light's shadow filter).
 * `uv` is the shadow map uv (spot) or the direction from the light (point); directional lights pick a cascade.
 * `updateBefore` draws the maps before the first receiver of each render. Twin: `vdbFrameShadowReceiver`.
 */
export class VDBFrameVolumeShadowNode extends Node {
  static get type(): string {
    return "VDBFrameVolumeShadowNode";
  }

  readonly shadows: VDBFrameVolumeShadows;
  readonly entry: VDBFrameShadowEntry;
  readonly uv: N;

  constructor(shadows: VDBFrameVolumeShadows, entry: VDBFrameShadowEntry, uv: N) {
    super("float");

    this.shadows = shadows;
    this.entry = entry;
    this.uv = uv;
    (this as unknown as { updateBeforeType: string }).updateBeforeType = vdbFrameRenderUpdateType();
  }

  // NOTE Spectral maps: transmittance of each channel. ShadowMaterial has one shadow mask, so it gets the mean.
  isSpectral(builder: { material?: { isShadowNodeMaterial?: boolean } }): boolean {
    return this.entry.spectralMap !== null && builder.material?.isShadowNodeMaterial !== true;
  }

  override getNodeType(builder: never): string {
    return this.isSpectral(builder) ? "vec3" : "float";
  }

  override updateBefore(frame: never): boolean | undefined {
    const { renderer, scene, camera, renderId } = frame as { renderer: RendererLike; scene: THREE.Scene; camera: THREE.Camera; renderId: number };

    this.shadows.render(renderer, scene, camera, renderId, this.entry);

    return undefined;
  }

  override setup(builder: never): N {
    const { entry } = this;
    const { map, spectralMap, isPoint, receiver } = entry;
    const spectral = this.isSpectral(builder);
    let { uv } = this;
    let { rayUniforms } = receiver;
    let inside: N = receiver.enabled.equal(1);

    if (receiver.cascades) {
      const lookup = getCascadeLookup(receiver, entry.size);

      uv = lookup.uv;
      rayUniforms = lookup.rayUniforms;
      inside = inside.and(lookup.inside);
    } else if (!isPoint) {
      inside = inside.and(uv.x.greaterThanEqual(0.0)).and(uv.x.lessThanEqual(1.0)).and(uv.y.greaterThanEqual(0.0)).and(uv.y.lessThanEqual(1.0));
    }

    return Fn(() => {
      const result = (spectral ? vec3(1.0) : float(1.0)).toVar();

      If(inside, () => {
        const texel = map.sample(uv).level(0).toVar();
        const opticalDepth = texel.w;

        If(opticalDepth.greaterThan(frameShadowMinOpticalDepth), () => {
          const far = rayUniforms.frameShadowFar;
          const first = texel.x.div(opticalDepth).mul(far);
          const last = texel.y.div(opticalDepth).mul(far);
          const depth = frameShadowDepth(rayUniforms, positionWorld);

          // NOTE Beer shadow map: optical depth grows linearly from the first to the last density
          const along = clamp(depth.sub(first).div(max(last.sub(first), 1e-4)), 0.0, 1.0);

          if (spectral) {
            result.assign(exp(spectralMap.sample(uv).level(0).rgb.mul(along).negate()));
          } else if (spectralMap) {
            // NOTE ShadowMaterial with spectral maps: mean transmittance of the channels
            result.assign(dot(exp(spectralMap.sample(uv).level(0).rgb.mul(along).negate()), vec3(1.0 / 3.0)));
          } else {
            result.assign(exp(opticalDepth.mul(along).negate()));
          }
        });
      });

      return result;
    })();
  }
}

// NOTE An object and all its parents are visible
const isVisible = (object: THREE.Object3D): boolean => {
  for (let current: THREE.Object3D | null = object; current; current = current.parent) {
    if (!current.visible) {
      return false;
    }
  }

  return true;
};

// NOTE The object is in a scene. A removed light is not.
const isInScene = (object: THREE.Object3D): boolean => {
  let root = object;

  while (root.parent) {
    root = root.parent;
  }

  return (root as THREE.Scene).isScene === true;
};

// NOTE Compares the signature with the one of the item. Stores it when it is different.
const changed = (item: { signature: number[] }): boolean => {
  const same = signature.length === item.signature.length && signature.every((value, i) => value === item.signature[i]);

  if (!same) {
    item.signature.length = 0;
    item.signature.push(...signature);
  }

  return !same;
};

interface WorkPiece {
  item: { signature: number[]; lastDraw: number };
  cost: number;
  draw(): void;
}

/** Shadow filter marker (his `isFrameVolumeShadowFilter`). */
type VolumeFilter = ((inputs: N) => N) & { isVDBFrameVolumeShadowFilter?: true };

/** Shadows of `VDBFrameVolume`s on meshes, owned by a {@link VDBFrameRenderer} (`frameRenderer.volumeShadows`). */
export class VDBFrameVolumeShadows {
  readonly entries = new Map<THREE.Light, VDBFrameShadowEntry>();
  readonly scene = new THREE.Scene();
  readonly rayUniforms = createShadowRayUniforms() as unknown as ShadowRayUniforms;
  readonly material: VDBFrameVolumeShadowMaterial;
  readonly mesh: THREE.Mesh;
  readonly frameRenderer: VDBFrameRenderer;
  // NOTE Replaced and freed maps and shadow nodes. update() runs mid-render, so the next update() disposes them.
  retired: { dispose(): void }[] = [];
  renderCount = 0;
  mapsRenderId = -1;
  mapsCamera: THREE.Camera | null = null;
  /** Map pieces drawn in the last render, and how many waited for the budget (not his; tests and timing). */
  readonly lastUpdate = { drawn: 0, waiting: 0 };

  constructor(frameRenderer: VDBFrameRenderer) {
    this.frameRenderer = frameRenderer;
    this.material = new VDBFrameVolumeShadowMaterial(frameRenderer, this.rayUniforms);
    this.mesh = new THREE.Mesh(createFullscreenGeometry(), this.material);
    this.mesh.frustumCulled = false;
    this.scene.matrixWorldAutoUpdate = false;
    this.scene.add(this.mesh);
  }

  getMapSize(): number {
    const { settings } = this.frameRenderer;

    return settings.castShadowCoarse ? frameShadowCoarseMapSize : Math.max(1, Math.round(settings.castShadowMapSize));
  }

  isSpectral(): boolean {
    return this.frameRenderer.settings.spectral === true;
  }

  getSteps(): number {
    const { settings } = this.frameRenderer;

    return settings.castShadowCoarse ? frameShadowCoarseSteps : THREE.MathUtils.clamp(Math.round(settings.castShadowSteps), 1, frameMaxCastShadowSteps);
  }

  getEntry(renderer: RendererLike, light: ShadowLight): VDBFrameShadowEntry {
    let entry = this.entries.get(light);

    if (!entry) {
      const isPoint = (light as THREE.PointLight).isPointLight === true;
      const size = this.getMapSize();
      const spectral = this.isSpectral();
      const target = initTarget(renderer, createTarget(size, light), isPoint);
      const spectralTarget = spectral ? initTarget(renderer, createTarget(size, light), isPoint) : null;

      entry = {
        light,
        isPoint,
        target,
        size,
        // NOTE settings.spectral: a second map with the optical depth of each channel
        spectral,
        spectralTarget,
        spectralMap: spectralTarget ? (isPoint ? cubeTexture(spectralTarget.texture) : texture(spectralTarget.texture)) : null,
        renderId: -1,
        cubeCamera: isPoint ? new THREE.CubeCamera(0.5, 500.0, target as unknown as THREE.WebGLCubeRenderTarget) : null,
        map: isPoint ? cubeTexture(target.texture) : texture(target.texture),
        enabled: uniform(0, "int"),
        // NOTE Current camera values. drawn: the values of the last draw, for the receivers.
        rayUniforms: createShadowRayUniforms() as unknown as ShadowRayUniforms,
        drawn: {
          origin: new THREE.Vector3(),
          forward: new THREE.Vector3(0.0, 0.0, -1.0),
          perspective: 0,
          far: 1.0,
          cascadeCount: 0,
          cascadeBreaks: new THREE.Vector4(),
        },
        // NOTE Spot map or cube map (all faces in one draw), or the split layout of the cascades
        item: { signature: [], lastDraw: -1 },
        cascadeCount: 0,
        cascadeBreaks: new THREE.Vector4(),
        cascades: (light as THREE.DirectionalLight).isDirectionalLight
          ? Array.from({ length: frameMaxShadowMapCascades }, () => ({
              camera: new THREE.OrthographicCamera(),
              viewProjection: new THREE.Matrix4(),
              origin: new THREE.Vector3(),
              far: 1.0,
              signature: [],
              lastDraw: -1,
              drawn: { viewProjection: new THREE.Matrix4(), origin: new THREE.Vector3(), far: 1.0 },
            }))
          : null,
        receiver: null as unknown as Receiver,
        userFilter: null,
      };

      entry.receiver = createReceiverUniforms(entry);
      this.entries.set(light, entry);
    }

    return entry;
  }

  // NOTE A new map for a new size or a spectral change: a new texture, texture node and shadow node (shadow nodes keep texture bindings)
  resizeMap(renderer: RendererLike, entry: VDBFrameShadowEntry, size: number, spectral: boolean): void {
    const oldTargets = entry.spectralTarget ? [entry.target, entry.spectralTarget] : [entry.target];

    entry.target = initTarget(renderer, createTarget(size, entry.light), entry.isPoint);
    entry.size = size;
    entry.map = entry.isPoint ? cubeTexture(entry.target.texture) : texture(entry.target.texture);
    entry.spectral = spectral;
    entry.spectralTarget = spectral ? initTarget(renderer, createTarget(size, entry.light), entry.isPoint) : null;
    entry.spectralMap = entry.spectralTarget ? (entry.isPoint ? cubeTexture(entry.spectralTarget.texture) : texture(entry.spectralTarget.texture)) : null;

    entry.item.signature.length = 0;
    entry.cascades?.forEach((cascade) => (cascade.signature.length = 0));
    entry.drawn.cascadeCount = 0;

    if (entry.cubeCamera) {
      entry.cubeCamera.renderTarget = entry.target as unknown as THREE.WebGLCubeRenderTarget;
    }

    this.retire(resetLightShadow(entry.light), oldTargets);
  }

  // NOTE Old maps are disposed in the next update()
  retire(oldShadowNode: { dispose(): void } | null, targets: THREE.RenderTarget[]): void {
    this.retired.push(...targets);

    if (oldShadowNode) {
      this.retired.push(oldShadowNode);
    }
  }

  // NOTE Frees the maps of a light and puts back its own filter. Receivers build again without the volume shadow.
  freeEntry(entry: VDBFrameShadowEntry): void {
    const { light } = entry;
    const targets = entry.spectralTarget ? [entry.target, entry.spectralTarget] : [entry.target];
    const shadow = light.shadow as THREE.LightShadow & { filterNode?: VolumeFilter | null };

    // NOTE Receivers that still have the volume shadow node until they build again read no volume shadow
    entry.enabled.value = 0;

    if (shadow.filterNode?.isVDBFrameVolumeShadowFilter) {
      shadow.filterNode = entry.userFilter as VolumeFilter | null;
    }

    this.retire(resetLightShadow(light), targets);
    this.entries.delete(light);
  }

  disposeRetired(): void {
    this.retired.forEach((resource) => resource.dispose());
    this.retired.length = 0;
  }

  // NOTE Puts the volume filter on each shadow light. Each frame, then new lights and new filters are found.
  // Frees the entries of lights with no shadow or not in a scene, and all entries when castShadows is off.
  update(renderer: RendererLike, scene: THREE.Object3D): void {
    this.disposeRetired();

    if (!this.frameRenderer.settings.castShadows) {
      this.entries.forEach((entry) => this.freeEntry(entry));

      return;
    }

    // NOTE The light-shadow reset (three internal) is self-tested before the first filter goes on a light
    vdbFrameCheckLightReset();

    const size = this.getMapSize();
    const spectral = this.isSpectral();

    this.entries.forEach((entry) => {
      if (!isShadowLight(entry.light) || !isInScene(entry.light)) {
        this.freeEntry(entry);
      } else if (entry.size !== size || entry.spectral !== spectral) {
        this.resizeMap(renderer, entry, size, spectral);
      }
    });

    scene.traverse((light) => {
      if (isShadowLight(light)) {
        this.install(renderer, light);
      }
    });
  }

  install(renderer: RendererLike, light: ShadowLight): void {
    const shadow = light.shadow as THREE.LightShadow & { filterNode?: VolumeFilter | null; map: THREE.RenderTarget | null };

    if (shadow.filterNode?.isVDBFrameVolumeShadowFilter) {
      return;
    }

    const entry = this.getEntry(renderer, light);
    const userFilter = (shadow.filterNode as VolumeFilter | null | undefined) ?? null;
    const filters = vdbFrameShadowFilters();

    entry.userFilter = userFilter;

    // NOTE ShadowNode calls the filter when it builds. The base filter is three's or the user's.
    // Point lights give the direction from the light (bd3D), other lights give shadowCoord.
    const filter: VolumeFilter = (inputs) => {
      vdbFrameCheckFilterInputs(inputs, entry.isPoint);

      const type = renderer.shadowMap.type as number;
      const baseFilter = (userFilter ?? (entry.isPoint ? (type === THREE.BasicShadowMap ? filters.basicPoint : filters.point) : filters.maps[type])) as ((i: unknown) => N) | null;

      if (!baseFilter) {
        throw new Error("THREE.WebGPURenderer: Shadow map type not supported yet.");
      }

      // NOTE PointShadowNode calls the filter in a TSL function, and its parameter proxy loses the shadow object.
      // So the base filter gets a plain object with the light shadow.
      const baseInputs: Record<string, unknown> = entry.isPoint
        ? { depthTexture: inputs.depthTexture, bd3D: inputs.bd3D, dp: inputs.dp, distance: inputs.distance }
        : { depthTexture: inputs.depthTexture, shadowCoord: inputs.shadowCoord, depthLayer: inputs.depthLayer };

      baseInputs.shadow = shadow;

      const uv = entry.isPoint ? inputs.bd3D : (inputs.shadowCoord as N).xy;

      return baseFilter(baseInputs).mul(new VDBFrameVolumeShadowNode(this, entry, uv));
    };

    filter.isVDBFrameVolumeShadowFilter = true;
    shadow.filterNode = filter;

    // NOTE The map exists when the shadow node was built before this filter
    if (shadow.map) {
      this.retire(resetLightShadow(light), []);
    }
  }

  // NOTE Spot lights: the light shadow camera. Point lights: a cube camera at the light. Sets the ray uniforms.
  prepareCamera(entry: VDBFrameShadowEntry): THREE.Camera {
    const { light, rayUniforms } = entry;
    const shadowCamera = light.shadow.camera as THREE.PerspectiveCamera;

    if (!entry.isPoint) {
      light.shadow.updateMatrices(light);

      rayUniforms.frameShadowOrigin.value.setFromMatrixPosition(shadowCamera.matrixWorld);
      shadowCamera.getWorldDirection(rayUniforms.frameShadowForward.value);
      rayUniforms.frameShadowPerspective.value = shadowCamera.isPerspectiveCamera ? 1 : 0;
      rayUniforms.frameShadowFar.value = shadowCamera.far;

      return shadowCamera;
    }

    const cubeCamera = entry.cubeCamera!;
    const near = shadowCamera.near;
    const far = (light as THREE.PointLight).distance || shadowCamera.far;

    for (const camera of cubeCamera.children as THREE.PerspectiveCamera[]) {
      if (camera.near !== near || camera.far !== far) {
        camera.near = near;
        camera.far = far;
        camera.updateProjectionMatrix();
      }
    }

    cubeCamera.position.setFromMatrixPosition(light.matrixWorld);
    cubeCamera.updateMatrixWorld();

    rayUniforms.frameShadowOrigin.value.copy(cubeCamera.position);
    rayUniforms.frameShadowPerspective.value = 1;
    rayUniforms.frameShadowFar.value = far;

    return cubeCamera.children[0] as THREE.Camera;
  }

  // NOTE Cascade cameras of a directional light, the method of three's CSMShadowNode: practical split of the view
  // depth, a light-aligned orthographic camera for each slice, size from the slice diagonal, position snapped to
  // texels. The depth range of each camera covers the slice and the volume box (the last grid cascade). Cascades end
  // at the far corner of the volume box plus one box diagonal, up to castShadowDistance, rounded to half octaves.
  fitCascades(entry: VDBFrameShadowEntry, camera: THREE.Camera, renderer: RendererLike): number {
    const { light, size, rayUniforms } = entry;
    const cascades = entry.cascades!;
    const { settings, gridUniforms } = this.frameRenderer;
    const count = THREE.MathUtils.clamp(Math.round(settings.castShadowCascades), 1, frameMaxShadowMapCascades);
    const last = (gridUniforms.frameGridCount.value as number) - 1;
    const cam = camera as THREE.PerspectiveCamera;
    const near = cam.near;
    const gridMin = gridUniforms.frameGridMin.array as THREE.Vector3[];
    const gridSize = gridUniforms.frameGridSize.array as THREE.Vector3[];

    casterWorldBox.min.copy(gridMin[last]!);
    casterWorldBox.max.copy(gridMin[last]!).add(gridSize[last]!);

    let boxFar = 0.0;

    for (let i = 0; i < 8; i++) {
      point
        .set(i & 1 ? casterWorldBox.max.x : casterWorldBox.min.x, i & 2 ? casterWorldBox.max.y : casterWorldBox.min.y, i & 4 ? casterWorldBox.max.z : casterWorldBox.min.z)
        .applyMatrix4(camera.matrixWorldInverse);

      boxFar = Math.max(boxFar, -point.z);
    }

    const far = vdbFrameShadowCascadeFar(near, cam.far, settings.castShadowDistance, boxFar, casterWorldBox.getSize(point).length());
    const breaks = vdbFrameShadowCascadeBreaks(count, near, far);

    lightDirection.setFromMatrixPosition((light as THREE.DirectionalLight).target.matrixWorld).sub(point.setFromMatrixPosition(light.matrixWorld)).normalize();

    // NOTE Rotation of a camera that looks along the light (Object3D.lookAt of a camera). Up is +z for vertical light.
    lightUp.set(0.0, 1.0, 0.0);

    if (Math.abs(lightDirection.y) > 0.999) {
      lightUp.set(0.0, 0.0, 1.0);
    }

    lightView.lookAt(point.set(0.0, 0.0, 0.0), lightDirection, lightUp);
    lightViewInverse.copy(lightView).invert();

    casterBox.copy(casterWorldBox).applyMatrix4(lightViewInverse);

    entry.cascadeCount = count;
    entry.cascadeBreaks.set(breaks[Math.min(0, count - 1)]!, breaks[Math.min(1, count - 1)]!, breaks[Math.min(2, count - 1)]!, breaks[Math.min(3, count - 1)]!);

    for (let i = 0; i < count; i++) {
      const cascade = cascades[i]!;
      const cascadeCamera = cascade.camera;

      getSliceCorners(camera, i === 0 ? near : breaks[i - 1]!, breaks[i]!);

      const width = Math.max(farCorners[0]!.distanceTo(farCorners[2]!), farCorners[0]!.distanceTo(nearCorners[2]!));
      const texel = width / size;

      sliceBox.makeEmpty();
      for (const corner of [...nearCorners, ...farCorners]) {
        sliceBox.expandByPoint(point.copy(corner).applyMatrix4(lightViewInverse));
      }

      const centerX = Math.floor(((sliceBox.min.x + sliceBox.max.x) * 0.5) / texel) * texel;
      const centerY = Math.floor(((sliceBox.min.y + sliceBox.max.y) * 0.5) / texel) * texel;
      const front = Math.max(sliceBox.max.z, casterBox.max.z) + frameShadowCascadeMargin;
      const back = Math.min(sliceBox.min.z, casterBox.min.z) - frameShadowCascadeMargin;

      cascadeCamera.left = -width * 0.5;
      cascadeCamera.right = width * 0.5;
      cascadeCamera.top = width * 0.5;
      cascadeCamera.bottom = -width * 0.5;
      cascadeCamera.near = 0.0;
      cascadeCamera.far = front - back;
      cascadeCamera.coordinateSystem = renderer.coordinateSystem as THREE.CoordinateSystem;
      cascadeCamera.updateProjectionMatrix();

      // NOTE Light view looks along -z, so the camera is at the front (largest z)
      cascadeCamera.position.set(centerX, centerY, front).applyMatrix4(lightView);
      cascadeCamera.quaternion.setFromRotationMatrix(lightView);
      cascadeCamera.updateMatrixWorld();

      cascade.viewProjection.multiplyMatrices(cascadeCamera.projectionMatrix, cascadeCamera.matrixWorldInverse);
      cascade.origin.copy(cascadeCamera.position);
      cascade.far = cascadeCamera.far;
    }

    cascades[0]!.camera.getWorldDirection(rayUniforms.frameShadowForward.value);
    rayUniforms.frameShadowPerspective.value = 0;

    return count;
  }

  // NOTE Copies ray values to the uniforms of the pass material
  setPassRay(origin: THREE.Vector3, forward: THREE.Vector3, perspective: number, far: number): void {
    this.rayUniforms.frameShadowOrigin.value.copy(origin);
    this.rayUniforms.frameShadowForward.value.copy(forward);
    this.rayUniforms.frameShadowPerspective.value = perspective;
    this.rayUniforms.frameShadowFar.value = far;
  }

  // NOTE Draws the pass into the map of the entry: one tile of it (viewport), or all cube faces (point lights)
  drawPass(renderer: RendererLike, entry: VDBFrameShadowEntry, camera: THREE.Camera, viewport: THREE.Vector4): void {
    const currentRenderTarget = renderer.getRenderTarget();
    const currentActiveCubeFace = renderer.getActiveCubeFace();
    const currentActiveMipmapLevel = renderer.getActiveMipmapLevel();
    const currentAutoClear = renderer.autoClear;
    const currentClearAlpha = renderer.getClearAlpha();
    const currentScissorTest = renderer.getScissorTest();

    renderer.getClearColor(clearColor);
    renderer.autoClear = false;
    renderer.setScissorTest(false);
    this.mesh.layers.mask = camera.layers.mask;

    // NOTE Spectral: the second draw writes the optical depth of each channel into the second map
    const targets = entry.spectralTarget ? [entry.target, entry.spectralTarget] : [entry.target];

    targets.forEach((target, output) => {
      this.material.frameShadowOutput.value = output;

      if (entry.isPoint) {
        const cubeCamera = entry.cubeCamera!;

        cubeCamera.renderTarget = target as unknown as THREE.WebGLCubeRenderTarget;
        cubeCamera.update(renderer as never, this.scene);
      } else {
        target.viewport.copy(viewport);
        renderer.setRenderTarget(target);
        renderer.render(this.scene, camera);
      }
    });

    if (entry.isPoint) {
      entry.cubeCamera!.renderTarget = entry.target as unknown as THREE.WebGLCubeRenderTarget;
    }

    renderer.setScissorTest(currentScissorTest);
    renderer.autoClear = currentAutoClear;
    renderer.setClearColor(clearColor, currentClearAlpha);
    renderer.setRenderTarget(currentRenderTarget, currentActiveCubeFace, currentActiveMipmapLevel);
  }

  // NOTE Adds the changed map pieces of one light to the work list
  collectWork(entry: VDBFrameShadowEntry, camera: THREE.Camera, renderer: RendererLike, work: WorkPiece[]): void {
    const castVersion = this.frameRenderer.gridState.castVersion;
    const steps = this.getSteps();
    const { drawn, rayUniforms } = entry;

    if (entry.cascades) {
      const cascades = entry.cascades;
      const count = this.fitCascades(entry, camera, renderer);
      const forward = rayUniforms.frameShadowForward.value as THREE.Vector3;

      const drawCascade = (i: number) => {
        const cascade = cascades[i]!;

        this.setPassRay(cascade.origin, forward, 0, cascade.far);
        this.drawPass(renderer, entry, cascade.camera, tileViewport.set(i * entry.size, 0, entry.size, entry.size));
        cascade.drawn.viewProjection.copy(cascade.viewProjection);
        cascade.drawn.origin.copy(cascade.origin);
        cascade.drawn.far = cascade.far;
        cascade.lastDraw = this.renderCount;
        drawn.forward.copy(forward);
      };

      const cascadeChanged = (i: number) => {
        const cascade = cascades[i]!;

        signature.length = 0;
        signature.push(entry.size, castVersion, steps, ...cascade.camera.matrixWorld.elements);
        signature.push(...cascade.camera.projectionMatrix.elements);

        return changed(cascade);
      };

      // NOTE A new split layout: one piece draws all cascades, then the receivers switch to the layout.
      // Receivers never mix tiles of two layouts.
      signature.length = 0;
      signature.push(entry.size, count, ...entry.cascadeBreaks.toArray());

      if (changed(entry.item)) {
        for (let i = 0; i < count; i++) {
          cascadeChanged(i);
        }

        work.push({
          item: entry.item,
          cost: count,
          draw: () => {
            for (let i = 0; i < count; i++) {
              drawCascade(i);
            }

            drawn.cascadeCount = count;
            drawn.cascadeBreaks.copy(entry.cascadeBreaks);
          },
        });

        return;
      }

      for (let i = 0; i < count; i++) {
        if (cascadeChanged(i)) {
          work.push({ item: cascades[i]!, cost: 1, draw: () => drawCascade(i) });
        }
      }

      return;
    }

    const shadowCamera = this.prepareCamera(entry);

    signature.length = 0;
    signature.push(entry.size, castVersion, steps, ...(entry.isPoint ? entry.cubeCamera! : shadowCamera).matrixWorld.elements);
    signature.push(...(shadowCamera as THREE.PerspectiveCamera).projectionMatrix.elements);

    if (!changed(entry.item)) {
      return;
    }

    work.push({
      item: entry.item,
      // NOTE All six cube faces in one draw, then the faces always match
      cost: entry.isPoint ? 6 : 1,
      draw: () => {
        this.setPassRay(rayUniforms.frameShadowOrigin.value, rayUniforms.frameShadowForward.value, rayUniforms.frameShadowPerspective.value, rayUniforms.frameShadowFar.value);
        this.drawPass(renderer, entry, shadowCamera, tileViewport.set(0, 0, entry.size, entry.size));
        drawn.origin.copy(rayUniforms.frameShadowOrigin.value);
        drawn.forward.copy(rayUniforms.frameShadowForward.value);
        drawn.perspective = rayUniforms.frameShadowPerspective.value;
        drawn.far = rayUniforms.frameShadowFar.value;
      },
    });
  }

  // NOTE Volume shadow maps of all lights, one time in each render, before the first receiver.
  // Changed pieces are drawn oldest first, up to castShadowUpdatesPerFrame (a tile or spot map costs 1, a cube map 6).
  // The oldest piece is always drawn. Then a large piece does not wait forever.
  updateMaps(renderer: RendererLike, scene: THREE.Scene, camera: THREE.Camera, renderId: number): void {
    if (this.mapsRenderId === renderId && this.mapsCamera === camera) {
      return;
    }

    this.mapsRenderId = renderId;
    this.mapsCamera = camera;
    this.renderCount++;
    this.lastUpdate.drawn = 0;
    this.lastUpdate.waiting = 0;

    const { frameRenderer } = this;
    const ready = frameRenderer.settings.castShadows && !!renderer.isWebGPURenderer && frameRenderer.prepareShadowGrid(renderer, scene, camera, renderId);

    if (!ready) {
      this.entries.forEach((entry) => {
        entry.enabled.value = 0;
        entry.item.signature.length = 0;
        entry.cascades?.forEach((cascade) => (cascade.signature.length = 0));
      });

      return;
    }

    this.material.setSpectral(this.isSpectral());
    this.material.frameShadowDensityMap.value = frameRenderer.shadowDensityTarget!.textures[0];
    this.material.frameShadowSteps.value = this.getSteps();

    const work: WorkPiece[] = [];

    this.entries.forEach((entry) => {
      const active = entry.light.castShadow && isVisible(entry.light);

      entry.enabled.value = active ? 1 : 0;

      if (active) {
        this.collectWork(entry, camera, renderer, work);
      }
    });

    work.sort((a, b) => a.item.lastDraw - b.item.lastDraw);

    let budget = Math.max(1, Math.round(frameRenderer.settings.castShadowUpdatesPerFrame));

    frameRenderer.passHook?.begin?.("volumeShadows");

    work.forEach((piece, index) => {
      if (index > 0 && piece.cost > budget) {
        // NOTE Not drawn: the next check sees the change again
        piece.item.signature.length = 0;
        this.lastUpdate.waiting++;

        return;
      }

      piece.draw();
      piece.item.lastDraw = this.renderCount;
      budget -= piece.cost;
      this.lastUpdate.drawn++;
    });

    frameRenderer.passHook?.end?.("volumeShadows");
  }

  // NOTE Called by the receivers of a light, one time in each render
  render(renderer: RendererLike, scene: THREE.Scene, camera: THREE.Camera, renderId: number, entry: VDBFrameShadowEntry): void {
    if (entry.renderId === renderId) {
      return;
    }

    entry.renderId = renderId;
    this.updateMaps(renderer, scene, camera, renderId);
  }

  // NOTE Puts back the filter of each light. Then a new VDBFrameVolumeShadows installs again.
  dispose(): void {
    if (this.entries.size > 0) {
      this.entries.forEach((entry) => this.freeEntry(entry));
    }

    this.disposeRetired();
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}
