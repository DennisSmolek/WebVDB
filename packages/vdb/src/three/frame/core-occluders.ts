/**
 * K6.5b: three's light shadow maps as the core renderer's occluder maps (mesh → volume shadows, design §3.7). For each
 * directional, spot and point light with `castShadow` (and `renderer.shadowMap.enabled`) whose map three has drawn, the
 * adapter passes the map's depth `GPUTexture` (a `'2d-array'` view: one layer, or the point light's 6 cube faces), the
 * view-projections it was drawn with, `shadow.bias`, the renderer's reversed depth and the filter (`BasicShadowMap`:
 * one texel, else 2 × 2 PCF).
 *
 * **No lag.** three draws a light's map inside its own render, when the first lit material of the frame draws (the
 * light's `ShadowNode.updateBefore`). Before the core update the adapter calls that `updateBefore` itself, with three's
 * frame id, so three draws (and submits) the map first and skips it later in the same frame; the core then reads this
 * frame's map with this frame's matrices. A map three does not redraw this frame (`autoUpdate` off) is read as it is,
 * with the matrices it was drawn with (they stay on the shadow camera), so map and matrices always agree.
 *
 * three r187 internals used, checked by a self-test per backend and map kind (`VDB_ENGINE_MISMATCH` otherwise):
 * 1. `light.shadow.map.depthTexture`'s GPU texture (`backend.get(t).texture`): a depth format with `TEXTURE_BINDING`,
 *    1 layer (directional, spot) or 6 (point, `CubeDepthTexture`);
 * 2. `light.shadow.matrix` is the WebGPU bias matrix × `shadow.camera` projection × view of the last draw
 *    (`LightShadow._updateMatrix`); a point light's is the translation to the light of the last draw
 *    (`PointShadowNode.renderShadow`), which leaves `shadow.camera` on face 5, and its faces follow three's WebGPU
 *    table (checked on face 5);
 * 3. the shared `AnalyticLightNode` of a light (`three-internals.ts` internal 1) has the `ShadowNode` in
 *    `shadowNode`, whose `updateBefore(frame)` draws the map once per frame and camera (`_cameraFrameId`), reading
 *    `frame.renderer / scene / camera / frameId`.
 * Not supported: logarithmic depth buffers (the maps store log depth; warned, skipped); `CSMShadowNode` (its cascades
 * are lights of their own; not mapped); `shadow.normalBias` (a volume has no normal).
 */

import * as THREE from "three";
import { REVISION } from "three";
import { VDBError } from "../../core/errors.js";
import type { VDBFrameOccluder } from "../../core/frame/types.js";
import { vdbFrameSharedLightNode } from "./three-internals.js";

interface BackendLike {
  get(object: object): Record<string, unknown>;
}
type RendererLike = THREE.EventDispatcher & {
  readonly backend: object;
  shadowMap: { enabled: boolean; type: THREE.ShadowMapType };
  reversedDepthBuffer?: boolean;
  logarithmicDepthBuffer?: boolean;
  _nodes?: { nodeFrame?: { frameId?: number; renderId?: number } };
};
type ShadowLight = (THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight) & { isDirectionalLight?: boolean; isSpotLight?: boolean; isPointLight?: boolean };
type ShadowNodeLike = { updateBefore?: (frame: object) => unknown; _cameraFrameId?: WeakMap<object, number> };

/** three r187 PointShadowNode's WebGPU cube faces: directions and ups (`_cubeDirectionsWebGPU`, `_cubeUpsWebGPU`). */
const CUBE_DIRECTIONS = [new THREE.Vector3(-1, 0, 0), new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1)];
const CUBE_UPS = [new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 1, 0)];
/** `LightShadow._updateMatrix`'s WebGPU bias (uv = ndc · ½ + ½, z kept). */
const WEBGPU_BIAS = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 1);
const TEXTURE_BINDING = 0x04;

const mismatch = (what: string): VDBError =>
  new VDBError(
    "VDB_ENGINE_MISMATCH",
    `three adapter (core occluder maps): three r${REVISION} ${what}. backend 'core' reads three's light shadow maps through three r187 ` +
      "internals (peer range >=0.187.0 <0.188.0); this three version is unsupported for mesh → volume shadows (set frameRenderer.occlusion = false).",
  );

const near = (a: ArrayLike<number>, b: ArrayLike<number>, eps: number) => {
  for (let i = 0; i < 16; i++) if (Math.abs(a[i]! - b[i]!) > eps * Math.max(1, Math.abs(b[i]!))) return false;
  return true;
};

/** Self-test results per backend and kind ('2d' | 'cube' | 'flush'). */
const selfTested = new WeakMap<object, Map<string, true | VDBError>>();

function check(backend: object, kind: string, test: () => string | null): void {
  let byKind = selfTested.get(backend);
  if (!byKind) selfTested.set(backend, (byKind = new Map()));
  const cached = byKind.get(kind);
  if (cached === true) return;
  if (cached) throw cached;
  let problem: string | null;
  try {
    problem = test();
  } catch (e) {
    problem = `failed the occluder self-test (${e instanceof Error ? e.message : String(e)})`;
  }
  const result = problem ? mismatch(problem) : true;
  byKind.set(kind, result);
  if (result !== true) throw result;
}

/** One light's state across updates: its revision (bumped when three drew the map) and the frame that bumped it. */
interface LightState {
  revision: number;
  frame: number;
}

/** The occluder inputs of a render (one instance per `VDBFrameRenderer`). */
export class VDBCoreOccluders {
  readonly #views = new WeakMap<GPUTexture, GPUTextureView>();
  readonly #state = new WeakMap<object, LightState>();
  readonly #nodes = new WeakMap<object, { shadowNode?: unknown } | null>();
  readonly #camera = new THREE.PerspectiveCamera();
  #warned = new Set<string>();

  /**
   * The occluder maps of `lights` (three lights in the camera's layers that the core lights; ids = `light.id`) for this
   * render: draws three's pending maps first (no lag), then reads each map that exists.
   */
  input(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera, lights: readonly THREE.Light[]): VDBFrameOccluder[] {
    if (!renderer.shadowMap?.enabled) return [];
    if (renderer.logarithmicDepthBuffer) {
      this.#warnOnce("log", "backend 'core': mesh → volume shadows do not read logarithmic-depth shadow maps; occlusion is off.");
      return [];
    }
    const backend = renderer.backend as BackendLike;
    const frameId = renderer._nodes?.nodeFrame?.frameId;
    const out: VDBFrameOccluder[] = [];
    for (const object of lights) {
      const light = object as ShadowLight;
      if (!light.castShadow || !(light.isDirectionalLight || light.isSpotLight || light.isPointLight)) continue;
      const shadow = light.shadow as THREE.LightShadow & { autoUpdate?: boolean; needsUpdate?: boolean };
      if (!shadow) continue;
      const state = this.#state.get(light) ?? { revision: 0, frame: -1 };
      this.#state.set(light, state);
      // three draws this frame's map now (if it will at all), so the core reads it in this render
      if (this.#flush(renderer, scene, camera, light, frameId)) {
        if (state.frame !== frameId) state.revision++;
        state.frame = frameId ?? -1;
      }
      const depthTexture = (shadow.map as (THREE.RenderTarget & { depthTexture?: THREE.Texture | null }) | null)?.depthTexture;
      if (!depthTexture) continue;
      const gpu = backend.get(depthTexture)?.texture as GPUTexture | undefined;
      if (!gpu) continue;
      const point = !!light.isPointLight;
      check(backend, point ? "cube" : "2d", () => {
        if (!String(gpu.format).startsWith("depth")) return `stores a light's shadow map as ${gpu.format}, not a depth format`;
        if (!(gpu.usage & TEXTURE_BINDING)) return "made a shadow map without TEXTURE_BINDING";
        if (gpu.depthOrArrayLayers !== (point ? 6 : 1)) return `made a ${point ? "point" : "directional / spot"} shadow map with ${gpu.depthOrArrayLayers} layers`;
        return point ? this.#checkCube(light as THREE.PointLight) : this.#check2d(shadow);
      });
      let view = this.#views.get(gpu);
      if (!view) this.#views.set(gpu, (view = gpu.createView({ dimension: "2d-array", aspect: "depth-only", label: `vdb occluder ${light.name || light.id}` })));
      out.push({
        light: light.id,
        depth: view,
        viewProjection: point ? this.#faces(light as THREE.PointLight) : [new THREE.Matrix4().multiplyMatrices(shadow.camera.projectionMatrix, shadow.camera.matrixWorldInverse).elements],
        bias: shadow.bias,
        reversedDepth: !!renderer.reversedDepthBuffer,
        filter: renderer.shadowMap.type === THREE.BasicShadowMap ? "nearest" : "bilinear",
        revision: state.revision,
      });
    }
    return out;
  }

  /**
   * Runs the light's `ShadowNode.updateBefore` with three's frame (internal 3): three draws the map now if it would draw
   * it this frame, and not again later. Returns whether the map was drawn this frame for this camera.
   */
  #flush(renderer: RendererLike, scene: THREE.Object3D, camera: THREE.Camera, light: ShadowLight, frameId: number | undefined): boolean {
    let lightNode = this.#nodes.get(light);
    if (lightNode === undefined) this.#nodes.set(light, (lightNode = vdbFrameSharedLightNode(light) as { shadowNode?: unknown } | null));
    const shadowNode = lightNode?.shadowNode as ShadowNodeLike | null | undefined;
    // No shadow node: no lit material uses the light yet, so three has no map (nothing to read)
    if (!shadowNode) return false;
    check(renderer.backend, "flush", () => {
      if (typeof frameId !== "number") return "has no renderer._nodes.nodeFrame.frameId";
      if (typeof shadowNode.updateBefore !== "function") return "has no ShadowNode.updateBefore";
      if (!(shadowNode._cameraFrameId instanceof WeakMap)) return "has no ShadowNode._cameraFrameId (its once-per-frame guard)";
      return null;
    });
    shadowNode.updateBefore!({ renderer, scene, camera, frameId, renderId: renderer._nodes?.nodeFrame?.renderId });
    return shadowNode._cameraFrameId!.get(camera) === frameId;
  }

  /** Internal 2 for a directional or spot map: shadow.matrix = bias × the camera's projection × view. */
  #check2d(shadow: THREE.LightShadow): string | null {
    const expected = new THREE.Matrix4().multiplyMatrices(WEBGPU_BIAS, new THREE.Matrix4().multiplyMatrices(shadow.camera.projectionMatrix, shadow.camera.matrixWorldInverse));
    return near(shadow.matrix.elements, expected.elements, 1e-5) ? null : "keeps a shadow.matrix that is not bias × the shadow camera's projection × view (the map's draw state is unknown)";
  }

  /** Internal 2 for a point map: face 5 of the table from the light position of the last draw equals the shadow camera. */
  #checkCube(light: THREE.PointLight): string | null {
    const faces = this.#faceViews(light);
    return near(faces[5]!.elements, light.shadow.camera.matrixWorldInverse.elements, 1e-5) ? null : "draws point shadow faces another way (face 5 does not match the shadow camera)";
  }

  /** The 6 face views of a point light's last draw (its position from `shadow.matrix`, three's WebGPU table). */
  #faceViews(light: THREE.PointLight): THREE.Matrix4[] {
    const position = new THREE.Vector3().setFromMatrixPosition(light.shadow.matrix).negate();
    const c = this.#camera;
    return CUBE_DIRECTIONS.map((direction, face) => {
      c.position.copy(position);
      c.up.copy(CUBE_UPS[face]!);
      c.lookAt(position.clone().add(direction));
      c.updateMatrixWorld(true);
      return c.matrixWorldInverse.clone();
    });
  }

  /** The 6 face view-projections (one projection: the shadow camera's, as drawn). */
  #faces(light: THREE.PointLight): Float32Array[] {
    const projection = light.shadow.camera.projectionMatrix;
    return this.#faceViews(light).map((view) => Float32Array.from(new THREE.Matrix4().multiplyMatrices(projection, view).elements));
  }

  #warnOnce(key: string, message: string): void {
    if (this.#warned.has(key)) return;
    this.#warned.add(key);
    console.warn(`@pmndrs/vdb/three: ${message}`);
  }
}
