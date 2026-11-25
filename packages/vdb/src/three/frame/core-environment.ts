/**
 * K6.2b: `scene.environment` as the core renderer's environment input (design §2.3, Q3). The core reads a host equirect
 * or cube `GPUTexture` directly, so the adapter passes the scene environment's own texture (no readback, no PMREM):
 *
 * - an equirect texture (`EquirectangularReflectionMapping` / `RefractionMapping`): its GPUTexture; three uploads
 *   `flipY` data bottom row first, so the map's v runs up: a reflection of y joins the rotation;
 * - a cube texture: its GPUTexture as a 6-layer cube; three's WebGPU cube lookups negate x (`CubeTextureNode`), so a
 *   reflection of x joins the rotation;
 * - anything else (a PMREM `CubeUVReflectionMapping` atlas, a render-target cube): drawn once into a 1024 × 512 float
 *   lat-long target through three's PMREM sampler at roughness 0 (the scene rotation applied there), and that target's
 *   GPUTexture is passed (redrawn only when the environment or its rotation changes).
 *
 * The rotation is `scene.environmentRotation` the way three applies it (`materialEnvRotation`: the transposed Euler
 * matrix), so the core's lights and SH sit where three's PMREM lookups put them. Intensity: `scene.environmentIntensity`.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import type { VDBFrameEnvironment } from "../../core/frame/types.js";
import { getGPUTexture } from "../device.js";
import { VDBFrameEnvironmentMaterial } from "./environment-material.js";
import { frameFullscreenVertex } from "./chunks.js";

const { Fn, cos, float, screenCoordinate, sin, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

/** The lat-long fallback's size: about one texel per PMREM mip-0 texel (0.35°), so a small sun is not lost between point samples. */
const LATLONG = [1024, 512] as const;

type SceneLike = THREE.Object3D & { environment?: THREE.Texture | null; environmentIntensity?: number; environmentRotation?: THREE.Euler };
type RendererLike = { initTexture?(t: THREE.Texture): void; setRenderTarget(t: THREE.RenderTarget | null): void; getRenderTarget(): THREE.RenderTarget | null; render(s: THREE.Object3D, c: THREE.Camera): void };

/** A lat-long of the scene environment in the core's equirect convention (row 0 the zenith, u = atan2(z, x) / 2π + ½). */
class VDBCoreLatLongMaterial extends VDBFrameEnvironmentMaterial {
  constructor() {
    super();
    this.name = "VDBCoreLatLongMaterial";
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
    return Fn(() => {
      // WebGPU: screenCoordinate.y is the render target row (0 at the top)
      const uv = vec2(screenCoordinate.x, screenCoordinate.y).div(vec2(LATLONG[0], LATLONG[1])).toVar();
      const theta = uv.y.mul(Math.PI);
      const phi = uv.x.sub(0.5).mul(2.0 * Math.PI);
      const direction = vec3(sin(theta).mul(cos(phi)), cos(theta), sin(theta).mul(sin(phi)));
      if (!this.environment) return vec4(0.0, 0.0, 0.0, 1.0);
      return vec4(this.environmentRadiance(direction, 0.0), float(1.0));
    })();
  }
}

/** @internal The environment input of the `./three` core backend. */
export class VDBCoreEnvironment {
  #material: VDBCoreLatLongMaterial | null = null;
  #scene: THREE.Scene | null = null;
  #camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  #target: THREE.RenderTarget | null = null;
  #drawnKey = "";
  #drawn = 0;
  #initialized = new WeakMap<THREE.Texture, number>();
  readonly #euler = new THREE.Matrix4();

  /** The core input for `scene` (null without an environment). */
  input(renderer: RendererLike, scene: SceneLike): VDBFrameEnvironment | null {
    const env = scene.environment;
    if (!env) return null;
    const intensity = scene.environmentIntensity ?? 1;
    const t = env as THREE.Texture & { isCubeTexture?: boolean; isRenderTargetTexture?: boolean; isDataTexture?: boolean };
    const equirect = (t.mapping === THREE.EquirectangularReflectionMapping || t.mapping === THREE.EquirectangularRefractionMapping) && !t.isRenderTargetTexture;
    const cube = !!t.isCubeTexture && !t.isRenderTargetTexture && (t.mapping === THREE.CubeReflectionMapping || t.mapping === THREE.CubeRefractionMapping);
    // three's lookup direction m = Rᵀ d (materialEnvRotation is the transposed Euler matrix)
    this.#euler.makeRotationFromEuler(scene.environmentRotation ?? new THREE.Euler());
    const e = this.#euler.elements;
    if (equirect || cube) {
      if (this.#initialized.get(t) !== t.version) {
        renderer.initTexture?.(t);
        this.#initialized.set(t, t.version);
      }
      // world → map: F · Rᵀ (column-major; Rᵀ's column c is R's row c)
      const flip = cube ? [-1, 1, 1] : [1, t.flipY ? -1 : 1, 1];
      const rotation = new Float32Array(9);
      for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) rotation[c * 3 + r] = flip[r]! * e[r * 4 + c]!;
      return { id: t, revision: t.version, texture: getGPUTexture(renderer as never, t), projection: cube ? "cube" : "equirect", rotation, intensity };
    }
    // A PMREM atlas or a render-target cube: the lat-long fallback (rotation applied by three's PMREM sampler)
    const r = scene.environmentRotation;
    const key = `${t.uuid}:${t.version}:${r ? `${r.x}:${r.y}:${r.z}:${r.order}` : ""}`;
    const target = this.#latLong(renderer, scene, key);
    return { id: this, revision: this.#drawn, texture: getGPUTexture(renderer as never, target.texture), projection: "equirect", intensity };
  }

  #latLong(renderer: RendererLike, scene: SceneLike, key: string): THREE.RenderTarget {
    if (!this.#target) {
      this.#target = new THREE.RenderTarget(LATLONG[0], LATLONG[1], { type: THREE.HalfFloatType, depthBuffer: false, magFilter: THREE.NearestFilter, minFilter: THREE.NearestFilter, generateMipmaps: false });
      this.#material = new VDBCoreLatLongMaterial();
      this.#scene = new THREE.Scene();
      const mesh = new THREE.Mesh(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3)), this.#material as unknown as NodeMaterial);
      mesh.frustumCulled = false;
      this.#scene.add(mesh);
    }
    if (key !== this.#drawnKey) {
      const s = this.#scene!;
      s.environment = scene.environment ?? null;
      s.environmentIntensity = 1;
      if (scene.environmentRotation) s.environmentRotation.copy(scene.environmentRotation);
      this.#material!.setEnvironment(scene.environment);
      const previous = renderer.getRenderTarget();
      renderer.setRenderTarget(this.#target);
      renderer.render(s, this.#camera);
      renderer.setRenderTarget(previous);
      this.#drawnKey = key;
      this.#drawn++;
    }
    return this.#target!;
  }

  dispose(): void {
    this.#target?.dispose();
    this.#material?.dispose();
    this.#target = null;
    this.#material = null;
    this.#scene = null;
    this.#drawnKey = "";
  }
}
