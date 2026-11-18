/**
 * `VDBVolumeMaterial`: the fragment raymarch of one grid in a `VDBGridPool`,
 * through a `VDBGridNode` (task K2.3, docs/handoffs/K2.md). It replaces v1's
 * `NanoVDBVolumeMaterial`, which bound one grid image as its own storage
 * attribute and rewrote the vendored library's buffer global.
 *
 *   const grids = new VDBGridNode(renderer, pool, { helpers: ['rayClipSupport'] });
 *   const material = new VDBVolumeMaterial(renderer, grids, id, { densityScale: 40 });
 *   scene.add(new Mesh(vdbProxyGeometry(pool.info(id)), material));
 *   // per frame: pool.update(encoder) (or pool.flush()), then grids.update(), then render.
 *
 * The march is one entry `wgslFn` over the public `vdb*` WGSL API, with
 * `grids.module` as its include (the module carries the pool binding):
 * - the world ray goes into the grid's index space through the record's
 *   affine (`vdbWorldToIndex`/`vdbWorldToIndexDir`), so `pool.setTransform`
 *   instances render where they are;
 * - it is clipped to the sampler's support box (`vdbRayClipSupport`,
 *   `[min - 1, max + 1]` for trilinear), not the voxel-cell box, so the
 *   min-side half voxel the trilinear taps reach is marched too;
 * - fixed-step density accumulation with `vdbSample` (the node's sampler),
 *   premultiplied front-to-back compositing, early-out at low transmittance.
 *   The step lattice is anchored at the voxel-box entry (`vdbRayClip`, where
 *   v1 started) and extended back over the support shell, so samples inside
 *   the box sit where v1's did; only the shell samples are new (v1 skipped
 *   that half voxel, which shows where a grid's min-side voxels are dense);
 * - lighting as in v1: Henyey-Greenstein phase on a directional sun, an
 *   N-step shadow march toward the sun (a second accessor), a constant
 *   ambient floor, and a per-fragment budget on the total samples.
 *
 * Rendered on a proxy box with `side: BackSide` (the camera can be inside);
 * entry and exit come from the analytic clip, not the rasterized faces. The
 * material is `transparent`, `depthWrite: false`, premultiplied alpha. Scene
 * depth compositing is not done (v1 neither).
 *
 * Every tunable is a live TSL uniform (`material.densityScale.value = ...`).
 * The grid is a `uint` uniform too when the constructor got a number
 * (`material.gridId.value = otherId`), so one material can show any grid of
 * the types its `VDBGridNode` compiles without a rebuild.
 *
 * Units and assumptions:
 * - The march runs in the grid's **index space**: `stepSize` is in index
 *   voxels, and `densityScale` integrates per index voxel. Opacity therefore
 *   does not change with the world size of the grid: `pool.setTransform` (or
 *   a NanoVDB map) that scales the grid scales its footprint, not how opaque
 *   it looks.
 * - The proxy mesh is not tied to the grid: after `material.gridId.value = id`
 *   or `pool.setTransform(id, ...)`, build a new `vdbProxyGeometry(pool.info(id))`
 *   for the mesh, or fragments outside the old box are never rasterized.
 * - The view ray is `positionWorld - cameraPosition`: a perspective camera.
 *   An orthographic camera's rays are parallel and do not start at
 *   `cameraPosition`, so it renders wrongly.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import { cameraPosition, code, positionWorld, screenCoordinate, uniform, wgslFn } from "three/tsl";
import { VDBError } from "../core/errors.js";
import { getDevice, type VDBThreeRendererLike } from "./device.js";
import { VDBGridNode, type VDBTslNode } from "./grid-node.js";

/** Default hard caps of the compiled loops (the live uniforms clamp under them). */
export const VDB_VOLUME_MAX_STEPS_CAP = 1024;
export const VDB_VOLUME_SHADOW_STEPS_CAP = 128;
/**
 * Default cap on the total samples (primary + shadow) one fragment takes.
 * `maxSteps` and `shadowSteps` are independent live uniforms, so their product
 * can reach `maxStepsCap * shadowStepsCap` (~131k at the defaults), a driver
 * timeout risk. Once the budget is spent the shadow loop stops first, then the
 * primary loop, and the fragment returns its partial march.
 */
export const VDB_VOLUME_SAMPLE_BUDGET_CAP = 16384;

/** Parameters of {@link VDBVolumeMaterial}: initial uniform values and compile-time caps. */
export interface VDBVolumeMaterialParameters {
  densityScale?: number;
  /** March step in index voxels. */
  stepSize?: number;
  maxSteps?: number;
  sunDirection?: THREE.Vector3;
  sunColor?: THREE.Color;
  sunIntensity?: number;
  /** Henyey-Greenstein anisotropy g in (-1, 1). */
  anisotropy?: number;
  shadowSteps?: number;
  /** Density multiplier along the shadow ray only. */
  shadowDensity?: number;
  ambient?: number;
  /** Per-pixel jitter of the first sample. Default true. */
  jitter?: boolean;
  /** Hard cap on the primary loop. Default {@link VDB_VOLUME_MAX_STEPS_CAP}. */
  maxStepsCap?: number;
  /** Hard cap on the shadow loop. Default {@link VDB_VOLUME_SHADOW_STEPS_CAP}. */
  shadowStepsCap?: number;
  /** Hard cap on the total samples per fragment. Default {@link VDB_VOLUME_SAMPLE_BUDGET_CAP}. */
  sampleBudgetCap?: number;
}

const DEFAULTS = {
  densityScale: 40,
  stepSize: 0.75,
  maxSteps: 512,
  sunIntensity: 3,
  anisotropy: 0.3,
  shadowSteps: 12,
  shadowDensity: 1,
  ambient: 0.15,
} as const;

/** A live TSL uniform with its value typed (three types `uniform()`'s value as unknown). */
type ScalarUniform = ReturnType<typeof uniform> & { value: number };
type Vec3Uniform = ReturnType<typeof uniform> & { value: THREE.Vector3 };
type ColorUniform = ReturnType<typeof uniform> & { value: THREE.Color };
/** The `uint` uniform `VDBVolumeMaterial` makes for a numeric grid id (`material.gridId.value = otherId`). */
export type VDBUintUniform = ReturnType<typeof uniform> & { value: number };

/** Entry name of the march `wgslFn`. */
const ENTRY = "vdbThreeVolumeMarch";

/**
 * Module-scope helpers of the march, a second include next to `grids.module`
 * (`wgslFn` parses only its first `fn`, so the entry can't hold them).
 */
const VOLUME_HELPERS_WGSL = /* wgsl */ `
fn vdbThreeHgPhase(cos_theta: f32, g: f32) -> f32 {
    let gg = g * g;
    let denom = 1.0 + gg - 2.0 * g * cos_theta;
    return (1.0 - gg) / (12.566370614359172 * pow(max(denom, 1.0e-4), 1.5));
}
fn vdbThreeHash12(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
`;

/** The march entry, the one `fn` of its `wgslFn`. Caps are compiled in. */
function marchWgsl(maxStepsCap: number, shadowStepsCap: number, sampleBudgetCap: number): string {
  return /* wgsl */ `fn ${ENTRY}(
    grid_id: u32,
    ray_origin_w: vec3f,
    ray_dir_w: vec3f,
    sun_dir_w: vec3f,
    sun_color: vec3f,
    sun_intensity: f32,
    density_scale: f32,
    step_size: f32,
    max_steps: f32,
    g: f32,
    shadow_steps: f32,
    shadow_density: f32,
    ambient: f32,
    jitter: f32,
    pix: vec2f
) -> vec4f {
    // World -> index through the pool record (instance transform included).
    let o_idx = vdbWorldToIndex(grid_id, ray_origin_w);
    let d_idx = normalize(vdbWorldToIndexDir(grid_id, ray_dir_w));
    let sun_idx = normalize(vdbWorldToIndexDir(grid_id, sun_dir_w));

    // Clip to the trilinear support box; a dead id misses.
    var span = vec2f(0.0, 1.0e30);
    if !vdbRayClipSupport(grid_id, o_idx, d_idx, &span) {
        return vec4f(0.0);
    }
    // t0 >= 0: no sample behind the eye (camera inside the support box).
    let t0 = max(span.x, 0.0);
    let t1 = span.y;

    var acc = vdbAccessorInit(grid_id);
    var sacc = vdbAccessorInit(grid_id);

    // The phase uses the world-space view/sun angle.
    let phase = vdbThreeHgPhase(dot(normalize(ray_dir_w), normalize(sun_dir_w)), g);

    // The step lattice is anchored where the ray enters the voxel box
    // (vdbRayClip, v1's entry) and extended back to the support entry, so the
    // samples inside the box sit exactly where v1's did.
    var t = t0;
    var cell = vec2f(0.0, 1.0e30);
    if vdbRayClip(grid_id, o_idx, d_idx, &cell) {
        let anchor = max(cell.x, 0.0);
        t = anchor - ceil((anchor - t0) / step_size) * step_size;
    }
    t += step_size * jitter * vdbThreeHash12(pix);
    // The lattice can start before t0 (the anchor is stepped back in whole
    // steps, plus jitter). Skip those steps on the lattice: they are behind the
    // eye (camera in the support shell) or outside the support box.
    if t < t0 {
        t += ceil((t0 - t) / step_size) * step_size;
    }
    var transmittance = 1.0;
    var color = vec3f(0.0);
    let max_i = i32(clamp(max_steps, 1.0, ${maxStepsCap}.0));
    let shadow_n = i32(clamp(shadow_steps, 0.0, ${shadowStepsCap}.0));
    let budget = ${sampleBudgetCap};
    var samples = 0;

    for (var i = 0; i < max_i; i++) {
        if t > t1 || transmittance < 0.003 || samples >= budget {
            break;
        }
        let pos = o_idx + d_idx * t;
        let density = max(vdbSample(&acc, pos), 0.0) * density_scale;
        samples++;
        if density > 0.0 {
            let a = 1.0 - exp(-density * step_size);
            var tau = 0.0;
            var st = step_size;
            for (var s = 0; s < shadow_n; s++) {
                if samples >= budget {
                    break;
                }
                let sd = max(vdbSample(&sacc, pos + sun_idx * st), 0.0) * density_scale * shadow_density;
                samples++;
                tau += sd * step_size;
                st += step_size;
            }
            let lit = sun_color * (sun_intensity * exp(-tau) * phase) + vec3f(ambient);
            color += transmittance * a * lit;
            transmittance *= 1.0 - a;
        }
        t += step_size;
    }
    return vec4f(color, 1.0 - transmittance);
}`;
}

function bad(message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", `VDBVolumeMaterial: ${message}`);
}

function positiveInt(name: string, value: number | undefined, fallback: number): number {
  const v = value ?? fallback;
  if (!Number.isInteger(v) || v < 1) throw bad(`${name} must be an integer >= 1, got ${JSON.stringify(value)}.`);
  return v;
}

/**
 * A fragment raymarch of one grid of a `VDBGridPool` (see the file comment for
 * the march). Put it on a `vdbProxyGeometry(pool.info(id))` mesh; each frame
 * run `pool.update(encoder)` (or `pool.flush()`), then `grids.update()`, then
 * render.
 *
 *   const grids = new VDBGridNode(renderer, pool, { helpers: ['rayClipSupport'] });
 *   const material = new VDBVolumeMaterial(renderer, grids, id, { densityScale: 40 });
 *   const mesh = new Mesh(vdbProxyGeometry(pool.info(id)), material);
 *
 * - `stepSize` is in index voxels and `densityScale` integrates per index
 *   voxel, so opacity ignores the grid's world scale (`setTransform` scaling
 *   changes the size on screen, not the opacity).
 * - After `gridId.value = otherId` or `pool.setTransform(id, ...)`, give the
 *   mesh a new `vdbProxyGeometry(pool.info(id))`.
 * - Perspective cameras only (the ray starts at `cameraPosition`).
 * - The material must be rendered by the renderer `grids` was made with.
 *
 * @typeParam TGridId What the constructor got as the id: a number (default)
 *   makes {@link gridId} a `uint` uniform; a node is kept as given.
 */
export class VDBVolumeMaterial<TGridId extends number | VDBTslNode = number> extends NodeMaterial {
  readonly isVDBVolumeMaterial = true;

  /** The pool node this material reads (its `module` is the march's include). */
  readonly grids: VDBGridNode;
  /**
   * The grid id: a `uint` uniform when the constructor got a number
   * (`material.gridId.value = otherId`; then give the mesh a new
   * `vdbProxyGeometry`), else the caller's node.
   */
  readonly gridId: TGridId extends number ? VDBUintUniform : VDBTslNode;
  /** The entry `fn` text (for inspection and tests). */
  readonly entrySource: string;

  /** Extinction per index voxel at density 1 (opacity of a step: `1 - exp(-density * densityScale * stepSize)`). Default 40. */
  readonly densityScale: ScalarUniform;
  /** March step in index voxels (live; the lattice re-anchors per fragment). Default 0.75. */
  readonly stepSize: ScalarUniform;
  /** Primary march steps, clamped to `[1, maxStepsCap]`. Default 512. */
  readonly maxSteps: ScalarUniform;
  /** Direction toward the sun, world space (normalized at construction; keep it normalized). */
  readonly sunDirection: Vec3Uniform;
  /** Sun colour (linear). Default (1, 0.95, 0.9). */
  readonly sunColor: ColorUniform;
  /** Sun intensity. Default 3. */
  readonly sunIntensity: ScalarUniform;
  /** Henyey-Greenstein anisotropy g in (-1, 1). Default 0.3. */
  readonly anisotropy: ScalarUniform;
  /** Shadow-ray steps toward the sun per lit sample, clamped to `[0, shadowStepsCap]`. Default 12. */
  readonly shadowSteps: ScalarUniform;
  /** Density multiplier along the shadow ray only. Default 1. */
  readonly shadowDensity: ScalarUniform;
  /** Constant ambient term added to the sun light. Default 0.15. */
  readonly ambient: ScalarUniform;
  /** 1 = per-pixel jitter of the first sample on, 0 = off. */
  readonly jitter: ScalarUniform;

  /**
   * @param renderer The initialized `WebGPURenderer` the pool and `grids` live on.
   * @param grids A `VDBGridNode` made on this same renderer, with profile
   *   `sample` or `march` and `helpers: ['rayClipSupport']`. Share one per
   *   pool: one module per shader.
   * @param gridId The grid's pool id (a number, made a `uint` uniform) or a `uint` node.
   * @throws VDBError `VDB_BAD_OPTION` for a `grids` this material can't use
   *   (another renderer, profile, helpers), a bad id or cap;
   *   `VDB_UNSUPPORTED_DEVICE` for an uninitialized renderer.
   */
  constructor(renderer: VDBThreeRendererLike, grids: VDBGridNode, gridId: TGridId, params: VDBVolumeMaterialParameters = {}) {
    super();
    if (!(grids instanceof VDBGridNode)) throw bad("grids must be a VDBGridNode.");
    getDevice(renderer); // VDB_UNSUPPORTED_DEVICE before init
    if (grids.renderer !== renderer) {
      throw bad(
        "grids was made for another renderer. A VDBGridNode binds the pool through the renderer it was made with " +
          "(another renderer, even on the same GPUDevice, would read 0): create one VDBGridNode per renderer " +
          "(new VDBGridNode(renderer, pool, { helpers: ['rayClipSupport'] })).",
      );
    }
    if (grids.options.profile === "read") {
      throw bad("grids needs profile 'sample' or 'march' (vdbSample); this VDBGridNode was made with profile 'read'.");
    }
    if (!grids.options.helpers?.includes("rayClipSupport")) {
      throw bad("grids must compile the 'rayClipSupport' helper: new VDBGridNode(renderer, pool, { helpers: ['rayClipSupport'] }).");
    }
    if (typeof gridId === "number") {
      if (!Number.isInteger(gridId) || gridId < 0) throw bad(`gridId must be a non-negative integer, got ${gridId}.`);
    } else if (gridId === null || typeof gridId !== "object") {
      throw bad("gridId must be a number or a uint TSL node.");
    }
    const maxStepsCap = positiveInt("maxStepsCap", params.maxStepsCap, VDB_VOLUME_MAX_STEPS_CAP);
    const shadowStepsCap = positiveInt("shadowStepsCap", params.shadowStepsCap, VDB_VOLUME_SHADOW_STEPS_CAP);
    const sampleBudgetCap = positiveInt("sampleBudgetCap", params.sampleBudgetCap, VDB_VOLUME_SAMPLE_BUDGET_CAP);

    this.grids = grids;
    this.gridId = (typeof gridId === "number" ? uniform(gridId, "uint") : gridId) as this["gridId"];
    this.entrySource = marchWgsl(maxStepsCap, shadowStepsCap, sampleBudgetCap);

    this.densityScale = uniform(params.densityScale ?? DEFAULTS.densityScale) as ScalarUniform;
    this.stepSize = uniform(params.stepSize ?? DEFAULTS.stepSize) as ScalarUniform;
    this.maxSteps = uniform(params.maxSteps ?? DEFAULTS.maxSteps) as ScalarUniform;
    this.sunDirection = uniform((params.sunDirection ?? new THREE.Vector3(0.5, 0.8, 0.3)).clone().normalize()) as Vec3Uniform;
    this.sunColor = uniform(params.sunColor ?? new THREE.Color(1, 0.95, 0.9)) as ColorUniform;
    this.sunIntensity = uniform(params.sunIntensity ?? DEFAULTS.sunIntensity) as ScalarUniform;
    this.anisotropy = uniform(params.anisotropy ?? DEFAULTS.anisotropy) as ScalarUniform;
    this.shadowSteps = uniform(params.shadowSteps ?? DEFAULTS.shadowSteps) as ScalarUniform;
    this.shadowDensity = uniform(params.shadowDensity ?? DEFAULTS.shadowDensity) as ScalarUniform;
    this.ambient = uniform(params.ambient ?? DEFAULTS.ambient) as ScalarUniform;
    this.jitter = uniform(params.jitter === false ? 0 : 1) as ScalarUniform;

    const march = wgslFn(this.entrySource, [grids.module, code(VOLUME_HELPERS_WGSL)] as never);
    this.fragmentNode = march({
      grid_id: this.gridId,
      ray_origin_w: cameraPosition,
      ray_dir_w: positionWorld.sub(cameraPosition).normalize(),
      sun_dir_w: this.sunDirection,
      sun_color: this.sunColor,
      sun_intensity: this.sunIntensity,
      density_scale: this.densityScale,
      step_size: this.stepSize,
      max_steps: this.maxSteps,
      g: this.anisotropy,
      shadow_steps: this.shadowSteps,
      shadow_density: this.shadowDensity,
      ambient: this.ambient,
      jitter: this.jitter,
      pix: screenCoordinate,
    });

    this.transparent = true;
    this.depthWrite = false;
    this.depthTest = true;
    this.side = THREE.BackSide;
    this.premultipliedAlpha = true;
    this.toneMapped = false;
  }
}

/** What {@link vdbProxyGeometry} reads from `pool.info(id)`. */
export interface VDBProxyGridInfo {
  readonly indexBBox: { readonly min: readonly number[]; readonly max: readonly number[] };
  /** 3x4 row-major index-to-world affine (`pool.info(id).indexToWorld`). */
  readonly indexToWorld: ArrayLike<number>;
}

/**
 * A world-space box for a grid's proxy mesh (add it with an identity
 * transform): the AABB of the index box's voxels `[min - pad, max + 1 + pad]`
 * through the grid's index-to-world affine. `pad: 0` (default) is
 * `pool.info(id).worldBBox`, the box v1 used; `pad: 1` also covers the
 * min-side half voxel of the trilinear support that `VDBVolumeMaterial`
 * marches (rays that cross only that shell are otherwise not rasterized).
 */
export function vdbProxyGeometry(info: VDBProxyGridInfo, options: { pad?: number } = {}): THREE.BoxGeometry {
  const pad = options.pad ?? 0;
  if (!Number.isFinite(pad) || pad < 0) throw bad(`vdbProxyGeometry pad must be a finite number >= 0, got ${pad}.`);
  const m = info.indexToWorld;
  const lo = [0, 1, 2].map((a) => info.indexBBox.min[a]! - pad);
  const hi = [0, 1, 2].map((a) => info.indexBBox.max[a]! + 1 + pad);
  const box = new THREE.Box3();
  const p = new THREE.Vector3();
  for (let k = 0; k < 8; k++) {
    const x = k & 1 ? hi[0]! : lo[0]!;
    const y = k & 2 ? hi[1]! : lo[1]!;
    const z = k & 4 ? hi[2]! : lo[2]!;
    p.set(
      m[0]! * x + m[1]! * y + m[2]! * z + m[3]!,
      m[4]! * x + m[5]! * y + m[6]! * z + m[7]!,
      m[8]! * x + m[9]! * y + m[10]! * z + m[11]!,
    );
    box.expandByPoint(p);
  }
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
  geometry.translate(center.x, center.y, center.z);
  return geometry;
}
