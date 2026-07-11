/**
 * `NanoVDBVolumeMaterial` — the fragment-stage sparse-volume raymarch that is
 * WebVDB's stated main goal (SPEC §3.2, docs/PLAN.md Phase 3). Extends
 * three.js `NodeMaterial` (three/webgpu): a `fragmentNode` marches a ray
 * through a `NanoVDBGrid`'s storage buffer directly (no 3D-texture
 * intermediary), sampling the vendored `pnanovdb.wgsl` traversal library.
 *
 * ## What ships in v1
 *
 * - Renders on the grid's proxy box, `side: BackSide` (so the camera can enter
 *   the volume without the box being culled; entry/exit come from an analytic
 *   bbox clip, not the rasterized face).
 * - Ray setup: `cameraPosition` + `positionWorld` → world ray → transformed to
 *   INDEX space by the library's own `Map` accessors → clipped against the root
 *   index bbox (`pnanovdb_hdda_ray_clip`).
 * - March: fixed-step density accumulation with the per-grid-type trilinear
 *   sampler (selected at build time — no runtime switch), premultiplied
 *   transmittance/emission compositing, early-out at low transmittance, hard
 *   `maxSteps` cap. HDDA node-skipping is a documented stretch goal, NOT shipped
 *   here — plain fixed step per the gate's allowance.
 * - Lighting: Henyey–Greenstein phase (param `g`) on a directional sun +
 *   N-step secondary shadow march + a constant ambient floor. Output is
 *   premultiplied `vec4(rgb, alpha)` (material uses premultiplied-alpha
 *   blending).
 * - Depth/scene compositing is deliberately SKIPPED (SPEC says v1.1): the
 *   material is `transparent`, `depthWrite: false`.
 *
 * All tunables are live-updatable TSL `uniform()` nodes exposed as public
 * fields (`material.densityScale.value = ...`).
 *
 * The integration strategy that makes the full library run inside a NodeMaterial
 * lives in `./wgsl.ts` (`assembleVolumeWgsl`); this file does the TSL wiring.
 */

import * as THREE from "three";
import { NodeMaterial } from "three/webgpu";
import {
  cameraPosition,
  code,
  positionWorld,
  screenCoordinate,
  storage,
  uniform,
  wgslFn,
} from "three/tsl";
import type { NanoVDBGrid } from "./grid.js";
import { assembleVolumeWgsl } from "./wgsl.js";
import type { AssembledVolumeWgsl } from "./wgsl.js";

export interface NanoVDBVolumeMaterialParameters {
  /** The grid to raymarch. */
  grid: NanoVDBGrid;
  /**
   * The vendored `pnanovdb.wgsl` source text. Supplied by the caller so this
   * package stays free of bundler-specific `?raw` imports (browser: pass
   * `import src from "nanovdb-wgsl/pnanovdb.wgsl?raw"`; node/tests: read the
   * file via `pnanovdbWgslUrl`). Required.
   */
  pnanovdbSource: string;

  // --- Initial values for the live uniforms (all optional; defaults below) ---
  densityScale?: number;
  stepSize?: number;
  maxSteps?: number;
  sunDirection?: THREE.Vector3;
  sunColor?: THREE.Color;
  sunIntensity?: number;
  /** Henyey–Greenstein anisotropy g in (-1, 1). */
  anisotropy?: number;
  shadowSteps?: number;
  /** Density multiplier used along the shadow ray only. */
  shadowDensity?: number;
  ambient?: number;
  /** Per-pixel jitter toggle: 1 = on, 0 = off. */
  jitter?: boolean;

  /** Hard compile-time caps (uniforms clamp under these). */
  maxStepsCap?: number;
  shadowStepsCap?: number;
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

/**
 * Live TSL uniforms. Update via `.value = ...`; the change is picked up on the
 * next render without rebuilding the material. Intersected with the concrete
 * `.value` type (three types `uniform()` as `value: unknown`) so callers get a
 * typed handle while the node still satisfies `wgslFn` argument positions.
 */
type ScalarUniform = ReturnType<typeof uniform> & { value: number };
type Vec3Uniform = ReturnType<typeof uniform> & { value: THREE.Vector3 };
type ColorUniform = ReturnType<typeof uniform> & { value: THREE.Color };

export class NanoVDBVolumeMaterial extends NodeMaterial {
  readonly isNanoVDBVolumeMaterial = true;

  readonly grid: NanoVDBGrid;
  /** The assembled WGSL (library + entry) — exposed for inspection/testing. */
  readonly assembled: AssembledVolumeWgsl;
  /** The build-time-selected trilinear sampler function name. */
  readonly samplerFn: string;

  // Live uniforms (SPEC §3.2 param list). Public for live tweaking.
  readonly densityScale: ScalarUniform;
  readonly stepSize: ScalarUniform;
  readonly maxSteps: ScalarUniform;
  readonly sunDirection: Vec3Uniform;
  readonly sunColor: ColorUniform;
  readonly sunIntensity: ScalarUniform;
  readonly anisotropy: ScalarUniform;
  readonly shadowSteps: ScalarUniform;
  readonly shadowDensity: ScalarUniform;
  readonly ambient: ScalarUniform;
  readonly jitter: ScalarUniform;

  constructor(params: NanoVDBVolumeMaterialParameters) {
    super();

    const { grid, pnanovdbSource } = params;
    if (!pnanovdbSource || typeof pnanovdbSource !== "string") {
      throw new Error(
        "NanoVDBVolumeMaterial: `pnanovdbSource` (the vendored pnanovdb.wgsl text) is required — " +
          'pass `import src from "nanovdb-wgsl/pnanovdb.wgsl?raw"` in a bundler, or read the file via ' +
          "`pnanovdbWgslUrl` in node.",
      );
    }
    this.grid = grid;

    // Build-time WGSL assembly (throws upstream for unsupported grid types via
    // samplerForGridType; NanoVDBGrid already gates the type at construction).
    this.assembled = assembleVolumeWgsl(pnanovdbSource, {
      gridTypeId: grid.gridTypeId,
      ...(params.maxStepsCap !== undefined ? { maxStepsCap: params.maxStepsCap } : {}),
      ...(params.shadowStepsCap !== undefined ? { shadowStepsCap: params.shadowStepsCap } : {}),
    });
    this.samplerFn = this.assembled.samplerFn;

    // Live uniforms.
    this.densityScale = uniform(params.densityScale ?? DEFAULTS.densityScale) as ScalarUniform;
    this.stepSize = uniform(params.stepSize ?? DEFAULTS.stepSize) as ScalarUniform;
    this.maxSteps = uniform(params.maxSteps ?? DEFAULTS.maxSteps) as ScalarUniform;
    this.sunDirection = uniform(
      (params.sunDirection ?? new THREE.Vector3(0.5, 0.8, 0.3)).clone().normalize(),
    ) as Vec3Uniform;
    this.sunColor = uniform(params.sunColor ?? new THREE.Color(1, 0.95, 0.9)) as ColorUniform;
    this.sunIntensity = uniform(params.sunIntensity ?? DEFAULTS.sunIntensity) as ScalarUniform;
    this.anisotropy = uniform(params.anisotropy ?? DEFAULTS.anisotropy) as ScalarUniform;
    this.shadowSteps = uniform(params.shadowSteps ?? DEFAULTS.shadowSteps) as ScalarUniform;
    this.shadowDensity = uniform(params.shadowDensity ?? DEFAULTS.shadowDensity) as ScalarUniform;
    this.ambient = uniform(params.ambient ?? DEFAULTS.ambient) as ScalarUniform;
    this.jitter = uniform(params.jitter === false ? 0 : 1) as ScalarUniform;

    // Grid storage node. setName() forces the emitted WGSL buffer identifier so
    // the rewritten library's `<bufferName>.value[...]` reads resolve against
    // it (strategy B in wgsl.ts). toReadOnly() + passing it as the entry's
    // pointer param is what makes TSL emit + bind it. Same `storage(attr,'uint',
    // count)` incantation demo 01 proved.
    const gridStorage = storage(grid.storageAttribute, "uint", grid.image.length)
      .toReadOnly()
      .setName(this.assembled.bufferName);

    const march = wgslFn(this.assembled.entrySource, [code(this.assembled.librarySource)]);

    const rayDir = positionWorld.sub(cameraPosition).normalize();

    // The wgslFn is typed as returning a bare Node; the WGSL returns vec4<f32>.
    this.fragmentNode = march({
      grid_ptr: gridStorage,
      ray_origin_w: cameraPosition,
      ray_dir_w: rayDir,
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

    // Transparent volume: premultiplied output, no depth write (v1: no scene
    // depth compositing — SPEC §3.2 item 5 is v1.1).
    this.transparent = true;
    this.depthWrite = false;
    this.depthTest = true;
    this.side = THREE.BackSide;
    this.premultipliedAlpha = true;
    this.toneMapped = false;
  }
}
