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
import { NodeMaterial, StorageBufferAttribute } from "three/webgpu";
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

  /**
   * Pre-allocate the grid storage buffer at this many BYTES so the grid can be
   * swapped for a DIFFERENT-SIZED one later via `rebindGrid()` without a
   * material/node rebuild (the paddable-buffer strategy — see `rebindGrid`).
   *
   * WHY THIS EXISTS: three's WebGPU backend allocates the storage GPUBuffer
   * once, sized to the attribute array's byteLength at first upload, and never
   * resizes it; subsequent updates are `device.queue.writeBuffer` into that
   * fixed buffer. So a same-size grid swap is free, but a bigger grid would
   * overflow the buffer. Sizing the buffer up front to the largest frame you
   * will ever bind (e.g. `max(frame.image.byteLength)` across a sequence) makes
   * every rebind an in-place sub-fill. Rounded up to a multiple of 4 bytes.
   *
   * Omit for a single static grid (the buffer is sized exactly to `grid`, and
   * `rebindGrid` then only accepts a grid of the same-or-smaller byte length).
   * The vendored traversal never bounds-checks reads against the buffer length,
   * so the trailing padding is inert.
   */
  maxGridBytes?: number;

  /** Hard compile-time caps (uniforms clamp under these). */
  maxStepsCap?: number;
  shadowStepsCap?: number;
  /**
   * Hard cap on the TOTAL trilinear taps (primary + shadow) a single fragment
   * may perform per frame. `maxSteps`/`shadowSteps` are independently-clamped
   * LIVE uniforms, so their product can otherwise reach `maxStepsCap *
   * shadowStepsCap` (~65k at the defaults) — a TDR risk on real hardware. This
   * budget is enforced inside the generated WGSL with a per-fragment counter:
   * once exhausted, the shadow loop breaks first, then the main march loop,
   * so the fragment returns its partial march instead of a hard black pixel.
   * Default 16384 (see `DEFAULT_SAMPLE_BUDGET_CAP` in `./wgsl.js`).
   */
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

  /** The assembled WGSL (library + entry) — exposed for inspection/testing. */
  readonly assembled: AssembledVolumeWgsl;
  /** The build-time-selected trilinear sampler function name. */
  readonly samplerFn: string;

  /** Currently-bound grid (mutated by `rebindGrid`). */
  private _grid: NanoVDBGrid;
  /** The bound storage attribute — owns the (possibly padded) backing array. */
  private readonly _gridAttribute: StorageBufferAttribute;
  /** Capacity of the storage buffer in u32 words (>= any grid we can rebind to). */
  private readonly _capacityWords: number;
  /** True when we allocated our own padded backing (vs. binding `grid`'s own attribute zero-copy). */
  private readonly _ownsBacking: boolean;

  /** The grid currently bound for raymarching. Swap it with `rebindGrid`. */
  get grid(): NanoVDBGrid {
    return this._grid;
  }

  /** Storage-buffer capacity in bytes (the largest grid image this material can rebind to). */
  get capacityBytes(): number {
    return this._capacityWords * 4;
  }

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
    this._grid = grid;

    // Storage-buffer capacity. Default: exactly the grid's image (zero-copy —
    // bind the grid's own attribute, no copy). If `maxGridBytes` asks for
    // headroom, allocate our own padded backing so a larger grid can be
    // sub-filled in later without a rebuild (see `maxGridBytes` / `rebindGrid`).
    const gridWords = grid.image.length;
    const requestedWords = Math.ceil((params.maxGridBytes ?? 0) / 4);
    this._capacityWords = Math.max(gridWords, requestedWords);
    this._ownsBacking = this._capacityWords > gridWords;

    if (this._ownsBacking) {
      const backing = new Uint32Array(this._capacityWords);
      backing.set(grid.image);
      this._gridAttribute = new StorageBufferAttribute(backing, 1);
    } else {
      this._gridAttribute = grid.storageAttribute;
    }

    // Build-time WGSL assembly (throws upstream for unsupported grid types via
    // samplerForGridType; NanoVDBGrid already gates the type at construction).
    this.assembled = assembleVolumeWgsl(pnanovdbSource, {
      gridTypeId: grid.gridTypeId,
      ...(params.maxStepsCap !== undefined ? { maxStepsCap: params.maxStepsCap } : {}),
      ...(params.shadowStepsCap !== undefined ? { shadowStepsCap: params.shadowStepsCap } : {}),
      ...(params.sampleBudgetCap !== undefined ? { sampleBudgetCap: params.sampleBudgetCap } : {}),
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
    const gridStorage = storage(this._gridAttribute, "uint", this._capacityWords)
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

  /**
   * Swap the grid this material raymarches WITHOUT rebuilding the material or
   * its node graph — the Phase 7 sequence-playback primitive (docs/SPEC §3.5,
   * the grid-rebind flag from docs/handoffs/PHASE-3.md).
   *
   * ## The mechanism (three r185, verified against source)
   *
   * The `fragmentNode`/`storage()` node — and the trilinear sampler baked into
   * the WGSL at construction — are fixed. What actually changes on the GPU is
   * only the contents of the storage buffer. three's WebGPU backend allocates
   * that GPUBuffer exactly once (sized to the attribute array's byteLength at
   * first upload, `WebGPUAttributeUtils.createBuffer`) and thereafter re-uploads
   * on any attribute version bump via `device.queue.writeBuffer(buffer, 0,
   * array, 0)` into that SAME fixed buffer (`WebGPUAttributeUtils.updateAttribute`,
   * gated by `Attributes.update`'s `version <` check). Consequences:
   *
   * - **Same or smaller byteLength**: copy the new image into the bound array
   *   and bump `needsUpdate` — the cheapest path, no reallocation. (A smaller
   *   image leaves a stale tail; the vendored traversal reads only via internal
   *   grid offsets and never bounds-checks against the buffer length, so the
   *   tail is inert.)
   * - **Larger byteLength**: the fixed GPUBuffer can't grow, and a longer
   *   `writeBuffer` would overflow it. This throws UNLESS the material was
   *   constructed with `maxGridBytes` headroom covering the new image, in which
   *   case the pre-sized buffer absorbs it as an in-place sub-fill.
   *
   * The grid TYPE cannot change: the sampler (`pnanovdb_sample_trilinear_*`) is
   * selected at build time, so rebinding a different-typed grid throws — build a
   * new material for a different grid type.
   *
   * @param grid The grid to bind. Must be the same grid TYPE and fit the
   *   storage capacity (`capacityBytes`).
   */
  rebindGrid(grid: NanoVDBGrid): void {
    if (grid.gridTypeId !== this._grid.gridTypeId) {
      throw new Error(
        `NanoVDBVolumeMaterial.rebindGrid: grid type changed ` +
          `(${this._grid.metadata.gridType} -> ${grid.metadata.gridType}). The trilinear sampler is baked ` +
          `into the shader at construction, so a different grid type needs a new material, not a rebind.`,
      );
    }

    const needWords = grid.image.length;
    if (needWords > this._capacityWords) {
      throw new Error(
        `NanoVDBVolumeMaterial.rebindGrid: new grid image is ${needWords * 4} bytes but the storage buffer ` +
          `capacity is ${this._capacityWords * 4} bytes. three's WebGPU backend fixes the GPUBuffer size at ` +
          `first upload and cannot grow it. Construct the material with ` +
          `\`maxGridBytes >= ${needWords * 4}\` (e.g. the largest frame in your sequence) to enable in-place ` +
          `rebinds, or rebuild the material for this frame.`,
      );
    }

    // In-place sub-fill of the bound backing array, then flag a re-upload. When
    // we don't own the backing (no headroom requested) this writes into the
    // grid's own attribute array — fine, we're replacing that grid anyway.
    const backing = this._gridAttribute.array as Uint32Array;
    backing.set(grid.image);
    this._gridAttribute.needsUpdate = true;
    this._grid = grid;
  }
}
