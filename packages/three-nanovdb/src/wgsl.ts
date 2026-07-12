/**
 * `assembleVolumeWgsl` — the one function that owns THE Phase 3 integration
 * decision: how to get the full vendored `pnanovdb.wgsl` traversal library
 * (`packages/nanovdb-wgsl/vendor/pnanovdb.wgsl`) running inside a three.js
 * `NodeMaterial` fragment shader with its module-scope buffer global
 * satisfied.
 *
 * ## The problem
 *
 * The vendored module's read path bottoms out in a SINGLE function,
 * `pnanovdb_buf_read_uint32`, which reads a module-scope global the consumer
 * must declare:
 *
 * ```wgsl
 * // (module scope, provided by the consumer)
 * var<storage, read> nanovdb_buffer : array<u32>;
 * fn pnanovdb_buf_read_uint32(buf, byte_offset) -> u32 {
 *   return nanovdb_buffer[(buf.byte_offset / 4u) + (byte_offset / 4u)];
 * }
 * ```
 *
 * The raw-WebGPU harness (Phase 2) just hand-declares that binding in a
 * footer. A `NodeMaterial` can't: TSL OWNS the bind-group layout, so a
 * hand-written `@group/@binding` storage declaration would (a) collide with
 * TSL's auto-assigned bindings and (b) never actually be bound to our grid
 * buffer (TSL doesn't know it exists) — the dispatch reads zeroes or fails
 * validation.
 *
 * ## The strategy that works (a B+C hybrid — see the phase report)
 *
 * 1. **B (force the emitted name).** Create the grid's storage node with
 *    `storage(grid.storageAttribute, "uint", n).toReadOnly().setName(BUFFER_NAME)`.
 *    three's `WGSLNodeBuilder` uses `uniformNode.name` verbatim for the emitted
 *    buffer variable (`WGSLNodeBuilder.getUniformFromNode` → line
 *    `uniformNode.name = name ? name : 'NodeBuffer_' + id`). A (non-custom)
 *    storage buffer is wrapped in a struct, so TSL emits:
 *
 *    ```wgsl
 *    struct <BUFFER_NAME>Struct { value : array<u32> };
 *    @binding(B) @group(G) var<storage, read> <BUFFER_NAME> : <BUFFER_NAME>Struct;
 *    ```
 *
 *    and accesses it as `<BUFFER_NAME>.value[i]` — TSL manages the binding and
 *    uploads our grid image, so the buffer is genuinely bound.
 *
 * 2. **C (one-token textual rewrite of the vendored source).** Rewrite the
 *    library's ONE buffer-global identifier `nanovdb_buffer` → `<BUFFER_NAME>.value`
 *    so every read resolves against the TSL-managed buffer. The vendored file
 *    is NOT edited; the rewrite runs at material-build time on the imported
 *    string (`rewriteBufferGlobal`, unit-tested below).
 *
 * 3. The rewritten library is included into the entry `wgslFn` via a
 *    `code(librarySource)` node (emitted verbatim at module scope, ahead of the
 *    entry function). The entry function additionally takes the same storage
 *    node as a `ptr<storage, array<u32>, read>` parameter — the proven Phase 1
 *    binding mechanism — which is what actually triggers TSL to emit + bind the
 *    `<BUFFER_NAME>` global. The library then reads that global directly.
 *
 * This module is pure (no three.js import): it takes the vendored WGSL text as
 * a string and returns assembled strings, so it is fully node/vitest-testable.
 * `material.ts` supplies the source and does the TSL wiring.
 */

/**
 * PNanoVDB `GridType` ids for the v1 supported subset (matches
 * `grid.ts` GRID_TYPE_IDS and `pnanovdb.wgsl`'s `PNANOVDB_GRID_TYPE_*`).
 */
export const GRID_TYPE_FLOAT = 1;
export const GRID_TYPE_FP8 = 14;
export const GRID_TYPE_FPN = 16;

/** Default emitted name for the TSL-managed grid storage buffer. */
export const DEFAULT_BUFFER_NAME = "nvdbGrid";

/** Default name of the assembled fragment entry function. */
export const DEFAULT_ENTRY_NAME = "nvdb_volume_march";

/** Hard compile-time caps on the march/shadow loops (uniforms clamp under these). */
export const DEFAULT_MAX_STEPS_CAP = 1024;
export const DEFAULT_SHADOW_STEPS_CAP = 128;

/**
 * Hard compile-time cap on the TOTAL trilinear taps (primary + shadow) a
 * single fragment may perform. `max_steps` (cap 1024) and `shadow_steps` (cap
 * 128) are independently-clamped LIVE uniforms — a user can drive them to
 * ~65k taps/fragment (1024 * 128), which risks a driver TDR on real hardware.
 * This budget is enforced INSIDE the generated WGSL (see `buildEntrySource`):
 * a counter increments on every trilinear tap, and once the budget is
 * exhausted the shadow loop breaks first, then the main march loop — a
 * graceful early-out on the partial march accumulated so far, not a hard
 * black pixel.
 */
export const DEFAULT_SAMPLE_BUDGET_CAP = 16384;

export interface VolumeWgslOptions {
  /** PNanoVDB `GridType` id (selects the trilinear sampler at build time). */
  gridTypeId: number;
  /** Emitted name for the TSL-managed grid storage buffer. Default `nvdbGrid`. */
  bufferName?: string;
  /** Name of the generated entry function. Default `nvdb_volume_march`. */
  entryName?: string;
  /** Hard cap on the primary march loop. Default 1024. */
  maxStepsCap?: number;
  /** Hard cap on the sun shadow-march loop. Default 128. */
  shadowStepsCap?: number;
  /**
   * Hard cap on the TOTAL trilinear taps (primary + shadow) per fragment.
   * Default 16384. See `DEFAULT_SAMPLE_BUDGET_CAP`.
   */
  sampleBudgetCap?: number;
}

export interface AssembledVolumeWgsl {
  /** Vendored library, buffer-global rewritten, plus the appended helpers. */
  librarySource: string;
  /** The entry `fn <entryName>(...) -> vec4<f32>` WGSL source (single function). */
  entrySource: string;
  entryName: string;
  bufferName: string;
  /** The build-time-selected trilinear sampler function name. */
  samplerFn: string;
  gridTypeId: number;
}

/**
 * Build-time sampler selection (SPEC §3.2: "pick per gridTypeId at material
 * build time — string-select the right sampler, no runtime switch needed").
 * Throws on any grid type `NanoVDBGrid` wouldn't have accepted.
 */
export function samplerForGridType(gridTypeId: number): string {
  switch (gridTypeId) {
    case GRID_TYPE_FLOAT:
      return "pnanovdb_sample_trilinear_float";
    case GRID_TYPE_FP8:
      return "pnanovdb_sample_trilinear_fp8";
    case GRID_TYPE_FPN:
      return "pnanovdb_sample_trilinear_fpn";
    default:
      throw new Error(
        `assembleVolumeWgsl: no trilinear sampler for grid type id ${gridTypeId} — ` +
          `v1 supports Float(1)/Fp8(14)/FpN(16) only (SPEC §2.1/§7).`,
      );
  }
}

/**
 * Rewrite the vendored source's single buffer-global identifier
 * `nanovdb_buffer` to `<bufferName>.value` (matching TSL's struct-wrapped
 * storage emission). Whole-word only, so the surrounding function/struct names
 * are untouched; the one live read site is `pnanovdb_buf_read_uint32`, plus a
 * header comment (harmless). Pure + exported for unit testing.
 */
export function rewriteBufferGlobal(pnanovdbSource: string, bufferName: string): string {
  return pnanovdbSource.replace(/\bnanovdb_buffer\b/g, `${bufferName}.value`);
}

/**
 * Guards against an empty/wrong `pnanovdbSource` string up front, rather than
 * letting it surface as an opaque WGSL compile error (or, worse, a silently
 * unrewritten/undeclared global) later. Shared by `assembleVolumeWgsl` (the
 * TSL integration, which rewrites this identifier — see the module header)
 * and `compute.ts`'s raw-WebGPU compute passes (which declare
 * `nanovdb_buffer` directly, unrewritten, exactly like the Phase 2 harness
 * footer) — both need the real vendored `pnanovdb.wgsl` text, not an
 * empty/wrong string, and this is the one check both care about.
 */
export function assertHasBufferGlobal(pnanovdbSource: string): void {
  if (!/\bnanovdb_buffer\b/.test(pnanovdbSource)) {
    throw new Error(
      "assertHasBufferGlobal: `pnanovdbSource` does not contain the `nanovdb_buffer` read site every " +
        "consumer of the vendored library must satisfy — pass the real vendored pnanovdb.wgsl text " +
        "(packages/nanovdb-wgsl/vendor/pnanovdb.wgsl), not an empty/wrong string.",
    );
  }
}

/**
 * Small WGSL helpers appended to the (rewritten) library so they live at module
 * scope alongside it — the entry `wgslFn` can only itself be a single parsed
 * function, so shared helpers can't live in the entry source. Prefixed
 * `nvdbx_` to avoid colliding with `pnanovdb_*`.
 */
export const VOLUME_HELPERS_WGSL = /* wgsl */ `
// --- three-nanovdb volume-material helpers (nvdbx_*) ---
fn nvdbx_hg_phase(cos_theta : f32, g : f32) -> f32 {
  let gg = g * g;
  let denom = 1.0 + gg - 2.0 * g * cos_theta;
  // 4*pi = 12.566370614359172
  return (1.0 - gg) / (12.566370614359172 * pow(max(denom, 1.0e-4), 1.5));
}

// Hash-based per-pixel jitter (Dave Hoskins hash12), in [0,1).
fn nvdbx_hash12(p : vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
`;

/**
 * Build the fragment-stage raymarch entry function. It is the SOLE WGSL `fn`
 * in the returned string (three's `WGSLNodeFunction` parses the first `fn` as
 * the entry), so all shared helpers live in the library string instead.
 *
 * March (SPEC §3.2): everything happens in the grid's INDEX space. The ray is
 * transformed world→index via the library's own `Map` accessors (exact, no
 * uniform matrix needed), clipped against the root index bbox with
 * `pnanovdb_hdda_ray_clip`, then fixed-step density accumulation with
 * premultiplied transmittance/emission compositing and early-out at low
 * transmittance. Lighting: Henyey–Greenstein phase on a directional sun +
 * an N-step secondary shadow march + a constant ambient floor.
 */
function buildEntrySource(
  entryName: string,
  samplerFn: string,
  maxStepsCap: number,
  shadowStepsCap: number,
  sampleBudgetCap: number,
): string {
  return /* wgsl */ `
fn ${entryName}(
  grid_ptr : ptr<storage, array<u32>, read>,
  ray_origin_w : vec3<f32>,
  ray_dir_w : vec3<f32>,
  sun_dir_w : vec3<f32>,
  sun_color : vec3<f32>,
  sun_intensity : f32,
  density_scale : f32,
  step_size : f32,
  max_steps : f32,
  g : f32,
  shadow_steps : f32,
  shadow_density : f32,
  ambient : f32,
  jitter : f32,
  pix : vec2<f32>
) -> vec4<f32> {
  // Grid handles. arrayLength(grid_ptr) is the one meaningful use of the
  // pointer param: it forces TSL to emit + bind the storage global the
  // library reads (see wgsl.ts header).
  let buf = pnanovdb_make_buffer(0u, arrayLength(grid_ptr));
  let grid = pnanovdb_grid_handle_t(0u);
  let tree = pnanovdb_grid_get_tree(grid);
  let root = pnanovdb_tree_get_root(buf, tree);
  var acc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&acc, root);
  var sacc : pnanovdb_readaccessor_t;
  pnanovdb_readaccessor_init(&sacc, root);

  // World -> index: origin as a point, directions via the map jacobian.
  let o_idx = pnanovdb_grid_world_to_indexf(buf, grid, ray_origin_w);
  let d_idx = normalize(pnanovdb_grid_world_to_index_dirf(buf, grid, ray_dir_w));
  let sun_idx = normalize(pnanovdb_grid_world_to_index_dirf(buf, grid, sun_dir_w));

  // Clip against the root index bbox (inclusive max coord -> +1 for far face).
  let bmin = vec3<f32>(pnanovdb_root_get_bbox_min(buf, root));
  let bmax = vec3<f32>(pnanovdb_root_get_bbox_max(buf, root) + vec3<i32>(1, 1, 1));
  var t0 = 0.0;
  var t1 = 1.0e30;
  let hit = pnanovdb_hdda_ray_clip(bmin, bmax, o_idx, &t0, d_idx, &t1);
  if (!hit || t1 <= t0) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  t0 = max(t0, 0.0);

  // Phase uses the true world-space view/sun angle.
  let cos_theta = dot(normalize(ray_dir_w), normalize(sun_dir_w));
  let phase = nvdbx_hg_phase(cos_theta, g);

  // Per-pixel jitter of the first sample (toggle via jitter in {0,1}).
  var t = t0 + step_size * jitter * nvdbx_hash12(pix);

  var transmittance = 1.0;
  var color = vec3<f32>(0.0, 0.0, 0.0);

  let max_i = i32(clamp(max_steps, 1.0, ${maxStepsCap}.0));
  let shadow_n = i32(clamp(shadow_steps, 0.0, ${shadowStepsCap}.0));

  // Per-fragment total-sample budget (unbounded maxSteps x shadowSteps product
  // guard — see wgsl.ts DEFAULT_SAMPLE_BUDGET_CAP doc). Counts every trilinear
  // tap (primary + shadow); when exhausted the shadow loop breaks first, then
  // the main loop, returning the partial march accumulated so far rather than
  // a hard black pixel.
  const nvdbx_sample_budget : i32 = ${sampleBudgetCap};
  var nvdbx_sample_count : i32 = 0;

  for (var i = 0; i < max_i; i = i + 1) {
    if (t > t1 || transmittance < 0.003 || nvdbx_sample_count >= nvdbx_sample_budget) {
      break;
    }
    let pos = o_idx + d_idx * t;
    let density = max(${samplerFn}(buf, &acc, pos), 0.0) * density_scale;
    nvdbx_sample_count = nvdbx_sample_count + 1;
    if (density > 0.0) {
      let a = 1.0 - exp(-density * step_size);

      // Secondary shadow march toward the sun.
      var tau = 0.0;
      var st = step_size;
      for (var s = 0; s < shadow_n; s = s + 1) {
        if (nvdbx_sample_count >= nvdbx_sample_budget) {
          break;
        }
        let spos = pos + sun_idx * st;
        let sd = max(${samplerFn}(buf, &sacc, spos), 0.0) * density_scale * shadow_density;
        nvdbx_sample_count = nvdbx_sample_count + 1;
        tau = tau + sd * step_size;
        st = st + step_size;
      }
      let sun_vis = exp(-tau);
      let lit = sun_color * (sun_intensity * sun_vis * phase) + vec3<f32>(ambient, ambient, ambient);

      // Premultiplied front-to-back compositing.
      color = color + transmittance * a * lit;
      transmittance = transmittance * (1.0 - a);
    }
    t = t + step_size;
  }

  let alpha = 1.0 - transmittance;
  return vec4<f32>(color, alpha);
}
`;
}

/** Throws unless `value` is a positive (>= 1) integer. */
function assertPositiveIntCap(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(
      `assembleVolumeWgsl: \`${name}\` must be an integer >= 1 (got ${JSON.stringify(value)}).`,
    );
  }
}

/**
 * Assemble the WGSL for a `NanoVDBVolumeMaterial` fragment raymarch. See the
 * module header for the integration strategy this encapsulates.
 *
 * Validates its inputs at construction time rather than letting a bad input
 * surface as an opaque WGSL compile error at render time:
 * - `pnanovdbSource` must contain the `nanovdb_buffer` read site that
 *   `rewriteBufferGlobal` is supposed to rewrite (an empty/wrong string would
 *   otherwise silently produce a library with no rewritten buffer global).
 * - `maxStepsCap`, `shadowStepsCap`, and `sampleBudgetCap` must each be
 *   integers >= 1.
 */
export function assembleVolumeWgsl(
  pnanovdbSource: string,
  opts: VolumeWgslOptions,
): AssembledVolumeWgsl {
  const bufferName = opts.bufferName ?? DEFAULT_BUFFER_NAME;
  const entryName = opts.entryName ?? DEFAULT_ENTRY_NAME;
  const maxStepsCap = opts.maxStepsCap ?? DEFAULT_MAX_STEPS_CAP;
  const shadowStepsCap = opts.shadowStepsCap ?? DEFAULT_SHADOW_STEPS_CAP;
  const sampleBudgetCap = opts.sampleBudgetCap ?? DEFAULT_SAMPLE_BUDGET_CAP;

  assertPositiveIntCap("maxStepsCap", maxStepsCap);
  assertPositiveIntCap("shadowStepsCap", shadowStepsCap);
  assertPositiveIntCap("sampleBudgetCap", sampleBudgetCap);

  assertHasBufferGlobal(pnanovdbSource);

  const samplerFn = samplerForGridType(opts.gridTypeId);

  const librarySource = rewriteBufferGlobal(pnanovdbSource, bufferName) + "\n" + VOLUME_HELPERS_WGSL;
  const entrySource = buildEntrySource(entryName, samplerFn, maxStepsCap, shadowStepsCap, sampleBudgetCap);

  return {
    librarySource,
    entrySource,
    entryName,
    bufferName,
    samplerFn,
    gridTypeId: opts.gridTypeId,
  };
}
