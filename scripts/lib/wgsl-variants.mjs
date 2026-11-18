// The K1.4a/K1.4b generator variant matrix, a check kernel per variant, and the
// §3.5 size measure. Shared by scripts/check-wgsl-variants.mjs (browser
// compile) and packages/vdb/test/core/wgsl-generate.test.ts (static checks),
// so both cover the same variants.

import { gzipSync } from "node:zlib";

export const TYPE_SETS = [["float"], ["fp4"], ["fp8"], ["fp16"], ["fpn"], ["float", "fp8", "fpn"]];

/** The slot the check kernels use for `slot` mode, and the host binding for `buffer` mode. */
export const CHECK_SLOT = { group: 1, firstBinding: 3 };
/** What three's TSL emits for a storage node named `vdbPool` (design note §6). */
export const CHECK_BUFFER_EXPRESSION = "vdbPool.value";

/** The opt-in helpers (K1.7a) each profile can take: all of them in the matrix's helper variants. */
export const SAMPLE_HELPERS = ["rayClipSupport"];
export const MARCH_HELPERS = ["cellBoundTrilinear", "hddaCellEnd", "rayClipSupport"];
/** K3.2: the sparse volume source's module (`vdbSparseSource` with `lod: 'node-average'`). Its own variant, so the ids above keep their bytes. */
export const MARCH_FRAME_HELPERS = ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage"];
/** K3.2d: vdbSampleFast, alone on `sample` and in the sparse volume source's module (its own variants, appended). */
export const FAST_HELPERS = ["fastTrilinear"];
export const MARCH_FRAME_FAST_HELPERS = [...MARCH_FRAME_HELPERS, "fastTrilinear"];

/**
 * Every generator variant that must compile: read/sample/march × type set ×
 * slot/buffer × isActive on/off, plus sample with `sampler: 'nearest'` and
 * with `outside: 'zero'`, march with `outside: 'zero'` (trilinear and
 * nearest), and sample/march with every helper they allow. Returns
 * `{ id, options }[]`.
 */
export function variantMatrix() {
  const out = [];
  for (const profile of ["read", "sample", "march"]) {
    for (const types of TYPE_SETS) {
      for (const mode of ["slot", "buffer"]) {
        const binding = mode === "slot" ? { slot: CHECK_SLOT } : { buffer: CHECK_BUFFER_EXPRESSION };
        const extras = [{ isActive: true }, { isActive: false }];
        if (profile === "sample") extras.push({ sampler: "nearest" }, { outside: "zero" });
        if (profile === "sample" || profile === "march") extras.push({ outside: "zero", sampler: "nearest" });
        if (profile === "march") extras.push({ outside: "zero" });
        // K1.7a opt-in helpers: every helper the profile allows (appended, so the
        // ids above, and their snapshot entries, are unchanged).
        if (profile === "sample") extras.push({ helpers: SAMPLE_HELPERS });
        if (profile === "march") extras.push({ helpers: MARCH_HELPERS }, { helpers: MARCH_FRAME_HELPERS });
        if (profile === "sample") extras.push({ helpers: FAST_HELPERS });
        if (profile === "march") extras.push({ helpers: MARCH_FRAME_FAST_HELPERS });
        for (const extra of extras) {
          const options = { ...binding, types, profile, ...extra };
          const tag = Object.entries(extra)
            .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join("+") : v}`)
            .join(",");
          out.push({ id: `${profile}/${types.join("+")}/${mode}/${tag}`, options });
        }
      }
    }
  }
  return out;
}

/**
 * A complete compute shader around a generated module: the host binding for
 * `buffer` mode, an output buffer, and an entry point that calls every public
 * function of the profile so nothing is pruned before the backend compiles it.
 */
export function checkKernel(module, options) {
  const host =
    options.buffer !== undefined
      ? `struct vdbPoolStruct {
    value: array<u32>,
}
@group(${CHECK_SLOT.group}) @binding(${CHECK_SLOT.firstBinding}) var<storage, read> vdbPool: vdbPoolStruct;
`
      : "";
  const isActive = options.isActive ?? true;
  const sample = options.profile !== "read";
  const march = options.profile === "march";
  const helpers = new Set(options.helpers ?? []);
  const helperCalls = [
    helpers.has("rayClipSupport")
      ? `    var ts = vec2f(0.0, 1.0e30);
    if vdbRayClipSupport(g, p, d, &ts) {
        s += ts.y;
    }
`
      : "",
    helpers.has("cellBoundTrilinear") ? "    s += vdbCellBoundTrilinear(&a, ijk, 8u);\n" : "",
    helpers.has("hddaCellEnd") ? "    s += vdbHddaCellEnd(&h);\n" : "",
    helpers.has("nodeAverage") ? "    s += vdbNodeAverage(&a, ijk, 8u) + vdbNodeAverage(&a, ijk, 128u);\n" : "",
    helpers.has("fastTrilinear") ? "    s += vdbSampleFast(&a, p);\n" : "",
  ].join("");
  return `${module}
${host}@group(0) @binding(0) var<storage, read_write> checkOut: array<f32>;

@compute @workgroup_size(1)
fn checkMain(@builtin(global_invocation_id) gid: vec3u) {
    let g = gid.x;
    var s = f32(vdbGridCount()) + f32(vdbGridType(g));
    var a = vdbAccessorInit(g);
    let ijk = vec3i(i32(gid.y), 2, 3);
    s += vdbReadValue(&a, ijk);
    s += f32(vdbIndexMin(g).x + vdbIndexMax(g).y);
${isActive ? "    if vdbIsActive(&a, ijk) {\n        s += 1.0;\n    }\n" : ""}${
    sample
      ? `    let p = vdbWorldToIndex(g, vec3f(0.5, 1.5, f32(gid.z)));
    let d = vdbWorldToIndexDir(g, vec3f(0.0, 0.0, 1.0));
    s += vdbIndexToWorld(g, p).x + vdbIndexToWorldDir(g, d).y;
    s += vdbSample(&a, p);
    var t = vec2f(0.0, 1.0e30);
    if vdbRayClip(g, p, d, &t) {
        s += t.x;
    }
`
      : ""
  }${
    march
      ? `    var h: VdbHdda;
    vdbHddaInit(&h, p, t.x, d, t.y, vdbGetDim(&a, ijk));
    for (var i = 0u; i < 8u; i++) {
        let dim = max(vdbGetDim(&a, h.voxel), 8u);
        if vdbHddaUpdate(&h, p, d, dim) {
            continue;
        }
        s += vdbNodeMax(&a, h.voxel, dim) + h.tmin;
        if !vdbHddaStep(&h) {
            break;
        }
    }
    if vdbHasStats(g) {
        s += vdbValueMax(g);
    }
`
      : ""
  }${helperCalls}    checkOut[g] = s;
}
`;
}

/**
 * §3.5 size measure: "min" = comments stripped, whitespace collapsed (and
 * dropped next to punctuation), no identifier mangling; "gz" = gzip -9 of min.
 */
export function minifyWgsl(code) {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\n]*/g, "")
    .replace(/\s+/g, " ")
    .replace(/ ?([{}()[\];,:=<>+\-*/%&|^!~.@]) ?/g, "$1")
    .trim();
}

export function wgslSizes(code) {
  const min = minifyWgsl(code);
  return {
    raw: Buffer.byteLength(code),
    min: Buffer.byteLength(min),
    gz: gzipSync(Buffer.from(min), { level: 9 }).length,
  };
}
