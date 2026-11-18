// The K1.4a generator variant matrix, a check kernel per variant, and the
// §3.5 size measure. Shared by scripts/check-wgsl-variants.mjs (browser
// compile) and packages/vdb/test/core/wgsl-generate.test.ts (static checks),
// so both cover the same variants.

import { gzipSync } from "node:zlib";

export const TYPE_SETS = [["float"], ["fp4"], ["fp8"], ["fp16"], ["fpn"], ["float", "fp8", "fpn"]];

/** The slot the check kernels use for `slot` mode, and the host binding for `buffer` mode. */
export const CHECK_SLOT = { group: 1, firstBinding: 3 };
/** What three's TSL emits for a storage node named `vdbPool` (design note §6). */
export const CHECK_BUFFER_EXPRESSION = "vdbPool.value";

/**
 * Every generator variant K1.4a must compile: read/sample × type set ×
 * slot/buffer × isActive on/off, plus sample with `sampler: 'nearest'` and
 * with `outside: 'zero'`. Returns `{ id, options }[]`.
 */
export function variantMatrix() {
  const out = [];
  for (const profile of ["read", "sample"]) {
    for (const types of TYPE_SETS) {
      for (const mode of ["slot", "buffer"]) {
        const binding = mode === "slot" ? { slot: CHECK_SLOT } : { buffer: CHECK_BUFFER_EXPRESSION };
        const extras = [{ isActive: true }, { isActive: false }];
        if (profile === "sample") extras.push({ sampler: "nearest" }, { outside: "zero" });
        for (const extra of extras) {
          const options = { ...binding, types, profile, ...extra };
          const tag = Object.entries(extra)
            .map(([k, v]) => `${k}=${v}`)
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
  }    checkOut[g] = s;
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
