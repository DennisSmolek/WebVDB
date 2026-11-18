import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  checkModule,
  VDBGridPool,
  VDB_GPU_LAYOUT_VERSION,
  VDB_MODULE_BANNER,
  VDBError,
  VDB_ALL_SCALAR_TYPES,
  VDB_GRID_TYPE_NAMES,
  vdbModuleEntries,
  vdbModuleWgsl,
  vdbPoolLayoutWgsl,
  type VDBModuleOptions,
  type VDBSlotModuleOptions,
} from "../../src/core/index.js";
import { vdbModuleDeclarations } from "../../src/core/wgsl/generate.js";
// @ts-expect-error: repo-level ESM helper without type declarations
import { CHECK_BUFFER_EXPRESSION, CHECK_SLOT, checkKernel, variantMatrix } from "../../../../scripts/lib/wgsl-variants.mjs";
import { FakeDevice } from "./fake-gpu.js";
import { makeImage } from "./synthetic-image.js";
import { wgslNameProblems } from "./wgsl-names.js";

const VARIANTS: { id: string; options: VDBModuleOptions }[] = variantMatrix();
const SLOT_DECLARATION = `@group(${CHECK_SLOT.group}) @binding(${CHECK_SLOT.firstBinding}) var<storage, read> vdbPool: array<u32>;\n`;

const count = (text: string, needle: string) => text.split(needle).length - 1;
const declaredNames = (code: string) => [...code.matchAll(/^(?:fn|struct|const|alias)\s+(\w+)/gm)].map((m) => m[1] as string);

function expectCode(fn: () => unknown, code: string) {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(VDBError);
    expect((e as VDBError).code).toBe(code);
    return;
  }
  throw new Error(`expected ${code}`);
}

describe("wgslNameProblems (helper)", () => {
  it("flags reserved words and function/variable clashes", () => {
    expect(wgslNameProblems("fn f(active: u32) -> u32 { return active; }")).toEqual(['"active" is a reserved word']);
    expect(wgslNameProblems("fn g(p: ptr<function, vdb_readaccessor_t>, set: vdb_buf_t) {}")).toEqual(['"set" is a reserved word']);
    expect(wgslNameProblems("@group(0) @binding(0) var<storage, read> h: array<u32>;\nfn h() {}")).toEqual(['"h" is both a function and a variable']);
  });
});

describe("vdbModuleWgsl: every variant", () => {
  it("covers the K1.4a + K1.4b matrix (read/sample/march x 6 type sets x slot/buffer x isActive, + nearest / outside zero) + K1.7a helpers", () => {
    expect(VARIANTS.filter((v) => v.options.helpers === undefined)).toHaveLength(132);
    expect(VARIANTS.filter((v) => v.options.helpers !== undefined)).toHaveLength(24);
  });

  it.each(VARIANTS)("$id is clean", ({ options }) => {
    const code = vdbModuleWgsl(options);
    // Deterministic: a second call with a fresh, reordered options object gives the same bytes.
    const again = vdbModuleWgsl({ ...options, types: [...options.types].reverse() });
    expect(again).toBe(code);

    expect(code.split("\n")[0]).toBe(VDB_MODULE_BANNER);
    expect(count(code, "SPDX-License-Identifier")).toBe(1);
    // The banner names "PNanoVDB.h"; no identifier keeps the vendored prefix.
    expect(code).not.toContain("pnanovdb");
    expect(code).not.toContain("PNANOVDB");
    expect(code).not.toContain("nanovdb_buffer");
    expect(code).not.toContain("`");
    expect(code).not.toMatch(/\/\/(?!.*SPDX)/); // the banner is the only comment
    expect(wgslNameProblems(code)).toEqual([]);
    expect(wgslNameProblems(checkKernel(code, options))).toEqual([]);

    // No duplicate declarations (the rename must not collide with the API layer).
    const names = declaredNames(code);
    expect(new Set(names).size).toBe(names.length);

    // Exactly one buffer access site.
    expect(count(code, "var<storage")).toBe(options.slot ? 1 : 0);
    expect(count(code, "fn vdb_load(")).toBe(1);
    if (options.slot) {
      expect(count(code, "vdbPool[")).toBe(1);
      expect([...code.matchAll(/\bvdbPool\b/g)]).toHaveLength(2); // declaration + access
    } else {
      expect(count(code, `${options.buffer}[`)).toBe(1);
      expect([...code.matchAll(/\bvdbPool\b/g)]).toHaveLength(1);
    }

    // Type table specialized; no dynamically indexed const array.
    expect(code).not.toMatch(/array<vdb_grid_type_constants_t/);
    expect(code).not.toMatch(/\[grid_type\]/);
  });
});

describe("vdbModuleWgsl: binding modes", () => {
  it.each(VARIANTS.filter((v) => v.options.slot))("$id: slot and buffer differ only in the declaration and the access site", ({ options }) => {
    const slot = vdbModuleWgsl(options);
    const buffer = vdbModuleWgsl({ ...options, slot: undefined, buffer: CHECK_BUFFER_EXPRESSION });
    expect(slot).toContain(SLOT_DECLARATION);
    const normalized = slot.replace(SLOT_DECLARATION, "").replace("return vdbPool[vdb_i];", `return ${CHECK_BUFFER_EXPRESSION}[vdb_i];`);
    expect(normalized).toBe(buffer);
  });

  it("declares the binding at the requested slot", () => {
    const code = vdbModuleWgsl({ slot: { group: 2, firstBinding: 20 }, types: ["float"], profile: "read" });
    expect(code).toContain("@group(2) @binding(20) var<storage, read> vdbPool: array<u32>;");
  });
});

describe("vdbModuleWgsl: profiles and compile-out", () => {
  const read = { slot: CHECK_SLOT, types: ["float"], profile: "read" } as const;

  it("read pulls in the design note's vendored roots and nothing of sample", () => {
    const decls = vdbModuleDeclarations(read);
    for (const root of [
      "pnanovdb_readaccessor_init",
      "pnanovdb_readaccessor_get_value_float",
      "pnanovdb_readaccessor_is_active",
      "pnanovdb_grid_get_tree",
      "pnanovdb_tree_get_root",
    ]) {
      expect(decls).toContain(root);
    }
    for (const absent of ["pnanovdb_sample_trilinear_typed", "pnanovdb_hdda_ray_clip", "pnanovdb_map_apply", "pnanovdb_hdda_zero_crossing"]) {
      expect(decls).not.toContain(absent);
    }
    const code = vdbModuleWgsl(read);
    const api = [...code.matchAll(/^fn (vdb[A-Z]\w*)/gm)].map((m) => m[1]);
    expect(api).toEqual(["vdbGridCount", "vdbGridType", "vdbIndexMin", "vdbIndexMax", "vdbAccessorInit", "vdbReadValue", "vdbIsActive"]);
    expect(code).toContain("struct VdbAccessor {");
  });

  it("sample adds transforms, vdbSample and vdbRayClip; the per-type trilinear wrappers are dropped", () => {
    const options = { ...read, profile: "sample" } as const;
    const decls = vdbModuleDeclarations(options);
    expect(decls).toContain("pnanovdb_sample_trilinear_typed");
    expect(decls).toContain("pnanovdb_hdda_ray_clip");
    for (const dropped of ["pnanovdb_sample_trilinear_float", "pnanovdb_sample_trilinear_fp8", "pnanovdb_sample_trilinear_fpn", "pnanovdb_map_apply"]) {
      expect(decls).not.toContain(dropped);
    }
    const api = [...vdbModuleWgsl(options).matchAll(/^fn (vdb[A-Z]\w*)/gm)].map((m) => m[1]);
    expect(api.slice(7)).toEqual(["vdbWorldToIndex", "vdbWorldToIndexDir", "vdbIndexToWorld", "vdbIndexToWorldDir", "vdbSample", "vdbRayClip"]);
  });

  it("isActive: false drops the is_active family", () => {
    const code = vdbModuleWgsl({ ...read, isActive: false });
    expect(code).not.toContain("vdbIsActive");
    expect(code).not.toContain("is_active");
    expect(code.length).toBeLessThan(vdbModuleWgsl(read).length - 3000);
  });

  it("sampler: 'nearest' rounds to the closest voxel and drops the trilinear path", () => {
    const code = vdbModuleWgsl({ ...read, profile: "sample", sampler: "nearest" });
    expect(code).toContain("vec3i(floor(xyz + vec3f(0.5)))");
    expect(code).not.toContain("vdb_sample_trilinear_typed");
  });

  it("outside: 'zero' zeroes the field outside the index bbox [min, max] carried in the accessor", () => {
    const zero = vdbModuleWgsl({ ...read, profile: "sample", outside: "zero" });
    expect(zero).toMatch(/lo: vec3i,\n {4}hi: vec3i,/);
    // Trilinear: every tap outside [lo, hi] reads 0; no tap inside -> 0 early.
    expect(zero).toContain("fn vdb_tap0(");
    expect(zero).toContain("if any(ijk < (*acc).lo) || any(ijk > (*acc).hi) {");
    expect(zero).toContain("any(b < (*acc).lo - vec3i(1)) || any(b > (*acc).hi)");
    expect(zero).not.toContain("vdb_sample_trilinear_typed");
    // Nearest: the rounded voxel must be inside [lo, hi].
    const nearest = vdbModuleWgsl({ ...read, profile: "sample", outside: "zero", sampler: "nearest" });
    expect(nearest).toContain("let ijk = vec3i(floor(xyz + vec3f(0.5)));\n    if !(*acc).ok || any(ijk < (*acc).lo) || any(ijk > (*acc).hi) {");
    expect(nearest).not.toContain("vdb_tap0");
    expect(vdbModuleWgsl({ ...read, profile: "sample" })).not.toMatch(/\(\*acc\)\.(lo|hi)\b/);
  });

  it("every per-grid function checks the id is live and below vdbGridCount()", () => {
    const code = vdbModuleWgsl({ ...read, profile: "march" });
    expect(code).toContain("fn vdb_live(g: u32) -> bool {\n    return g < vdbGridCount() && (vdb_rec(g, VDB_REC_FLAGS) & VDB_REC_FLAG_LIVE) != 0u;\n}");
    for (const fn of ["vdbGridType", "vdbIndexMin", "vdbIndexMax", "vdb_affine", "vdbRayClip", "vdbValueMax"]) {
      const body = new RegExp(`fn ${fn}\\([^)]*\\)[^{]*\\{\\n {4}if !vdb_live\\(g\\)`);
      expect(code, fn).toMatch(body);
    }
    expect(code).toContain("return vdb_live(g) && (vdb_rec(g, VDB_REC_FLAGS) & VDB_REC_FLAG_HAS_STATS) != 0u;");
    // A dead id gives an empty box (max < min) and no ray hit.
    expect(code).toMatch(/fn vdbIndexMax[^{]*\{\n {4}if !vdb_live\(g\) \{\n {8}return vec3i\(-1\);/);
    expect(code).toMatch(/fn vdbRayClip[^{]*\{\n {4}if !vdb_live\(g\) \{\n {8}return false;/);
  });

  it("march adds the HDDA, vdbGetDim, node stats and the record's stats flag / valueMax", () => {
    const options = { ...read, profile: "march" } as const;
    const decls = vdbModuleDeclarations(options);
    for (const root of [
      "pnanovdb_hdda_init",
      "pnanovdb_hdda_update",
      "pnanovdb_hdda_step",
      "pnanovdb_readaccessor_get_dim",
      "pnanovdb_root_get_max_float",
      "pnanovdb_upper_get_max_float",
      "pnanovdb_lower_get_max_float",
      "pnanovdb_leaf_get_max_float",
      "pnanovdb_sample_trilinear_typed",
    ]) {
      expect(decls).toContain(root);
    }
    for (const absent of ["pnanovdb_hdda_zero_crossing", "pnanovdb_leaf_get_min_float", "pnanovdb_leaf_get_ave_float", "pnanovdb_hdda_pos_to_ijk"]) {
      expect(decls).not.toContain(absent);
    }
    const code = vdbModuleWgsl(options);
    const api = [...code.matchAll(/^fn (vdb[A-Z]\w*)/gm)].map((m) => m[1]);
    expect(api.slice(13)).toEqual(["vdbHddaInit", "vdbHddaUpdate", "vdbHddaStep", "vdbGetDim", "vdbHasStats", "vdbValueMax", "vdbNodeMax"]);
    expect(code).toContain("alias VdbHdda = vdb_hdda_t;");
    expect(code).toMatch(/stats: bool,\n {4}vmax: f32,/);
    expect(code).toContain("a.vmax = bitcast<f32>(vdb_rec(g, VDB_REC_VALUE_MAX));");
    expect(code).toContain("const VDB_REC_VALUE_MAX: u32 = 35u;");
    expect(code).toContain("const VDB_REC_FLAG_HAS_STATS: u32 = 0x2u;");
  });

  it("march leaf stats: Float only reads f32, Fp only decodes uint16 codes, mixed branches on the compiled Fp types", () => {
    const march = { ...read, profile: "march" } as const;
    const float = vdbModuleWgsl(march);
    expect(float).toMatch(/fn vdb_leaf_get_stat_float\([^)]*\) -> f32 \{\n {4}return vdb_read_float\(buf, address\);\n\}/);
    expect(float).not.toContain("vdb_read_uint16");
    expect(float).not.toContain("vdb_leaf_fp_get_stat_float");

    const fpn = vdbModuleWgsl({ ...march, types: ["fpn"] });
    expect(fpn).toMatch(/fn vdb_leaf_get_stat_float\([^)]*\) -> f32 \{\n {4}return vdb_leaf_fp_get_stat_float\(grid_type, buf, node, address, with_minimum\);\n\}/);
    expect(fpn).toContain("fn vdb_read_uint16(");
    expect(fpn).toContain("return (raw >> ((address & 2u) << 3u)) & 65535u;");

    const mixed = vdbModuleWgsl({ ...march, types: ["float", "fp8", "fpn"] });
    expect(mixed).toContain("    if grid_type == VDB_GRID_TYPE_FP8 || grid_type == VDB_GRID_TYPE_FPN {\n        return vdb_leaf_fp_get_stat_float(");
  });

  it("sample-only options do not change a read module", () => {
    expect(vdbModuleWgsl({ ...read, sampler: "nearest", outside: "zero" })).toBe(vdbModuleWgsl(read));
  });

  it("Float only: no Fp decoders, one type-table row, no branch on type in the decode", () => {
    const code = vdbModuleWgsl(read);
    expect(code).not.toContain("vdb_leaf_fp");
    expect(code).not.toContain("VDB_GRID_TYPE_FP");
    expect(code).toMatch(/fn vdb_gtc\(grid_type: u32\) -> vdb_grid_type_constants_t \{\n {4}return vdb_grid_type_constants_t\(/);
    expect(code).toMatch(/fn vdb_read_float_typed\([^)]*\) -> f32 \{\n {4}return vdb_read_float\(buf, address\);\n\}/);
  });

  it("Fp8 only: the Fp8 decoder alone, as the unconditional decode", () => {
    const code = vdbModuleWgsl({ ...read, types: ["fp8"] });
    expect(code).toContain("fn vdb_leaf_fp8_read_float");
    for (const other of ["fp4", "fp16", "fpn"]) expect(code).not.toContain(`vdb_leaf_${other}_read_float`);
    expect(code).not.toContain("VDB_GRID_TYPE_FLOAT");
    expect(code).toMatch(/fn vdb_read_float_typed\([^)]*\) -> f32 \{\n {4}return vdb_root_fp8_read_float\(buf, address, ijk, level\);\n\}/);
  });

  it("Float+Fp8+FpN: one branch per decoder, type-table rows shared where equal", () => {
    const code = vdbModuleWgsl({ ...read, types: ["fpn", "float", "fp8"] });
    expect(code).toContain("if grid_type == VDB_GRID_TYPE_FP8 {");
    expect(code).toContain("if grid_type == VDB_GRID_TYPE_FPN {");
    expect(code).toContain("return t == VDB_GRID_TYPE_FLOAT || t == VDB_GRID_TYPE_FP8 || t == VDB_GRID_TYPE_FPN;");
    const gtc = /fn vdb_gtc[\s\S]*?\n\}/.exec(code)?.[0] ?? "";
    // Float and the Fp types differ (value_stride_bits); Fp8 and FpN share a row.
    expect(count(gtc, "vdb_grid_type_constants_t(")).toBe(2);
  });

  it("keeps every type-table field the emitted code reads", () => {
    const fields = [
      "root_off_background",
      "root_size",
      "value_stride_bits",
      "table_stride",
      "root_tile_off_value",
      "root_tile_size",
      "upper_off_table",
      "lower_off_table",
      "leaf_off_table",
    ];
    const code = vdbModuleWgsl({ ...read, profile: "sample", types: ["float", "fp8"] });
    for (const f of fields) expect(code).toContain(`.${f}`);
    // Float row, vendored order: background 28, root_size 64, stride bits 32, table_stride 8, tile value 20, tile size 32, upper table 8256, lower table 1088, leaf table 96.
    expect(code).toContain("vdb_grid_type_constants_t(28u, 0u, 0u, 0u, 0u, 64u, 32u, 8u, 20u, 32u, 0u, 0u, 0u, 0u, 8256u, 0u, 0u, 0u, 0u, 0u, 1088u, 0u, 0u, 0u, 0u, 0u, 96u, 0u)");
  });

  it("emits only the layout constants it uses", () => {
    const code = vdbModuleWgsl(read);
    for (const m of code.matchAll(/^const (VDB_(?:POOL|REC)_\w+)/gm)) {
      expect(count(code, m[1] as string)).toBeGreaterThan(1);
    }
    expect(code).toContain("const VDB_POOL_TABLE_WORD: u32 = 8u;");
    expect(code).toContain("const VDB_POOL_RECORD_WORDS: u32 = 40u;");
    expect(code).not.toContain("VDB_REC_VALUE_MAX");
  });
});

describe("vdbModuleWgsl: options", () => {
  const base = { slot: CHECK_SLOT, types: ["float"], profile: "read" } as const;

  it("rejects bad options with VDB_BAD_OPTION", () => {
    const cases: unknown[] = [
      { ...base, slot: undefined },
      { ...base, buffer: "vdbPool.value" },
      { ...base, slot: undefined, buffer: "vdbPool[0]" },
      { ...base, slot: undefined, buffer: "" },
      { ...base, slot: { group: -1, firstBinding: 0 } },
      { ...base, slot: { group: 0 } },
      { ...base, group: 0, firstBinding: 1 },
      { ...base, types: [] },
      { ...base, types: "float" },
      { ...base, types: ["half"] },
      { ...base, profile: "majorant" },
      { ...base, slot: undefined, buffer: "__vdbPool.value" },
      { ...base, slot: undefined, buffer: "vdbPool.__value" },
      { ...base, slot: undefined, buffer: "loop.value" },
      { ...base, slot: undefined, buffer: "vdbPool.active" },
      { ...base, slot: undefined, buffer: "_" },
      { ...base, profile: "full" },
      { ...base, profile: "sample", sampler: "cubic" },
      { ...base, profile: "sample", outside: "clamp" },
      { ...base, isActive: "yes" },
      { ...base, profile: "march", helpers: "hddaCellEnd" },
      { ...base, profile: "march", helpers: ["cellBound"] },
      { ...base, profile: "march", helpers: [null] },
      { ...base, profile: "sample", helpers: ["cellBoundTrilinear"] },
      { ...base, profile: "sample", helpers: ["hddaCellEnd"] },
      { ...base, profile: "read", helpers: ["rayClipSupport"] },
      null,
    ];
    for (const options of cases) expectCode(() => vdbModuleWgsl(options as VDBModuleOptions), "VDB_BAD_OPTION");
  });

  it("says majorant is deferred past 0.1 (D22)", () => {
    expect(() => vdbModuleWgsl({ ...base, profile: "majorant" })).toThrow(/deferred past 0\.1 \(D22\)/);
  });

  it("names the offending part of a reserved buffer expression", () => {
    expect(() => vdbModuleWgsl({ ...base, slot: undefined, buffer: "vdbPool.set" })).toThrow(/'set', which WGSL reserves/);
    expect(() => vdbModuleWgsl({ ...base, slot: undefined, buffer: "__pool" })).toThrow(/must not start with '__'/);
    expect(vdbModuleWgsl({ ...base, slot: undefined, buffer: "pool_.value_" })).toContain("return pool_.value_[vdb_i];");
  });

  it("treats types as a set", () => {
    expect(vdbModuleWgsl({ ...base, types: new Set(["fp8", "float", "fp8"] as const) })).toBe(vdbModuleWgsl({ ...base, types: ["float", "fp8"] }));
  });
});

describe("vdbModuleEntries", () => {
  it("is one read-only storage entry at firstBinding, visible to compute and fragment by default", () => {
    const want = [{ binding: 20, visibility: 0x4 | 0x2, buffer: { type: "read-only-storage" } }];
    expect(vdbModuleEntries({ slot: { group: 2, firstBinding: 20 } })).toEqual(want);
    // The bare VDBSlot, the same value vdbModuleWgsl({ slot }) and pool.bindGroupEntry(slot) take.
    expect(vdbModuleEntries({ group: 2, firstBinding: 20 })).toEqual(want);
  });

  it("an explicit visibility overrides the default", () => {
    expect(vdbModuleEntries({ slot: { group: 0, firstBinding: 1 }, visibility: 0x2 })[0]?.visibility).toBe(0x2);
    expect(vdbModuleEntries({ slot: { group: 0, firstBinding: 1 }, visibility: 0x4 })[0]?.visibility).toBe(0x4);
  });

  it("takes a module's slot without a non-null assertion", () => {
    const options: VDBSlotModuleOptions = { slot: { group: 1, firstBinding: 7 }, types: ["float"], profile: "read" };
    expect(vdbModuleEntries(options.slot)[0]?.binding).toBe(7);
    expect(vdbModuleEntries({ slot: options.slot, visibility: 0x4 })[0]?.binding).toBe(7);
  });

  it("rejects malformed slots and mixed shapes", () => {
    expectCode(() => vdbModuleEntries({ group: 0 } as never), "VDB_BAD_OPTION");
    expectCode(() => vdbModuleEntries({ group: 0, firstBinding: -1 }), "VDB_BAD_OPTION");
    expectCode(() => vdbModuleEntries({ group: 0, firstBinding: 1, visibility: 0x2 } as never), "VDB_BAD_OPTION");
    expectCode(() => vdbModuleEntries({ slot: { group: 0, firstBinding: 1 }, group: 0 } as never), "VDB_BAD_OPTION");
    expectCode(() => vdbModuleEntries({ slot: { group: 0, firstBinding: 1 }, visibility: 0 }), "VDB_BAD_OPTION");
    expectCode(() => vdbModuleEntries(null as never), "VDB_BAD_OPTION");
  });
});

describe("opt-in helpers (K1.7a)", () => {
  const march = { slot: CHECK_SLOT, types: ["float", "fp8"], profile: "march" } as const;
  const HELPER_FNS = ["vdbCellBoundTrilinear", "vdbHddaCellEnd", "vdbRayClipSupport"];

  it("emit nothing unless requested: no helpers and helpers: [] give the same bytes", () => {
    const plain = vdbModuleWgsl(march);
    expect(vdbModuleWgsl({ ...march, helpers: [] })).toBe(plain);
    for (const fn of HELPER_FNS) expect(plain).not.toContain(fn);
    for (const v of VARIANTS.filter((x) => x.options.helpers === undefined)) {
      const code = vdbModuleWgsl(v.options);
      for (const fn of HELPER_FNS) expect(code, v.id).not.toContain(fn);
    }
  });

  it("each helper appends exactly its own function to the plain module", () => {
    const plain = vdbModuleWgsl(march);
    for (const [helper, fn] of [
      ["cellBoundTrilinear", "vdbCellBoundTrilinear"],
      ["hddaCellEnd", "vdbHddaCellEnd"],
      ["rayClipSupport", "vdbRayClipSupport"],
    ] as const) {
      const code = vdbModuleWgsl({ ...march, helpers: [helper] });
      expect(code.startsWith(plain.slice(0, -1)), helper).toBe(true);
      expect(declaredNames(code.slice(plain.length - 1))).toEqual([fn]);
    }
  });

  it("are a set: order and duplicates don't change the bytes", () => {
    expect(vdbModuleWgsl({ ...march, helpers: ["hddaCellEnd", "cellBoundTrilinear", "hddaCellEnd"] })).toBe(
      vdbModuleWgsl({ ...march, helpers: new Set(["cellBoundTrilinear", "hddaCellEnd"] as const) }),
    );
  });

  it("cellBoundTrilinear is the 2x2x2 vdbNodeMax block; hddaCellEnd is min(next, tmax)", () => {
    const code = vdbModuleWgsl({ ...march, helpers: ["cellBoundTrilinear", "hddaCellEnd"] });
    expect(code).toContain("fn vdbCellBoundTrilinear(acc: ptr<function, VdbAccessor>, ijk: vec3i, dim: u32) -> f32 {");
    expect(count(code, "vdbNodeMax(acc, ijk")).toBe(2); // the cell, then the 7 neighbours in a loop
    expect(code).toContain("fn vdbHddaCellEnd(h: ptr<function, VdbHdda>) -> f32 {\n    return min(min((*h).next.x, (*h).next.y), min((*h).next.z, (*h).tmax));");
  });

  it("rayClipSupport pads the index box by the sampler's support (1 voxel trilinear, 0.5 nearest)", () => {
    const sample = { slot: CHECK_SLOT, types: ["float"], profile: "sample", helpers: ["rayClipSupport"] } as const;
    expect(vdbModuleWgsl(sample)).toContain("vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)) - vec3f(1.0), vec3f(vdbIndexMax(g)) + vec3f(1.0),");
    expect(vdbModuleWgsl({ ...sample, sampler: "nearest" })).toContain("vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)) - vec3f(0.5), vec3f(vdbIndexMax(g)) + vec3f(0.5),");
    // The plain vdbRayClip keeps its voxel-cell box.
    expect(vdbModuleWgsl(sample)).toContain("vdb_hdda_ray_clip(vec3f(vdbIndexMin(g)), vec3f(vdbIndexMax(g) + vec3i(1)),");
  });

  it("say which profile a helper needs", () => {
    expect(() => vdbModuleWgsl({ ...march, profile: "sample", helpers: ["hddaCellEnd"] })).toThrow(/helper 'hddaCellEnd' needs profile 'march'/);
    expect(() => vdbModuleWgsl({ ...march, profile: "read", helpers: ["rayClipSupport"] })).toThrow(/helper 'rayClipSupport' needs profile 'sample'/);
  });
});

describe("VDB_ALL_SCALAR_TYPES", () => {
  it("is every grid type the pool can hold, and generates a module checkModule accepts for any pool", () => {
    expect([...VDB_ALL_SCALAR_TYPES]).toEqual([...VDB_GRID_TYPE_NAMES]);
    expect(Object.isFrozen(VDB_ALL_SCALAR_TYPES)).toBe(true);
    const options = { slot: CHECK_SLOT, types: VDB_ALL_SCALAR_TYPES, profile: "march" } as const;
    expect(vdbModuleWgsl(options)).toContain("fn vdbNodeMax(");
    expect(() => checkModule({ types: VDB_GRID_TYPE_NAMES, layoutVersion: VDB_GPU_LAYOUT_VERSION }, options)).not.toThrow();
  });
});

describe("checkModule", () => {
  const options = { slot: CHECK_SLOT, types: ["float", "fp8"], profile: "sample" } as const;

  it("accepts a pool whose types the module compiles", () => {
    expect(() => checkModule({ types: new Set(["fp8"] as const), layoutVersion: 1 }, options)).not.toThrow();
    expect(() => checkModule({ types: [], layoutVersion: 1 }, options)).not.toThrow();
  });

  it("throws VDB_TYPE_NOT_COMPILED naming the missing types", () => {
    expectCode(() => checkModule({ types: ["float", "fpn"], layoutVersion: 1 }, options), "VDB_TYPE_NOT_COMPILED");
    expect(() => checkModule({ types: ["fpn"], layoutVersion: 1 }, options)).toThrow(/fpn/);
  });

  it("takes just { types, layoutVersion? } for a type/layout-only check (engine adapters)", () => {
    const pool = { types: ["float", "fp8"] as const, layoutVersion: VDB_GPU_LAYOUT_VERSION };
    expect(() => checkModule(pool, { types: ["float", "fp8", "fpn"] })).not.toThrow();
    expect(() => checkModule(pool, { types: new Set(["fp8", "float"] as const), layoutVersion: VDB_GPU_LAYOUT_VERSION })).not.toThrow();
    expectCode(() => checkModule(pool, { types: ["float"] }), "VDB_TYPE_NOT_COMPILED");
    expectCode(() => checkModule(pool, { types: ["float", "fp8"], layoutVersion: VDB_GPU_LAYOUT_VERSION + 1 }), "VDB_LAYOUT_MISMATCH");
    expectCode(() => checkModule(pool, { types: [] }), "VDB_BAD_OPTION");
    expectCode(() => checkModule(pool, { types: ["half"] } as never), "VDB_BAD_OPTION");
    expectCode(() => checkModule(pool, { types: ["float"], layoutVersion: -1 }), "VDB_BAD_OPTION");
    // The full form is still validated as module options.
    expectCode(() => checkModule(pool, { types: ["float", "fp8"], profile: "read" } as never), "VDB_BAD_OPTION");
  });

  it("VDB_ENGINE_MISMATCH is a VDBError code", () => {
    expect(new VDBError("VDB_ENGINE_MISMATCH", "x").code).toBe("VDB_ENGINE_MISMATCH");
  });

  it("says to generate for every type the host may add, or regenerate and rebuild the pipeline", () => {
    expect(() => checkModule({ types: ["fpn"], layoutVersion: 1 }, options)).toThrow(/every type the host may add/);
    expect(() => checkModule({ types: ["fpn"], layoutVersion: 1 }, options)).toThrow(/VDB_ALL_SCALAR_TYPES/);
    expect(() => checkModule({ types: ["fpn"], layoutVersion: 1 }, options)).toThrow(/rebuild the pipeline/);
  });

  it("throws VDB_LAYOUT_MISMATCH on another layout version", () => {
    expectCode(() => checkModule({ types: ["float"], layoutVersion: 2 }, options), "VDB_LAYOUT_MISMATCH");
  });

  it("validates the options too", () => {
    expectCode(() => checkModule({ types: ["float"], layoutVersion: 1 }, { ...options, types: [] }), "VDB_BAD_OPTION");
  });

  it("compares against the layout.ts version, not a copy", () => {
    expectCode(() => checkModule({ types: ["float"], layoutVersion: VDB_GPU_LAYOUT_VERSION + 1 }, options), "VDB_LAYOUT_MISMATCH");
    expect(() => checkModule({ types: ["float"], layoutVersion: VDB_GPU_LAYOUT_VERSION }, options)).not.toThrow();
  });

  it("accepts a real VDBGridPool structurally", async () => {
    const pool = await VDBGridPool.create(new FakeDevice().asGPU(), { initialBytes: 16384 });
    pool.add(makeImage({ bytes: 1024, gridType: 1 }));
    pool.add(makeImage({ bytes: 1024, gridType: 14 }));
    expect([...pool.types].sort()).toEqual(["float", "fp8"]);
    // The right set passes, both for the exact set and a superset.
    expect(() => checkModule(pool, { slot: CHECK_SLOT, types: [...pool.types], profile: "read" })).not.toThrow();
    expect(() => checkModule(pool, { slot: CHECK_SLOT, types: ["float", "fp8", "fpn"], profile: "sample" })).not.toThrow();
    // A narrower set names what is missing.
    expectCode(() => checkModule(pool, { slot: CHECK_SLOT, types: ["float"], profile: "read" }), "VDB_TYPE_NOT_COMPILED");
    expect(() => checkModule(pool, { slot: CHECK_SLOT, types: ["float"], profile: "read" })).toThrow(/fp8/);
    pool.dispose();
  });
});

describe("layout constants come from core/pool/layout.ts", () => {
  it("every emitted VDB_ const line is a line of vdbPoolLayoutWgsl()", () => {
    const layoutLines = new Set(vdbPoolLayoutWgsl().split("\n"));
    const names = Object.keys(Object.fromEntries([...vdbPoolLayoutWgsl().matchAll(/^const (\w+)/gm)].map((m) => [m[1], 1])));
    for (const v of VARIANTS) {
      const code = vdbModuleWgsl(v.options);
      for (const m of code.matchAll(/^const ((?:VDB_REC_|VDB_HEADER_|VDB_POOL_|VDB_GPU_)\w*): .*$/gm)) {
        expect(layoutLines.has(m[0]), `${v.id}: ${m[1]}`).toBe(true);
        expect(names).toContain(m[1]);
      }
    }
  });

  it("the WGSL names the API layers use exist in layout.ts and are emitted", () => {
    const code = vdbModuleWgsl({ slot: CHECK_SLOT, types: ["float"], profile: "sample" });
    for (const n of ["VDB_POOL_TABLE_WORD", "VDB_POOL_RECORD_WORDS", "VDB_HEADER_GRID_COUNT", "VDB_REC_FLAGS", "VDB_REC_FLAG_LIVE", "VDB_REC_GRID_TYPE", "VDB_REC_IMAGE_OFFSET", "VDB_REC_WORLD_TO_INDEX"]) {
      expect(code, n).toContain(`const ${n}: u32 =`);
    }
  });
});

describe("VENDOR.md generation transforms", () => {
  it("logs every declaration the generator replaces", async () => {
    const doc = await readFile(new URL("../../vendor/VENDOR.md", import.meta.url), "utf8");
    expect(doc).toContain("## Generation transforms");
    for (const name of ["pnanovdb_buf_read_uint32", "pnanovdb_grid_type_constants", "pnanovdb_read_float_typed", "pnanovdb_sample_trilinear_{float,fp8,fpn}"]) {
      expect(doc).toContain(name);
    }
  });
});
