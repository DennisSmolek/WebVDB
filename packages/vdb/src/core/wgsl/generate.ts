/**
 * WGSL module generator (docs/design/K1-core.md §3, task K1.4a).
 *
 * Roots for the profile and options (the vendored names the API layer uses)
 * -> closure over the declaration table extracted from the unmodified vendored
 * `pnanovdb.wgsl` -> generation transforms -> `pnanovdb_`→`vdb_` rename ->
 * append the API layer -> prepend one SPDX line.
 *
 * Generation transforms (each one is a row in vendor/VENDOR.md, "Generation
 * transforms"; none edits the vendored file):
 *   1. `pnanovdb_buf_read_uint32` reads through `vdb_load`, the module's single
 *      buffer access site (§2).
 *   2. The 27-row `pnanovdb_grid_type_constants` array becomes
 *      `fn vdb_gtc(grid_type)` returning only the compiled types' rows; its
 *      `[grid_type]` call sites become `vdb_gtc(grid_type)`.
 *   3. `pnanovdb_read_float_typed` keeps only the compiled types' branches, so
 *      the other decoders drop out of the closure.
 *   4. Declarations unreachable from the profile's roots are not emitted (the
 *      `pnanovdb_sample_trilinear_{float,fp8,fpn}` wrappers among them).
 *   5. Every `pnanovdb_` / `PNANOVDB_` becomes `vdb_` / `VDB_`.
 *   6. Comments are dropped (done at extraction).
 *
 * Output is deterministic: the same options give the same bytes, and `types`
 * is a set (order and duplicates do not matter).
 */

import { VDBError } from "../errors.js";
import { apiWgsl, type ApiOptions } from "./api.js";
import { PNANOVDB_CHUNKS, type PnanovdbChunk } from "./pnanovdb.chunks.generated.js";
import { recordConstantsWgsl, VDB_GPU_LAYOUT_VERSION } from "./record-constants.js";
import {
  VDB_GRID_TYPE_IDS,
  VDB_GRID_TYPE_NAMES,
  type VDBGridTypeName,
  type VDBModuleOptions,
  type VDBModulePoolLike,
  type VDBModuleProfile,
  type VDBSlot,
} from "./types.js";

/** First line of every generated module (§3.3). */
export const VDB_MODULE_BANNER =
  "// SPDX-License-Identifier: Apache-2.0 AND MIT. Derived from PNanoVDB.h (OpenVDB) via emcfarlane/webgpu-nanovdb; see NOTICE.";

/** Name of the storage binding `slot` mode declares. */
export const VDB_POOL_BINDING_NAME = "vdbPool";

/** Profiles in inclusion order; only the first `IMPLEMENTED_PROFILES` generate today. */
const PROFILES: readonly VDBModuleProfile[] = ["read", "sample", "march", "majorant"];
const IMPLEMENTED_PROFILES = 2;

/** `GPUShaderStage.COMPUTE`; the global does not exist in Node. */
const SHADER_STAGE_COMPUTE = 0x4;

const IDENTIFIER = /[A-Za-z_][A-Za-z0-9_]*/g;
const BUFFER_EXPRESSION = /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$/;

type Binding = { readonly kind: "slot"; readonly slot: VDBSlot } | { readonly kind: "buffer"; readonly expression: string };

interface NormalizedOptions extends ApiOptions {
  readonly profile: VDBModuleProfile;
  readonly profileRank: number;
  readonly binding: Binding;
}

function bad(message: string): VDBError {
  return new VDBError("VDB_BAD_OPTION", message);
}

function checkSlot(slot: unknown, where: string): VDBSlot {
  const s = slot as Partial<VDBSlot> | null | undefined;
  if (
    s === null ||
    typeof s !== "object" ||
    !Number.isInteger(s.group) ||
    !Number.isInteger(s.firstBinding) ||
    (s.group as number) < 0 ||
    (s.firstBinding as number) < 0
  ) {
    throw bad(`${where}: slot must be { group, firstBinding } with non-negative integers, got ${JSON.stringify(slot)}.`);
  }
  return { group: s.group as number, firstBinding: s.firstBinding as number };
}

/** The old bvh-style positional slot (`{ group, firstBinding }` at the top level) would be silently misread. */
function rejectPositionalSlot(where: string, options: object): void {
  if ("group" in options || "firstBinding" in options) {
    throw bad(`${where}: got 'group' or 'firstBinding' at the top level. Pass { slot: { group, firstBinding } }.`);
  }
}

function normalizeTypes(types: unknown, where: string): VDBGridTypeName[] {
  if (types === undefined || types === null || typeof types === "string" || typeof (types as Iterable<unknown>)[Symbol.iterator] !== "function") {
    throw bad(`${where}: types must be an array or set of grid type names (${VDB_GRID_TYPE_NAMES.join(", ")}), e.g. [...pool.types].`);
  }
  const seen = new Set<string>();
  for (const t of types as Iterable<unknown>) {
    if (typeof t !== "string" || !(VDB_GRID_TYPE_NAMES as readonly string[]).includes(t)) {
      throw bad(`${where}: unknown grid type ${JSON.stringify(t)}; use ${VDB_GRID_TYPE_NAMES.join(", ")}.`);
    }
    seen.add(t);
  }
  if (seen.size === 0) {
    throw bad(`${where}: types is empty. Compile at least one grid type, e.g. ['float'], or pass [...pool.types] after adding a grid.`);
  }
  return VDB_GRID_TYPE_NAMES.filter((t) => seen.has(t));
}

function normalize(options: VDBModuleOptions, where: string): NormalizedOptions {
  if (options === null || typeof options !== "object") {
    throw bad(`${where}: expected an options object { slot | buffer, types, profile }.`);
  }
  rejectPositionalSlot(where, options);

  const hasSlot = options.slot !== undefined;
  const hasBuffer = options.buffer !== undefined;
  if (hasSlot === hasBuffer) {
    throw bad(`${where}: pass exactly one of slot: { group, firstBinding } (declares the binding) or buffer: '<wgsl expression>' (reads a host-declared array<u32>).`);
  }
  let binding: Binding;
  if (hasSlot) {
    binding = { kind: "slot", slot: checkSlot(options.slot, where) };
  } else {
    const expression = options.buffer;
    if (typeof expression !== "string" || !BUFFER_EXPRESSION.test(expression)) {
      throw bad(`${where}: buffer must be a WGSL identifier or member chain such as 'vdbPool.value', got ${JSON.stringify(expression)}.`);
    }
    binding = { kind: "buffer", expression };
  }

  const profileRank = PROFILES.indexOf(options.profile);
  if (profileRank < 0) {
    throw bad(`${where}: profile must be one of ${PROFILES.map((p) => `'${p}'`).join(", ")}, got ${JSON.stringify(options.profile)}.`);
  }
  if (profileRank >= IMPLEMENTED_PROFILES) {
    throw bad(`${where}: profile '${options.profile}' lands in K1.4b. Use 'read' or 'sample' for now.`);
  }

  const sampler = options.sampler ?? "trilinear";
  if (sampler !== "trilinear" && sampler !== "nearest") {
    throw bad(`${where}: sampler must be 'trilinear' or 'nearest', got ${JSON.stringify(sampler)}.`);
  }
  const outside = options.outside ?? "background";
  if (outside !== "background" && outside !== "zero") {
    throw bad(`${where}: outside must be 'background' or 'zero', got ${JSON.stringify(outside)}.`);
  }
  const isActive = options.isActive ?? true;
  if (typeof isActive !== "boolean") {
    throw bad(`${where}: isActive must be a boolean, got ${JSON.stringify(isActive)}.`);
  }

  // Sample-only options do not change a `read` module; normalize them away so
  // equal modules share one cache key.
  const sampleOnly = profileRank >= 1;
  return {
    profile: options.profile,
    profileRank,
    binding,
    types: normalizeTypes(options.types, where),
    isActive,
    sampler: sampleOnly ? sampler : "trilinear",
    outside: sampleOnly ? outside : "background",
  };
}

// ---------------------------------------------------------------------------
// Declaration table and transforms

const CHUNKS = new Map<string, PnanovdbChunk>(PNANOVDB_CHUNKS.map((c) => [c.name, c]));
const ORDER = new Map<string, number>(PNANOVDB_CHUNKS.map((c, i) => [c.name, i]));

function chunk(name: string): PnanovdbChunk {
  const c = CHUNKS.get(name);
  if (!c) throw new Error(`@pmndrs/vdb: vendored declaration ${name} is missing; re-run scripts/extract-wgsl-chunks.mjs`);
  return c;
}

const TYPE_TABLE = "pnanovdb_grid_type_constants";
const TYPE_TABLE_FN = "pnanovdb_gtc";
const TYPE_TABLE_CALL_SITE = /\bpnanovdb_grid_type_constants\[([^\]]+)\]/g;
/** Names the transforms introduce, resolved to the declaration they replace. */
const ALIASES = new Map<string, string>([[TYPE_TABLE_FN, TYPE_TABLE]]);

const GRID_TYPE_CONSTANT_PRE: Readonly<Record<VDBGridTypeName, string>> = {
  float: "PNANOVDB_GRID_TYPE_FLOAT",
  fp4: "PNANOVDB_GRID_TYPE_FP4",
  fp8: "PNANOVDB_GRID_TYPE_FP8",
  fp16: "PNANOVDB_GRID_TYPE_FP16",
  fpn: "PNANOVDB_GRID_TYPE_FPN",
};

/** Field names of the vendored `pnanovdb_grid_type_constants_t`, in order. */
const TYPE_TABLE_FIELDS: readonly string[] = [...chunk("pnanovdb_grid_type_constants_t").text.matchAll(/^\s+(\w+)\s*:/gm)].map((m) => m[1] as string);

/** The 27 rows of the vendored type table, as argument lists. */
const TYPE_TABLE_ROWS: readonly (readonly string[])[] = (() => {
  const rows = [...chunk(TYPE_TABLE).text.matchAll(/pnanovdb_grid_type_constants_t\(([^)]*)\)/g)].map((m) =>
    (m[1] as string).split(",").map((v) => v.trim()),
  );
  if (rows.length !== 27 || rows.some((r) => r.length !== TYPE_TABLE_FIELDS.length)) {
    throw new Error(`@pmndrs/vdb: unexpected vendored grid type table shape; re-check generate.ts against vendor/pnanovdb.wgsl`);
  }
  return rows;
})();

/**
 * Generation context: the options, plus the type-table fields the emitted
 * code reads (`null` = all, used for the first closure pass).
 */
interface Context {
  readonly o: NormalizedOptions;
  readonly fields: ReadonlySet<string> | null;
}

/** Transform 1: the single buffer access. */
function bufReadOverride(): string {
  return `fn pnanovdb_buf_read_uint32(buf: pnanovdb_buf_t, byte_offset: u32) -> u32 {
    return vdb_load((buf.byte_offset >> 2u) + (byte_offset >> 2u));
}`;
}

/**
 * Transform 2: the type table cut to the compiled types. Fields no emitted code
 * reads are written as 0u, and types whose remaining rows are equal share one
 * branch (Fp4/Fp8/Fp16/FpN differ only in the unread `leaf_size`). One distinct
 * row: a constant, no branch.
 */
function typeTableOverride({ o, fields }: Context): string {
  const groups: { row: string; types: VDBGridTypeName[] }[] = [];
  for (const t of o.types) {
    const values = TYPE_TABLE_ROWS[VDB_GRID_TYPE_IDS[t]] as readonly string[];
    const row = `pnanovdb_grid_type_constants_t(${values
      .map((v, i) => (fields === null || fields.has(TYPE_TABLE_FIELDS[i] as string) ? v : "0u"))
      .join(", ")})`;
    const group = groups.find((g) => g.row === row);
    if (group) group.types.push(t);
    else groups.push({ row, types: [t] });
  }
  const lines = [`fn ${TYPE_TABLE_FN}(grid_type: u32) -> pnanovdb_grid_type_constants_t {`];
  groups.forEach((g, i) => {
    if (i === groups.length - 1) {
      lines.push(`    return ${g.row};`);
    } else {
      lines.push(`    if ${g.types.map((t) => `grid_type == ${GRID_TYPE_CONSTANT_PRE[t]}`).join(" || ")} {`);
      lines.push(`        return ${g.row};`);
      lines.push(`    }`);
    }
  });
  lines.push("}");
  return lines.join("\n");
}

/** Transform 3: typed decode limited to the compiled types. */
function readFloatTypedOverride({ o }: Context): string {
  const text = chunk("pnanovdb_read_float_typed").text;
  const signature = text.slice(0, text.indexOf("{") + 1);
  const fpTypes = o.types.filter((t) => t !== "float");
  const decode = (t: VDBGridTypeName) => `pnanovdb_root_${t}_read_float(buf, address, ijk, level)`;
  const lines = [signature];
  // Without Float, the last Fp type is the fallthrough: `ok` already rejects
  // every other type before a read reaches here.
  const branches = o.types.includes("float") ? fpTypes : fpTypes.slice(0, -1);
  for (const t of branches) {
    lines.push(`    if grid_type == ${GRID_TYPE_CONSTANT_PRE[t]} {`);
    lines.push(`        return ${decode(t)};`);
    lines.push(`    }`);
  }
  lines.push(o.types.includes("float") ? "    return pnanovdb_read_float(buf, address);" : `    return ${decode(fpTypes[fpTypes.length - 1] as VDBGridTypeName)};`);
  lines.push("}");
  return lines.join("\n");
}

const OVERRIDES = new Map<string, (ctx: Context) => string>([
  ["pnanovdb_buf_read_uint32", bufReadOverride],
  [TYPE_TABLE, typeTableOverride],
  ["pnanovdb_read_float_typed", readFloatTypedOverride],
]);

function effectiveText(c: PnanovdbChunk, ctx: Context): string {
  const override = OVERRIDES.get(c.name);
  if (override) return override(ctx);
  return c.text.replace(TYPE_TABLE_CALL_SITE, `${TYPE_TABLE_FN}($1)`);
}

function referencedChunks(text: string, self?: string): string[] {
  const out = new Set<string>();
  for (const m of text.matchAll(IDENTIFIER)) {
    const name = ALIASES.get(m[0]) ?? m[0];
    if (name !== self && CHUNKS.has(name)) out.add(name);
  }
  return [...out];
}

/** Map a final (`vdb_` / `VDB_`) name back to its vendored declaration, if it is one. */
function vendoredName(finalName: string): string | undefined {
  let name: string | undefined;
  if (finalName.startsWith("vdb_")) name = `pnanovdb_${finalName.slice(4)}`;
  else if (finalName.startsWith("VDB_")) name = `PNANOVDB_${finalName.slice(4)}`;
  if (name === undefined) return undefined;
  name = ALIASES.get(name) ?? name;
  return CHUNKS.has(name) ? name : undefined;
}

function rename(text: string): string {
  return text.replace(/\bpnanovdb_/g, "vdb_").replace(/\bPNANOVDB_/g, "VDB_");
}

/** Vendored declarations a module needs, in file order. */
function closurePass(api: string, ctx: Context): string[] {
  const roots = new Set<string>();
  for (const m of api.matchAll(IDENTIFIER)) {
    const name = vendoredName(m[0]);
    if (name) roots.add(name);
  }
  const included = new Set<string>();
  const stack = [...roots];
  while (stack.length > 0) {
    const name = stack.pop() as string;
    if (included.has(name)) continue;
    included.add(name);
    const c = chunk(name);
    for (const d of referencedChunks(effectiveText(c, ctx), name)) if (!included.has(d)) stack.push(d);
  }
  return [...included].sort((a, b) => (ORDER.get(a) as number) - (ORDER.get(b) as number));
}

/**
 * Vendored declarations a module needs, in file order, and the context to
 * emit them with. Two passes: the first (whole type-table rows) finds which
 * type-table fields the closure reads; the second emits only those, which can
 * only shrink the closure.
 */
function closure(api: string, o: NormalizedOptions): { names: string[]; ctx: Context } {
  const first = closurePass(api, { o, fields: null });
  const fields = new Set<string>();
  for (const name of first) {
    if (name === TYPE_TABLE || name === "pnanovdb_grid_type_constants_t") continue;
    for (const m of chunk(name).text.matchAll(/\.([A-Za-z_]\w*)/g)) {
      if (TYPE_TABLE_FIELDS.includes(m[1] as string)) fields.add(m[1] as string);
    }
  }
  const ctx: Context = { o, fields };
  return { names: closurePass(api, ctx), ctx };
}

const cache = new Map<string, string>();

function build(o: NormalizedOptions): string {
  const api = apiWgsl(o.profileRank, o);
  const access = o.binding.kind === "slot" ? VDB_POOL_BINDING_NAME : o.binding.expression;
  const load = `fn vdb_load(vdb_i: u32) -> u32 {
    return ${access}[vdb_i];
}`;
  const { names, ctx } = closure(api, o);
  const vendored = names.map((name) => rename(effectiveText(chunk(name), ctx))).join("\n");
  const body = [vendored, load, api].join("\n");
  const constants = recordConstantsWgsl().filter((line) => {
    const name = /^const\s+(\w+)/.exec(line)?.[1] as string;
    return new RegExp(`\\b${name}\\b`).test(body);
  });
  const head = [VDB_MODULE_BANNER];
  if (o.binding.kind === "slot") {
    head.push(`@group(${o.binding.slot.group}) @binding(${o.binding.slot.firstBinding}) var<storage, read> ${VDB_POOL_BINDING_NAME}: array<u32>;`);
  }
  return [...head, ...constants, body].join("\n") + "\n";
}

/**
 * Generate a WGSL module for the pool (docs/design/K1-core.md §3.1).
 * Deterministic: the same options give the same string.
 *
 * @throws VDBError `VDB_BAD_OPTION` for invalid options.
 */
export function vdbModuleWgsl(options: VDBModuleOptions): string {
  const o = normalize(options, "vdbModuleWgsl");
  const key = JSON.stringify(o);
  let code = cache.get(key);
  if (code === undefined) {
    code = build(o);
    cache.set(key, code);
  }
  return code;
}

/**
 * The bind group layout entries a `slot` module declares: one read-only
 * storage buffer at `slot.firstBinding` (layout 1). `visibility` defaults to
 * `GPUShaderStage.COMPUTE`.
 */
export function vdbModuleEntries(options: { readonly slot: VDBSlot; readonly visibility?: GPUShaderStageFlags | undefined }): GPUBindGroupLayoutEntry[] {
  if (options === null || typeof options !== "object") throw bad("vdbModuleEntries: expected { slot, visibility? }.");
  rejectPositionalSlot("vdbModuleEntries", options);
  const slot = checkSlot(options.slot, "vdbModuleEntries");
  return [{ binding: slot.firstBinding, visibility: options.visibility ?? SHADER_STAGE_COMPUTE, buffer: { type: "read-only-storage" } }];
}

/**
 * Check that a module generated with `options` can read every grid in `pool`.
 *
 * @throws VDBError `VDB_LAYOUT_MISMATCH` if the pool's layout is not the one
 *   the generator targets; `VDB_TYPE_NOT_COMPILED` if the pool holds a grid
 *   type the module would not decode; `VDB_BAD_OPTION` for invalid options.
 */
export function checkModule(pool: VDBModulePoolLike, options: VDBModuleOptions): void {
  const o = normalize(options, "checkModule");
  if (pool.layoutVersion !== VDB_GPU_LAYOUT_VERSION) {
    throw new VDBError(
      "VDB_LAYOUT_MISMATCH",
      `the pool uses layout ${pool.layoutVersion} but this generator targets layout ${VDB_GPU_LAYOUT_VERSION}. Use one @pmndrs/vdb version for both.`,
    );
  }
  const missing = [...new Set(pool.types)].filter((t) => !o.types.includes(t));
  if (missing.length > 0) {
    throw new VDBError(
      "VDB_TYPE_NOT_COMPILED",
      `the pool holds ${missing.join(", ")} grids but the module compiles only ${o.types.join(", ")}; those grids would read 0. Regenerate with types: [...pool.types].`,
    );
  }
}

/** @internal Vendored declarations (vendored names) a module with these options includes, in file order. For tests. */
export function vdbModuleDeclarations(options: VDBModuleOptions): string[] {
  const o = normalize(options, "vdbModuleDeclarations");
  return closure(apiWgsl(o.profileRank, o), o).names;
}
