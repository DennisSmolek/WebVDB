/**
 * Test helper: names in generated WGSL that would stop the shader compiling.
 * A reserved word used as a name (`active` and `target` broke bvh kernels;
 * `active` and `set` are in this repo's Traps), or a function with the same
 * name as a module-scope variable (WGSL has one namespace for both).
 *
 * Ported from @pmndrs/bvh `src/wgsl-names.ts` (design note §3.4). Parameter
 * detection also covers lowercase `*_t` struct types (the vendored handles).
 */

// WGSL's reserved words (WGSL spec, "Reserved Words").
const RESERVED = new Set([
  "NULL", "Self", "abstract", "active", "alignas", "alignof", "as", "asm", "asm_fragment", "async", "attribute", "auto", "await", "become",
  "cast", "catch", "class", "co_await", "co_return", "co_yield", "coherent", "column_major", "common", "compile", "compile_fragment", "concept",
  "const_cast", "consteval", "constexpr", "constinit", "crate", "debugger", "decltype", "delete", "demote", "demote_to_helper", "do",
  "dynamic_cast", "enum", "explicit", "export", "extends", "extern", "external", "fallthrough", "filter", "final", "finally", "friend", "from",
  "fxgroup", "get", "goto", "groupshared", "highp", "impl", "implements", "import", "inline", "instanceof", "interface", "layout", "lowp",
  "macro", "macro_rules", "match", "mediump", "meta", "mod", "module", "move", "mut", "mutable", "namespace", "new", "nil", "noexcept",
  "noinline", "nointerpolation", "non_coherent", "noncoherent", "noperspective", "null", "nullptr", "of", "operator", "package", "packoffset",
  "partition", "pass", "patch", "pixelfragment", "precise", "precision", "premerge", "priv", "protected", "pub", "public", "readonly", "ref",
  "regardless", "register", "reinterpret_cast", "require", "resource", "restrict", "self", "set", "shared", "sizeof", "smooth", "snorm",
  "static", "static_assert", "static_cast", "std", "subroutine", "super", "target", "template", "this", "thread_local", "throw", "trait", "try",
  "type", "typedef", "typeid", "typename", "typeof", "union", "unless", "unorm", "unsafe", "unsized", "use", "using", "varying", "virtual",
  "volatile", "wgsl", "where", "with", "writeonly", "yield",
]);

/** Problems with the names `code` declares, empty if none. */
export function wgslNameProblems(code: string): string[] {
  const declared = [...code.matchAll(/\b(?:let|var|fn|const|struct|alias)\s*(?:<[^>]*>)?\s+([A-Za-z_]\w*)/g)].map((m) => m[1] as string);
  const parameters = [
    ...code.matchAll(/\b([A-Za-z_]\w*)\s*:\s*(?:u32|i32|f32|bool|vec[234][uif]|mat\dx\d[f]|ptr<[^>]*>|array<[^>]*>|[A-Z]\w*|[a-z]\w*_t)\b/g),
  ].map((m) => m[1] as string);
  const problems = [...declared, ...parameters].filter((name) => RESERVED.has(name)).map((name) => `"${name}" is a reserved word`);
  const functions = new Set([...code.matchAll(/\bfn\s+([A-Za-z_]\w*)/g)].map((m) => m[1] as string));
  const globals = [...code.matchAll(/^(?:@[^\n]*?\s)?var(?:<[^>]*>)?\s+([A-Za-z_]\w*)/gm)].map((m) => m[1] as string);
  for (const name of globals) if (functions.has(name)) problems.push(`"${name}" is both a function and a variable`);
  return problems;
}
