/**
 * Test helper: names in generated WGSL that would stop the shader compiling.
 * A reserved word used as a name (`active` and `target` broke bvh kernels;
 * `active` and `set` are in this repo's Traps), or a function with the same
 * name as a module-scope variable (WGSL has one namespace for both).
 *
 * Ported from @pmndrs/bvh `src/wgsl-names.ts` (design note §3.4). Parameter
 * detection also covers lowercase `*_t` struct types (the vendored handles).
 */

import { WGSL_RESERVED_WORDS as RESERVED } from "../../src/core/wgsl/reserved.js";

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
