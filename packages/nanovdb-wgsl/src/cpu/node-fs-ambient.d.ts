/**
 * Minimal ambient declaration for the one `node:fs` function `stride-tables.ts`
 * needs. This package's tsconfig deliberately limits `compilerOptions.types`
 * to `@webgpu/types` (see tsconfig.base.json) and doesn't depend on
 * `@types/node`, so without this shim `tsc --build` can't resolve
 * `node:fs`'s types even though the function works fine at runtime under
 * Vitest/Node. Keep this narrow — it is not a general `node:fs` types stand-in.
 */
declare module "node:fs" {
  export function readFileSync(path: URL | string, encoding: "utf8"): string;
}
