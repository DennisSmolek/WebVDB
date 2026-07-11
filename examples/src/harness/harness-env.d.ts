/**
 * Ambient module declaration for Vite's `?raw` import suffix (returns the
 * file's contents as a plain string at build time). The workspace tsconfig
 * (`tsconfig.base.json`) deliberately limits `compilerOptions.types` to
 * `@webgpu/types` and doesn't pull in `vite/client`, so without this shim
 * `tsc --build` can't type `import pnanovdbWgsl from
 * "nanovdb-wgsl/pnanovdb.wgsl?raw"` even though Vite resolves it fine at
 * dev/build time. Scoped to `*.wgsl?raw` only — not a general vite/client
 * stand-in.
 */
declare module "*.wgsl?raw" {
  const src: string;
  export default src;
}
