/**
 * K6.R: the reference renderer's GPU half (design K6-core-renderer.md §7.4). Internal (`core/internal.ts`); no public
 * export. The scene types and the CPU twin are in `tools/reference/`.
 */
export * from "./wgsl.js";
export * from "./renderer.js";
