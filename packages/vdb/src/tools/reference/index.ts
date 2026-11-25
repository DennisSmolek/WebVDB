/**
 * K6.R: the reference renderer's engine-free half (design K6-core-renderer.md §7.4): the scene description, the
 * random stream it shares with the GPU kernel, the CPU twin (a brute-force volumetric path tracer), the media the
 * twin reads and the analytic cases. Internal (`tools/internal.ts`); no public export.
 */
export * from "./rng.js";
export * from "./scene.js";
export * from "./trace.js";
export * from "./media.js";
export * from "./analytic.js";
export * from "./tables.js";
