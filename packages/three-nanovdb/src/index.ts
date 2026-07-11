/**
 * three-nanovdb — TSL/three.js layer over the nanovdb-wgsl core.
 *
 * Phase 0 stub: API surface only, no three.js import yet. `NanoVDBGrid`
 * and the compute utilities land in Phases 1–4; `NanoVDBVolumeMaterial`
 * is the Phase 3 main goal (docs/PLAN.md, docs/SPEC.md §3).
 */

export interface VolumeRendererOptions {
  /** Grid bytes the device must be able to bind; raises `maxStorageBufferBindingSize`. */
  gridBytes?: number;
  canvas?: HTMLCanvasElement;
}

export interface CapabilityReport {
  adapterLimits: Record<string, number>;
  requestedLimits: Record<string, number>;
  features: string[];
}

/**
 * Device-first renderer bootstrap (decision D4): requests the adapter,
 * creates the `GPUDevice` ourselves with `requiredLimits` raised to
 * `min(adapter.limits, needed(gridBytes))`, then constructs
 * `WebGPURenderer({ device })` with the shared device. Three.js will NOT
 * raise limits for us — documented trap (docs/FEASIBILITY.md §5).
 *
 * Phase 3 deliverable — currently throws.
 */
export async function createVolumeRenderer(
  _opts: VolumeRendererOptions = {},
): Promise<{ renderer: unknown; report: CapabilityReport }> {
  throw new Error("createVolumeRenderer: not implemented (Phase 3)");
}

/** Wraps one NanoVDB grid image for GPU use. Phase 1+ deliverable. */
export class NanoVDBGrid {
  constructor(_gridImage: Uint32Array) {
    throw new Error("NanoVDBGrid: not implemented (Phase 1)");
  }
}
