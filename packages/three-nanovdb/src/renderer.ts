/**
 * `createVolumeRenderer` — device-first `WebGPURenderer` bootstrap (decision
 * D4, SPEC §3.4): request the adapter and create the `GPUDevice` ourselves
 * with `requiredLimits` raised to cover the grid(s) we intend to bind, then
 * hand that device to `WebGPURenderer` at construction. Three.js will not
 * raise limits for us on our behalf — this sidesteps that entirely and keeps
 * device ownership with us (pattern proven in
 * `examples/src/demos/01-hello-nvdb/main.ts`).
 */

import { WebGPURenderer } from "three/webgpu";

/** Never request less than this for the two size-ish limits, however small the grid is. */
const DEFAULT_LIMIT_BYTES = 128 * 1024 * 1024; // 128 MiB

/** `shader-f16` and `float32-filterable` are requested by default (SPEC §3.4). */
const DEFAULT_REQUIRED_FEATURES: readonly GPUFeatureName[] = ["shader-f16", "float32-filterable"];

export interface VolumeRendererOptions {
  /** Grid bytes the device must be able to bind; raises `maxStorageBufferBindingSize`/`maxBufferSize`. */
  gridBytes?: number;
  canvas?: HTMLCanvasElement;
  /** Features to request (intersected with what the adapter actually supports). Defaults to shader-f16 + float32-filterable. */
  requiredFeatures?: GPUFeatureName[];
}

export interface RequiredLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

export interface CapabilityReport {
  /** The two limits the adapter actually offers (the ceiling we can't exceed). */
  adapterLimits: RequiredLimits;
  /** What we asked `requestDevice` for (`min(adapterLimits, needed(gridBytes))`). */
  requestedLimits: RequiredLimits;
  /** Features we asked for (default or `opts.requiredFeatures`). */
  requestedFeatures: GPUFeatureName[];
  /** Subset of `requestedFeatures` the adapter actually supports and that were requested from the device. */
  grantedFeatures: GPUFeatureName[];
  /** `adapter.info` vendor/architecture, when the adapter reports them (both fields are "" otherwise). */
  adapterInfo?: { vendor: string; architecture: string };
}

/** Smallest power of two >= n (n <= 0 yields 1). Mirrors demo 01's `nextPow2`. */
export function nextPow2(n: number): number {
  if (n <= 1) return 1;
  return 2 ** Math.ceil(Math.log2(n));
}

/**
 * Pure limits computation, split out so it's unit-testable without a GPU
 * (Vitest runs headlessly in node, per-package): `min(adapter limit,
 * max(128 MiB, nextPow2(gridBytes)))` for each of the two size limits, exactly
 * demo 01's math (SPEC §3.4).
 */
export function computeRequiredLimits(adapterLimits: RequiredLimits, gridBytes = 0): RequiredLimits {
  const needed = Math.max(DEFAULT_LIMIT_BYTES, nextPow2(gridBytes));
  return {
    maxStorageBufferBindingSize: Math.min(adapterLimits.maxStorageBufferBindingSize, needed),
    maxBufferSize: Math.min(adapterLimits.maxBufferSize, needed),
  };
}

/**
 * Device-first renderer bootstrap (decision D4): requests the adapter,
 * creates the `GPUDevice` ourselves with `requiredLimits` raised to
 * `min(adapter.limits, needed(gridBytes))`, then constructs
 * `WebGPURenderer({ device })` with the shared device. Three.js will NOT
 * raise limits for us — documented trap (docs/FEASIBILITY.md §5).
 */
export async function createVolumeRenderer(
  opts: VolumeRendererOptions = {},
): Promise<{ renderer: WebGPURenderer; device: GPUDevice; report: CapabilityReport }> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error(
      "createVolumeRenderer: WebGPU is unavailable (no `navigator.gpu`). This requires a secure " +
        "context (https:// or localhost) in a browser that ships WebGPU (Chrome/Edge 113+, or " +
        "another browser behind a flag) — it cannot run in Node or an insecure context.",
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error(
      "createVolumeRenderer: navigator.gpu.requestAdapter() returned null — no WebGPU adapter is " +
        "available (unsupported GPU/driver, WebGPU disabled, or a headless environment without a " +
        "software adapter such as SwiftShader).",
    );
  }

  const adapterLimits: RequiredLimits = {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  };
  const requestedLimits = computeRequiredLimits(adapterLimits, opts.gridBytes ?? 0);

  const requestedFeatures = [...(opts.requiredFeatures ?? DEFAULT_REQUIRED_FEATURES)];
  const grantedFeatures = requestedFeatures.filter((f) => adapter.features.has(f));

  const device = await adapter.requestDevice({
    requiredLimits: { ...requestedLimits },
    requiredFeatures: grantedFeatures,
  });

  const renderer = new WebGPURenderer({ device, canvas: opts.canvas, antialias: false });
  await renderer.init();

  const report: CapabilityReport = {
    adapterLimits,
    requestedLimits,
    requestedFeatures,
    grantedFeatures,
    ...(adapter.info.vendor || adapter.info.architecture
      ? { adapterInfo: { vendor: adapter.info.vendor, architecture: adapter.info.architecture } }
      : {}),
  };

  return { renderer, device, report };
}
