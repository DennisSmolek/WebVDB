/**
 * Device-first `WebGPURenderer` for the demos (D8): the page requests the
 * adapter and the device (with `VDBGridPool.requiredLimits(adapter)` so the
 * pool can grow to the adapter's maxima) and hands the device to three, so
 * `renderer.dispose()` never destroys it or the pool's buffer.
 *
 * Replaces the v1 `createVolumeRenderer` the package no longer exports.
 */
import { WebGPURenderer } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";

/** Requested when the adapter has them (as v1's bootstrap did). */
const WANTED_FEATURES: readonly GPUFeatureName[] = ["shader-f16", "float32-filterable"];

export interface DeviceRenderer {
  renderer: WebGPURenderer;
  device: GPUDevice;
  /** `adapter.info` vendor and architecture, or "" when the adapter reports neither. */
  adapterName: string;
  /** Features the device was created with. */
  features: GPUFeatureName[];
}

export async function createDeviceRenderer(options: { canvas?: HTMLCanvasElement; trackTimestamp?: boolean } = {}): Promise<DeviceRenderer> {
  if (!navigator.gpu) throw new Error("WebGPU is unavailable (no navigator.gpu): use a secure context (https or localhost) in a WebGPU browser.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("navigator.gpu.requestAdapter() returned null: no WebGPU adapter.");
  const wanted = [...WANTED_FEATURES, ...(options.trackTimestamp ? (["timestamp-query"] as const) : [])];
  const features = wanted.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter), requiredFeatures: features });
  const renderer = new WebGPURenderer({
    device,
    antialias: false,
    trackTimestamp: options.trackTimestamp ?? false,
    ...(options.canvas ? { canvas: options.canvas } : {}),
  });
  await renderer.init();
  const adapterName = [adapter.info.vendor, adapter.info.architecture].filter(Boolean).join(" ");
  return { renderer, device, adapterName, features };
}
