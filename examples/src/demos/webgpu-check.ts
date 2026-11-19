/** Pure WebGPU availability check (unit-tested in `examples/test/webgpu-check.test.ts`). */

export const NO_WEBGPU_MESSAGE =
  "This demo needs WebGPU: Chrome or Edge 113+, Safari 26+, or Firefox with WebGPU enabled.";

export type WebGPUCheck = { ok: true } | { ok: false; reason: "no-api" | "no-adapter" | "error"; detail: string };

/** `navigator.gpu` missing, `requestAdapter()` null or throwing: the demo can't run. */
export async function checkWebGPU(gpu: { requestAdapter(): Promise<unknown> } | undefined | null): Promise<WebGPUCheck> {
  if (!gpu) return { ok: false, reason: "no-api", detail: "navigator.gpu is not available (use a WebGPU browser, over https or localhost)." };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no-adapter", detail: "The browser exposes WebGPU but found no usable GPU adapter (blocklisted GPU, or WebGPU disabled)." };
    return { ok: true };
  } catch (err) {
    return { ok: false, reason: "error", detail: `requestAdapter() failed: ${err instanceof Error ? err.message : String(err)}` };
  }
}
