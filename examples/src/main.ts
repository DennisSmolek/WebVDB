import { NVDB_SUPPORTED_GRID_TYPES, WVDB_SUPPORTED_VERSIONS } from "@pmndrs/vdb/tools";

// Phase 0 smoke: prove the workspace wiring (TS imports across packages,
// Vite dev server) and report WebGPU capability without requiring it —
// CI runs headless where an adapter may be absent.
const report = document.querySelector<HTMLDivElement>("#webgpu-report");
if (!report) throw new Error("missing #webgpu-report element");

function line(text: string): void {
  const p = document.createElement("p");
  p.textContent = text;
  report!.appendChild(p);
}

report.textContent = "";
line(
  `workspace ok — @pmndrs/vdb/tools .nvdb grid types: ${NVDB_SUPPORTED_GRID_TYPES.join(", ")}, ` +
    `.wvdb versions: ${WVDB_SUPPORTED_VERSIONS.join(", ")}`,
);

if (!navigator.gpu) {
  line("WebGPU: not available in this browser.");
  report.dataset["webgpu"] = "unavailable";
} else {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      line("WebGPU: API present, but no adapter (headless/software env?).");
      report.dataset["webgpu"] = "no-adapter";
    } else {
      const max = adapter.limits.maxStorageBufferBindingSize;
      line(
        `WebGPU: adapter ready — maxStorageBufferBindingSize ` +
          `${(max / (1024 * 1024)).toFixed(0)} MiB (default is 128 MiB; ` +
          `we raise it at device creation, per D4).`,
      );
      report.dataset["webgpu"] = "ok";
    }
  } catch (err) {
    line(`WebGPU: adapter request failed — ${String(err)}`);
    report.dataset["webgpu"] = "error";
  }
}
