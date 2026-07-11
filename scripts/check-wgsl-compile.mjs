// Standalone WGSL compile check for the vendored PNanoVDB fork.
//
// WebGPU is secure-context only, so we cannot compile a shader module from
// about:blank. This script spins up a throwaway local HTTP server, navigates
// a real Chromium (SwiftShader-backed WebGPU) to http://localhost:<port>,
// reads vendor/pnanovdb.wgsl, appends the storage-buffer binding the file's
// header comment asks for plus a @compute entry point that references the
// Phase 2 additions (so nothing is dead-stripped before validation), calls
// device.createShaderModule, and prints every getCompilationInfo() message.
//
// Exit code is non-zero if any message has severity "error".
//
// Run:  node scripts/check-wgsl-compile.mjs

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const WGSL_URL = new URL(
  "../packages/nanovdb-wgsl/vendor/pnanovdb.wgsl",
  import.meta.url,
);
const CHROMIUM_EXECUTABLE =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?? "/opt/pw-browsers/chromium";

// The binding the vendored file's header comment requires, plus a @compute
// entry point that touches a representative call from every Phase 2 work item
// so the validator sees them (WGSL prunes unreferenced functions).
const HARNESS_FOOTER = /* wgsl */ `

// ---- compile-check harness (appended, not part of the vendored file) ----
@group(0) @binding(0) var<storage, read> nanovdb_buffer: array<u32>;
@group(0) @binding(1) var<storage, read_write> compile_check_out: array<f32>;

@compute @workgroup_size(1)
fn compile_check_main() {
    let buf = pnanovdb_make_buffer(0u, arrayLength(&nanovdb_buffer));
    let grid = pnanovdb_grid_handle_t(0u);
    let grid_type = PNANOVDB_GRID_TYPE_FLOAT;

    let tree = pnanovdb_grid_get_tree(grid);
    let root = pnanovdb_tree_get_root(buf, tree);

    var acc: pnanovdb_readaccessor_t;
    pnanovdb_readaccessor_init(&acc, root);

    let ijk = vec3i(1, 2, 3);
    let xyzf = vec3f(1.5, 2.5, 3.5);

    var sum = 0.0;

    // Item 2 — FP leaf decoders + typed dispatch.
    sum += pnanovdb_leaf_fp4_read_float(buf, 0u, ijk);
    sum += pnanovdb_leaf_fp8_read_float(buf, 0u, ijk);
    sum += pnanovdb_leaf_fp16_read_float(buf, 0u, ijk);
    sum += pnanovdb_leaf_fpn_read_float(buf, 0u, ijk);
    sum += pnanovdb_readaccessor_get_value_float(grid_type, buf, &acc, ijk);

    // Item 3 — active-state queries (pre-existing, exercised for parity).
    if pnanovdb_readaccessor_is_active(grid_type, buf, &acc, ijk) { sum += 1.0; }

    // Item 4 — world<->index map.
    sum += pnanovdb_grid_world_to_indexf(buf, grid, xyzf).x;
    sum += pnanovdb_grid_index_to_worldf(buf, grid, xyzf).y;
    sum += pnanovdb_grid_world_to_index_dirf(buf, grid, xyzf).z;
    sum += pnanovdb_grid_index_to_world_dirf(buf, grid, xyzf).x;

    // Item 5 — trilinear sampling (WebVDB extension).
    sum += pnanovdb_sample_trilinear_float(buf, &acc, xyzf);
    sum += pnanovdb_sample_trilinear_fp8(buf, &acc, xyzf);
    sum += pnanovdb_sample_trilinear_fpn(buf, &acc, xyzf);

    // Item 6 — HDDA family (pre-existing).
    var thit = 0.0;
    var vhit = 0.0;
    if pnanovdb_hdda_zero_crossing(grid_type, buf, &acc, xyzf, 0.0, vec3f(0.0, 0.0, 1.0), 100.0, &thit, &vhit) {
        sum += thit + vhit;
    }

    // Item 7 — stats readers.
    sum += pnanovdb_root_get_min_float(grid_type, buf, root);
    sum += pnanovdb_root_get_max_float(grid_type, buf, root);
    sum += pnanovdb_root_get_ave_float(grid_type, buf, root);
    sum += pnanovdb_root_get_stddev_float(grid_type, buf, root);
    let upper = pnanovdb_upper_handle_t(0u);
    sum += pnanovdb_upper_get_min_float(grid_type, buf, upper);
    sum += pnanovdb_upper_get_max_float(grid_type, buf, upper);
    let lower = pnanovdb_lower_handle_t(0u);
    sum += pnanovdb_lower_get_min_float(grid_type, buf, lower);
    sum += pnanovdb_lower_get_max_float(grid_type, buf, lower);
    let leaf = pnanovdb_leaf_handle_t(0u);
    sum += pnanovdb_leaf_get_min_float(grid_type, buf, leaf);
    sum += pnanovdb_leaf_get_max_float(grid_type, buf, leaf);
    sum += pnanovdb_leaf_get_ave_float(grid_type, buf, leaf);
    sum += pnanovdb_leaf_get_stddev_float(grid_type, buf, leaf);

    compile_check_out[0] = sum;
}
`;

async function main() {
  const vendored = await readFile(fileURLToPath(WGSL_URL), "utf8");
  const shaderSource = vendored + HARNESS_FOOTER;

  // Trivial secure-ish origin: a localhost HTTP server (localhost counts as a
  // secure context for WebGPU even over plain http).
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end("<!doctype html><meta charset=utf-8><title>wgsl compile check</title><body>ok</body>");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  const pageUrl = `http://localhost:${port}/`;

  const browser = await chromium.launch({
    executablePath: CHROMIUM_EXECUTABLE,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--enable-unsafe-swiftshader",
    ],
  });

  let exitCode = 0;
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => console.log(`[page:${msg.type()}] ${msg.text()}`));
    await page.goto(pageUrl);

    const result = await page.evaluate(async (source) => {
      if (!("gpu" in navigator)) {
        return { ok: false, fatal: "navigator.gpu unavailable in this Chromium" };
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return { ok: false, fatal: "no WebGPU adapter (SwiftShader not active?)" };
      }
      const device = await adapter.requestDevice();

      // Surface device-scope validation errors too.
      const scopeErrors = [];
      device.pushErrorScope("validation");

      const module = device.createShaderModule({ code: source });
      const info = await module.getCompilationInfo();

      const scopeError = await device.popErrorScope();
      if (scopeError) scopeErrors.push(scopeError.message);

      const messages = info.messages.map((m) => ({
        type: m.type,
        lineNum: m.lineNum,
        linePos: m.linePos,
        message: m.message,
      }));
      const hasError = messages.some((m) => m.type === "error");
      return { ok: !hasError, messages, scopeErrors };
    }, shaderSource);

    if (result.fatal) {
      console.error(`FATAL: ${result.fatal}`);
      exitCode = 2;
    } else {
      const errors = result.messages.filter((m) => m.type === "error");
      const warnings = result.messages.filter((m) => m.type === "warning");
      const infos = result.messages.filter((m) => m.type === "info");

      for (const m of result.messages) {
        console.log(
          `  [${m.type}] ${m.lineNum}:${m.linePos}  ${m.message.trim()}`,
        );
      }
      for (const e of result.scopeErrors ?? []) {
        console.log(`  [device-validation] ${e.trim()}`);
      }

      console.log(
        `\ngetCompilationInfo: ${errors.length} error(s), ` +
          `${warnings.length} warning(s), ${infos.length} info.`,
      );

      if (errors.length > 0 || (result.scopeErrors ?? []).length > 0) {
        console.error("WGSL compile check FAILED.");
        exitCode = 1;
      } else {
        console.log("WGSL compile check PASSED (zero errors).");
      }
    }
  } finally {
    await browser.close();
    server.close();
  }

  process.exit(exitCode);
}

main().catch((err) => {
  console.error(err);
  process.exit(3);
});
