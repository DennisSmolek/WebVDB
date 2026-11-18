// Compile every K1.4a generator variant in a real WebGPU implementation
// (headless Chromium, SwiftShader), and print each variant's §3.5 sizes.
//
// Sibling of scripts/check-wgsl-compile.mjs (which checks the raw vendored
// file). For each variant in scripts/lib/wgsl-variants.mjs it builds the check
// kernel, then under pushErrorScope('validation'):
//   - createShaderModule + getCompilationInfo (zero errors),
//   - createComputePipelineAsync with an explicit layout whose pool group comes
//     from vdbModuleEntries (slot mode) or the host binding (buffer mode).
// Exit code is non-zero if any variant has an error.
//
// Run:  pnpm build && node scripts/check-wgsl-variants.mjs [--sizes-only]
// Env:  PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH (e.g. a chromium_headless_shell under
//       ~/Library/Caches/ms-playwright).

import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { checkModule, vdbModuleEntries, vdbModuleWgsl } from "../packages/vdb/dist/core/index.js";
import { buildMarchProbe, compareMarchProbe, marchProbeAvailable, runMarchProbeInPage } from "./lib/march-probe.mjs";
import { CHECK_SLOT, MARCH_HELPERS, SAMPLE_HELPERS, TYPE_SETS, checkKernel, variantMatrix, wgslSizes } from "./lib/wgsl-variants.mjs";

const DIST = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../packages/vdb/dist");

const CHROMIUM_EXECUTABLE = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ?? "/opt/pw-browsers/chromium";
const SIZES_ONLY = process.argv.includes("--sizes-only");
const NO_PROBE = process.argv.includes("--no-probe");

function printSizes() {
  const kb = (n) => (n / 1024).toFixed(1);
  const rows = [
    ["read", { isActive: true }],
    ["read", { isActive: false }],
    ["sample", { isActive: true }],
    ["sample", { isActive: false }],
    ["sample", { sampler: "nearest" }],
    ["march", { isActive: true }],
    ["march", { isActive: false }],
    ["march", { outside: "zero" }],
    ["sample", { helpers: SAMPLE_HELPERS }],
    ["march", { helpers: MARCH_HELPERS }],
  ];
  console.log("\nSizes (slot mode; KiB = 1024 B; min / gz per §3.5, gz = gzip -9 of min):\n");
  console.log(`| Variant | ${TYPE_SETS.map((t) => t.join("+")).join(" | ")} |`);
  console.log(`|---|${TYPE_SETS.map(() => "---").join("|")}|`);
  for (const [profile, extra] of rows) {
    const cells = TYPE_SETS.map((types) => {
      const s = wgslSizes(vdbModuleWgsl({ slot: CHECK_SLOT, types, profile, ...extra }));
      return `${kb(s.min)} / ${kb(s.gz)}`;
    });
    const tag = Object.entries(extra).map(([k, v]) => `${k}: ${v}`).join(", ");
    console.log(`| \`${profile}\`, ${tag} | ${cells.join(" | ")} |`);
  }
  console.log("\nExact bytes (min, gz):");
  for (const [profile, extra] of rows) {
    for (const types of TYPE_SETS) {
      const s = wgslSizes(vdbModuleWgsl({ slot: CHECK_SLOT, types, profile, ...extra }));
      console.log(`  ${profile} ${JSON.stringify(extra)} ${types.join("+")}: raw ${s.raw}, min ${s.min}, gz ${s.gz}`);
    }
  }
}

async function main() {
  const variants = variantMatrix().map(({ id, options }) => {
    // Sanity: the JS-side check agrees the module covers its own types.
    checkModule({ types: options.types, layoutVersion: 1 }, options);
    const module = vdbModuleWgsl(options);
    return {
      id,
      code: checkKernel(module, options),
      poolEntries:
        options.slot !== undefined
          ? vdbModuleEntries({ slot: options.slot, visibility: 0x4 })
          : [{ binding: CHECK_SLOT.firstBinding, visibility: 0x4, buffer: { type: "read-only-storage" } }],
    };
  });

  printSizes();
  if (SIZES_ONLY) return 0;

  const probe = !NO_PROBE && marchProbeAvailable() ? await buildMarchProbe() : null;
  if (!NO_PROBE && probe === null) console.log("\nmarch probe skipped: fixtures/primitives missing (pnpm fixtures:bake).");

  const server = createServer(async (req, res) => {
    const url = req.url ?? "/";
    if (probe && url.startsWith("/probe-image/")) {
      const bytes = probe.imageBytes[Number(url.slice("/probe-image/".length))];
      res.writeHead(bytes ? 200 : 404, { "content-type": "application/octet-stream" });
      res.end(bytes ?? "");
      return;
    }
    if (url.startsWith("/dist/") && url.endsWith(".js") && !url.includes("..")) {
      try {
        const body = await readFile(path.join(DIST, url.slice("/dist/".length)));
        res.writeHead(200, { "content-type": "text/javascript; charset=utf-8" });
        res.end(body);
      } catch {
        res.writeHead(404);
        res.end();
      }
      return;
    }
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end("<!doctype html><meta charset=utf-8><title>wgsl variants</title><body>ok</body>");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  const browser = await chromium.launch({
    executablePath: CHROMIUM_EXECUTABLE,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-unsafe-swiftshader"],
  });

  let failed = 0;
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => console.log(`[page:${msg.type()}] ${msg.text()}`));
    await page.goto(`http://localhost:${port}/`);

    const results = await page.evaluate(
      async ({ variants, poolGroup }) => {
        if (!("gpu" in navigator)) return { fatal: "navigator.gpu unavailable" };
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return { fatal: "no WebGPU adapter (SwiftShader not active?)" };
        const device = await adapter.requestDevice();
        const info = adapter.info ?? {};
        const outLayout = device.createBindGroupLayout({
          entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
        });
        const out = [];
        for (const v of variants) {
          const t0 = performance.now();
          device.pushErrorScope("validation");
          const module = device.createShaderModule({ code: v.code });
          const compile = await module.getCompilationInfo();
          const messages = compile.messages
            .filter((m) => m.type !== "info")
            .map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
          const poolLayout = device.createBindGroupLayout({ entries: v.poolEntries });
          const bindGroupLayouts = [];
          bindGroupLayouts[0] = outLayout;
          bindGroupLayouts[poolGroup] = poolLayout;
          let pipelineError = null;
          try {
            await device.createComputePipelineAsync({
              layout: device.createPipelineLayout({ bindGroupLayouts }),
              compute: { module, entryPoint: "checkMain" },
            });
          } catch (e) {
            pipelineError = String(e && e.message ? e.message : e);
          }
          const scope = await device.popErrorScope();
          out.push({
            id: v.id,
            errors: compile.messages.filter((m) => m.type === "error").length,
            messages,
            pipelineError,
            scopeError: scope ? scope.message : null,
            ms: Math.round(performance.now() - t0),
          });
        }
        return { adapter: `${info.vendor ?? "?"} ${info.architecture ?? ""} ${info.description ?? ""}`.trim(), out };
      },
      { variants, poolGroup: CHECK_SLOT.group },
    );

    if (results.fatal) {
      console.error(`FATAL: ${results.fatal}`);
      return 2;
    }
    console.log(`\nAdapter: ${results.adapter}; browser ${browser.version()}`);
    for (const r of results.out) {
      const ok = r.errors === 0 && !r.pipelineError && !r.scopeError;
      if (!ok) failed++;
      console.log(`${ok ? "ok  " : "FAIL"} ${r.id} (${r.ms} ms)`);
      for (const m of r.messages) console.log(`       ${m}`);
      if (r.pipelineError) console.log(`       pipeline: ${r.pipelineError}`);
      if (r.scopeError) console.log(`       validation: ${r.scopeError}`);
    }
    console.log(`\n${results.out.length - failed}/${results.out.length} variants compiled with zero errors.`);

    if (probe) {
      const got = await page.evaluate(runMarchProbeInPage, { probe: probe.page, dist: "/dist" });
      const { fails, summary } = compareMarchProbe(probe, got);
      console.log(`\n${summary}`);
      for (const f of fails.slice(0, 20)) console.log(`  FAIL ${f}`);
      if (fails.length > 0) failed++;
    }
  } finally {
    await browser.close();
    server.close();
  }
  return failed > 0 ? 1 : 0;
}

main().then(
  (code) => process.exit(code),
  (err) => {
    console.error(err);
    process.exit(3);
  },
);
