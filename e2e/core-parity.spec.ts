import { expect, test } from "@playwright/test";

// K1.6: the K1 regression gate (docs/design/K1-core.md §7, docs/handoffs/K1-6.md).
// One VDBGridPool (4 KiB start, maxGrids 2) holds the nine primitive fixtures, two synthetic quantized grids (sphere_fog_fp4 / sphere_fog_fp16, K5.4b) plus
// torus_fog_fp8 again with an instance transform; every generated-module variant of
// the matrix is dispatched and compared with the CPU twins. The page does the
// comparing and publishes counts, max deltas and failure lines.
const PAGE_URL = "/src/harness/core-parity.html";
const PROBE_FIXTURE_URL = "/fixtures/primitives/box_fog_float.nvdb";

interface CatStats {
  total: number;
  failed: number;
  maxDelta: number;
  maxUlp: number;
}
interface CoreParity {
  done: boolean;
  error?: string;
  adapter?: string;
  isSwiftShader?: boolean;
  pool?: { versions: number[]; gridCapacity: number; gridCount: number; offsets: number[] };
  expected?: { variants: string[] };
  variants: { id: string; compileErrors: string[]; cats: Record<string, CatStats>; failures: string[]; ms: number }[];
  totals: Record<string, CatStats>;
  counts: { read: number; sample: number; xform: number; rays: number; cells: number; nodeMax: number };
  poolChecks: { name: string; ok: boolean; detail?: string }[];
  validationError?: string;
  missing?: string[];
  totalMs: number;
}

// Categories every variant of a profile must have exercised (non-empty), so a silently skipped check fails the gate.
const READ_CATS = ["read-value", "record", "dead", "sidecar-value"];
const SAMPLE_CATS = ["sample"];
const XFORM_CATS = ["xform", "world-sample", "clip-hit"];
const MARCH_CATS = [
  "nodemax-leaf",
  "nodemax-lower",
  "nodemax-upper",
  "nodemax-root",
  "nodemax-tile",
  "nodemax-far",
  "walk-extent",
  "walk-intervals-active",
  "walk-cell-active",
  "walk-cell-max",
];

test("core parity: generated modules + VDBGridPool vs the CPU twins (K1 gate)", async ({ page }) => {
  const probe = await page.request.get(PROBE_FIXTURE_URL);
  test.skip(!probe.ok(), `fixtures missing (${PROBE_FIXTURE_URL} -> ${probe.status()})`);
  test.setTimeout(300_000);

  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __CORE_PARITY__?: { done: boolean } }).__CORE_PARITY__?.done === true, undefined, {
    timeout: 240_000,
  });
  const r = (await page.evaluate(() => (window as { __CORE_PARITY__?: unknown }).__CORE_PARITY__)) as CoreParity;

  const failures = r.variants.flatMap((v) => v.failures);
  const detail = JSON.stringify(
    { error: r.error, validationError: r.validationError, consoleErrors, failures: failures.slice(0, 20), poolChecks: r.poolChecks.filter((c) => !c.ok) },
    null,
    2,
  );

  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  expect(r.validationError, detail).toBeUndefined();
  expect(
    r.poolChecks.filter((c) => !c.ok),
    detail,
  ).toEqual([]);
  expect(r.pool!.versions[r.pool!.versions.length - 1]!, detail).toBeGreaterThanOrEqual(4);
  expect(r.pool!.gridCapacity, detail).toBeGreaterThanOrEqual(16);
  expect(
    r.variants.map((v) => v.id),
    detail,
  ).toEqual(r.expected!.variants);
  expect(r.variants.length).toBe(29); // 21 base + 2 fastTrilinear (K3.2d) + 6 fp4/fp16 (K5.4)

  for (const v of r.variants) {
    expect(v.compileErrors, `${v.id} compile`).toEqual([]);
    const profile = v.id.split("/")[0];
    // fp4 / fp16 exist only as synthetic quantize() grids (no native sidecar), so their single-type modules have no sidecar-value rows.
    const readCats = /\/fp(4|16)$/.test(v.id) ? READ_CATS.filter((c) => c !== "sidecar-value") : READ_CATS;
    const required = [...readCats, ...(profile !== "read" ? SAMPLE_CATS : []), ...(profile === "march" ? MARCH_CATS : [])];
    if (["sample/all", "march/all", "march/all/buffer"].includes(v.id)) required.push(...XFORM_CATS);
    // K1.7a opt-in helpers (vdbHddaCellEnd feeds the walk categories above).
    if (v.id === "march/all/helpers") required.push("cellbound", "clip-support-hit", "clip-support-t", "clip-hit");
    if (v.id === "sample/all/nearest,helpers") required.push("clip-support-hit", "clip-support-t");
    // K3.2d: vdbSampleFast bit-identical to vdbSample.
    if (v.id.endsWith("fast")) required.push("fast-sample");
    for (const cat of required) {
      expect(v.cats[cat]?.total ?? 0, `${v.id}: category ${cat} ran`).toBeGreaterThan(0);
    }
    for (const [cat, s] of Object.entries(v.cats)) {
      expect(s.failed, `${v.id} [${cat}] ${v.failures.filter((f) => f.includes(`[${cat}]`)).join(" | ")}`).toBe(0);
    }
  }
  expect(failures, detail).toEqual([]);
  // Negative cases ran in the single-type modules (grids of another type read as nothing).
  expect(r.totals["negative"]?.total ?? 0).toBeGreaterThan(0);

  // Precision: SwiftShader is bit-exact on every f32-faithful twin; other GPUs may differ by 1 ulp (the harness allows that).
  // `sidecar-value` is excluded: the native sidecars decode Fp leaves with a fused multiply-add (sphere_fog_fp8/fpn at
  // [-23,37,-16]: 0.99999994 native vs 1.0 from the unfused twin and the GPU), so they are 1 ulp apart by design.
  if (r.isSwiftShader) {
    for (const cat of ["read-value", "nodemax-leaf", "nodemax-lower", "nodemax-upper", "nodemax-root", "nodemax-tile", "walk-cell-max"]) {
      expect(r.totals[cat]?.maxUlp, `${cat} max ulp on SwiftShader`).toBe(0);
    }
  }

  expect(r.totalMs, "runtime budget").toBeLessThan(180_000); // 29 variants; headroom for loaded machines and small CI runners
  const t = r.totals;
  test.info().annotations.push({
    type: "core-parity",
    description:
      `${r.adapter}: ${Math.round(r.totalMs)} ms; read ${t["read-value"]?.total} (+${t["sidecar-value"]?.total} sidecar), sample ${t["sample"]?.total} (max delta ${t["sample"]?.maxDelta.toExponential(2)}), ` +
      `xform ${t["xform"]?.total} (max rel ${t["xform"]?.maxDelta.toExponential(2)}), clip ${t["clip-hit"]?.total}, nodeMax ${r.counts.nodeMax}, rays ${r.counts.rays} / ${r.counts.cells} cells, record ${t["record"]?.total}, dead ${t["dead"]?.total}, negative ${t["negative"]?.total}`,
  });
});
