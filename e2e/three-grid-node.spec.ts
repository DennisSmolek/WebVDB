import { expect, test } from "@playwright/test";

// K2.2 gate (docs/dev/design/K1-core.md §6): VDBGridNode + vdbReadValue / vdbSample
// in a three r187 TSL compute shader, against the tools CPU twins; pool growth
// rebinds without a new shader; renderer.dispose() leaves the pool's buffer alone.
// The page is examples/src/harness/three-grid-node.ts.
const PAGE_URL = "/src/harness/three-grid-node.html";
const PROBE_FIXTURE_URL = "/fixtures/primitives/box_fog_float.nvdb";

interface ModeStats {
  grid: string;
  mode: string;
  count: number;
  maxAbs: number;
  exact: number;
  exactF32: number;
  bad: number;
}
interface PhaseResult {
  version: number;
  rebound?: boolean;
  stats: ModeStats[];
  dead: { count: number; bad: number };
  validationError?: string;
}
interface Result {
  done: boolean;
  error?: string;
  missing?: string[];
  sameDevice?: boolean;
  adapter?: string;
  phase1?: PhaseResult;
  phase2?: PhaseResult;
  shaderModulesPhase1?: number;
  shaderModulesPhase2Delta?: number;
  computePipelinesPhase2Delta?: number;
  bindingVerified?: boolean;
  bufferRefs?: string[];
  retiredStandIns?: number;
  afterDispose?: { identical: boolean[]; poolUsable: boolean; validationError?: string };
  uncaptured: string[];
}

test("three VDBGridNode: TSL reads match the CPU twins, survive growth and renderer.dispose()", async ({ page }) => {
  const probe = await page.request.get(PROBE_FIXTURE_URL);
  test.skip(!probe.ok(), `fixtures missing (${PROBE_FIXTURE_URL} -> ${probe.status()})`);
  test.setTimeout(180_000);

  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_THREE_GRID_NODE__?: { done: boolean } }).__VDB_THREE_GRID_NODE__?.done === true, undefined, {
    timeout: 170_000,
  });
  const r = (await page.evaluate(() => (window as { __VDB_THREE_GRID_NODE__?: unknown }).__VDB_THREE_GRID_NODE__)) as Result;
  const detail = JSON.stringify(r, null, 2);

  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  expect(r.sameDevice, detail).toBe(true);
  expect(r.uncaptured, detail).toEqual([]);

  // Phase 1: four grids (Float, Fp8, FpN, a transformed Float instance) x read / sample index / sample world.
  const p1 = r.phase1!;
  expect(p1.validationError, detail).toBeUndefined();
  expect(p1.stats, detail).toHaveLength(12);
  expect(r.bindingVerified, detail).toBe(true);
  // One module text: three r187 names the storage node's property 'vdbPool.value' (setName), the same on every load.
  expect(r.bufferRefs, detail).toEqual(["vdbPool.value"]);
  // Phase 2: growth bumped the version, update() rebound, the same shader ran (no new module or pipeline).
  const p2 = r.phase2!;
  expect(p2.validationError, detail).toBeUndefined();
  expect(p2.version, detail).toBeGreaterThan(p1.version);
  expect(p2.rebound, detail).toBe(true);
  expect(p2.stats, detail).toHaveLength(18);
  expect(r.shaderModulesPhase2Delta, detail).toBe(0);
  expect(r.computePipelinesPhase2Delta, detail).toBe(0);

  // Tolerances as the core parity work (read 1e-5, trilinear 1e-3 vs the f64 twins). On SwiftShader the
  // read and index-space sample probes must also be bit-exact against the f32 twins (WGSL leaves mix() and
  // FMA contraction to the implementation, so hardware is held to the tolerances only).
  const swiftShader = /swiftshader/i.test(r.adapter ?? "");
  for (const phase of [p1, p2]) {
    expect(phase.dead.count, detail).toBeGreaterThan(0);
    expect(phase.dead.bad, detail).toBe(0);
    for (const s of phase.stats) {
      expect(s.count, `${s.grid} ${s.mode}`).toBe(256);
      expect(s.bad, `v${phase.version} ${s.grid} ${s.mode}: ${JSON.stringify(s)}`).toBe(0);
      if (swiftShader && s.exactF32 >= 0) expect(s.exactF32, `v${phase.version} ${s.grid} ${s.mode} f32 twin: ${JSON.stringify(s)}`).toBe(s.count);
    }
  }

  // Phase 3: renderer.dispose() swapped throwaways in for the stand-ins; the pool's buffer is intact and usable.
  expect(r.retiredStandIns, detail).toBeGreaterThanOrEqual(2);
  expect(r.afterDispose?.validationError, detail).toBeUndefined();
  expect(r.afterDispose?.identical, detail).toEqual([true, true, true, true, true, true]);
  expect(r.afterDispose?.poolUsable, detail).toBe(true);

  const exact = p2.stats.reduce((n, s) => n + s.exact, 0);
  const total = p2.stats.reduce((n, s) => n + s.count, 0);
  const worst = Math.max(...p2.stats.map((s) => s.maxAbs));
  const inexact = p2.stats
    .filter((s) => s.exact < s.count || (s.exactF32 >= 0 && s.exactF32 < s.count))
    .map((s) => `${s.grid} ${s.mode} ${s.exact}/${s.count} (f32 twin ${s.exactF32}) max ${s.maxAbs}`);
  test.info().annotations.push({ type: "parity", description: `${exact}/${total} bit-exact, max |Δ| ${worst}, versions ${p1.version} -> ${p2.version}` });
  test.info().annotations.push({ type: "inexact", description: inexact.join("; ") || "none" });
  test.info().annotations.push({ type: "adapter", description: `${r.adapter} (f32-twin exactness ${swiftShader ? "gated" : "not gated"})` });
});
