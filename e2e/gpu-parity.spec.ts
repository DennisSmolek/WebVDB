import { expect, test } from "@playwright/test";

// GPU parity harness — the Phase 2 gate (docs/PLAN.md): thousands of
// deterministic voxel probes and trilinear samples across all nine fixtures,
// run through the extended vendored WGSL module and compared against the
// proven CPU reference. SwiftShader (software WebGPU) compiles + runs the
// nine fixtures slowly (2000 probes + 500 trilinear samples each), so this
// gets a generous timeout.
const HARNESS_URL = "/src/harness/index.html";
const PROBE_FIXTURE_URL = "/fixtures/primitives/box_fog_float.nvdb";

const EXPECTED_FIXTURES = [
  "box_fog_float",
  "box_fog_fp8",
  "box_fog_fpn",
  "sphere_fog_float",
  "sphere_fog_fp8",
  "sphere_fog_fpn",
  "torus_fog_float",
  "torus_fog_fp8",
  "torus_fog_fpn",
];
const EXPECTED_PROBE_COUNT = 2000;
const EXPECTED_TRILINEAR_COUNT = 500;

test("GPU parity harness — 9 fixtures x (2000 probes + 500 trilinear samples)", async ({ page }) => {
  // Fixtures are git-ignored; skip cleanly if this machine doesn't have them.
  const probe = await page.request.get(PROBE_FIXTURE_URL);
  test.skip(!probe.ok(), `fixtures missing (${PROBE_FIXTURE_URL} -> ${probe.status()})`);

  test.setTimeout(120_000);

  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto(HARNESS_URL);

  // Wait for the harness to publish its result (SwiftShader compile + 9x
  // compute+readback dispatch).
  await page.waitForFunction(() => window.__GPU_PARITY__?.done === true, undefined, {
    timeout: 110_000,
  });

  const result = await page.evaluate(() => window.__GPU_PARITY__);
  expect(result, "window.__GPU_PARITY__ should be set").toBeTruthy();

  const detail = JSON.stringify(
    {
      error: result!.error,
      consoleErrors,
      fixtures: result!.fixtures.map((f) => ({
        name: f.name,
        gridType: f.gridType,
        probes: f.probes,
        trilinear: f.trilinear,
        timingMs: Math.round(f.timingMs),
        error: f.error,
      })),
    },
    null,
    2,
  );

  // Surface the harness-level error (e.g. a WGSL compile failure — the
  // debugging channel getCompilationInfo() messages are forwarded into) up
  // front, since it usually means the fixtures array is empty.
  expect(result!.error, `harness reported an error:\n${detail}`).toBeFalsy();

  expect(
    result!.fixtures.map((f) => f.name),
    `expected all 9 fixtures to be exercised:\n${detail}`,
  ).toEqual(EXPECTED_FIXTURES);

  for (const fixture of result!.fixtures) {
    test.info().annotations.push({
      type: "gpu-parity",
      description:
        `${fixture.name} (${fixture.gridType}): probes ${fixture.probes.total - fixture.probes.failed}/` +
        `${fixture.probes.total} (maxΔ ${fixture.probes.maxDelta.toExponential(3)}), ` +
        `trilinear ${fixture.trilinear.total - fixture.trilinear.failed}/${fixture.trilinear.total} ` +
        `(maxΔ ${fixture.trilinear.maxDelta.toExponential(3)}), ${Math.round(fixture.timingMs)} ms`,
    });
    // eslint-disable-next-line no-console
    console.log(`[gpu-parity] ${test.info().annotations.at(-1)?.description}`);

    expect(fixture.error, `${fixture.name} reported an error:\n${detail}`).toBeFalsy();
    expect(fixture.probes.total, `${fixture.name}: probe count:\n${detail}`).toBe(EXPECTED_PROBE_COUNT);
    expect(fixture.trilinear.total, `${fixture.name}: trilinear count:\n${detail}`).toBe(
      EXPECTED_TRILINEAR_COUNT,
    );
    expect(fixture.probes.failed, `${fixture.name}: all probes must match:\n${detail}`).toBe(0);
    expect(fixture.trilinear.failed, `${fixture.name}: all trilinear samples must match:\n${detail}`).toBe(0);
  }
});
