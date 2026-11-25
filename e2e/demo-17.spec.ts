import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

/**
 * Demo 17 — rendering without three (K6.9): the core frame renderer on raw WebGPU, the page's own passes around it.
 * Readback presentation (headless SwiftShader has no WebGPU canvas compositor), `?test=1` runs the page's statistical
 * checks: each turns one feature off and counts the pixels that change against the full frame (sum of |Δ| over RGB > 6,
 * sRGB bytes) with their mean signed luma change. CI has no WDAS cloud or icosahedron, so the spec pins the committed CC0
 * cloud and the procedural level set (`?cloud=cc0&levelset=procedural`); a second case runs the fixtures when present.
 *
 * What each check proves:
 * - volumes off: the volumes draw (many pixels change);
 * - volume → mesh shadows off: the ground under the cloud gets brighter (the receivers read the core's Beer map);
 * - mesh → volume occluders off: the cloud gets brighter where the obelisk's shadow crossed it (the core reads our map);
 * - emission off: the ground and props near the explosion get darker (the receivers read the emission probes);
 * - level set off: the level-set pass drew a surface;
 * - our composite shader (`vdbFrameIntegrateWgsl`) vs the core's composite pass: the same image within f16 rounding;
 * - a second identical frame is byte-identical and builds no pipeline or shader module.
 */
const DEMO = "/src/demos/17-raw-webgpu/index.html";
const BASE = "presentation=readback&test=1&still=1&quality=low&panel=0";
const SHOTS = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "artifacts", "k6", "demo-17");

interface Diff {
  changed: number;
  meanLuma: number;
  maxChannel: number;
  over2: number;
  pixels: number;
}
interface State {
  ready: boolean;
  done?: boolean;
  error?: string;
  presentation: string;
  uncaptured: string[];
  info: {
    cloud: string;
    levelSet: string;
    lighting: string;
    volumes: number;
    recorded: boolean;
    froxels: number[] | null;
    shadowMap: { layers: number; size: number; enabled: boolean } | null;
    emissionProbes: { resolution: number; emitters: number } | null;
    occluders: number;
    checks: { repeat: { identical: boolean; builds: number }; noVolumes: Diff; noVolumeShadows: Diff; noOccluders: Diff; noEmission: Diff; noLevelSet: Diff; hostComposite: Diff };
    snapshot: { spread: number; mean: number[] };
    finalEqualsBase: boolean;
  };
}

test.use({ viewport: { width: 960, height: 540 } });
test.setTimeout(900_000);

async function openDone(page: Page, query: string): Promise<State> {
  await page.goto(`${DEMO}?${query}`);
  await page.waitForFunction(() => {
    const s = (window as unknown as { __DEMO17__?: { error?: string; done?: boolean } }).__DEMO17__;
    return s?.error !== undefined || s?.done === true;
  }, undefined, { timeout: 600_000 });
  return (await page.evaluate(() => (window as unknown as { __DEMO17__: unknown }).__DEMO17__)) as State;
}

function checkScene(state: State, errors: string[], lighting: string): void {
  const detail = JSON.stringify({ ...state, info: { ...state.info, snapshot: undefined } }, null, 2) + `\nerrors: ${errors.join("\n")}`;
  expect(state.error, detail).toBeUndefined();
  expect(state.presentation, detail).toBe("readback");
  expect(state.uncaptured, detail).toEqual([]);
  const i = state.info;
  expect(i.lighting, detail).toBe(lighting);
  expect(i.recorded, detail).toBe(true);
  expect(i.volumes, detail).toBe(2);
  expect(i.shadowMap?.enabled, detail).toBe(true);
  expect(i.emissionProbes?.emitters, detail).toBeGreaterThan(0);
  expect(i.occluders, detail).toBe(1);
  const c = i.checks;
  const px = c.noVolumes.pixels;
  // determinism and zero builds in a second identical frame
  expect(c.repeat.identical, detail).toBe(true);
  expect(c.repeat.builds, detail).toBe(0);
  expect(i.finalEqualsBase, detail).toBe(true);
  // the volumes draw
  expect(c.noVolumes.changed, detail).toBeGreaterThan(px * 0.05);
  // volume → mesh: without the receivers' Beer map the ground under the cloud is brighter
  expect(c.noVolumeShadows.changed, detail).toBeGreaterThan(px * 0.01);
  expect(c.noVolumeShadows.meanLuma, detail).toBeGreaterThan(4);
  // mesh → volume: without our depth map as an occluder the cloud is brighter
  expect(c.noOccluders.changed, detail).toBeGreaterThan(px * 0.002);
  expect(c.noOccluders.meanLuma, detail).toBeGreaterThan(2);
  // the explosion's light: without it the ground near it is darker
  expect(c.noEmission.changed, detail).toBeGreaterThan(px * 0.002);
  expect(c.noEmission.meanLuma, detail).toBeLessThan(-2);
  // the level set draws
  expect(c.noLevelSet.changed, detail).toBeGreaterThan(px * 0.001);
  // our composite shader equals the core's pass within f16 rounding of the HDR target
  expect(c.hostComposite.maxChannel, detail).toBeLessThanOrEqual(3);
  expect(c.hostComposite.over2, detail).toBeLessThanOrEqual(px * 0.001);
  expect(i.snapshot.spread, detail).toBeGreaterThan(20);
  expect(errors, detail).toEqual([]);
}

test.describe("demo 17 — rendering without three", () => {
  test.skip(({ browserName }) => browserName !== "chromium", "WebGPU/SwiftShader path is chromium-only");

  for (const lighting of ["frame", "volume"] as const) {
    test(`CC0 cloud + explosion, procedural level set, '${lighting}' lighting: shadows both ways, fire light, both composites`, async ({ page }) => {
      const errors = collectErrors(page);
      const state = await openDone(page, `${BASE}&cloud=cc0&levelset=procedural&lighting=${lighting}`);
      mkdirSync(SHOTS, { recursive: true });
      await page.screenshot({ path: path.join(SHOTS, `cc0-${lighting}.png`) });
      console.log(`demo 17 (cc0, ${lighting}): ${JSON.stringify({ ...state.info.checks, froxels: state.info.froxels })}`);
      checkScene(state, errors, lighting);
      expect(state.info.cloud).toBe("cc0");
      expect(state.info.levelSet).toBe("procedural");
    });
  }

  test("the WDAS cloud and the OpenVDB icosahedron (git-ignored fixtures)", async ({ page }) => {
    const wdas = await page.request.head("/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb");
    const ico = await page.request.head("/fixtures/vdb-samples/icosahedron.vdb");
    test.skip(!wdas.ok() || !ico.ok(), "fixtures missing: pnpm fixtures:wdas (then Fp8) and pnpm fixtures:vdb-samples");
    const errors = collectErrors(page);
    const state = await openDone(page, BASE);
    mkdirSync(SHOTS, { recursive: true });
    await page.screenshot({ path: path.join(SHOTS, "wdas-frame.png") });
    console.log(`demo 17 (wdas): ${JSON.stringify({ ...state.info.checks, froxels: state.info.froxels })}`);
    checkScene(state, errors, "frame");
    expect(state.info.cloud).toBe("wdas");
    expect(state.info.levelSet).toBe("icosahedron");
  });
});
