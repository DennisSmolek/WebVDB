import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { encodePng } from "../scripts/k3/png.mjs";

// K6.6 (design §3.9): level-set surfaces and point sprites through the core's raw passes
// (`vdbCreateLevelSetPassGpu`, `vdbCreatePointPassGpu`, no three objects), on the K3.7 harness with
// `?backend=core` (examples/src/harness/k3-levelset.ts + k6-surface-core.ts). Gates: the raw pass against the
// K6.6 surface twins (IoU, hits, normals, NDC depth, shade), against three's VDBLevelSetMaterial on the same
// device (`&vsThree=1`: the K3 path), the points' drawn count and disc coverage, the sprite WGSL against its f32
// twin, and a fail-before mutation for each twin term (`?mutate=`). PNGs of the core renders go to
// artifacts/k6/ours-step7-core/ (`node scripts/k3/compare-step7.mjs --ours artifacts/k6/ours-step7-core`).
const PAGE_URL = "/src/harness/k3-levelset.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k6", "ours-step7-core");
const has = (rel: string) => existsSync(path.join(repoRoot, "fixtures", rel));

interface Twin {
  gpuHits: number;
  cpuHits: number;
  iou: number;
  hitMaxVoxels: number;
  normalMeanDeg: number;
  normalP99Deg: number;
  normalMaxDeg: number;
  normalInteriorMaxDeg: number;
  analytic?: { meanDeg: number; maxDeg: number; radiusSpreadVoxels: number };
  depthMaxAbs: number;
  shadeMaxAbs: number;
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  twin?: Twin;
  vsThree?: { coreHits: number; threeHits: number; iou: number; hitMaxVoxels: number; normalMaxDeg: number };
  points?: { count: number; totalCount: number; covered: number; centresCovered: number; centresTotal: number; stray: number; drawn?: number };
  colors?: { checked: number; bad: number };
  sprite?: { compared: number; maxUlp: number; over: number; bound: number };
  pixelScale?: { core: number; three: number };
  uncaptured: string[];
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_LEVELSET__?: { done: boolean } }).__VDB_K3_LEVELSET__?.done === true, undefined, { timeout: 600_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_LEVELSET__?: unknown }).__VDB_K3_LEVELSET__)) as Result;
}

const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64 chars` : undefined }, null, 2);

function save(name: string, r: Result): Buffer {
  const png = Buffer.from(encodePng({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) }) as Uint8Array);
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, `${name}.png`), png);
  return png;
}

/** The level-set gates of a core probe run (the K3.7 bars plus depth and shade). */
function checkLevelSet(r: Result, minHits: number): void {
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.inPageIdentical, detail).toBe(true);
  const t = r.twin!;
  expect(t.gpuHits, detail).toBeGreaterThan(minHits);
  expect(t.iou, detail).toBeGreaterThanOrEqual(0.99);
  expect(t.hitMaxVoxels, detail).toBeLessThan(0.01);
  expect(t.normalP99Deg, detail).toBeLessThan(0.01);
  expect(t.normalInteriorMaxDeg, detail).toBeLessThan(0.1);
  // frag_depth (depth32float) vs the twin's clip.z / clip.w of the GPU hit: f32 rounding of the projection.
  expect(t.depthMaxAbs, detail).toBeLessThan(1e-6);
  // The shade pipeline vs Lambert + ambient of the G-buffer pipeline's normal (same trace).
  expect(t.shadeMaxAbs, detail).toBeLessThan(1e-4);
}

test("K6.6 level set (raw pass): the teapot matches the surface twins and three's material on the same device", async ({ page }) => {
  test.skip(!has("vdb-samples/utahteapot.vdb"), "fixtures/vdb-samples/utahteapot.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(900_000);
  const r = await run(page, "?scene=teapot&twin=1&backend=core&vsThree=1");
  checkLevelSet(r, 5_000);
  const v = r.vsThree!;
  // The K3 path (VDBLevelSetMaterial, now on the same core WGSL): the same pixels, hits and normals.
  expect(v.iou, detailOf(r)).toBeGreaterThanOrEqual(0.99);
  expect(v.hitMaxVoxels, detailOf(r)).toBeLessThan(0.01);
  expect(v.normalMaxDeg, detailOf(r)).toBeLessThan(0.1);
  const again = await run(page, "?scene=teapot&backend=core");
  expect(save("k6-levelset-teapot-run1", r).equals(save("k6-levelset-teapot-run2", again)), "two page loads render byte-identical images").toBe(true);
  const t = r.twin!;
  console.log(
    `[k6-surface] ${r.adapter}: teapot raw pass IoU ${t.iou.toFixed(5)} (${t.gpuHits}/${t.cpuHits} px), hit max ${t.hitMaxVoxels.toExponential(2)} voxels, normal p99 ${t.normalP99Deg.toExponential(2)}° interior max ${t.normalInteriorMaxDeg.toExponential(2)}°, depth max ${t.depthMaxAbs.toExponential(2)}, shade max ${t.shadeMaxAbs.toExponential(2)}; vs three IoU ${v.iou.toFixed(5)} (${v.coreHits}/${v.threeHits}), hit max ${v.hitMaxVoxels.toExponential(2)} voxels, normal max ${v.normalMaxDeg.toExponential(2)}°`,
  );
});

test("K6.6 level set (raw pass): a rotated, non-uniformly scaled sphere: twins, analytic W^T normals, three", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere.vdb"), "fixtures/vdb-samples/sphere.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(600_000);
  const r = await run(page, "?scene=sphere&twin=1&aniso=1&w=192&h=108&backend=core&vsThree=1");
  checkLevelSet(r, 2_000);
  expect(r.twin!.analytic!.meanDeg, detailOf(r)).toBeLessThan(0.1);
  expect(r.twin!.analytic!.maxDeg, detailOf(r)).toBeLessThan(1);
  expect(r.vsThree!.iou, detailOf(r)).toBeGreaterThanOrEqual(0.99);
  console.log(`[k6-surface] sphere (aniso): analytic mean ${r.twin!.analytic!.meanDeg.toFixed(4)}° max ${r.twin!.analytic!.maxDeg.toFixed(4)}°; vs three IoU ${r.vsThree!.iou.toFixed(5)}`);
});

// Fail-before: each mutation breaks one twin term; the gate it targets must fail on the same GPU output.
const MUTATIONS: { mutate: string; fails: (r: Result) => boolean }[] = [
  { mutate: "pixel-corner", fails: (r) => r.twin!.hitMaxVoxels >= 0.01 || r.twin!.iou < 0.99 },
  { mutate: "ray-ortho", fails: (r) => r.twin!.iou < 0.99 },
  { mutate: "normal-view", fails: (r) => r.twin!.normalP99Deg >= 0.01 },
  { mutate: "depth-linear", fails: (r) => r.twin!.depthMaxAbs >= 1e-6 },
  { mutate: "shade-sign", fails: (r) => r.twin!.shadeMaxAbs >= 1e-4 },
];
for (const m of MUTATIONS) {
  test(`K6.6 level set (raw pass): mutation ${m.mutate} fails its gate`, async ({ page }) => {
    test.skip(!has("vdb-samples/utahteapot.vdb"), "fixtures/vdb-samples/utahteapot.vdb missing (pnpm fixtures:vdb-samples)");
    test.setTimeout(600_000);
    const r = await run(page, `?scene=teapot&twin=1&backend=core&w=192&h=108&mutate=${m.mutate}`);
    expect(r.error, detailOf(r)).toBeUndefined();
    expect(m.fails(r), detailOf(r)).toBe(true);
  });
}

test("K6.6 sprites: the core sprite WGSL matches its f32 twin; each mutation fails", async ({ page }) => {
  test.setTimeout(120_000);
  const r = await run(page, "?scene=sprite-probe&backend=core");
  expect(r.error, detailOf(r)).toBeUndefined();
  expect(r.uncaptured, detailOf(r)).toEqual([]);
  expect(r.sprite!.compared).toBe(4096 * 4);
  expect(r.sprite!.over, detailOf(r)).toBe(0);
  for (const mutate of ["sprite-half", "sprite-min", "sprite-w"]) {
    const bad = await run(page, `?scene=sprite-probe&backend=core&mutate=${mutate}`);
    expect(bad.error, detailOf(bad)).toBeUndefined();
    expect(bad.sprite!.over, `${mutate}: ${detailOf(bad)}`).toBeGreaterThan(0);
  }
  console.log(`[k6-surface] sprite probe: ${r.sprite!.compared} values, max ${r.sprite!.maxUlp.toFixed(2)} f32 ulp (bound ${r.sprite!.bound})`);
});

test("K6.6 points (raw pass): sphere_points draws every point; centres inside each disc covered, nothing outside", async ({ page }) => {
  test.skip(!has("vdb-samples/sphere_points.vdb"), "fixtures/vdb-samples/sphere_points.vdb missing (pnpm fixtures:vdb-samples)");
  test.setTimeout(300_000);
  const r = await run(page, "?scene=points&backend=core");
  const detail = detailOf(r);
  expect(r.error, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.inPageIdentical, detail).toBe(true);
  const p = r.points!;
  expect(p.totalCount, detail).toBe(9802);
  expect(p.drawn, detail).toBe(9802);
  expect(p.centresTotal, detail).toBeGreaterThan(9000);
  expect(p.centresCovered, detail).toBe(p.centresTotal);
  expect(p.stray, detail).toBe(0);
  // vdbPointPixelScale (projection[5]) equals VDBPoints' fov-based scale for a perspective camera.
  expect(Math.abs(r.pixelScale!.core / r.pixelScale!.three - 1), detail).toBeLessThan(1e-9);
  save("k6-points-sphere", r);
  console.log(`[k6-surface] points: ${p.drawn} drawn, ${p.centresCovered}/${p.centresTotal} centres covered, ${p.covered} px covered, ${p.stray} stray`);
});

test("K6.6 points (raw pass): the colour attribute reaches each point's pixels", async ({ page }) => {
  test.setTimeout(120_000);
  const r = await run(page, "?scene=points-color&backend=core");
  expect(r.error, detailOf(r)).toBeUndefined();
  expect(r.uncaptured, detailOf(r)).toEqual([]);
  expect(r.colors!.checked).toBe(64);
  expect(r.colors!.bad, detailOf(r)).toBe(0);
});

// His Step 7 parity scenes through the raw passes (1280x720). Opt in with K3_FRAME_PARITY=1, then
// `node scripts/k3/compare-step7.mjs --ours artifacts/k6/ours-step7-core`.
const PARITY_SCENES = ["step7-utah-teapot-surface-normals", "step7-venus-statue-surface-normals", "step7-sphere-points-isolated"];
for (const scene of PARITY_SCENES) {
  test(`K6.6 parity render (raw pass): ${scene}`, async ({ page }) => {
    test.skip(process.env.K3_FRAME_PARITY !== "1", "set K3_FRAME_PARITY=1");
    test.skip(scene.includes("venus") && !has("k3/venusstatue.vdb"), "fixtures/k3/venusstatue.vdb missing");
    test.skip(!scene.includes("venus") && !has(scene.includes("points") ? "vdb-samples/sphere_points.vdb" : "vdb-samples/utahteapot.vdb"), "fixture missing");
    test.setTimeout(1_200_000);
    const r = await run(page, `?scene=${scene}&backend=core`);
    expect(r.error, detailOf(r)).toBeUndefined();
    expect(r.uncaptured).toEqual([]);
    expect(r.inPageIdentical).toBe(true);
    save(scene, r);
    writeFileSync(path.join(OURS, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
  });
}
