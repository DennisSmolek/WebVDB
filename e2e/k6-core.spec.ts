import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages, diffHeatmap } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K6.1 (design §7.1): the core frame kernels on raw WebGPU (examples/src/harness/k6-core.ts), each against its
// CPU twin (tools/frame) within 2 f16 ulp; every probe's mutation must fail; every pipeline variant compiles;
// a second identical update builds nothing; two loads are deterministic.
const PAGE_URL = "/src/harness/k6-core.html";

interface Stats {
  compared: number;
  covered: number;
  over: number;
  maxUlp: number;
  maxAbs: number;
  bound: string;
  firstBad?: unknown;
}
interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  stats?: Stats;
  renderStats?: Stats;
  gridStats?: Stats;
  transmittanceStats?: Stats;
  workStats?: Stats;
  mapStats?: Stats;
  bufferStats?: Stats;
  environmentGridStats?: Stats;
  mipStats?: Stats;
  environment?: { runs: { prefilter: number; blur: number; reduce: number }[]; count: number; mode: number; valid: number; directions: number[][]; colors: number[][] };
  frame?: { cascades: number; redraw: unknown };
  builds?: { first: number; second: number };
  compile?: { variants: number; errors: string[] };
  receiverStats?: Stats;
  shadow?: { light: string; layers: number; shadowed: number; receiverShadowed: number; same: { drawn: number; waiting: number }; moved: { drawn: number; waiting: number }; cascades: number; enabled: boolean };
  reduceStats?: Stats;
  probeStats?: Stats;
  irradianceStats?: Stats;
  emission?: { state: { count: number; resolution: number; emitters: number }; first: { directions: number; probes: boolean }; second: { directions: number; probes: boolean }; hookErrors: string[]; outside: number };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 300_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_CORE__?: { done: boolean } }).__VDB_K6_CORE__?.done === true, undefined, { timeout });
  return (await page.evaluate(() => (window as { __VDB_K6_CORE__?: unknown }).__VDB_K6_CORE__)) as Result;
}

const detail = (r: Result) => JSON.stringify(r, null, 2);

test("K6.1 every frame pipeline variant compiles", async ({ page }) => {
  // K6.5 added the shadow and emission kernels (with hooks): ~43 variants, minutes on SwiftShader under load
  test.setTimeout(900_000);
  const errors = collectErrors(page);
  const r = await run(page, "?probe=compile", 900_000);
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.compile!.errors, detail(r)).toEqual([]);
  expect(r.compile!.variants).toBeGreaterThanOrEqual(15);
  expect(r.uncaptured).toEqual([]);
  expect(errors).toEqual([]);
});

const PROBES: { name: string; query: string; minCovered: number; mutations: string[] }[] = [
  { name: "view chunks", query: "?probe=view", minCovered: 512, mutations: ["coverage", "mie", "octaves", "two-sided"] },
  { name: "gather (mixture, one group)", query: "?probe=gather&mix=mixture&groups=1", minCovered: 300, mutations: ["mix", "phase", "extinction", "fade"] },
  { name: "gather (over, one group)", query: "?probe=gather&mix=over&groups=1", minCovered: 300, mutations: ["mix"] },
  { name: "gather (mixture, two groups)", query: "?probe=gather&mix=mixture&groups=2", minCovered: 300, mutations: ["mix"] },
  { name: "gather (over, two groups, no mips)", query: "?probe=gather&mix=over&groups=2&mips=0", minCovered: 300, mutations: ["extinction"] },
  { name: "resolve", query: "?probe=resolve&temporal=0", minCovered: 512, mutations: ["scan-stop"] },
  { name: "resolve (temporal)", query: "?probe=resolve&temporal=1", minCovered: 512, mutations: ["temporal-clamp", "scan-stop"] },
  { name: "integrate", query: "?probe=integrate", minCovered: 512, mutations: ["coverage", "taps"] },
  // K6.1d: the gather on the core sparse source (K6.7), volume A skipping empty space (hdda), B wrapped (no skip)
  { name: "gather sparse (lod none, one group, hdda)", query: "?probe=gather&source=sparse&lod=none&groups=1", minCovered: 200, mutations: ["skip", "extinction", "phase"] },
  { name: "gather sparse (node-average, two groups)", query: "?probe=gather&source=sparse&lod=node-average&groups=2", minCovered: 300, mutations: ["mix"] },
  { name: "gather sparse (coarse-pyramid, one group)", query: "?probe=gather&source=sparse&lod=coarse-pyramid&groups=1", minCovered: 300, mutations: ["extinction"] },
];

for (const p of PROBES) {
  test(`K6.1 probe: ${p.name} matches its CPU twin; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.stats!.compared, detail(r)).toBeGreaterThanOrEqual(512);
    expect(r.stats!.covered, detail(r)).toBeGreaterThanOrEqual(p.minCovered);
    expect(r.stats!.over, detail(r)).toBeLessThanOrEqual(Math.floor(r.stats!.compared * 0.001));
    if (r.renderStats) {
      expect(r.renderStats.over, detail(r)).toBe(0);
      expect(r.renderStats.covered, detail(r)).toBeGreaterThan(30);
    }
    console.log(`[k6-core] ${p.name} on ${r.adapter}: ${r.stats!.compared} texels, ${r.stats!.covered} non-zero, max ${r.stats!.maxUlp.toFixed(2)} f16 ulp, ${r.stats!.over} over`);
    for (const m of p.mutations) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad.stats!.over, `mutation ${m} must fail: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad.stats!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.2 (design §7.1): 'frame' lighting, each kernel against its twin from the read-back input of the kernel before it:
// the shadow grid (vdbFrameGridDensityVoxel), the transmittance (vdbFrameTransmittanceVoxel3d on the read-back grid) and
// the fused gather (his density + lighting twins on the read-back transmittance). `mutations` maps a twin term to the
// stats it must break. A second identical update skips the grid and the transmittance (his redraw signatures).
type StatsKey = "stats" | "gridStats" | "transmittanceStats";
const FRAME_PROBES: { name: string; query: string; cascades: number; mutations: Record<string, StatsKey> }[] = [
  {
    name: "dense, one group, one cascade",
    query: "?probe=frame",
    cascades: 1,
    mutations: { "grid-absorbance": "gridStats", "grid-lod": "gridStats", diffusion: "transmittanceStats", slots: "stats", sky: "stats", extinction: "stats" },
  },
  { name: "dense, two groups (partials), two cascades", query: "?probe=frame&groups=2&cascades=2", cascades: 2, mutations: { slots: "stats" } },
  { name: "sparse, one group", query: "?probe=frame&source=sparse", cascades: 1, mutations: { "grid-absorbance": "gridStats", diffusion: "transmittanceStats" } },
  { name: "sparse, two groups, single scattering", query: "?probe=frame&source=sparse&groups=2&ms=0", cascades: 1, mutations: { slots: "stats" } },
];

for (const p of FRAME_PROBES) {
  test(`K6.2 probe: 'frame' lighting (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.frame!.cascades, detail(r)).toBe(p.cascades);
    const redraw = r.frame!.redraw as { shadowDensity: { drawn: number; skipped: number }; transmittance: { drawn: number; skipped: number } };
    expect(redraw.shadowDensity, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    expect(redraw.transmittance, detail(r)).toEqual({ drawn: 1, skipped: 1 });
    for (const key of ["gridStats", "transmittanceStats", "stats"] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "stats" ? 200 : 250);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] frame ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.3 (design §3.6): phase functions (the Lorenz-Mie LUT, per-channel g) and every multiple scattering mode in both
// lightings, each kernel against its twin from the read-back input before it ('mlp' / 'multi-scale': grid → mips →
// transmittance). `mutations` maps a twin term to the stats it must break.
type K63Key = "stats" | "gridStats" | "transmittanceStats" | "environmentGridStats" | "mipStats";
const K63_PROBES: { name: string; query: string; keys: K63Key[]; mutations: Record<string, K63Key> }[] = [
  { name: "gather, 'lorenz-mie'", query: "?probe=gather&phase=lorenz-mie", keys: ["stats"], mutations: { mie: "stats" } },
  { name: "gather, 'lorenz-mie-rgb'", query: "?probe=gather&phase=lorenz-mie-rgb&source=sparse&lod=none", keys: ["stats"], mutations: { mie: "stats" } },
  { name: "gather, 'octaves'", query: "?probe=gather&ms=octaves", keys: ["stats"], mutations: { octaves: "stats" } },
  { name: "gather, 'octaves' with the environment ('directional')", query: "?probe=gather&ms=octaves&env=directional", keys: ["stats"], mutations: { octaves: "stats" } },
  { name: "gather, 'two-sided'", query: "?probe=gather&ms=two-sided", keys: ["stats"], mutations: { "two-sided": "stats" } },
  { name: "frame, 'lorenz-mie'", query: "?probe=frame&phase=lorenz-mie", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { mie: "stats" } },
  { name: "frame, 'lorenz-mie-rgb'", query: "?probe=frame&phase=lorenz-mie-rgb", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { mie: "stats" } },
  { name: "frame, henyey-greenstein, phaseGExponent 1.5", query: "?probe=frame&phase=henyey-greenstein&exponent=1.5", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { slots: "stats" } },
  { name: "frame, 'octaves'", query: "?probe=frame&ms=octaves", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { octaves: "transmittanceStats" } },
  {
    name: "frame, 'octaves', environment 'directional' (slot + grid)",
    query: "?probe=frame&ms=octaves&env=directional",
    keys: ["gridStats", "transmittanceStats", "environmentGridStats", "stats"],
    mutations: { octaves: "environmentGridStats" },
  },
  { name: "frame, 'two-sided'", query: "?probe=frame&ms=two-sided", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "two-sided": "transmittanceStats" } },
  { name: "frame, 'two-sided', sparse, environment 'spherical-harmonics'", query: "?probe=frame&ms=two-sided&source=sparse&env=spherical-harmonics", keys: ["gridStats", "transmittanceStats", "environmentGridStats", "stats"], mutations: {} },
  {
    name: "frame, 'mlp' (his network)",
    query: "?probe=frame&ms=mlp",
    keys: ["gridStats", "mipStats", "transmittanceStats", "stats"],
    mutations: { "mlp-mip": "mipStats", "mlp-weights": "transmittanceStats", "mlp-depth": "transmittanceStats" },
  },
  { name: "frame, 'neural' (= 'mlp'), sparse, two groups", query: "?probe=frame&ms=neural&source=sparse&groups=2", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "mlp-weights": "transmittanceStats" } },
  { name: "frame, 'multi-scale' (edge zero)", query: "?probe=frame&ms=multi-scale", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "ms-edge": "mipStats", "ms-levels": "transmittanceStats" } },
  { name: "frame, 'multi-scale' (edge clamp, 4 levels)", query: "?probe=frame&ms=multi-scale&edge=clamp&levels=4", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { "ms-edge": "mipStats" } },
];

for (const p of K63_PROBES) {
  test(`K6.3 probe: ${p.name} matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.warnings, detail(r)).toEqual([]);
    for (const key of p.keys) {
      const s = r[key]!;
      expect(s, `${key} missing: ${detail(r)}`).toBeDefined();
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "mipStats" ? 64 : 512);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] K6.3 ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// K6.5 (design §2.6, §3.8): volume shadow maps (each light kind) and the emission kernels, each against its twin from
// the read-back input of the kernel before it (grid → map → receivers; directions → reduce → probes → receivers).
type K65Key = "stats" | "receiverStats" | "reduceStats" | "probeStats" | "irradianceStats";
const K65_PROBES: { name: string; query: string; keys: K65Key[]; mutations: Record<string, K65Key> }[] = [
  { name: "shadow map, directional (2 cascades), dense", query: "?probe=shadow&light=dir&cascades=2", keys: ["stats", "receiverStats"], mutations: { "map-steps": "stats", "map-far": "stats", "receiver-far": "receiverStats", "receiver-layer": "receiverStats" } },
  { name: "shadow map, spot, dense", query: "?probe=shadow&light=spot", keys: ["stats", "receiverStats"], mutations: { "map-far": "stats", "receiver-far": "receiverStats" } },
  { name: "shadow map, point (cube), dense", query: "?probe=shadow&light=point", keys: ["stats", "receiverStats"], mutations: { "map-steps": "stats", "receiver-far": "receiverStats" } },
  { name: "shadow map, directional (4 cascades), sparse", query: "?probe=shadow&light=dir&cascades=4&source=sparse", keys: ["stats", "receiverStats"], mutations: { "receiver-layer": "receiverStats" } },
  {
    name: "emission, dense, soot LUT",
    query: "?probe=emission",
    keys: ["stats", "reduceStats", "probeStats", "irradianceStats"],
    mutations: { "emission-direction": "stats", "emission-color": "stats", "emission-reduce": "reduceStats", "emission-probe": "probeStats", "emission-irradiance": "irradianceStats" },
  },
  { name: "emission, sparse, 2 probe cascades", query: "?probe=emission&source=sparse&cascades=2", keys: ["stats", "reduceStats", "probeStats", "irradianceStats"], mutations: { "emission-color": "stats" } },
  { name: "emission, dense, user WGSL hook", query: "?probe=emission&hook=1", keys: ["stats", "reduceStats", "probeStats", "irradianceStats"], mutations: { "emission-color": "stats" } },
];

for (const p of K65_PROBES) {
  test(`K6.5 probe: ${p.name} matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.warnings, detail(r)).toEqual([]);
    for (const key of p.keys) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "reduceStats" ? 2 : key === "stats" && p.query.includes("emission") ? 32 : 512);
      expect(s.over, `${key}: ${detail(r)}`).toBe(0);
      console.log(`[k6-core] K6.5 ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toExponential(2)} (${s.bound}), max |d| ${s.maxAbs.toExponential(2)}, ${s.over} over`);
    }
    if (r.shadow) {
      console.log(`[k6-core] K6.5 ${p.name}: ${JSON.stringify(r.shadow)}`);
      expect(r.shadow.enabled, detail(r)).toBe(true);
      expect(r.shadow.shadowed, detail(r)).toBeGreaterThan(20);
      expect(r.shadow.receiverShadowed, detail(r)).toBeGreaterThan(20);
      // nothing changed: no piece; the light moved: one piece (a directional layout draws its cascades in one piece)
      expect(r.shadow.same, detail(r)).toEqual({ drawn: 0, waiting: 0 });
      expect(r.shadow.moved.drawn, detail(r)).toBeGreaterThanOrEqual(1);
    }
    if (r.emission) {
      console.log(`[k6-core] K6.5 ${p.name}: ${JSON.stringify(r.emission)}`);
      expect(r.emission.first, detail(r)).toEqual({ directions: 1, probes: true });
      expect(r.emission.second, detail(r)).toEqual({ directions: 0, probes: false });
      expect(r.emission.outside, detail(r)).toBeGreaterThan(20);
      if (p.query.includes("hook=1")) {
        expect(r.emission.hookErrors.length, detail(r)).toBe(3);
        expect(r.emission.hookErrors[0], detail(r)).toMatch(/VDB_BAD_WGSL.*line 1: top-level name 'emissionColour'/);
        expect(r.emission.hookErrors[1], detail(r)).toMatch(/VDB_BAD_WGSL.*line 3/);
        expect(r.emission.hookErrors[2], detail(r)).toMatch(/VDB_BAD_WGSL.*vdbUserEmissionColor\(f32, f32\)/);
      }
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(0);
    }
    expect(errors).toEqual([]);
  });
}

test("K6.1 a second identical update builds no pipeline or module; two loads agree", async ({ page }) => {
  test.setTimeout(600_000);
  const r = await run(page, "?probe=gather&swap=1");
  expect(r.error, detail(r)).toBeUndefined();
  expect(r.builds!.first, detail(r)).toBeGreaterThan(0);
  expect(r.builds!.second, detail(r)).toBe(0);
  const again = await run(page, "?probe=gather&swap=1");
  expect(again.stats, detail(again)).toEqual(r.stats);
});

// ---------------------------------------------------------------------------------------------------------------
// Gate 2 (design §7.2): his K3 Step 1 scenes through the k3-frame harness with `?backend=core` vs `?backend=tsl` on
// the same device: PSNR ≥ 45 dB and ≤ 0.2 % pixels > 4/255, with froxelMix 'mixture' and 'over'. The scenes need his
// CC BY-SA bunny (fixtures/k3/bunny_cloud.wvdb); full size (1280x720, 512x512x24): opt in with K6_CORE_AB=1.
// 'frame'-lighting scenes render with ?lighting=volume (K6.1 has 'volume' lighting only). PNGs, diff heatmaps and a
// JSON per scene go to artifacts/k6/ab/.
// ---------------------------------------------------------------------------------------------------------------
const AB_SCENES = [
  "step1-lighting-volume",
  "step1-lights-directional",
  "step1-multiple-scattering-off",
  "step1-volume-dir-ms-off",
  "step1-volume-dir-ms-off-isolated",
  "step1-volume-dir-ms-on",
  "step1-volume-dir-ms-on-isolated",
];
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const AB_DIR = path.join(REPO, "artifacts", "k6", "ab");

interface K3Result {
  done: boolean;
  error?: string;
  width?: number;
  height?: number;
  rgba?: string;
  inPageIdentical?: boolean;
  uncaptured: string[];
  warnings: string[];
}

async function k3(page: Page, query: string): Promise<K3Result> {
  await page.goto(`/src/harness/k3-frame.html${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, { timeout: 600_000 });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as K3Result;
}
const k3Image = (r: K3Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });
const k3Detail = (r: K3Result) => JSON.stringify({ ...r, rgba: undefined });

for (const scene of AB_SCENES) {
  for (const mix of ["mixture", "over"] as const) {
    test(`K6.1 A/B core vs TSL: ${scene} (${mix})`, async ({ page }) => {
      test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3/bunny_cloud.wvdb)");
      test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
      test.setTimeout(1_200_000);
      const base = `?scene=${scene}&lighting=volume`;
      const tsl = await k3(page, `${base}&backend=tsl`);
      const core = await k3(page, `${base}&backend=core&froxelMix=${mix}`);
      expect(tsl.error, k3Detail(tsl)).toBeUndefined();
      expect(core.error, k3Detail(core)).toBeUndefined();
      expect(core.uncaptured, k3Detail(core)).toEqual([]);
      expect(core.inPageIdentical, k3Detail(core)).toBe(true);
      const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
      mkdirSync(AB_DIR, { recursive: true });
      writeFileSync(path.join(AB_DIR, `${scene}-tsl.png`), encodePng(k3Image(tsl)));
      writeFileSync(path.join(AB_DIR, `${scene}-core-${mix}.png`), encodePng(k3Image(core)));
      writeFileSync(path.join(AB_DIR, `${scene}-core-${mix}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
      writeFileSync(path.join(AB_DIR, `${scene}-${mix}.json`), JSON.stringify({ scene, mix, ...cmp, coreWarnings: core.warnings }, null, 2));
      if (mix === "mixture") {
        // Gate 3 (his captures): scripts/k3/compare.mjs --ours artifacts/k6/ours-{core,tsl} --reference fixtures/k3-reference/75e7345
        for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
          mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
          writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), encodePng(k3Image(r)));
          writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
        }
      }
      console.log(`[k6-ab] ${scene} ${mix}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
      expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
    });
  }
}

// ---------------------------------------------------------------------------------------------------------------
// K6.2 gate 2 (design §7.2): his K3 Step 3 'frame' lighting scenes through `?backend=core` vs `?backend=tsl` (same
// device, same bar: PSNR ≥ 45 dB, ≤ 0.2 % pixels > 4/255), dense and (K6.1d) sparse sources. The environment scene
// (`*-env-isolated`) needs K6.2b's GPU environment and is not here. Opt in with K6_CORE_AB=1; PNGs, heatmaps and JSON
// go to artifacts/k6/ab-frame/, and the mixture images to artifacts/k6/ours-{core,tsl}/ for scripts/k3/compare.mjs.
// ---------------------------------------------------------------------------------------------------------------
const AB_FRAME = [
  { scene: "step3-default-dir-isolated", source: "dense" },
  { scene: "step3-shadow-cascades-4-dir-isolated", source: "dense" },
  { scene: "step3-shadow-grid-high-dir-isolated", source: "dense" },
  { scene: "step3-two-bunnies-dir-isolated", source: "dense" },
  { scene: "step3-default-dir-isolated", source: "sparse" },
  { scene: "step3-two-bunnies-dir-isolated", source: "sparse" },
  { scene: "step1-volume-dir-ms-on-isolated", source: "sparse" },
] as const;
const AB_FRAME_DIR = path.join(REPO, "artifacts", "k6", "ab-frame");

for (const { scene, source } of AB_FRAME) {
  test(`K6.2 A/B core vs TSL: ${scene} (${source})`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3/bunny_cloud.wvdb)");
    test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
    test.setTimeout(1_800_000);
    const base = `?scene=${scene}&source=${source}`;
    const tsl = await k3(page, `${base}&backend=tsl`);
    const core = await k3(page, `${base}&backend=core`);
    expect(tsl.error, k3Detail(tsl)).toBeUndefined();
    expect(core.error, k3Detail(core)).toBeUndefined();
    expect(core.uncaptured, k3Detail(core)).toEqual([]);
    expect(core.inPageIdentical, k3Detail(core)).toBe(true);
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    const name = `${scene}-${source}`;
    mkdirSync(AB_FRAME_DIR, { recursive: true });
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-tsl.png`), encodePng(k3Image(tsl)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-core.png`), encodePng(k3Image(core)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}.json`), JSON.stringify({ scene, source, ...cmp, coreWarnings: core.warnings }, null, 2));
    if (source === "dense") {
      for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
        mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
        writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), encodePng(k3Image(r)));
        writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
      }
    }
    console.log(`[k6-ab-frame] ${name}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
    expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// K6.3 gate 2 (design §7.2): his K3 Step 4 scenes (phase functions, multiple scattering modes, octave settings, his
// 'neural' default) through `?backend=core` vs `?backend=tsl`, same device, same bar (PSNR ≥ 45 dB, ≤ 0.2 % pixels >
// 4/255). Opt in with K6_CORE_AB=1; images and JSON go to artifacts/k6/ab-step4/ and the core / TSL frames to
// artifacts/k6/ours-{core,tsl}/ for scripts/k3/compare.mjs (his captures: the regression report).
// ---------------------------------------------------------------------------------------------------------------
const AB_STEP4 = [
  "step4-default-neural-dir-isolated",
  "step4-frame-phase-dual-lobe-dir-isolated",
  "step4-frame-phase-dual-lobe-back-g-high-dir-isolated",
  "step4-frame-phase-dual-lobe-back-weight-high-dir-isolated",
  "step4-frame-phase-g-exponent-dir-isolated",
  "step4-frame-phase-g-low-dir-isolated",
  "step4-frame-phase-isotropic-dir-isolated",
  "step4-frame-phase-lorenz-mie-dir-isolated",
  "step4-frame-phase-lorenz-mie-rgb-dir-isolated",
  "step4-lighting-volume-phase-g-low-dir-isolated",
  "step4-multiple-scattering-octaves-dir-isolated",
  "step4-multiple-scattering-two-sided-dir-isolated",
  "step4-octave-contribution-high-dir-isolated",
  "step4-octave-extinction-high-dir-isolated",
  "step4-octave-phase-low-dir-isolated",
  "step4-octaves-1-dir-isolated",
  "step4-octaves-8-dir-isolated",
];
const AB_STEP4_DIR = path.join(REPO, "artifacts", "k6", "ab-step4");

for (const scene of AB_STEP4) {
  test(`K6.3 A/B core vs TSL: ${scene}`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3/bunny_cloud.wvdb)");
    test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing");
    test.setTimeout(1_800_000);
    const tsl = await k3(page, `?scene=${scene}&backend=tsl`);
    const core = await k3(page, `?scene=${scene}&backend=core`);
    expect(tsl.error, k3Detail(tsl)).toBeUndefined();
    expect(core.error, k3Detail(core)).toBeUndefined();
    expect(core.uncaptured, k3Detail(core)).toEqual([]);
    expect(core.inPageIdentical, k3Detail(core)).toBe(true);
    // K6.3: no 'not in the core yet' warnings (phase, multiple scattering)
    expect(core.warnings.filter((w) => /K6\.3/.test(w)), k3Detail(core)).toEqual([]);
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    mkdirSync(AB_STEP4_DIR, { recursive: true });
    writeFileSync(path.join(AB_STEP4_DIR, `${scene}-tsl.png`), encodePng(k3Image(tsl)));
    writeFileSync(path.join(AB_STEP4_DIR, `${scene}-core.png`), encodePng(k3Image(core)));
    writeFileSync(path.join(AB_STEP4_DIR, `${scene}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
    writeFileSync(path.join(AB_STEP4_DIR, `${scene}.json`), JSON.stringify({ scene, ...cmp, coreWarnings: core.warnings }, null, 2));
    for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
      mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
      writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), encodePng(k3Image(r)));
      writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
    }
    console.log(`[k6-ab-step4] ${scene}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
    expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}

// K6.1d (design §7.2): the WDAS cloud (fixtures/wdas/wdas_cloud_quarter_fp8.nvdb, Fp8, sparse at full resolution) through
// examples/src/harness/k3-timing.html `?image=1`, core vs TSL on the same device, 'volume' and 'frame' lighting, the
// sparse source ('coarse-pyramid', HDDA on), 1280x720 with his 512x512x24 froxels (at 640x360 with 160² froxels the
// silhouette band of V80 alone is 0.3 % of the pixels). Opt in with K6_CORE_AB=1; skipped without the fixture.
for (const lighting of ["volume", "frame"] as const) {
  test(`K6.1d A/B core vs TSL: WDAS cloud, sparse, ${lighting} lighting`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1");
    test.skip(!existsSync(path.join(REPO, "fixtures", "wdas", "wdas_cloud_quarter_fp8.nvdb")), "fixtures/wdas/wdas_cloud_quarter_fp8.nvdb missing");
    test.setTimeout(1_800_000);
    const shot = async (backend: string) => {
      await page.goto(`/src/harness/k3-timing.html?asset=wdas&source=sparse&lod=pyramid&hdda=1&lighting=${lighting}&w=1280&h=720&froxels=512x512x24&image=1&backend=${backend}`);
      await page.waitForFunction(() => (window as { __VDB_K3_TIMING__?: { done: boolean } }).__VDB_K3_TIMING__?.done === true, undefined, { timeout: 1_800_000 });
      return (await page.evaluate(() => (window as { __VDB_K3_TIMING__?: unknown }).__VDB_K3_TIMING__)) as K3Result;
    };
    const tsl = await shot("tsl");
    const core = await shot("core");
    expect(tsl.error, k3Detail(tsl)).toBeUndefined();
    expect(core.error, k3Detail(core)).toBeUndefined();
    expect(core.uncaptured, k3Detail(core)).toEqual([]);
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    const name = `wdas-sparse-${lighting}`;
    mkdirSync(AB_FRAME_DIR, { recursive: true });
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-tsl.png`), encodePng(k3Image(tsl)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-core.png`), encodePng(k3Image(core)));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
    writeFileSync(path.join(AB_FRAME_DIR, `${name}.json`), JSON.stringify({ lighting, ...cmp, coreWarnings: core.warnings }, null, 2));
    console.log(`[k6-ab-wdas] ${name}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
    expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}

// K6.2b (design §3.5, Q3): the environment reduction on the GPU, each stage against its twin from the read-back input of
// the stage before it: the work map (exact-area equirect / supersampled cube), his 64 x 32 map (the GGX roughness-0.4
// convolution) and the environment buffer (his lights and the SH9). f32 outputs: 1e-3 relative (SwiftShader's
// transcendental and division precision; the twins are f64). The stages rerun only for their inputs.
type EnvKey = "workStats" | "mapStats" | "bufferStats";
const ENV_PROBES: { name: string; query: string; count: number; mutations: Record<string, EnvKey> }[] = [
  { name: "equirect rgba16float, 'directional'", query: "?probe=environment&mode=directional", count: 8, mutations: { "env-area": "workStats", "env-blur": "mapStats", "env-seeds": "bufferStats", "env-sh": "bufferStats" } },
  { name: "equirect, rotated and reflected, 'directional' pairs", query: "?probe=environment&mode=directional&pairs=1&rotate=1", count: 8, mutations: { "env-seeds": "bufferStats", "env-sh": "bufferStats" } },
  { name: "equirect rgba32float, rotated, SH L2 lights", query: "?probe=environment&mode=spherical-harmonics&order=2&rotate=1&format=rgba32float", count: 12, mutations: { "env-blur": "mapStats" } },
  { name: "equirect, SH L1 lights, 3 samples", query: "?probe=environment&mode=spherical-harmonics&order=1&samples=3", count: 6, mutations: {} },
  { name: "cube rgba16float, 'directional'", query: "?probe=environment&projection=cube&mode=directional&samples=5", count: 5, mutations: { "env-area": "workStats" } },
  { name: "equirect, 'outward' (SH only)", query: "?probe=environment&mode=outward", count: 0, mutations: { "env-sh": "bufferStats" } },
];

for (const p of ENV_PROBES) {
  test(`K6.2b probe: environment (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    for (const key of ["workStats", "mapStats", "bufferStats"] as const) console.log(`[k6-core] environment ${p.name} ${key}: ${JSON.stringify(r[key])}`);
    expect(r.environment!.valid, detail(r)).toBe(1);
    expect(r.environment!.count, detail(r)).toBe(p.count);
    for (const key of ["workStats", "mapStats", "bufferStats"] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(key === "bufferStats" ? 10 : 2048);
      // cube: a sub-sample near a face texel edge may pick the neighbour texel (its direction comes from WGSL sin / cos,
      // 2^-11 absolute): one sub-sample of k^2 per flipped texel, a few % of the work texels at most
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(key === "workStats" && p.query.includes("cube") ? Math.floor(s.compared * 0.02) : 0);
      console.log(`[k6-core] environment ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toExponential(2)} relative, ${s.over} over`);
    }
    // identical update: nothing; intensity: reduce; rotation: blur + reduce; revision: all three
    const runs = r.environment!.runs;
    expect(runs, detail(r)).toEqual([
      { prefilter: 1, blur: 1, reduce: 1 },
      { prefilter: 1, blur: 1, reduce: 1 },
      { prefilter: 1, blur: 1, reduce: 2 },
      { prefilter: 1, blur: 2, reduce: 3 },
      { prefilter: 2, blur: 3, reduce: 4 },
    ]);
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(0);
    }
    expect(errors).toEqual([]);
  });
}

// K6.2b: 'frame' lighting with an environment (a synthetic sky, rotated and reflected): the dominant 'directional' light in
// the next free slot (single scattering), the environment grid (vdbFrameEnvironmentGridVoxel3d on the read-back grid and
// buffer) and the 'outward' SH term of the fused gather; 'volume' lighting marches every environment light.
type FrameEnvKey = "stats" | "transmittanceStats" | "environmentGridStats";
const FRAME_ENV_PROBES: { name: string; query: string; grid: boolean; mutations: Record<string, FrameEnvKey> }[] = [
  { name: "'directional', single scattering (slot + grid)", query: "?probe=frame&env=directional&ms=0", grid: true, mutations: { "env-slot": "transmittanceStats", "env-grid": "environmentGridStats" } },
  { name: "'directional', diffusion (grid only), sparse", query: "?probe=frame&env=directional&source=sparse", grid: true, mutations: { "env-grid": "environmentGridStats" } },
  { name: "SH lights, single scattering, two groups", query: "?probe=frame&env=spherical-harmonics&ms=0&groups=2", grid: true, mutations: { "env-grid": "environmentGridStats" } },
  { name: "'outward' SH irradiance", query: "?probe=frame&env=outward", grid: false, mutations: { "env-outward": "stats" } },
];

for (const p of FRAME_ENV_PROBES) {
  test(`K6.2b probe: 'frame' lighting with an environment (${p.name}) matches its CPU twins; mutations fail`, async ({ page }) => {
    test.setTimeout(900_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.environment!.valid, detail(r)).toBe(1);
    const redraw = r.frame!.redraw as { environmentGrid: { drawn: number; skipped: number } };
    expect(redraw.environmentGrid, detail(r)).toEqual(p.grid ? { drawn: 1, skipped: 1 } : { drawn: 0, skipped: 0 });
    for (const key of ["transmittanceStats", "stats", ...(p.grid ? (["environmentGridStats"] as const) : [])] as const) {
      const s = r[key]!;
      expect(s.compared, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(512);
      expect(s.covered, `${key}: ${detail(r)}`).toBeGreaterThanOrEqual(200);
      expect(s.over, `${key}: ${detail(r)}`).toBeLessThanOrEqual(Math.floor(s.compared * 0.001));
      console.log(`[k6-core] frame env ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

const GATHER_ENV_PROBES: { name: string; query: string; mutations: string[] }[] = [
  { name: "'directional' lights, diffusion", query: "?probe=gather&env=directional", mutations: ["env-lights", "env-outward"] },
  { name: "SH lights, single scattering, sparse", query: "?probe=gather&env=spherical-harmonics&ms=0&source=sparse&lod=none", mutations: ["env-lights"] },
  { name: "'outward' SH irradiance", query: "?probe=gather&env=outward", mutations: [] },
];

for (const p of GATHER_ENV_PROBES) {
  test(`K6.2b probe: 'volume' gather with an environment (${p.name}) matches its CPU twin; mutations fail`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, p.query);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    expect(r.environment!.valid, detail(r)).toBe(1);
    expect(r.stats!.covered, detail(r)).toBeGreaterThanOrEqual(200);
    expect(r.stats!.over, detail(r)).toBeLessThanOrEqual(Math.floor(r.stats!.compared * 0.001));
    console.log(`[k6-core] gather env ${p.name} on ${r.adapter}: ${r.stats!.compared} texels, ${r.stats!.covered} non-zero, max ${r.stats!.maxUlp.toFixed(2)} f16 ulp, ${r.stats!.over} over`);
    for (const m of p.mutations) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad.stats!.over, `mutation ${m} must fail: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad.stats!.compared * 0.001));
    }
    expect(errors).toEqual([]);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// K6.2b gate 2 (design §7.2): the environment scenes through `?backend=core` vs `?backend=tsl`, same bar (PSNR ≥ 45 dB,
// ≤ 0.2 % pixels > 4/255). The TSL path reads its lights back from the GPU (they arrive frames later); its harness waits
// for them, so both images are settled frames. Extra rows exercise the slot (single scattering), SH lights, 'outward'
// and three's PMREM atlas as scene.environment (the core's lat-long fallback). Opt in with K6_CORE_AB=1; the
// environment needs his map in fixtures/k3/sky-midday-HDR.jpg. Images and JSON (with both backends' lights) go to
// artifacts/k6/ab-env/; the his-capture scenes' images to artifacts/k6/ours-{core,tsl}/.
// ---------------------------------------------------------------------------------------------------------------
const AB_ENV = [
  { scene: "step3-environment-sky-midday-env-isolated", extra: "", capture: true },
  { scene: "step3-environment-sky-midday-env-isolated", extra: "&source=sparse", capture: false },
  { scene: "step5-lighting-volume-sky-midday-env-isolated", extra: "", capture: true },
  { scene: "step5-lighting-volume-medium-cloud-sky-midday-env-isolated", extra: "", capture: true },
  { scene: "step3-environment-sky-midday-env-isolated", extra: "&ms=false", capture: false },
  { scene: "step3-environment-sky-midday-env-isolated", extra: "&envMode=spherical-harmonics", capture: false },
  { scene: "step3-environment-sky-midday-env-isolated", extra: "&envMode=outward", capture: false },
  { scene: "step5-lighting-volume-sky-midday-env-isolated", extra: "&envMode=outward", capture: false },
  { scene: "step3-environment-sky-midday-env-isolated", extra: "&envPmrem=1", capture: false },
] as const;
const AB_ENV_DIR = path.join(REPO, "artifacts", "k6", "ab-env");

for (const { scene, extra, capture } of AB_ENV) {
  test(`K6.2b A/B core vs TSL: ${scene}${extra}`, async ({ page }) => {
    test.skip(process.env.K6_CORE_AB !== "1", "set K6_CORE_AB=1 (needs fixtures/k3/bunny_cloud.wvdb and sky-midday-HDR.jpg)");
    test.skip(!existsSync(path.join(REPO, "fixtures", "k3", "sky-midday-HDR.jpg")), "fixtures/k3/sky-midday-HDR.jpg missing");
    test.setTimeout(1_800_000);
    const base = `?scene=${scene}${extra}`;
    const tsl = await k3(page, `${base}&backend=tsl`);
    const core = await k3(page, `${base}&backend=core`);
    expect(tsl.error, k3Detail(tsl)).toBeUndefined();
    expect(core.error, k3Detail(core)).toBeUndefined();
    expect(core.uncaptured, k3Detail(core)).toEqual([]);
    expect(core.inPageIdentical, k3Detail(core)).toBe(true);
    const cmp = compareImages(k3Image(core), k3Image(tsl), { minPsnr: 45, threshold: 4, maxFraction: 0.002 });
    const name = `${scene}${extra.replace(/[&=]/g, "-")}`;
    mkdirSync(AB_ENV_DIR, { recursive: true });
    writeFileSync(path.join(AB_ENV_DIR, `${name}-tsl.png`), encodePng(k3Image(tsl)));
    writeFileSync(path.join(AB_ENV_DIR, `${name}-core.png`), encodePng(k3Image(core)));
    writeFileSync(path.join(AB_ENV_DIR, `${name}-diff.png`), encodePng(diffHeatmap(k3Image(core), k3Image(tsl), { threshold: 4, fullScale: 16 })));
    const lights = (r: K3Result) => (r as unknown as { environment?: unknown }).environment ?? null;
    writeFileSync(path.join(AB_ENV_DIR, `${name}.json`), JSON.stringify({ scene, extra, ...cmp, coreEnvironment: lights(core), tslEnvironment: lights(tsl), coreWarnings: core.warnings }, null, 2));
    if (capture) {
      for (const [dir, r] of [["ours-core", core], ["ours-tsl", tsl]] as const) {
        mkdirSync(path.join(REPO, "artifacts", "k6", dir), { recursive: true });
        writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.png`), encodePng(k3Image(r)));
        writeFileSync(path.join(REPO, "artifacts", "k6", dir, `${scene}.json`), JSON.stringify({ ...r, rgba: undefined }, null, 2));
      }
    }
    console.log(`[k6-ab-env] ${name}: PSNR ${cmp.psnr.toFixed(2)} dB, ${(cmp.fraction * 100).toFixed(3)}% > 4/255 (max ${cmp.maxDiff})`);
    // Rows that exercise an intentional divergence ('outward' SH irradiance vs PMREM roughness 1, DIVERGES) report only
    if (!extra.includes("outward")) expect(cmp.failures, JSON.stringify(cmp)).toEqual([]);
  });
}
