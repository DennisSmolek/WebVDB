import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";

// K6.Q1 (V85): the shadowed ambient (`ambientOcclusion`) on raw WebGPU (examples/src/harness/k6-core.ts `?ao=<pairs>`),
// each kernel against its CPU twin within 2 f16 ulp (≤ 0.1 % over): 'frame' lighting's transmittance (the fourth
// target's hemispherical visibilities, vdbFrameTransmittanceVoxel3d) and fused gather (the sky totals replaced, ambient
// scaled), and the 'volume' gather (the direction pairs through the volume's shadow march). Every mode family: single
// scattering, diffusion (white-furnace pairs), octaves (normalised totals), the MLP. Each mutation must fail. With the
// setting off the frames are the K6.1–K6.5 frames (k6-core.spec.ts covers those unchanged).
const PAGE_URL = "/src/harness/k6-core.html";

interface Stats {
  compared: number;
  covered: number;
  over: number;
  maxUlp: number;
}
interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  stats?: Stats;
  gridStats?: Stats;
  transmittanceStats?: Stats;
  mipStats?: Stats;
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string, timeout = 300_000): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_CORE__?: { done: boolean } }).__VDB_K6_CORE__?.done === true, undefined, { timeout });
  return (await page.evaluate(() => (window as { __VDB_K6_CORE__?: unknown }).__VDB_K6_CORE__)) as Result;
}

const detail = (r: Result) => JSON.stringify(r, null, 2);

type Key = "stats" | "transmittanceStats" | "gridStats" | "mipStats";
const PROBES: { name: string; query: string; keys: Key[]; mutations: Record<string, Key> }[] = [
  { name: "frame, single scattering, 4 pairs", query: "?probe=frame&ms=0&ao=4", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "ao-pairs": "transmittanceStats", ao: "stats" } },
  { name: "frame, diffusion, 7 pairs, sparse, two groups", query: "?probe=frame&ao=7&source=sparse&groups=2", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "ao-pairs": "transmittanceStats", ao: "stats" } },
  { name: "frame, 'octaves', 3 pairs", query: "?probe=frame&ms=octaves&ao=3", keys: ["gridStats", "transmittanceStats", "stats"], mutations: { "ao-pairs": "transmittanceStats", octaves: "transmittanceStats" } },
  { name: "frame, 'neural', 1 pair (the sky marches only)", query: "?probe=frame&ms=neural&ao=1", keys: ["gridStats", "mipStats", "transmittanceStats", "stats"], mutations: { ao: "stats" } },
  { name: "gather, single scattering, 4 pairs", query: "?probe=gather&ms=0&ao=4", keys: ["stats"], mutations: { "ao-pairs": "stats", ao: "stats" } },
  { name: "gather, diffusion, 4 pairs, 6 steps, sparse", query: "?probe=gather&ao=4&aoSteps=6&source=sparse&lod=none", keys: ["stats"], mutations: { "ao-pairs": "stats", ao: "stats" } },
  { name: "gather, 'octaves', 2 pairs", query: "?probe=gather&ms=octaves&ao=2", keys: ["stats"], mutations: { ao: "stats", octaves: "stats" } },
];

for (const p of PROBES) {
  test(`K6.Q1 probe: shadowed ambient, ${p.name}, matches its CPU twins; mutations fail`, async ({ page }) => {
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
      console.log(`[k6-ambient] ${p.name} ${key} on ${r.adapter}: ${s.compared} values, ${s.covered} non-zero, max ${s.maxUlp.toFixed(2)} f16 ulp, ${s.over} over`);
    }
    for (const [m, key] of Object.entries(p.mutations)) {
      const bad = await run(page, `${p.query}&mutate=${m}`);
      expect(bad.error, detail(bad)).toBeUndefined();
      expect(bad[key]!.over, `mutation ${m} must fail ${key}: ${detail(bad)}`).toBeGreaterThan(Math.floor(bad[key]!.compared * 0.001));
      console.log(`[k6-ambient] ${p.name} mutation ${m}: ${bad[key]!.over} of ${bad[key]!.compared} over (${key})`);
    }
    expect(errors).toEqual([]);
  });
}
