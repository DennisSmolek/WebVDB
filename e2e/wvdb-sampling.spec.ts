import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3 (no type declarations)
import { above, psnr } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- plain .mjs helper shared with scripts/k3 (no type declarations)
import { encodePng } from "../scripts/k3/png.mjs";

// K4.21: .wvdb bake sampling A/B (nearest vs box) at resolution 128 and 64, image metric only (D25). The page is
// examples/src/harness/wvdb-sampling.ts: it bakes each volume in memory, renders every bake through the K3.0 dense
// source (emission-absorption march) and compares it with (a) the same march over the full-resolution sparse grid
// ("ea", like-for-like) and (b) VDBVolumeMaterial on the pool ("vvm", a different renderer: ranking only).
// WEBVDB_WRITE_K421=1 writes docs/handoffs/K4-21.json. WEBVDB_K421_VOLUMES / WEBVDB_K421_RES change the sets.
const PAGE_URL = "/src/harness/wvdb-sampling.html";
const VOLUMES = (process.env.WEBVDB_K421_VOLUMES ?? "smoke,bunny_cloud,wdas").split(",");
const RES = process.env.WEBVDB_K421_RES ?? "128,64";
// WEBVDB_K421_EXT: the march extinction (default 20; his FrameVolume default is 100, near-opaque at this size).
const EXT = process.env.WEBVDB_K421_EXT ?? "20";
const URLS: Record<string, string> = {
  smoke: "/fixtures/vdb-samples/smoke.vdb",
  bunny_cloud: "/fixtures/vdb-samples/bunny_cloud.vdb",
  wdas: "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb",
};

interface Variant {
  key: string;
  sampling: "nearest" | "box";
  resolution: number;
  dims: number[];
  fileBytes: number;
  meanDensity: number;
  nonZeroVoxels: number;
  texelMean: number;
  frames: string[];
  nonBackground: number[];
}
interface Volume {
  name: string;
  ea: string[];
  eaNonBackground: number[];
  vvm: string[];
  vvmNonBackground: number[];
  variants: Variant[];
}
interface Result {
  done: boolean;
  error?: string;
  missing?: string;
  adapter?: string;
  width?: number;
  height?: number;
  steps?: number;
  extinction?: number;
  volumes: Volume[];
  uncaptured: string[];
}

const image = (r: Result, b64: string) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(b64, "base64")) });
const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;

/** Signed mean luminance (a - b) over all pixels, 8-bit units: > 0 = `a` is brighter (more cloud). */
function bias(a: { data: Uint8Array }, b: { data: Uint8Array }): number {
  let sum = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    sum += 0.2126 * (a.data[i]! - b.data[i]!) + 0.7152 * (a.data[i + 1]! - b.data[i + 1]!) + 0.0722 * (a.data[i + 2]! - b.data[i + 2]!);
  }
  return sum / (a.data.length / 4);
}

async function run(page: Page): Promise<Result> {
  await page.goto(`${PAGE_URL}?res=${RES}&extinction=${EXT}&volumes=${VOLUMES.join(",")}`);
  await page.waitForFunction(() => (window as { __VDB_WVDB_SAMPLING__?: { done: boolean } }).__VDB_WVDB_SAMPLING__?.done === true, undefined, {
    timeout: 1_700_000,
  });
  return (await page.evaluate(() => (window as { __VDB_WVDB_SAMPLING__?: unknown }).__VDB_WVDB_SAMPLING__)) as Result;
}

test("K4.21 .wvdb bake sampling A/B: nearest vs box against the full-resolution sparse render", async ({ page }, testInfo) => {
  test.setTimeout(1_800_000);
  for (const v of VOLUMES) {
    const url = URLS[v]!;
    const head = await page.request.head(url).catch(() => null);
    const got = head && head.ok() ? head : await page.request.get(url);
    test.skip(!got.ok(), `${url} missing (pnpm fixtures:vdb-samples / fixtures:wdas)`);
  }

  const r = await run(page);
  const err = JSON.stringify({ error: r.error, missing: r.missing, uncaptured: r.uncaptured });
  expect(r.error, err).toBeUndefined();
  expect(r.missing, err).toBeUndefined();
  expect(r.uncaptured, err).toEqual([]);
  expect(r.volumes.map((v) => v.name)).toEqual(VOLUMES);

  const rows: Record<string, unknown>[] = [];
  const outDir = testInfo.outputPath("frames");
  mkdirSync(outDir, { recursive: true });
  for (const vol of r.volumes) {
    for (let pose = 0; pose < vol.ea.length; pose++) {
      expect(vol.eaNonBackground[pose], `${vol.name} reference pose ${pose} is empty`).toBeGreaterThan(500);
      writeFileSync(path.join(outDir, `${vol.name}-ea-pose${pose}.png`), encodePng(image(r, vol.ea[pose]!)));
      writeFileSync(path.join(outDir, `${vol.name}-vvm-pose${pose}.png`), encodePng(image(r, vol.vvm[pose]!)));
    }
    for (const v of vol.variants) {
      for (let pose = 0; pose < v.frames.length; pose++) {
        const frame = image(r, v.frames[pose]!);
        const ea = image(r, vol.ea[pose]!);
        const vvm = image(r, vol.vvm[pose]!);
        const m = above(frame, ea, 8);
        const mv = above(frame, vvm, 8);
        rows.push({
          volume: vol.name,
          sampling: v.sampling,
          resolution: v.resolution,
          pose,
          psnr: Number(psnr(frame, ea).toFixed(2)),
          pctAbove8: Number((m.fraction * 100).toFixed(3)),
          bias: Number(bias(frame, ea).toFixed(3)),
          psnrVvm: Number(psnr(frame, vvm).toFixed(2)),
          pctAbove8Vvm: Number((mv.fraction * 100).toFixed(3)),
          px: v.nonBackground[pose],
          refPx: vol.eaNonBackground[pose],
          fileBytes: v.fileBytes,
          dims: v.dims.join("x"),
          headerMeanDensity: v.meanDensity,
          nonZeroVoxels: v.nonZeroVoxels,
          texelMean: Number(v.texelMean.toFixed(6)),
        });
        expect(v.nonBackground[pose], `${vol.name} ${v.key} pose ${pose} is empty`).toBeGreaterThan(200);
        writeFileSync(path.join(outDir, `${vol.name}-${v.sampling}-${v.resolution}-pose${pose}.png`), encodePng(frame));
      }
    }
  }

  const summary: Record<string, unknown>[] = [];
  const resolutions = [...new Set(rows.map((x) => x.resolution as number))];
  for (const resolution of resolutions) {
    for (const sampling of ["nearest", "box"]) {
      for (const volume of [...r.volumes.map((v) => v.name), "all"]) {
        const sel = rows.filter((x) => x.sampling === sampling && x.resolution === resolution && (volume === "all" || x.volume === volume));
        const f = (k: string) => sel.map((x) => x[k] as number);
        summary.push({
          resolution,
          sampling,
          volume,
          psnr: Number(mean(f("psnr")).toFixed(2)),
          psnrWorst: Math.min(...f("psnr")),
          pctAbove8: Number(mean(f("pctAbove8")).toFixed(3)),
          bias: Number(mean(f("bias")).toFixed(3)),
          psnrVvm: Number(mean(f("psnrVvm")).toFixed(2)),
          pctAbove8Vvm: Number(mean(f("pctAbove8Vvm")).toFixed(3)),
          fileBytes: volume === "all" ? Math.round(mean(f("fileBytes"))) : f("fileBytes")[0],
        });
      }
    }
  }

  const table = { adapter: r.adapter, width: r.width, height: r.height, steps: r.steps, extinction: r.extinction, rows, summary };
  writeFileSync(testInfo.outputPath("wvdb-sampling.json"), JSON.stringify(table, null, 2));
  if (process.env.WEBVDB_WRITE_K421) writeFileSync(path.join("docs", "handoffs", `K4-21${EXT === "20" ? "" : `-ext${EXT}`}.json`), JSON.stringify(table, null, 2));
  console.log(`[wvdb-sampling] ${r.adapter}, ${r.width}x${r.height}, ${r.steps} steps, extinction ${r.extinction}`);
  console.table(summary);

  // Sanity only: every bake resembles the reference. The ranking is the finding, not a gate.
  for (const row of rows) expect(row.psnr as number, JSON.stringify(row)).toBeGreaterThan(12);
});
