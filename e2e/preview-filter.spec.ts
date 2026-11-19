import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3 (no type declarations)
import { above, psnr } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- plain .mjs helper shared with scripts/k3 (no type declarations)
import { encodePng } from "../scripts/k3/png.mjs";

// K4.19: coarse-first preview filter A/B, image metric only (timings are deferred, D25). The page is
// examples/src/harness/preview-filter.ts. For smoke and the WDAS quarter cloud it renders the full grid and the
// resolution-48 previews (box / point / trilinear; the active cell box and the same lattice shifted by half a preview
// voxel) from two camera poses, each preview with densityScale x voxelRatio (K4.16). This spec compares every preview
// frame to the full-resolution frame of the same pose (PSNR, % pixels > 8/255, scripts/k3/metrics.mjs) and writes the
// JSON table (test output dir, and docs/archive/handoffs/K4-19.json when WEBVDB_WRITE_K419=1).
const PAGE_URL = "/src/harness/preview-filter.html";
// WEBVDB_K419_VOLUMES=smoke,wdas,explosion,fire adds the high-frequency samples (exploration; the committed table is smoke + wdas).
const VOLUMES = (process.env.WEBVDB_K419_VOLUMES ?? "smoke,wdas").split(",");

interface Variant {
  key: string;
  filter: string;
  shift: "aligned" | "shifted";
  dims: number[];
  voxelCount: number;
  imageBytes: number;
  voxelRatio: number;
  densityScale: number;
  frames: string[];
  nonBackground: number[];
}
interface Volume {
  name: string;
  source: string;
  fullDensityScale: number;
  fullVoxelCount: number;
  full: string[];
  fullNonBackground: number[];
  variants: Variant[];
}
interface Result {
  done: boolean;
  error?: string;
  missing?: string;
  adapter?: string;
  width?: number;
  height?: number;
  resolution?: number;
  volumes: Volume[];
  uncaptured: string[];
}

const image = (r: Result, b64: string) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(b64, "base64")) });

async function run(page: Page): Promise<Result> {
  await page.goto(`${PAGE_URL}?resolution=${process.env.WEBVDB_K419_RES ?? 48}&volumes=${VOLUMES.join(",")}`);
  await page.waitForFunction(() => (window as { __VDB_PREVIEW_FILTER__?: { done: boolean } }).__VDB_PREVIEW_FILTER__?.done === true, undefined, {
    timeout: 560_000,
  });
  return (await page.evaluate(() => (window as { __VDB_PREVIEW_FILTER__?: unknown }).__VDB_PREVIEW_FILTER__)) as Result;
}

/** Signed mean luminance (preview - full) over all pixels, 8-bit units: > 0 = the preview is brighter (more cloud). */
function bias(a: { data: Uint8Array }, b: { data: Uint8Array }): number {
  let sum = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    sum += 0.2126 * (a.data[i]! - b.data[i]!) + 0.7152 * (a.data[i + 1]! - b.data[i + 1]!) + 0.0722 * (a.data[i + 2]! - b.data[i + 2]!);
  }
  return sum / (a.data.length / 4);
}

const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length;

test("K4.19 preview filter A/B: box vs point vs trilinear previews against the full-resolution render", async ({ page }, testInfo) => {
  test.setTimeout(600_000);
  for (const url of ["/fixtures/vdb-samples/smoke.vdb", "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb"]) {
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
    for (let pose = 0; pose < vol.full.length; pose++) {
      expect(vol.fullNonBackground[pose], `${vol.name} full pose ${pose} is empty`).toBeGreaterThan(500);
      writeFileSync(path.join(outDir, `${vol.name}-full-pose${pose}.png`), encodePng(image(r, vol.full[pose]!)));
    }
    for (const v of vol.variants) {
      for (let pose = 0; pose < v.frames.length; pose++) {
        const full = image(r, vol.full[pose]!);
        const prev = image(r, v.frames[pose]!);
        const m = above(prev, full, 8);
        const p = psnr(prev, full);
        rows.push({
          volume: vol.name,
          filter: v.filter,
          shift: v.shift,
          pose,
          psnr: Number(p.toFixed(2)),
          pctAbove8: Number((m.fraction * 100).toFixed(3)),
          maxDiff: m.maxDiff,
          bias: Number(bias(prev, full).toFixed(3)),
          previewPx: v.nonBackground[pose],
          fullPx: vol.fullNonBackground[pose],
          voxels: v.voxelCount,
          dims: v.dims.join("x"),
          imageBytes: v.imageBytes,
          voxelRatio: Number(v.voxelRatio.toFixed(3)),
        });
        expect(v.nonBackground[pose], `${vol.name} ${v.key} pose ${pose} is empty`).toBeGreaterThan(200);
        writeFileSync(path.join(outDir, `${vol.name}-${v.filter}-${v.shift}-pose${pose}.png`), encodePng(prev));
      }
    }
  }

  // Summaries: per filter, over volumes x poses, for the aligned box, the half-voxel-shifted box, and the worst of both.
  const summary: Record<string, unknown>[] = [];
  for (const filter of ["box", "point", "trilinear"]) {
    for (const volume of [...r.volumes.map((v) => v.name), "all"]) {
      const sel = rows.filter((x) => x.filter === filter && (volume === "all" || x.volume === volume));
      const aligned = sel.filter((x) => x.shift === "aligned");
      const shifted = sel.filter((x) => x.shift === "shifted");
      const f = (xs: Record<string, unknown>[], k: string) => xs.map((x) => x[k] as number);
      summary.push({
        filter,
        volume,
        psnrAligned: Number(mean(f(aligned, "psnr")).toFixed(2)),
        psnrShifted: Number(mean(f(shifted, "psnr")).toFixed(2)),
        psnrWorst: Math.min(...f(sel, "psnr")),
        pctAbove8Aligned: Number(mean(f(aligned, "pctAbove8")).toFixed(3)),
        pctAbove8Shifted: Number(mean(f(shifted, "pctAbove8")).toFixed(3)),
        pctAbove8Worst: Math.max(...f(sel, "pctAbove8")),
        meanBias: Number(mean(f(sel, "bias")).toFixed(3)),
        // How much the half-voxel shift moves the preview's error (the aliasing check): |shifted - aligned| per pose.
        psnrShiftSwing: Number(mean(aligned.map((a) => Math.abs((a.psnr as number) - (shifted.find((s) => s.volume === a.volume && s.pose === a.pose)!.psnr as number)))).toFixed(2)),
      });
    }
  }

  const table = { adapter: r.adapter, width: r.width, height: r.height, resolution: r.resolution, rows, summary };
  writeFileSync(testInfo.outputPath("preview-filter.json"), JSON.stringify(table, null, 2));
  if (process.env.WEBVDB_WRITE_K419) writeFileSync(path.join("docs", "archive", "handoffs", "K4-19.json"), JSON.stringify(table, null, 2));
  console.log(`[preview-filter] ${r.adapter}, ${r.width}x${r.height}, preview resolution ${r.resolution}`);
  console.table(rows);
  console.table(summary);

  // Sanity only: every preview resembles its full render. The ranking is the finding, not a gate.
  for (const row of rows) expect(row.psnr as number, JSON.stringify(row)).toBeGreaterThan(12);
});
