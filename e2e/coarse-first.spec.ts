import { writeFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helper shared with scripts/k3 (no type declarations)
import { encodePng } from "../scripts/k3/png.mjs";

// K4.16 / K4.18 (docs/dev/design/K4-5-coarse-first.md, the settling test's step 2). The page is
// examples/src/harness/coarse-first.ts: smoke.vdb baked to [density.lod48, density] in the page,
// streamed through a throttled pull-driven ReadableStream into vdbStreamCoarseFirst on a real pool.
// Gates: the preview is added after <= 14.8 KiB of transfer (his smoke LOD0) with ZIP inside the
// file (no HTTP compression) and with gzip on the wire; the preview renders non-trivially; the full
// grid replaces it in place (same id, no pool growth) and the final frame equals a render of the
// full grid alone, byte for byte.
const PAGE_URL = "/src/harness/coarse-first.html";
const GATE = Math.floor(14.8 * 1024); // 15155 B: his smoke.wvdb before LOD0 can draw

interface Run {
  firstPaintTransfer: number;
  firstPaintFile: number;
  transferTotal: number;
  chunk: number;
  grids: string[];
  id: number;
  actions: string[];
}
interface Frame {
  rgba: string;
  nonBackground: number;
}
interface Result {
  done: boolean;
  error?: string;
  missing?: string;
  adapter?: string;
  width?: number;
  height?: number;
  sizes?: { zipFile: number; noneFile: number; zipGzip: number; noneGzip: number };
  grids?: { name: string; gridType: string; voxelCount: number; gridByteSize: number; voxelSize: number[] }[];
  runs?: { zipRaw: Run; noneGzip: Run; zipGzip: Run };
  placement?: { sameId: boolean; inPlace: boolean; finalWorldBBox: number[][]; fullWorldBBox: number[][]; previewCellMin: number[]; fullCellMin: number[] };
  preview?: Frame & { densityScale: number };
  final?: Frame;
  reference?: Frame;
  finalEqualsReference?: boolean;
  previewDiffersFromFinal?: boolean;
  uncaptured: string[];
}

async function run(page: Page): Promise<Result> {
  await page.goto(PAGE_URL);
  await page.waitForFunction(() => (window as { __VDB_COARSE_FIRST__?: { done: boolean } }).__VDB_COARSE_FIRST__?.done === true, undefined, {
    timeout: 170_000,
  });
  return (await page.evaluate(() => (window as { __VDB_COARSE_FIRST__?: unknown }).__VDB_COARSE_FIRST__)) as Result;
}

const png = (r: Result, f: Frame) => encodePng({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(f.rgba, "base64")) });

test("K4.16 coarse-first: preview after <= 14.8 KiB, in-place swap, final frame == full grid alone", async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  const fixture = await page.request.get("/fixtures/vdb-samples/smoke.vdb");
  test.skip(!fixture.ok(), "fixtures/vdb-samples/smoke.vdb missing (pnpm fixtures:vdb-samples)");

  const r = await run(page);
  const frames = { preview: r.preview?.rgba, final: r.final?.rgba, reference: r.reference?.rgba };
  const detail = JSON.stringify({ ...r, preview: { ...r.preview, rgba: undefined }, final: { ...r.final, rgba: undefined }, reference: { ...r.reference, rgba: undefined } }, null, 2);
  expect(r.error, detail).toBeUndefined();
  expect(r.missing, detail).toBeUndefined();
  expect(r.uncaptured, detail).toEqual([]);
  expect(r.grids!.map((g) => [g.name, g.gridType]), detail).toEqual([
    ["density.lod48", "FpN"],
    ["density", "FpN"],
  ]);

  // Bytes to first paint.
  const { zipRaw, noneGzip, zipGzip } = r.runs!;
  for (const run of [zipRaw, noneGzip, zipGzip]) {
    expect(run.grids, detail).toEqual(["density.lod48", "density"]);
    expect(run.actions, detail).toEqual(["add", "replace"]);
    expect(run.firstPaintTransfer, detail).toBeGreaterThan(0);
    expect(run.firstPaintTransfer, detail).toBeLessThanOrEqual(GATE);
  }
  // Without the ZIP codec the preview needs its raw ~321 KiB (the upper node): host compression is required then.
  expect(noneGzip.firstPaintFile, detail).toBeGreaterThan(300 * 1024);
  expect(zipRaw.firstPaintFile, detail).toBeLessThanOrEqual(zipRaw.firstPaintTransfer);

  // The swap: same id, in place (no pool growth), world placement from the full grid's own Map.
  expect(r.placement!.sameId, detail).toBe(true);
  expect(r.placement!.inPlace, detail).toBe(true);
  for (let a = 0; a < 3; a++) {
    expect(r.placement!.finalWorldBBox[0]![a], detail).toBeCloseTo(r.placement!.fullWorldBBox[0]![a]!, 4);
    expect(r.placement!.finalWorldBBox[1]![a], detail).toBeCloseTo(r.placement!.fullWorldBBox[1]![a]!, 4);
    expect(r.placement!.previewCellMin[a], detail).toBeCloseTo(r.placement!.fullCellMin[a]!, 6);
  }

  // Frames: the preview draws something, the final frame is the full grid's, byte for byte.
  expect(r.final!.nonBackground, detail).toBeGreaterThan(2_000);
  expect(r.preview!.nonBackground, detail).toBeGreaterThan(r.final!.nonBackground * 0.5);
  expect(r.previewDiffersFromFinal, detail).toBe(true);
  expect(r.finalEqualsReference, detail).toBe(true);
  expect(frames.final).toBe(frames.reference);

  for (const [name, f] of [["preview", r.preview!], ["final", r.final!], ["reference", r.reference!]] as const) {
    writeFileSync(testInfo.outputPath(`coarse-first-${name}.png`), png(r, f));
  }
  const kib = (n: number) => `${(n / 1024).toFixed(1)} KiB`;
  console.log(
    `[coarse-first] ${r.adapter}: first paint zip/raw ${kib(zipRaw.firstPaintTransfer)} (file offset ${zipRaw.firstPaintFile} B), ` +
      `none/gzip ${kib(noneGzip.firstPaintTransfer)}, zip/gzip ${kib(zipGzip.firstPaintTransfer)}; files: zip ${kib(r.sizes!.zipFile)}, ` +
      `none ${kib(r.sizes!.noneFile)}, none.gz ${kib(r.sizes!.noneGzip)}, zip.gz ${kib(r.sizes!.zipGzip)}; ` +
      `preview ${r.preview!.nonBackground} px (density x${(r.preview!.densityScale / 4).toFixed(2)}), final ${r.final!.nonBackground} px`,
  );
});
