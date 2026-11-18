import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
// @ts-expect-error -- plain .mjs helpers shared with scripts/k3/compare.mjs (no type declarations)
import { compareImages } from "../scripts/k3/metrics.mjs";
// @ts-expect-error -- see above
import { encodePng } from "../scripts/k3/png.mjs";

// K3.1 (R1 §6 Step 1): mjurczyk's froxel pipeline with 'volume' lighting, ported onto the dense source
// (examples/src/harness/k3-frame.ts). Gates until his reference captures exist (K3.0a): no validation
// errors; the TSL chunks match their CPU twins; the integrate pass matches its CPU twin; the volume renders
// non-trivially; it is deterministic (two page loads byte-identical); an opaque mesh in front occludes it.
// PNGs go to artifacts/k3/ours/ (k3-frame-*.png) for scripts/k3/compare.mjs.
const PAGE_URL = "/src/harness/k3-frame.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OURS = path.join(repoRoot, "artifacts", "k3", "ours");

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  scene?: string;
  width?: number;
  height?: number;
  rgba?: string;
  volumeRect?: number[];
  inPageIdentical?: boolean;
  nonBackground?: number;
  chunkProbe?: { count: number; maxAbs: number; maxRel: number; over: number; firstBad?: unknown };
  twin?: { pixels: number; covered: number; maxAbs: number; over: number; firstBad?: unknown };
  warnings: string[];
  uncaptured: string[];
  [k: string]: unknown;
}

async function run(page: Page, query = ""): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K3_FRAME__?: { done: boolean } }).__VDB_K3_FRAME__?.done === true, undefined, {
    timeout: 170_000,
  });
  return (await page.evaluate(() => (window as { __VDB_K3_FRAME__?: unknown }).__VDB_K3_FRAME__)) as Result;
}

const image = (r: Result) => ({ width: r.width!, height: r.height!, data: new Uint8Array(Buffer.from(r.rgba!, "base64")) });
const detailOf = (r: Result) => JSON.stringify({ ...r, rgba: r.rgba ? `${r.rgba.length} b64 chars` : undefined }, null, 2);

function save(name: string, r: Result): Uint8Array {
  const png = encodePng(image(r)) as Uint8Array;
  mkdirSync(OURS, { recursive: true });
  writeFileSync(path.join(OURS, `${name}.png`), png);
  return png;
}

test("K3.1 froxels, 'volume' lighting: renders, chunks and integrate match their CPU twins, deterministic", async ({ page }) => {
  test.setTimeout(400_000);
  // Froxel grid = screen size, so the integrate twin reads one unfiltered froxel column per pixel.
  const q = "?w=160&h=96&froxels=160x96x24&twin=1";
  const first = await run(page, q);
  const detail = detailOf(first);
  expect(first.error, detail).toBeUndefined();
  expect(first.uncaptured, detail).toEqual([]);
  expect(first.inPageIdentical, detail).toBe(true);
  // The cloud covers a good part of the 160x96 frame.
  expect(first.nonBackground!, detail).toBeGreaterThan(2_000);
  expect(first.chunkProbe!.count, detail).toBe(1024);
  expect(first.chunkProbe!.over, detail).toBe(0);
  expect(first.twin!.covered, detail).toBeGreaterThan(50);
  expect(first.twin!.over, detail).toBe(0);
  // Unported features must not warn in the e2e scene (no environment, 'volume' lighting).
  expect(first.warnings, detail).toEqual([]);

  const second = await run(page, q);
  expect(second.error).toBeUndefined();
  const cmp = compareImages(image(first), image(second));
  expect(cmp.psnr, JSON.stringify(cmp)).toBe(Infinity);
  const a = save("k3-frame-smoke-run1", first);
  const b = save("k3-frame-smoke-run2", second);
  expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);
  console.log(
    `[k3-frame] ${first.adapter}: ${first.nonBackground} volume px; chunk probe max rel ${first.chunkProbe!.maxRel.toExponential(2)}; integrate twin max ${first.twin!.maxAbs.toExponential(2)} over ${first.twin!.covered} covered px; two loads identical`,
  );
});

test("K3.1 froxels: multiple scattering on and off differ; an opaque mesh in front occludes the volume", async ({ page }) => {
  test.setTimeout(400_000);
  const base = "?w=160&h=96&froxels=96x64x24";
  const msOn = await run(page, `${base}&ms=1`);
  const msOff = await run(page, `${base}&ms=0`);
  expect(msOn.error, detailOf(msOn)).toBeUndefined();
  expect(msOff.error, detailOf(msOff)).toBeUndefined();
  const ms = compareImages(image(msOn), image(msOff));
  // Diffusion brightens the shadowed cloud: many pixels change.
  expect(ms.above, JSON.stringify(ms)).toBeGreaterThan(200);
  // ...and only brightens it (total transmittance >= direct): the mean of the frame goes up.
  const mean = (r: Result) => image(r).data.reduce((s: number, v: number, i: number) => (i % 4 === 3 ? s : s + v), 0);
  expect(mean(msOn)).toBeGreaterThan(mean(msOff));

  const occluded = await run(page, `${base}&ms=1&occluder=1`);
  const occluderOnly = await run(page, `${base}&ms=1&occluder=1&novolume=1`);
  expect(occluded.error, detailOf(occluded)).toBeUndefined();
  expect(occluderOnly.error, detailOf(occluderOnly)).toBeUndefined();
  expect(occluded.uncaptured).toEqual([]);
  // Pixels of the occluder (its colour in the occluder-only frame) are identical with the volume behind it,
  // and the volume does change those pixels when the occluder is absent.
  const occ = image(occluderOnly).data;
  const withVolume = image(occluded).data;
  const plain = image(msOn).data;
  let occluderPixels = 0;
  let changedBehind = 0;
  let leaked = 0;
  for (let i = 0; i < occ.length; i += 4) {
    const isOccluder = occ[i]! > 200 && occ[i + 1]! < 120 && occ[i + 2]! > 60 && occ[i + 2]! < 140;
    if (!isOccluder) continue;
    occluderPixels++;
    if (withVolume[i] !== occ[i] || withVolume[i + 1] !== occ[i + 1] || withVolume[i + 2] !== occ[i + 2]) leaked++;
    if (Math.abs(plain[i]! - occ[i]!) + Math.abs(plain[i + 1]! - occ[i + 1]!) + Math.abs(plain[i + 2]! - occ[i + 2]!) > 30) changedBehind++;
  }
  expect(occluderPixels).toBeGreaterThan(50);
  expect(leaked, `${leaked} of ${occluderPixels} occluder pixels changed by the volume behind`).toBe(0);
  expect(changedBehind).toBeGreaterThan(occluderPixels * 0.5);
  save("k3-frame-smoke-occluded", occluded);
});

// His parity scenes need his CC BY-SA .wvdb files (copied from his clone into fixtures/k3/) and are full size
// (1280x720, 512x512x24 froxels): slow on SwiftShader. Opt in with K3_FRAME_PARITY=1; the PNGs are what
// scripts/k3/compare.mjs compares with fixtures/k3-reference/75e7345/ (K3.0a).
const PARITY_SCENES = [
  "step1-volume-dir-ms-off-isolated",
  "step1-volume-dir-ms-on-isolated",
  "step1-volume-dir-ms-off",
  "step1-volume-dir-ms-on",
  "step1-lighting-volume",
  "step1-lights-directional",
  "step1-multiple-scattering-off",
];
for (const scene of PARITY_SCENES) {
  test(`K3.1 parity render: ${scene}`, async ({ page }) => {
    test.skip(process.env.K3_FRAME_PARITY !== "1", "set K3_FRAME_PARITY=1 (needs fixtures/k3/bunny_cloud.wvdb)");
    test.skip(!existsSync(path.join(repoRoot, "fixtures", "k3", "bunny_cloud.wvdb")), "fixtures/k3/bunny_cloud.wvdb missing (copy it from his clone)");
    test.setTimeout(1_200_000);
    const r1 = await run(page, `?scene=${scene}`);
    expect(r1.error, detailOf(r1)).toBeUndefined();
    expect(r1.uncaptured).toEqual([]);
    expect(r1.inPageIdentical).toBe(true);
    save(scene, r1);
    writeFileSync(
      path.join(OURS, `${scene}.json`),
      JSON.stringify({ ...r1, rgba: undefined }, null, 2),
    );
  });
}
