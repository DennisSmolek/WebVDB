import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test, type Page } from "@playwright/test";
import { collectErrors } from "./console-errors";
// @ts-expect-error -- plain .mjs helper shared with scripts/k3 (no type declarations)
import { encodePng } from "../scripts/k3/png.mjs";

// K6.5b gate (design §3.7): mesh → volume shadows through `backend: 'core'` (examples/src/harness/k6-occluders.ts). A box
// on layer 1 (in the light's shadow map, not in the view) sits between the light and the cloud. Per pixel the harness
// classifies the view ray's samples inside the volume box against the box's shadow: all in the eroded box's shadow
// (`inside`), none in the dilated box's shadow (`outside`). Gate: inside pixels darker (occlusion on vs off), outside
// pixels unchanged; a map with nothing in it (the box not casting) gives the occlusion-off frame byte for byte; moving
// the box changes the very next frame as much as the settled one (no lag: the core reads this frame's map). The harness
// reports which backend drew and the occluder maps the core got. PNGs: artifacts/k6/occluders/.
const PAGE_URL = "/src/harness/k6-occluders.html";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(repoRoot, "artifacts", "k6", "occluders");

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  volume?: string;
  width?: number;
  height?: number;
  on?: string;
  off?: string;
  noCaster?: string;
  moved?: string;
  settled?: string;
  mask?: string;
  backend?: { core: boolean; occluders: number[]; layers: number[] };
  warnings: string[];
  uncaptured: string[];
}

async function run(page: Page, query: string): Promise<Result> {
  await page.goto(`${PAGE_URL}${query}`);
  await page.waitForFunction(() => (window as { __VDB_K6_OCCLUDERS__?: { done: boolean } }).__VDB_K6_OCCLUDERS__?.done === true, undefined, { timeout: 300_000 });
  return (await page.evaluate(() => (window as { __VDB_K6_OCCLUDERS__?: unknown }).__VDB_K6_OCCLUDERS__)) as Result;
}

const bytes = (b64: string) => new Uint8Array(Buffer.from(b64, "base64"));
const detail = (r: Result) => JSON.stringify({ ...r, on: undefined, off: undefined, noCaster: undefined, moved: undefined, settled: undefined, mask: undefined }, null, 2);
const luma = (d: Uint8Array, i: number) => 0.2126 * d[i * 4]! + 0.7152 * d[i * 4 + 1]! + 0.0722 * d[i * 4 + 2]!;

function save(name: string, r: Result, b64: string): void {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(encodePng({ width: r.width!, height: r.height!, data: bytes(b64) }) as Uint8Array));
}

const CASES = [
  { light: "sun", lighting: "frame", layers: 1 },
  { light: "sun", lighting: "volume", layers: 1 },
  { light: "spot", lighting: "frame", layers: 1 },
  { light: "point", lighting: "frame", layers: 6 },
  { light: "point", lighting: "volume", layers: 6 },
] as const;

for (const c of CASES) {
  test(`K6.5b gate: a box between the ${c.light} and the cloud shadows only the cloud behind it ('${c.lighting}' lighting)`, async ({ page }) => {
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    const r = await run(page, `?light=${c.light}&lighting=${c.lighting}`);
    expect(r.error, detail(r)).toBeUndefined();
    expect(r.uncaptured, detail(r)).toEqual([]);
    // the core drew, with three's map of this light
    expect(r.backend!.core, detail(r)).toBe(true);
    expect(r.backend!.occluders.length, detail(r)).toBe(1);
    expect(r.backend!.layers, detail(r)).toEqual([c.layers]);
    const name = `${c.light}-${c.lighting}`;
    save(`${name}-on`, r, r.on!);
    save(`${name}-off`, r, r.off!);

    const on = bytes(r.on!);
    const off = bytes(r.off!);
    const noCaster = bytes(r.noCaster!);
    const mask = bytes(r.mask!);
    let inside = 0;
    let darker = 0;
    let insideDrop = 0;
    let outside = 0;
    let outsideMax = 0;
    let outsideChanged = 0;
    for (let i = 0; i < mask.length; i++) {
      const d = luma(off, i) - luma(on, i);
      if (mask[i] === 1) {
        inside++;
        insideDrop += d;
        if (d >= 2) darker++;
      } else if (mask[i] === 2 || mask[i] === 0) {
        outside++;
        let m = 0;
        for (let k = 0; k < 3; k++) m = Math.max(m, Math.abs(on[i * 4 + k]! - off[i * 4 + k]!));
        outsideMax = Math.max(outsideMax, m);
        if (m > 0) outsideChanged++;
      }
    }
    let noCasterDiff = 0;
    for (let i = 0; i < off.length; i++) if (noCaster[i] !== off[i]) noCasterDiff++;
    const moved = bytes(r.moved!);
    const settled = bytes(r.settled!);
    let lagDiff = 0;
    let movedVsOn = 0;
    for (let i = 0; i < moved.length; i++) {
      if (moved[i] !== settled[i]) lagDiff++;
      if (moved[i] !== on[i]) movedVsOn++;
    }
    const stats = { volume: r.volume, inside, darker, meanDrop: +(insideDrop / Math.max(1, inside)).toFixed(2), outside, outsideChanged, outsideMax, noCasterDiff, lagDiff, movedVsOn };
    console.log(`[k6-occluders] ${name} on ${r.adapter}: ${JSON.stringify(stats)}`);
    // inside the box's shadow: darker (luma ≥ 2/255 down on ≥ 90 % of the pixels)
    expect(inside, JSON.stringify(stats)).toBeGreaterThan(30);
    expect(darker / inside, JSON.stringify(stats)).toBeGreaterThanOrEqual(0.9);
    // away from it (and off the volume): unchanged
    expect(outsideMax, JSON.stringify(stats)).toBeLessThanOrEqual(1);
    expect(outsideChanged / outside, JSON.stringify(stats)).toBeLessThanOrEqual(0.001);
    // an empty map leaves the frame as occlusion off does, byte for byte
    expect(noCasterDiff, JSON.stringify(stats)).toBe(0);
    // no lag: the frame after the move is the settled frame (and it did move)
    expect(movedVsOn, JSON.stringify(stats)).toBeGreaterThan(0);
    expect(lagDiff, JSON.stringify(stats)).toBe(0);
    expect(errors).toEqual([]);
  });
}
