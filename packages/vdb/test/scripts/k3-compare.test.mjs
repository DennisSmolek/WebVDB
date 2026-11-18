import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { PARITY_BARS, above, compareImages, diffHeatmap, mse, psnr } from "../../../../scripts/k3/metrics.mjs";
import { decodePng, encodePng } from "../../../../scripts/k3/png.mjs";

// K3.0: the parity metrics (scripts/k3/metrics.mjs), the PNG codec and the compare CLI.

const COMPARE = fileURLToPath(new URL("../../../../scripts/k3/compare.mjs", import.meta.url));

const solid = (w, h, rgb) => {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) data.set([...rgb, 255], i * 4);
  return { width: w, height: h, data };
};

describe("PSNR and threshold math", () => {
  it("identical images: MSE 0, PSNR Infinity, nothing above any threshold", () => {
    const a = solid(4, 3, [10, 20, 30]);
    expect(mse(a, a)).toBe(0);
    expect(psnr(a, a)).toBe(Infinity);
    expect(above(a, a, 0)).toEqual({ count: 0, fraction: 0, pixels: 12, maxDiff: 0 });
  });

  it("a uniform error of d per channel gives MSE d² and PSNR 20·log10(255/d)", () => {
    const a = solid(8, 8, [100, 100, 100]);
    for (const d of [1, 5, 16]) {
      const b = solid(8, 8, [100 + d, 100 - d, 100 + d]);
      expect(mse(a, b)).toBe(d * d);
      expect(psnr(a, b)).toBeCloseTo(20 * Math.log10(255 / d), 10);
    }
    // 40 dB is an RMS error of 255 / 100 = 2.55 levels.
    expect(10 * Math.log10((255 * 255) / 2.55 ** 2)).toBeCloseTo(40, 10);
  });

  it("alpha is ignored", () => {
    const a = solid(2, 2, [1, 2, 3]);
    const b = { ...a, data: a.data.map((v, i) => (i % 4 === 3 ? 0 : v)) };
    expect(psnr(a, b)).toBe(Infinity);
  });

  it("MSE averages over pixels and the three colour channels", () => {
    const a = solid(2, 1, [0, 0, 0]);
    const b = solid(2, 1, [0, 0, 0]);
    b.data[0] = 30; // one channel of one pixel: 900 / (2 px * 3 ch)
    expect(mse(a, b)).toBe(150);
    expect(psnr(a, b)).toBeCloseTo(10 * Math.log10((255 * 255) / 150), 10);
  });

  it("above() counts pixels by their largest channel difference, strictly greater than the threshold", () => {
    const a = solid(5, 1, [50, 50, 50]);
    const b = solid(5, 1, [50, 50, 50]);
    b.data.set([58, 50, 50], 0); // 8: not above 8
    b.data.set([50, 59, 50], 4); // 9: above
    b.data.set([45, 50, 50], 8); // 5
    b.data.set([50, 50, 0], 12); // 50: above
    const r = above(a, b, 8);
    expect(r.count).toBe(2);
    expect(r.fraction).toBe(0.4);
    expect(r.maxDiff).toBe(50);
    expect(above(a, b, 48).count).toBe(1);
  });

  it("compareImages applies the strict bar (40 dB, 0.5% > 8/255) and the JPEG smoke bar (1% > 48/255)", () => {
    const a = solid(100, 100, [80, 80, 80]);
    const b = solid(100, 100, [80, 80, 80]);
    for (let i = 0; i < 50; i++) b.data[i * 4] = 100; // 50 px (0.5%) off by 20
    const strict = compareImages(a, b, PARITY_BARS.strict);
    expect(strict.above).toBe(50);
    expect(strict.fraction).toBe(0.005);
    expect(strict.psnr).toBeGreaterThan(40); // MSE = 50*400/30000 = 0.667
    expect(strict.pass).toBe(true); // exactly at the limit passes
    b.data[50 * 4] = 100;
    const failing = compareImages(a, b, PARITY_BARS.strict);
    expect(failing.pass).toBe(false);
    expect(failing.failures.join()).toMatch(/pixels > 8\/255/);
    expect(compareImages(a, b, PARITY_BARS.jpegSmoke).pass).toBe(true);
    const noisy = solid(100, 100, [83, 77, 83]); // RMS 3 levels everywhere: PSNR 38.6 dB, nothing > 8
    const r = compareImages(a, noisy, PARITY_BARS.strict);
    expect(r.above).toBe(0);
    expect(r.pass).toBe(false);
    expect(r.failures.join()).toMatch(/PSNR/);
  });

  it("rejects size mismatches", () => {
    expect(() => psnr(solid(2, 2, [0, 0, 0]), solid(3, 2, [0, 0, 0]))).toThrow(/size mismatch/);
  });

  it("the heatmap is black where equal and marks pixels over the threshold", () => {
    const a = solid(5, 5, [10, 10, 10]);
    const b = solid(5, 5, [10, 10, 10]);
    b.data.set([10, 200, 10], (2 * 5 + 2) * 4);
    const h = diffHeatmap(a, b, { threshold: 8 });
    expect([...h.data.subarray(0, 4)]).toEqual([0, 0, 0, 255]);
    expect([...h.data.subarray((2 * 5 + 2) * 4, (2 * 5 + 2) * 4 + 3)]).toEqual([255, 255, 255]);
    expect([...h.data.subarray((2 * 5 + 1) * 4, (2 * 5 + 1) * 4 + 3)]).toEqual([255, 0, 255]); // outline
  });
});

describe("PNG codec", () => {
  it("round-trips RGBA8 and encodes identical pixels to identical bytes", () => {
    const img = { width: 7, height: 5, data: Uint8Array.from({ length: 140 }, (_, i) => (i * 37) & 0xff) };
    const png = encodePng(img);
    expect(Buffer.from(encodePng(img)).equals(Buffer.from(png))).toBe(true);
    const back = decodePng(png);
    expect(back.width).toBe(7);
    expect(back.height).toBe(5);
    expect([...back.data]).toEqual([...img.data]);
  });
});

describe("compare CLI", () => {
  it("exits 0 for identical PNGs, 1 for a failing pair, and writes a heatmap", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "k3-compare-"));
    const a = solid(20, 10, [40, 50, 60]);
    const b = solid(20, 10, [40, 50, 60]);
    for (let i = 0; i < 10; i++) b.data[i * 4 + 1] = 120;
    writeFileSync(path.join(dir, "a.png"), encodePng(a));
    writeFileSync(path.join(dir, "b.png"), encodePng(b));
    const same = execFileSync(process.execPath, [COMPARE, path.join(dir, "a.png"), path.join(dir, "a.png")], { encoding: "utf8" });
    expect(same).toMatch(/PASS a: PSNR inf \(identical\)/);
    const heat = path.join(dir, "diff.png");
    const run = spawnSync(process.execPath, [COMPARE, path.join(dir, "a.png"), path.join(dir, "b.png"), "--heatmap", heat, "--json"], { encoding: "utf8" });
    expect(run.status).toBe(1);
    const report = JSON.parse(run.stdout);
    expect(report.pass).toBe(false);
    expect(report.results[0].above).toBe(10);
    expect(decodePng(readFileSync(heat)).width).toBe(20);
    const timings = path.join(dir, "t.json");
    writeFileSync(timings, JSON.stringify({ passes: { render: 1.25 } }));
    const t = execFileSync(process.execPath, [COMPARE, path.join(dir, "a.png"), path.join(dir, "a.png"), "--timings", timings], { encoding: "utf8" });
    expect(t).toMatch(/PROVISIONAL, D25/);
    const bad = spawnSync(process.execPath, [COMPARE, path.join(dir, "a.png")], { encoding: "utf8" });
    expect(bad.status).toBe(2);
  });

  it("K3.1 directory mode: the harness sidecar masks to the volume rect and non-parity scenes do not gate", () => {
    const root = mkdtempSync(path.join(tmpdir(), "k3-compare-dir-"));
    const ours = path.join(root, "ours");
    const ref = path.join(root, "ref");
    mkdirSync(ours);
    mkdirSync(ref);
    // 20x10; the volume rect is the left half. A difference outside it must not count.
    const base = solid(20, 10, [40, 50, 60]);
    const outside = solid(20, 10, [40, 50, 60]);
    for (let y = 0; y < 10; y++) for (let x = 12; x < 20; x++) outside.data[(y * 20 + x) * 4] = 200;
    const inside = solid(20, 10, [40, 50, 60]);
    for (let y = 0; y < 10; y++) for (let x = 0; x < 4; x++) inside.data[(y * 20 + x) * 4] = 200;
    writeFileSync(path.join(ours, "masked.png"), encodePng(base));
    writeFileSync(path.join(ref, "masked.png"), encodePng(outside));
    writeFileSync(path.join(ours, "masked.json"), JSON.stringify({ volumeRect: [0, 0, 10, 10], parityExpected: true }));
    writeFileSync(path.join(ours, "info.png"), encodePng(base));
    writeFileSync(path.join(ref, "info.png"), encodePng(inside));
    writeFileSync(path.join(ours, "info.json"), JSON.stringify({ volumeRect: [0, 0, 10, 10], parityExpected: false, parityNote: "environment (K3.4)" }));
    writeFileSync(path.join(ours, "only-ours.png"), encodePng(base));
    const run = spawnSync(process.execPath, [COMPARE, "--ours", ours, "--reference", ref, "--json"], { encoding: "utf8" });
    expect(run.status).toBe(0);
    const report = JSON.parse(run.stdout);
    expect(report.results.map((r) => r.name)).toEqual(["info", "masked"]);
    const [info, masked] = report.results;
    expect(masked).toMatchObject({ pass: true, gates: true, mask: [0, 0, 10, 10], pixels: 100 });
    expect(info).toMatchObject({ pass: false, gates: false, note: "environment (K3.4)", above: 40 });
    // --gate all makes the INFO pair gate; --mask none compares the whole frame.
    expect(spawnSync(process.execPath, [COMPARE, "--ours", ours, "--reference", ref, "--gate", "all"], { encoding: "utf8" }).status).toBe(1);
    const whole = spawnSync(process.execPath, [COMPARE, "--ours", ours, "--reference", ref, "--mask", "none", "--scenes", "masked", "--json"], { encoding: "utf8" });
    expect(whole.status).toBe(1);
    expect(JSON.parse(whole.stdout).results[0]).toMatchObject({ name: "masked", mask: null, above: 80 });
  });
});
