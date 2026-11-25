import { describe, expect, it } from "vitest";
import { dilateMask, gaussian, luma, silhouetteMask, ssim } from "../../../../scripts/k6/metrics.mjs";

// K6.R1: the reference report's SSIM (scripts/k6/metrics.mjs) against closed forms and a direct 2D evaluation.

const image = (w, h, f) => {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set([...f(x, y), 255], (y * w + x) * 4);
  return { width: w, height: h, data };
};
const rand = (seed) => () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);

describe("K6.R1 SSIM", () => {
  it("is 1 for identical images and symmetric", () => {
    const r = rand(3);
    const a = image(30, 20, () => [r() * 255, r() * 255, r() * 255].map(Math.round));
    const b = image(30, 20, () => [r() * 255, r() * 255, r() * 255].map(Math.round));
    expect(ssim(a, a).ssim).toBeCloseTo(1, 12);
    expect(ssim(a, b).ssim).toBeCloseTo(ssim(b, a).ssim, 12);
    expect(ssim(a, a).pixels).toBe(20 * 10);
  });

  it("flat images: (2μxμy + C1) / (μx² + μy² + C1)", () => {
    const a = image(16, 16, () => [100, 100, 100]);
    const b = image(16, 16, () => [140, 140, 140]);
    const C1 = (0.01 * 255) ** 2;
    const mx = luma(a)[0];
    const my = luma(b)[0];
    expect(ssim(a, b).ssim).toBeCloseTo((2 * mx * my + C1) / (mx * mx + my * my + C1), 10);
  });

  it("equals the direct 2D window formula on one 11 × 11 window", () => {
    const r = rand(11);
    const a = image(11, 11, () => [r() * 255, r() * 255, r() * 255].map(Math.round));
    const b = image(11, 11, () => [r() * 255, r() * 255, r() * 255].map(Math.round));
    const g = gaussian();
    const x = luma(a);
    const y = luma(b);
    let ux = 0;
    let uy = 0;
    for (let j = 0; j < 11; j++) for (let i = 0; i < 11; i++) {
      ux += g[i] * g[j] * x[j * 11 + i];
      uy += g[i] * g[j] * y[j * 11 + i];
    }
    let vx = 0;
    let vy = 0;
    let cxy = 0;
    for (let j = 0; j < 11; j++) for (let i = 0; i < 11; i++) {
      const w = g[i] * g[j];
      vx += w * (x[j * 11 + i] - ux) ** 2;
      vy += w * (y[j * 11 + i] - uy) ** 2;
      cxy += w * (x[j * 11 + i] - ux) * (y[j * 11 + i] - uy);
    }
    const C1 = (0.01 * 255) ** 2;
    const C2 = (0.03 * 255) ** 2;
    const want = ((2 * ux * uy + C1) * (2 * cxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2));
    const got = ssim(a, b);
    expect(got.pixels).toBe(1);
    expect(got.ssim).toBeCloseTo(want, 10);
  });

  it("falls as noise grows, and a mask restricts the mean", () => {
    const base = image(40, 30, (x, y) => [128 + 80 * Math.sin(x / 3), 100 + 60 * Math.cos(y / 4), 90].map(Math.round));
    const noisy = (k) => {
      const r = rand(7);
      return image(40, 30, (x, y) => {
        const i = (y * 40 + x) * 4;
        return [0, 1, 2].map((c) => Math.min(255, Math.max(0, Math.round(base.data[i + c] + k * (r() - 0.5)))));
      });
    };
    const s = [4, 16, 64].map((k) => ssim(base, noisy(k)).ssim);
    expect(s[0]).toBeGreaterThan(s[1]);
    expect(s[1]).toBeGreaterThan(s[2]);
    const mask = new Uint8Array(40 * 30);
    for (let y = 5; y < 25; y++) for (let x = 5; x < 12; x++) mask[y * 40 + x] = 1;
    expect(ssim(base, noisy(16), mask).pixels).toBe(20 * 7);
  });

  it("silhouette and dilated masks", () => {
    const w = 8;
    const h = 6;
    const alpha = new Float64Array(w * h);
    for (let y = 1; y < 5; y++) for (let x = 2; x < 6; x++) alpha[y * w + x] = 0.5;
    const m = silhouetteMask(alpha, w, h);
    // the 4 × 4 block's outline (12 pixels) and the ring around it (20 in the image), not its 2 × 2 interior
    let count = 0;
    for (const v of m) count += v;
    expect(m[2 * w + 3]).toBe(0);
    expect(m[1 * w + 2]).toBe(1);
    expect(m[0 * w + 1]).toBe(1);
    expect(count).toBe(12 + 20);
    const d = dilateMask(m, w, h, 1);
    for (const v of d) expect(v).toBe(1);
  });
});
