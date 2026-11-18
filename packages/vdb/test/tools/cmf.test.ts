/**
 * D33: the CIE 1931 2° colour matching functions as the analytic multi-lobe fit of Wyman, Sloan & Shirley (JCGT 2(2),
 * 2013; `vdbCie1931Fit`), and the blackbody table integrated with it (`tools/frame/blackbody.ts`).
 *
 * - Shape: peaks and their heights within the paper's stated accuracy (max squared error 2.0e-4 / 6.4e-5 / 4.9e-4),
 *   the lobes continuous at their centres, equal-energy white at (⅓, ⅓).
 * - The Planck × soot integration (his 81-sample sum at 380–780 nm) against an independent 1 nm trapezoid of the full
 *   Planck law; the `cmf` argument path.
 * - Against the exact CIE table (test-only, CC BY-SA 4.0): `node scripts/fetch-cie-cmf.mjs` puts it in fixtures/cie/
 *   (git-ignored, never shipped). Skipped when absent; WEBVDB_REQUIRE_FIXTURES does not require it.
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  vdbCie1931Fit,
  vdbFrameBlackbodyColor,
  vdbFrameBlackbodyRgb,
  vdbFrameBlackbodyTable,
  vdbFrameBlackbodyWavelengthCount,
  vdbFrameBlackbodyWavelengthStart,
  vdbFrameBlackbodyWavelengthStep,
  type VDBColorMatchingFunctions,
  type VDBFrameVec3 as Vec3,
} from "../../src/tools/internal.js";

const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol);

/** The paper's Table 2, multi-lobe 1931 fit: max / mean squared error per curve vs the 1 nm CIE data. */
const PAPER_MAX_SQ = [2.0e-4, 6.4e-5, 4.9e-4];
const PAPER_MEAN_SQ = [3.1e-5, 7.1e-6, 1.6e-5];
const PAPER_MAX_ABS = PAPER_MAX_SQ.map(Math.sqrt);

/** The maximum of one curve on a 0.1 nm grid inside [lo, hi]. */
const peak = (k: number, lo: number, hi: number) => {
  let best = { wavelength: lo, value: -Infinity };
  for (let w = lo; w <= hi; w += 0.1) {
    const v = vdbCie1931Fit(w)[k]!;
    if (v > best.value) best = { wavelength: w, value: v };
  }
  return best;
};

const xyOf = ([X, Y, Z]: Vec3): [number, number] => [X / (X + Y + Z), Y / (X + Y + Z)];

/** Planck × soot XYZ with his 81-sample sum (the table's integration) for a given set of curves. */
const blackbodyXyz = (temperature: number, exponent: number, cmf: VDBColorMatchingFunctions): Vec3 => {
  const xyz: Vec3 = [0, 0, 0];
  for (let i = 0; i < vdbFrameBlackbodyWavelengthCount; i++) {
    const w = vdbFrameBlackbodyWavelengthStart + i * vdbFrameBlackbodyWavelengthStep;
    const radiance = ((w / 550) ** -exponent * (550 / w) ** 5) / Math.expm1(1.4388e7 / (w * temperature));
    const c = cmf(w);
    for (let k = 0; k < 3; k++) xyz[k] += radiance * c[k]!;
  }
  return xyz;
};

const temperatures = (n: number, lo = 1000, hi = 40000) => Array.from({ length: n }, (_, i) => lo * (hi / lo) ** (i / (n - 1)));

describe("D33: the Wyman–Sloan–Shirley 2013 multi-lobe fit of the CIE 1931 2° observer", () => {
  it("peaks where the CIE curves peak, within the paper's stated accuracy", () => {
    // ȳ is the photopic luminous efficiency: 1 at 555 nm by definition
    const y = peak(1, 500, 620);
    expect(y.wavelength).toBeGreaterThan(550);
    expect(y.wavelength).toBeLessThan(562);
    close(y.value, 1.0, PAPER_MAX_ABS[1]! * 2);
    close(vdbCie1931Fit(555)[1], 1.0, PAPER_MAX_ABS[1]!);
    // x̄: the red lobe near 600 nm (≈ 1.06), a blue lobe near 442 nm (≈ 0.35), a minimum near 505 nm
    const xRed = peak(0, 560, 640);
    expect(xRed.wavelength).toBeGreaterThan(595);
    expect(xRed.wavelength).toBeLessThan(605);
    close(xRed.value, 1.062, PAPER_MAX_ABS[0]!);
    const xBlue = peak(0, 400, 480);
    expect(xBlue.wavelength).toBeGreaterThan(438);
    expect(xBlue.wavelength).toBeLessThan(448);
    close(xBlue.value, 0.348, PAPER_MAX_ABS[0]!);
    expect(vdbCie1931Fit(505)[0]).toBeLessThan(0.01);
    // z̄: one flat-topped lobe near 445 nm (≈ 1.78)
    const z = peak(2, 400, 500);
    expect(z.wavelength).toBeGreaterThan(440);
    expect(z.wavelength).toBeLessThan(450);
    close(z.value, 1.78, PAPER_MAX_ABS[2]!);
    // tails: below the paper's max error outside 380–780 nm
    for (const w of [360, 370, 790, 830]) for (let k = 0; k < 3; k++) expect(Math.abs(vdbCie1931Fit(w)[k]!)).toBeLessThan(PAPER_MAX_ABS[k]!);
  });

  it("each lobe is continuous at its centre (the S selector of Equation 4)", () => {
    for (const centre of [442.0, 599.8, 501.1, 568.8, 530.9, 437.0, 459.0]) {
      const a = vdbCie1931Fit(centre - 1e-9);
      const b = vdbCie1931Fit(centre + 1e-9);
      for (let k = 0; k < 3; k++) close(a[k]!, b[k]!, 1e-9);
    }
  });

  it("equal-energy white integrates to (⅓, ⅓) and the three curves to ≈ the same area", () => {
    const area: Vec3 = [0, 0, 0];
    for (let w = 360; w <= 830; w++) {
      const c = vdbCie1931Fit(w);
      for (let k = 0; k < 3; k++) area[k] += c[k]!;
    }
    const [x, y] = xyOf(area);
    close(x, 1 / 3, 0.003);
    close(y, 1 / 3, 0.003);
    close(area[0] / area[1], 1, 0.01);
  });
});

describe("D33: the Planck integration with the fitted curves", () => {
  it("his 81-sample sum equals an independent 1 nm trapezoid of the full Planck law (soot 0, 1.39, 3)", () => {
    const h = 6.62607015e-34;
    const c = 2.99792458e8;
    const kB = 1.380649e-23;
    const trapezoid = (temperature: number, exponent: number): Vec3 => {
      const xyz: Vec3 = [0, 0, 0];
      for (let w = 380; w <= 780; w++) {
        const m = w * 1e-9;
        const planck = (2 * h * c * c) / m ** 5 / Math.expm1((h * c) / (m * kB * temperature));
        const weight = (w === 380 || w === 780 ? 0.5 : 1) * planck * (w / 550) ** -exponent;
        const f = vdbCie1931Fit(w);
        for (let k = 0; k < 3; k++) xyz[k] += weight * f[k]!;
      }
      return xyz;
    };
    const toRgb = ([X, Y, Z]: Vec3): Vec3 => {
      const r = Math.max(0, 3.2406 * X - 1.5372 * Y - 0.4986 * Z);
      const g = Math.max(0, -0.9689 * X + 1.8758 * Y + 0.0415 * Z);
      const b = Math.max(0, 0.0557 * X - 0.204 * Y + 1.057 * Z);
      const p = Math.max(r, g, b);
      return [r / p, g / p, b / p];
    };
    let worst = 0;
    for (const exponent of [0, 1.39, 3])
      for (const T of temperatures(40, 600, 40000)) {
        const a = vdbFrameBlackbodyRgb(T, exponent);
        const b = toRgb(trapezoid(T, exponent));
        for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(a[k]! - b[k]!));
      }
    // 5 nm rectangle vs 1 nm trapezoid of smooth curves: sampling error only
    expect(worst).toBeLessThan(2e-3);
  });

  it("'planck' stays on the Planckian locus with the fit (x, y at 3000 K and 6500 K within 0.002)", () => {
    const [x3, y3] = xyOf(blackbodyXyz(3000, 0, vdbCie1931Fit));
    close(x3, 0.4369, 0.002);
    close(y3, 0.4041, 0.002);
    const [x6, y6] = xyOf(blackbodyXyz(6500, 0, vdbCie1931Fit));
    close(x6, 0.3135, 0.002);
    close(y6, 0.3236, 0.002);
  });

  it("the cmf argument: the default is the fit; other curves give their own (cached) table", () => {
    expect(vdbFrameBlackbodyRgb(2500, 0.7)).toEqual(vdbFrameBlackbodyRgb(2500, 0.7, vdbCie1931Fit));
    expect(vdbFrameBlackbodyTable()).toBe(vdbFrameBlackbodyTable(vdbCie1931Fit));
    // a scaled set of curves: the same normalized colour
    const scaled: VDBColorMatchingFunctions = (w) => vdbCie1931Fit(w).map((v) => v * 3) as Vec3;
    const a = vdbFrameBlackbodyRgb(2500, 0.7, scaled);
    const b = vdbFrameBlackbodyRgb(2500, 0.7);
    for (let k = 0; k < 3; k++) close(a[k]!, b[k]!, 1e-12);
    // ȳ alone: XYZ = (0, Y, 0) → linear sRGB (0, 1.8758, 0) Y → (0, 1, 0)
    const lumaOnly: VDBColorMatchingFunctions = (w) => [0, vdbCie1931Fit(w)[1], 0];
    expect(vdbFrameBlackbodyRgb(4000, 0, lumaOnly)).toEqual([0, 1, 0]);
    const table = vdbFrameBlackbodyTable(lumaOnly);
    expect(table).not.toBe(vdbFrameBlackbodyTable());
    expect(vdbFrameBlackbodyTable(lumaOnly)).toBe(table);
    expect(vdbFrameBlackbodyColor(4000, 0, lumaOnly)).toEqual([0, 1, 0]);
    // the curves are read at his 81 wavelengths only
    const seen: number[] = [];
    vdbFrameBlackbodyRgb(3000, 0, (w) => (seen.push(w), vdbCie1931Fit(w)));
    expect(seen).toEqual(Array.from({ length: 81 }, (_, i) => 380 + 5 * i));
  });
});

// The exact CIE 1931 2° table (test-only fixture; CC BY-SA 4.0, never shipped: D33)
const CIE_CSV = fileURLToPath(new URL("../../../../fixtures/cie/CIE_xyz_1931_2deg.csv", import.meta.url));
const cieRows = existsSync(CIE_CSV)
  ? new Map(
      readFileSync(CIE_CSV, "utf8")
        .trim()
        .split(/\r?\n/)
        .map((line) => line.split(",").map(Number))
        .map(([w, x, y, z]) => [w!, [x!, y!, z!] as Vec3]),
    )
  : null;
const cieExact: VDBColorMatchingFunctions = (w) => {
  const row = cieRows!.get(w);
  if (!row) throw new Error(`no CIE row at ${w} nm`);
  return row;
};

describe.skipIf(!cieRows)("D33: the fit against the exact CIE 1931 table (fixtures/cie, test-only)", () => {
  it("curves: squared errors within the paper's Table 2 (1 nm, 360–830 nm)", () => {
    const maxSq = [0, 0, 0];
    const sumSq = [0, 0, 0];
    let n = 0;
    for (const [w, exact] of cieRows!) {
      const fit = vdbCie1931Fit(w);
      for (let k = 0; k < 3; k++) {
        const e = (fit[k]! - exact[k]!) ** 2;
        maxSq[k] = Math.max(maxSq[k]!, e);
        sumSq[k] += e;
      }
      n++;
    }
    const meanSq = sumSq.map((s) => s / n);
    console.info(`[D33] CMF fit vs CIE (${n} rows): max squared error ${maxSq.map((v) => v.toExponential(2)).join(" / ")}, mean ${meanSq.map((v) => v.toExponential(2)).join(" / ")}`);
    for (let k = 0; k < 3; k++) {
      expect(maxSq[k]).toBeLessThanOrEqual(PAPER_MAX_SQ[k]! * 1.05);
      expect(meanSq[k]).toBeLessThanOrEqual(PAPER_MEAN_SQ[k]! * 1.05);
    }
  });

  // The fit's x̄ / ȳ tails past ~680 nm fall off faster than the table's, so the coolest temperatures (whose radiance
  // sits in that tail) move most in chromaticity; their normalized colour is saturated red either way.
  it("blackbody chromaticity and linear-RGB colour: fit vs exact table over 1,000–40,000 K", () => {
    let maxXy = 0;
    let maxXyHot = 0;
    let sumXy = 0;
    let maxRgb = 0;
    let sumRgb = 0;
    let maxSoot = 0;
    const Ts = temperatures(1000);
    for (const T of Ts) {
      const [xa, ya] = xyOf(blackbodyXyz(T, 0, vdbCie1931Fit));
      const [xb, yb] = xyOf(blackbodyXyz(T, 0, cieExact));
      const dxy = Math.hypot(xa - xb, ya - yb);
      maxXy = Math.max(maxXy, dxy);
      if (T >= 3000) maxXyHot = Math.max(maxXyHot, dxy);
      sumXy += dxy;
      const a = vdbFrameBlackbodyRgb(T, 0);
      const b = vdbFrameBlackbodyRgb(T, 0, cieExact);
      const d = Math.max(...a.map((v, k) => Math.abs(v - b[k]!)));
      maxRgb = Math.max(maxRgb, d);
      sumRgb += d;
      const sa = vdbFrameBlackbodyRgb(T, 1.39);
      const sb = vdbFrameBlackbodyRgb(T, 1.39, cieExact);
      maxSoot = Math.max(maxSoot, ...sa.map((v, k) => Math.abs(v - sb[k]!)));
    }
    console.info(
      `[D33] blackbody 1,000–40,000 K (${Ts.length} temperatures): Δxy max ${maxXy.toExponential(2)} (≥ 3,000 K: ${maxXyHot.toExponential(2)}), mean ${(sumXy / Ts.length).toExponential(2)}; ` +
        `normalized linear RGB max |Δ| ${maxRgb.toExponential(2)}, mean ${(sumRgb / Ts.length).toExponential(2)}; soot 1.39 max |Δ| ${maxSoot.toExponential(2)}`,
    );
    expect(maxXy).toBeLessThan(0.02);
    expect(maxXyHot).toBeLessThan(0.002);
    expect(sumXy / Ts.length).toBeLessThan(0.003);
    expect(maxRgb).toBeLessThan(0.025);
    expect(sumRgb / Ts.length).toBeLessThan(0.008);
    expect(maxSoot).toBeLessThan(0.025);
  });

  it("the exact table through the cmf argument reproduces the locus more tightly", () => {
    const [x3, y3] = xyOf(blackbodyXyz(3000, 0, cieExact));
    close(x3, 0.4369, 0.0005);
    close(y3, 0.4041, 0.0005);
  });
});
