import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { compareNormals, decodeNormal, maskIou, silhouette } from "../../../../scripts/k3/compare-step7.mjs";
import { encodePng } from "../../../../scripts/k3/png.mjs";

// K3.7: the Step 7 parity metrics (scripts/k3/compare-step7.mjs): silhouette IoU and normal-angle error on images
// that encode view-space normals as sRGB(n * 0.5 + 0.5) on black, as both renderers write them.

const CLI = fileURLToPath(new URL("../../../../scripts/k3/compare-step7.mjs", import.meta.url));
const oetf = (x) => (x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055);

/** A disc of radius r (px) centred at (cx, cy) whose normals are a hemisphere, rotated by `tilt` radians about y. */
function sphereImage(w, h, cx, cy, r, tilt = 0) {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i + 3] = 255;
      const dx = (x + 0.5 - cx) / r;
      const dy = (cy - y - 0.5) / r;
      if (dx * dx + dy * dy > 1) continue;
      const nz = Math.sqrt(1 - dx * dx - dy * dy);
      const n = [dx * Math.cos(tilt) + nz * Math.sin(tilt), dy, -dx * Math.sin(tilt) + nz * Math.cos(tilt)];
      for (let c = 0; c < 3; c++) data[i + c] = Math.round(oetf(n[c] * 0.5 + 0.5) * 255);
    }
  }
  return { width: w, height: h, data };
}

describe("Step 7 metrics", () => {
  it("identical normal images: IoU 1, angle error within 8-bit quantization", () => {
    const a = sphereImage(64, 48, 32, 24, 18);
    const r = compareNormals(a, a);
    expect(r.iou).toBe(1);
    expect(r.normal.maxDeg).toBeLessThan(1e-4); // acos of a dot product of 1 - ulp
    expect(r.normal.pixels).toBeGreaterThan(800);
    // The decode recovers the normal to the 8-bit sRGB step (< 1 degree): pixel (40, 20) of the disc.
    const n = decodeNormal(a.data, 20 * 64 + 40);
    const dx = (40.5 - 32) / 18;
    const dy = (24 - 20.5) / 18;
    const dot = n[0] * dx + n[1] * dy + n[2] * Math.sqrt(1 - dx * dx - dy * dy);
    expect(Math.acos(Math.min(1, dot)) * (180 / Math.PI)).toBeLessThan(1);
  });

  it("a 10-degree tilt reads as ~10 degrees; a 1-pixel shift costs IoU", () => {
    const a = sphereImage(64, 48, 32, 24, 18);
    const tilted = compareNormals(a, sphereImage(64, 48, 32, 24, 18, (10 * Math.PI) / 180));
    expect(tilted.iou).toBe(1);
    expect(tilted.normal.medianDeg).toBeGreaterThan(9);
    expect(tilted.normal.medianDeg).toBeLessThan(11);
    const shifted = compareNormals(a, sphereImage(64, 48, 33, 24, 18));
    expect(shifted.iou).toBeLessThan(0.99);
    expect(shifted.iou).toBeGreaterThan(0.9);
  });

  it("silhouette: no unit normal is that dark; masks and IoU count pixels", () => {
    const m = silhouette(sphereImage(16, 16, 8, 8, 6));
    expect(maskIou(m, m)).toMatchObject({ iou: 1 });
    expect(maskIou(new Uint8Array([1, 1, 0, 0]), new Uint8Array([1, 0, 1, 0]))).toMatchObject({ iou: 1 / 3, intersection: 1, union: 3 });
  });

  it("CLI: SKIPPED (exit 0) without Step 7 captures; PASS / FAIL on the IoU gate", () => {
    const root = mkdtempSync(path.join(tmpdir(), "k3-step7-"));
    const ours = path.join(root, "ours");
    const ref = path.join(root, "ref");
    mkdirSync(ours);
    mkdirSync(ref);
    const run = () => spawnSync(process.execPath, [CLI, "--ours", ours, "--reference", ref], { encoding: "utf8" });
    let r = run();
    expect(r.status).toBe(0);
    expect(r.stdout).toContain("SKIPPED");
    const name = "step7-utah-teapot-surface-normals.png";
    writeFileSync(path.join(ref, name), encodePng(sphereImage(64, 48, 32, 24, 18)));
    writeFileSync(path.join(ours, name), encodePng(sphereImage(64, 48, 32, 24, 18, 0.05)));
    r = run();
    expect(r.status, r.stdout + r.stderr).toBe(0);
    expect(r.stdout).toMatch(/PASS\s+step7-utah-teapot-surface-normals\s+IoU 1\.0000/);
    writeFileSync(path.join(ours, name), encodePng(sphereImage(64, 48, 34, 24, 18)));
    r = run();
    expect(r.status).toBe(1);
    expect(r.stdout).toContain("FAIL");
  });
});
