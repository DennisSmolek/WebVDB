// K3 image-parity metrics (R1 §6). Inputs are RGBA8 images { width, height, data } of equal size;
// alpha is ignored (both renderers present opaque frames).
//
//   psnr        10·log10(255² / MSE) over the R, G and B channels; Infinity when identical.
//   above(t)    pixels whose largest |ΔR|, |ΔG|, |ΔB| exceeds t (his validation.spec.js countChangedPixels
//               uses the same per-pixel max, with t = 48 and ≤ 1% for his q90 JPEG smoke check).
//   heatmap     per-pixel max channel difference as a colour ramp; pixels over the threshold are outlined.

/** Parity bars. `strict` is Step 1+ on identical dense data; `jpegSmoke` is his own loose JPEG check. */
export const PARITY_BARS = Object.freeze({
  strict: Object.freeze({ minPsnr: 40, threshold: 8, maxFraction: 0.005 }),
  jpegSmoke: Object.freeze({ minPsnr: 0, threshold: 48, maxFraction: 0.01 }),
});

function checkPair(a, b) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`size mismatch: ${a.width}x${a.height} vs ${b.width}x${b.height} (capture both at the same resolution)`);
  }
  const n = a.width * a.height * 4;
  if (a.data.length !== n || b.data.length !== n) throw new Error("images must be RGBA8 with width*height*4 bytes");
}

/** Mean squared error over RGB (alpha ignored), in 8-bit units². */
export function mse(a, b) {
  checkPair(a, b);
  let sum = 0;
  const d = a.data;
  const e = b.data;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i] - e[i];
    const g = d[i + 1] - e[i + 1];
    const bl = d[i + 2] - e[i + 2];
    sum += r * r + g * g + bl * bl;
  }
  return sum / (a.width * a.height * 3);
}

/** PSNR in dB for 8-bit RGB; Infinity for identical images. */
export function psnr(a, b) {
  const m = mse(a, b);
  return m === 0 ? Infinity : 10 * Math.log10((255 * 255) / m);
}

/** Pixels whose largest channel difference is strictly greater than `threshold` (8-bit units). */
export function above(a, b, threshold = 8) {
  checkPair(a, b);
  let count = 0;
  let maxDiff = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const diff = Math.max(Math.abs(a.data[i] - b.data[i]), Math.abs(a.data[i + 1] - b.data[i + 1]), Math.abs(a.data[i + 2] - b.data[i + 2]));
    if (diff > maxDiff) maxDiff = diff;
    if (diff > threshold) count++;
  }
  const pixels = a.width * a.height;
  return { count, fraction: count / pixels, pixels, maxDiff };
}

/**
 * Diff heatmap: black (equal) → blue → red → yellow → white at `fullScale` (default 32/255) or more.
 * Pixels over `threshold` get a magenta 1-px outline marker so isolated failures stay visible.
 */
export function diffHeatmap(a, b, { threshold = 8, fullScale = 32 } = {}) {
  checkPair(a, b);
  const { width, height } = a;
  const out = new Uint8Array(width * height * 4);
  const over = new Uint8Array(width * height);
  const ramp = (t) => {
    // piecewise: 0 black, 0.25 blue, 0.5 red, 0.75 yellow, 1 white
    const stops = [
      [0, 0, 0],
      [0, 40, 200],
      [220, 30, 30],
      [255, 220, 0],
      [255, 255, 255],
    ];
    const x = Math.min(Math.max(t, 0), 1) * 4;
    const i = Math.min(Math.floor(x), 3);
    const f = x - i;
    return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
  };
  for (let p = 0; p < width * height; p++) {
    const i = p * 4;
    const diff = Math.max(Math.abs(a.data[i] - b.data[i]), Math.abs(a.data[i + 1] - b.data[i + 1]), Math.abs(a.data[i + 2] - b.data[i + 2]));
    const [r, g, bl] = ramp(diff / fullScale);
    out[i] = r;
    out[i + 1] = g;
    out[i + 2] = bl;
    out[i + 3] = 255;
    if (diff > threshold) over[p] = 1;
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (over[y * width + x]) continue;
      const near = (dx, dy) => {
        const xx = x + dx;
        const yy = y + dy;
        return xx >= 0 && yy >= 0 && xx < width && yy < height && over[yy * width + xx] === 1;
      };
      if (near(-1, 0) || near(1, 0) || near(0, -1) || near(0, 1)) {
        const i = (y * width + x) * 4;
        out[i] = 255;
        out[i + 1] = 0;
        out[i + 2] = 255;
      }
    }
  }
  return { width, height, data: out };
}

/**
 * Compare two images against a bar. Returns every metric plus `pass` and the reasons it failed.
 * @param bar `PARITY_BARS.strict` (default) or `PARITY_BARS.jpegSmoke` or `{ minPsnr, threshold, maxFraction }`
 */
export function compareImages(a, b, bar = PARITY_BARS.strict) {
  const p = psnr(a, b);
  const ab = above(a, b, bar.threshold);
  const failures = [];
  if (p < bar.minPsnr) failures.push(`PSNR ${p.toFixed(2)} dB < ${bar.minPsnr} dB`);
  if (ab.fraction > bar.maxFraction) {
    failures.push(`${(ab.fraction * 100).toFixed(3)}% pixels > ${bar.threshold}/255 (limit ${(bar.maxFraction * 100).toFixed(3)}%)`);
  }
  return { psnr: p, mse: mse(a, b), threshold: bar.threshold, above: ab.count, fraction: ab.fraction, maxDiff: ab.maxDiff, pixels: ab.pixels, pass: failures.length === 0, failures };
}
