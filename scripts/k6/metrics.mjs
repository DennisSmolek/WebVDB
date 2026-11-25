// K6.R1 report metrics on RGBA8 sRGB images { width, height, data } (alpha ignored).
//
//   ssim(a, b, mask?)   mean SSIM (Wang, Bovik, Sheikh & Simoncelli 2004) of the Rec. 601 luma (MATLAB rgb2gray), an
//                       11 × 11 Gaussian window (σ 1.5), K1 0.01, K2 0.03, L 255, over the pixels whose window fits
//                       inside the image ('valid'); with a mask (Uint8Array, 1 = in), over the valid pixels in it.
//   silhouetteMask      pixels within one pixel of the alpha = 0.02 contour of a coverage image (both sides).
//   dilateMask          a mask grown by r pixels (Chebyshev).
//
// FLIP (Andersson et al. 2020) is not here: without NVIDIA's reference implementation to check against, SSIM is the
// structural metric of the report (K6.R1 brief: "FLIP, or SSIM if FLIP is too costly to implement correctly").

const K1 = 0.01;
const K2 = 0.03;
const L = 255;
const C1 = (K1 * L) ** 2;
const C2 = (K2 * L) ** 2;

/** Rec. 601 luma of an RGBA8 image (rgb2gray), as floats in 0–255. */
export function luma(img) {
  const n = img.width * img.height;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = 0.298936 * img.data[i * 4] + 0.587043 * img.data[i * 4 + 1] + 0.114021 * img.data[i * 4 + 2];
  return y;
}

/** The normalised 1D Gaussian of `size` taps (σ). */
export function gaussian(size = 11, sigma = 1.5) {
  const r = (size - 1) / 2;
  const g = Array.from({ length: size }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
  const s = g.reduce((a, b) => a + b, 0);
  return g.map((v) => v / s);
}

/** 'valid' separable filtering of a w × h field: (w − size + 1) × (h − size + 1). */
function filterValid(f, w, h, g) {
  const k = g.length;
  const ow = w - k + 1;
  const oh = h - k + 1;
  const tmp = new Float64Array(ow * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < ow; x++) {
    let s = 0;
    for (let i = 0; i < k; i++) s += g[i] * f[y * w + x + i];
    tmp[y * ow + x] = s;
  }
  const out = new Float64Array(ow * oh);
  for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
    let s = 0;
    for (let i = 0; i < k; i++) s += g[i] * tmp[(y + i) * ow + x];
    out[y * ow + x] = s;
  }
  return { data: out, width: ow, height: oh };
}

/**
 * Mean SSIM of two RGBA8 images of equal size (luma). `mask` (Uint8Array w·h, optional) restricts the mean to the
 * window centres inside it. Returns { ssim, pixels } (NaN when no window fits / none is in the mask).
 */
export function ssim(a, b, mask = null, size = 11, sigma = 1.5) {
  if (a.width !== b.width || a.height !== b.height) throw new Error(`ssim: size ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  const { width: w, height: h } = a;
  const x = luma(a);
  const y = luma(b);
  const g = gaussian(size, sigma);
  const xx = x.map((v) => v * v);
  const yy = y.map((v) => v * v);
  const xy = x.map((v, i) => v * y[i]);
  const mx = filterValid(x, w, h, g);
  const my = filterValid(y, w, h, g);
  const sxx = filterValid(xx, w, h, g);
  const syy = filterValid(yy, w, h, g);
  const sxy = filterValid(xy, w, h, g);
  const r = (size - 1) / 2;
  let sum = 0;
  let n = 0;
  for (let j = 0; j < mx.height; j++) for (let i = 0; i < mx.width; i++) {
    if (mask && !mask[(j + r) * w + i + r]) continue;
    const k = j * mx.width + i;
    const ux = mx.data[k];
    const uy = my.data[k];
    const vx = sxx.data[k] - ux * ux;
    const vy = syy.data[k] - uy * uy;
    const cxy = sxy.data[k] - ux * uy;
    sum += ((2 * ux * uy + C1) * (2 * cxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2));
    n++;
  }
  return { ssim: n ? sum / n : NaN, pixels: n };
}

/** Pixels within one pixel of the `threshold` contour of a coverage image (alpha, w·h floats): both sides of the edge. */
export function silhouetteMask(alpha, w, h, threshold = 0.02) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const inside = alpha[y * w + x] > threshold;
    let edge = false;
    for (let dy = -1; dy <= 1 && !edge; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = x + dx;
      const Y = y + dy;
      if (X < 0 || Y < 0 || X >= w || Y >= h) continue;
      if (alpha[Y * w + X] > threshold !== inside) {
        edge = true;
        break;
      }
    }
    if (edge) out[y * w + x] = 1;
  }
  return out;
}

/** A mask grown by `r` pixels (Chebyshev distance). */
export function dilateMask(mask, w, h, r) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const X = x + dx;
      const Y = y + dy;
      if (X >= 0 && Y >= 0 && X < w && Y < h) out[Y * w + X] = 1;
    }
  }
  return out;
}
