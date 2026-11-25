/**
 * Demo 17's environment: a procedural HDR sky written on the CPU into an equirect `rgba16float` texture, the
 * convention the core reads (`VDBFrameEnvironment`: u = atan2(d.z, d.x) / 2π + ½ along x, row 0 the zenith). The same
 * texture is the page's background, so the sky the volumes are lit by is the sky you see. No sun disc in the map: the
 * sun is a directional light of its own (the background pass draws its disc), so the core's environment extraction
 * finds the sky, not a second sun.
 */
import { vdbFloatToHalf } from "../../../../packages/vdb/src/tools/internal.js";
import type { Vec3 } from "./raw-frame.js";

export const SKY_WIDTH = 512;
export const SKY_HEIGHT = 256;

/** The unit direction of equirect texel centre (x, y). */
export function skyDirection(x: number, y: number, w = SKY_WIDTH, h = SKY_HEIGHT): Vec3 {
  const phi = ((x + 0.5) / w - 0.5) * 2 * Math.PI;
  const theta = ((y + 0.5) / h) * Math.PI;
  const s = Math.sin(theta);
  return [Math.cos(phi) * s, Math.cos(theta), Math.sin(phi) * s];
}

/** Linear radiance of the sky toward `d` with the sun toward `toSun` (unit vectors). */
export function skyRadiance(d: Vec3, toSun: Vec3, sunColor: Vec3): Vec3 {
  const c = Math.max(0, d[0] * toSun[0] + d[1] * toSun[1] + d[2] * toSun[2]);
  const elevation = Math.max(0.05, toSun[1]);
  if (d[1] >= 0) {
    const t = Math.pow(1 - d[1], 3);
    const zenith: Vec3 = [0.045, 0.12, 0.38];
    const horizon: Vec3 = [0.4, 0.47, 0.58];
    // a warmer, brighter horizon when the sun is low
    const warm = 1 - elevation;
    const glow = 0.35 * Math.pow(c, 32) + 0.12 * Math.pow(c, 4);
    return [0, 1, 2].map((k) => (zenith[k]! + (horizon[k]! - zenith[k]!) * t) * (0.6 + 0.6 * elevation) + glow * sunColor[k]! + (k === 0 ? 0.08 : k === 1 ? 0.03 : 0) * warm * t) as Vec3;
  }
  // below the horizon: dim ground bounce, darker straight down
  const g = 0.35 + 0.65 * Math.exp(d[1] * 5);
  return [0.16 * g, 0.15 * g, 0.13 * g];
}

/** The equirect texels (rgba16float bits) of the sky. */
export function skyTexels(toSun: Vec3, sunColor: Vec3, w = SKY_WIDTH, h = SKY_HEIGHT): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const r = skyRadiance(skyDirection(x, y, w, h), toSun, sunColor);
      const i = (y * w + x) * 4;
      out[i] = vdbFloatToHalf(r[0]);
      out[i + 1] = vdbFloatToHalf(r[1]);
      out[i + 2] = vdbFloatToHalf(r[2]);
      out[i + 3] = vdbFloatToHalf(1);
    }
  }
  return out;
}

/** The hemisphere ambient a Lambert surface facing up (sky) and down (ground) gets from this sky (cosine-weighted mean). */
export function skyAmbient(toSun: Vec3, sunColor: Vec3): { sky: Vec3; ground: Vec3 } {
  const sky: Vec3 = [0, 0, 0];
  const ground: Vec3 = [0, 0, 0];
  let ws = 0;
  let wg = 0;
  const n = 32;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n * 2; x++) {
      const d = skyDirection(x, y, n * 2, n);
      const r = skyRadiance(d, toSun, sunColor);
      const solid = Math.sin(((y + 0.5) / n) * Math.PI);
      const up = Math.max(0, d[1]) * solid;
      const down = Math.max(0, -d[1]) * solid;
      for (let k = 0; k < 3; k++) {
        sky[k]! += r[k]! * up;
        ground[k]! += r[k]! * down;
      }
      ws += up;
      wg += down;
    }
  }
  return { sky: sky.map((v) => v / ws) as Vec3, ground: ground.map((v) => v / wg) as Vec3 };
}
