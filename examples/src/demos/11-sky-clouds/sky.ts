/**
 * Demo 11's sky: the time-of-day presets, the sun colour from its elevation, and the procedural sky used when the
 * gallery HDRs are absent (an equirect HalfFloat texture made on the CPU, used as background and environment, so the
 * froxel renderer's environment lights still have a sky to read).
 */
import * as THREE from "three/webgpu";

export type TimeOfDay = "midday" | "dusk" | "cloudy";

export interface TimePreset {
  label: string;
  /** `fixtures/gallery/sky-<hdr>-HDR.jpg` (UltraHDR). */
  hdr: string;
  /** Default sun elevation (degrees), near the HDR's own sun. */
  elevation: number;
  /** Sun intensity factor (overcast skies dim the key light). */
  sun: number;
  /** Environment intensity with the HDR, and with the procedural sky. */
  environment: number;
  /** Background intensity of the HDR (the dusk photo is blown out around its sun). */
  background: number;
  proceduralEnvironment: number;
  exposure: number;
}

export const TIMES: Record<TimeOfDay, TimePreset> = {
  midday: { label: "midday", hdr: "sky-midday", elevation: 50, sun: 1, environment: 0.3, background: 1, proceduralEnvironment: 0.6, exposure: 0.9 },
  dusk: { label: "dusk", hdr: "sky-dusk", elevation: 7, sun: 1.1, environment: 0.2, background: 0.6, proceduralEnvironment: 0.6, exposure: 0.8 },
  cloudy: { label: "overcast", hdr: "sky-cloudy", elevation: 35, sun: 0.35, environment: 0.55, background: 0.85, proceduralEnvironment: 0.9, exposure: 0.85 },
};

/** Kasten–Young air mass at an elevation (degrees). */
const airMass = (elevationDeg: number): number => {
  const e = Math.max(elevationDeg, -0.5);
  return 1 / (Math.sin(THREE.MathUtils.degToRad(e)) + 0.50572 * Math.pow(e + 6.07995, -1.6364));
};

/** Optical depth per channel (Rayleigh-like falloff plus a little aerosol), R G B. */
const TAU = [0.1, 0.17, 0.33];

/** Sunlight colour (max channel 1) and a brightness factor (0 below the horizon) for an elevation. */
export function sunColor(elevationDeg: number): { color: THREE.Color; brightness: number } {
  const m = Math.min(airMass(elevationDeg), 40);
  const t = TAU.map((tau) => Math.exp(-m * tau));
  const max = Math.max(...t);
  const color = new THREE.Color(t[0]! / max, t[1]! / max, t[2]! / max);
  const horizon = THREE.MathUtils.smoothstep(elevationDeg, -1, 6);
  const luminance = 0.2126 * t[0]! + 0.7152 * t[1]! + 0.0722 * t[2]!;
  return { color, brightness: horizon * (0.45 + 0.55 * Math.sqrt(luminance)) };
}

/** The sun direction (unit, pointing at the sun) for an azimuth and an elevation (degrees). */
export function sunDirection(azimuthDeg: number, elevationDeg: number, out = new THREE.Vector3()): THREE.Vector3 {
  const az = THREE.MathUtils.degToRad(azimuthDeg);
  const el = THREE.MathUtils.degToRad(elevationDeg);
  return out.set(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
}

/**
 * The azimuth (degrees, in `sunDirection`'s convention) of the brightest texel of an equirect HDR texture, so the key
 * light can stand where the photo's sun is. three's equirect lookup: u = atan(z, x) / 2π + 0.5.
 */
export function hdrSunAzimuth(texture: THREE.Texture): number | null {
  const image = texture.image as { data?: ArrayLike<number>; width?: number; height?: number } | undefined;
  const data = image?.data;
  const w = image?.width ?? 0;
  const h = image?.height ?? 0;
  if (!data || w === 0 || h === 0 || data.length < w * h * 4) return null;
  const half = texture.type === THREE.HalfFloatType;
  const value = (i: number): number => (half ? THREE.DataUtils.fromHalfFloat(data[i]!) : data[i]!);
  let best = -Infinity;
  let bestX = 0;
  // Coarse scan (every 4th texel) is enough to find the sun.
  for (let y = 0; y < h; y += 2) {
    for (let x = 0; x < w; x += 2) {
      const i = (y * w + x) * 4;
      const l = value(i) + value(i + 1) + value(i + 2);
      if (l > best) {
        best = l;
        bestX = x;
      }
    }
  }
  const u = (bestX + 0.5) / w;
  return THREE.MathUtils.radToDeg((u - 0.5) * Math.PI * 2);
}

const mix3 = (a: readonly number[], b: readonly number[], t: number): number[] => a.map((v, i) => v + (b[i]! - v) * t);

/**
 * A procedural equirect sky (linear HalfFloat RGBA): zenith-to-horizon gradient, a warm horizon toward a low sun, a
 * Mie glow and a small sun disc, and a hazy ground below the horizon. Rows run bottom (down) to top (up), the
 * DataTexture default (flipY false), so three's equirect lookup sees the sky above.
 */
export function proceduralSky(time: TimeOfDay, azimuthDeg: number, elevationDeg: number, width = 1024, height = 512): THREE.DataTexture {
  const sun = sunDirection(azimuthDeg, elevationDeg);
  const { color: sunRgb, brightness } = sunColor(elevationDeg);
  const low = 1 - THREE.MathUtils.smoothstep(elevationDeg, 2, 30);
  const overcast = time === "cloudy" ? 1 : 0;
  const zenithDay = [0.16, 0.32, 0.75];
  const zenithDusk = [0.07, 0.09, 0.22];
  const horizonDay = [0.62, 0.74, 0.92];
  const horizonDusk = [0.95, 0.52, 0.3];
  const grey = [0.55, 0.58, 0.62];
  const zenith = mix3(mix3(zenithDay, zenithDusk, low), mix3(grey, grey, 0).map((v) => v * 0.8), overcast);
  const horizon = mix3(mix3(horizonDay, horizonDusk, low), grey, overcast);
  const data = new Uint16Array(width * height * 4);
  const dir = new THREE.Vector3();
  for (let y = 0; y < height; y++) {
    const lat = ((y + 0.5) / height - 0.5) * Math.PI;
    for (let x = 0; x < width; x++) {
      const lon = ((x + 0.5) / width - 0.5) * Math.PI * 2;
      dir.set(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon));
      const up = Math.max(dir.y, 0);
      const cosSun = dir.dot(sun);
      // Gradient: horizon band widens toward the sun at dusk.
      const toward = Math.max(0, cosSun) * low;
      const t = Math.pow(1 - up, 4 + 2 * (1 - toward));
      let c = mix3(zenith, horizon, t);
      if (dir.y < 0) c = mix3(horizon, [0.18, 0.19, 0.2].map((v) => v * (1 - 0.5 * overcast) + 0.3 * overcast), Math.min(1, -dir.y * 6));
      const glow = (1 - overcast * 0.85) * brightness * (0.35 * Math.pow(Math.max(cosSun, 0), 6) + 1.4 * Math.pow(Math.max(cosSun, 0), 64));
      // A soft disc (~1.5° across) rather than a hard one: a hard disc is one texel of the equirect, a square.
      const disc = dir.y > -0.01 ? 60 * Math.pow(Math.max(cosSun, 0), 4000) * brightness * (1 - overcast * 0.95) : 0;
      const scale = 0.4 + 0.6 * brightness;
      const i = (y * width + x) * 4;
      data[i] = THREE.DataUtils.toHalfFloat(c[0]! * scale + (glow + disc) * sunRgb.r);
      data[i + 1] = THREE.DataUtils.toHalfFloat(c[1]! * scale + (glow + disc) * sunRgb.g);
      data[i + 2] = THREE.DataUtils.toHalfFloat(c[2]! * scale + (glow + disc) * sunRgb.b);
      data[i + 3] = THREE.DataUtils.toHalfFloat(1);
    }
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}
