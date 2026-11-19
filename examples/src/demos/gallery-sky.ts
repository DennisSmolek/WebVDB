/**
 * Sky helpers of the gallery demos (12, 16): the UltraHDR map loader, the procedural fallback sky and the horizon
 * colour a ground fog is matched to. Plain three; no shell dependency.
 */
import * as THREE from "three/webgpu";
import { UltraHDRLoader } from "three/addons/loaders/UltraHDRLoader.js";

/** An UltraHDR JPEG sky (his examples' maps) as an equirect environment and background. */
export async function loadHdrSky(url: string): Promise<THREE.Texture> {
  const texture = await new UltraHDRLoader().loadAsync(url);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  return texture;
}

/**
 * A procedural equirect sky (the fallback when no HDR map is present): a vertical gradient from `zenith` through
 * `horizon` to `ground`, with an optional warm glow band at the horizon. Linear-light values, so it lights as well.
 */
export function proceduralSky(stops: { zenith: string; horizon: string; ground: string; glow?: string }): THREE.DataTexture {
  const W = 256;
  const H = 128;
  const data = new Float32Array(W * H * 4);
  const c = (s: string) => new THREE.Color(s); // sRGB hex -> linear working colour
  const zenith = c(stops.zenith);
  const horizon = c(stops.horizon);
  const ground = c(stops.ground);
  const glow = stops.glow ? c(stops.glow) : null;
  const col = new THREE.Color();
  for (let y = 0; y < H; y++) {
    // row 0 = top of the texture (flipY false below): v = 1 at the zenith
    const v = 1 - (y + 0.5) / H;
    const el = (v - 0.5) * Math.PI; // elevation
    if (el >= 0) col.copy(horizon).lerp(zenith, Math.pow(Math.sin(el), 0.55));
    else col.copy(horizon).lerp(ground, Math.min(1, Math.pow(-Math.sin(el), 0.35) * 1.2));
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      let r = col.r;
      let g = col.g;
      let b = col.b;
      if (glow && el > -0.05) {
        const az = (x / W) * Math.PI * 2;
        const band = Math.exp(-Math.pow(el / 0.12, 2)) * (0.55 + 0.45 * Math.max(0, Math.cos(az - 0.6)));
        r += glow.r * band;
        g += glow.g * band;
        b += glow.b * band;
      }
      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      data[o + 3] = 1;
    }
  }
  const texture = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * The mean colour of an equirect map's horizon band (the rows within ±`band` of the middle), linear: a fog colour
 * that blends a ground plane into the sky. Reads a `DataTexture`'s float or half-float texels; null otherwise.
 */
export function horizonColor(texture: THREE.Texture, band = 0.03): THREE.Color | null {
  const image = texture.image as { data?: ArrayLike<number>; width?: number; height?: number } | undefined;
  const data = image?.data;
  const w = image?.width ?? 0;
  const h = image?.height ?? 0;
  if (!data || !w || !h || data.length < w * h * 4) return null;
  const half = data instanceof Uint16Array;
  const read = (i: number) => (half ? THREE.DataUtils.fromHalfFloat(data[i]!) : data[i]!);
  const mean = (y0: number, y1: number): THREE.Color => {
    const sum: [number, number, number] = [0, 0, 0];
    let n = 0;
    for (let y = Math.max(0, y0); y < Math.min(h, y1); y++) {
      for (let x = 0; x < w; x += 2) {
        const o = (y * w + x) * 4;
        sum[0] += read(o);
        sum[1] += read(o + 1);
        sum[2] += read(o + 2);
        n++;
      }
    }
    return new THREE.Color(sum[0]! / Math.max(1, n), sum[1]! / Math.max(1, n), sum[2]! / Math.max(1, n));
  };
  // The darker of the bands just above and just below the horizon (the map's ground side, whichever way up it is).
  const mid = Math.round(h / 2);
  const a = mean(mid - Math.ceil(h * band), mid);
  const b = mean(mid, mid + Math.ceil(h * band));
  const lum = (c: THREE.Color) => c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
  return lum(a) <= lum(b) ? a : b;
}
