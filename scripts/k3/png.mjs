// Minimal PNG codec for the K3 parity rig (no dependencies; node:zlib).
// Encode: RGBA8, filter 0, deterministic bytes for identical pixels.
// Decode: 8-bit greyscale, grey+alpha, RGB, RGBA (colour types 0, 4, 2, 6), non-interlaced,
// all five scanline filters (Chrome screenshots use them). Output is always RGBA8.
import { deflateSync, inflateSync } from "node:zlib";

const SIGNATURE = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** @param {{ width: number, height: number, data: Uint8Array }} image RGBA8, top row first */
export function encodePng({ width, height, data }) {
  if (data.length !== width * height * 4) throw new Error(`encodePng: ${data.length} bytes for ${width}x${height} RGBA8`);
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  const raw = new Uint8Array(height * (1 + width * 4));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + width * 4)] = 0;
    raw.set(data.subarray(y * width * 4, (y + 1) * width * 4), y * (1 + width * 4) + 1);
  }
  const parts = [SIGNATURE, chunk("IHDR", ihdr), chunk("IDAT", new Uint8Array(deflateSync(raw, { level: 9 }))), chunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

const paeth = (a, b, c) => {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
};

/** @returns {{ width: number, height: number, data: Uint8Array }} RGBA8, top row first */
export function decodePng(bytes) {
  bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  for (let i = 0; i < 8; i++) if (bytes[i] !== SIGNATURE[i]) throw new Error("decodePng: not a PNG");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 8;
  let width = 0;
  let height = 0;
  let colorType = -1;
  const idat = [];
  while (o < bytes.length) {
    const len = view.getUint32(o);
    const type = String.fromCharCode(...bytes.subarray(o + 4, o + 8));
    const data = bytes.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      width = view.getUint32(o + 8);
      height = view.getUint32(o + 12);
      const depth = data[8];
      colorType = data[9];
      if (depth !== 8) throw new Error(`decodePng: bit depth ${depth} not supported (8 only)`);
      if (data[12] !== 0) throw new Error("decodePng: interlaced PNGs are not supported");
      if (![0, 2, 4, 6].includes(colorType)) throw new Error(`decodePng: colour type ${colorType} not supported`);
    } else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
    o += 12 + len;
  }
  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType];
  const joined = new Uint8Array(idat.reduce((s, d) => s + d.length, 0));
  let j = 0;
  for (const d of idat) {
    joined.set(d, j);
    j += d.length;
  }
  const raw = new Uint8Array(inflateSync(joined));
  const stride = width * channels;
  const pixels = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const out = pixels.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? pixels.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[x - channels] : 0;
      const b = prev ? prev[x] : 0;
      const c = prev && x >= channels ? prev[x - channels] : 0;
      let v = line[x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) v += paeth(a, b, c);
      else if (filter !== 0) throw new Error(`decodePng: bad filter ${filter} on row ${y}`);
      out[x] = v & 0xff;
    }
  }
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const s = i * channels;
    if (channels === 4) data.set(pixels.subarray(s, s + 4), i * 4);
    else if (channels === 3) {
      data[i * 4] = pixels[s];
      data[i * 4 + 1] = pixels[s + 1];
      data[i * 4 + 2] = pixels[s + 2];
      data[i * 4 + 3] = 255;
    } else {
      data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = pixels[s];
      data[i * 4 + 3] = channels === 2 ? pixels[s + 1] : 255;
    }
  }
  return { width, height, data };
}
