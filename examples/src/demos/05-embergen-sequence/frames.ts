/**
 * Synthetic animated-fog frames for demo 05 — a self-contained stand-in for a
 * real EmberGen `.nvdb` sequence, so the whole sequence-playback pipeline is
 * exercised end-to-end in-sandbox (no network, no EmberGen download). Authored
 * with the project's OWN serializer (`vdb-web-tools.buildFromDense`), so every
 * frame is a genuine, loader-parseable NanoVDB FLOAT grid — the same contract a
 * real pack satisfies.
 *
 * The animation is a fog blob orbiting + undulating inside a FIXED 64^3 index
 * domain, deterministic in the frame index. Because the blob moves and the
 * background is a hard 0, each frame's ACTIVE topology (and therefore its grid
 * byte length) varies frame to frame — which is exactly the case that makes
 * `NanoVDBVolumeMaterial.rebindGrid` need pre-sized (`maxGridBytes`) capacity
 * rather than the same-size fast path. That's deliberate: the demo proves the
 * harder, realistic rebind story.
 *
 * Shared by the demo (in-memory + Blob-URL paths) and any node-side script.
 */
import { buildFromDense } from "vdb-web-tools";

/** Voxel resolution of the (fixed) animation domain. */
export const FRAME_DIM = 64;

/** Total frames in the looping synthetic sequence. */
export const FRAME_COUNT = 24;

/** Clamp helper. */
function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Deterministic dense fog field for `frameIndex` in `[0, frameCount)`: a soft
 * blob whose center orbits the domain (bobbing in Y) with an undulating radius,
 * plus a gentle swirl term. Values in [0,1]; a hard floor makes the background
 * exactly 0 so the built grid is genuinely sparse (topology varies with the
 * blob's position — the point of the demo).
 */
export function makeDenseFrame(frameIndex: number, dim = FRAME_DIM, frameCount = FRAME_COUNT): Float32Array {
  const values = new Float32Array(dim * dim * dim);
  const t = (frameIndex / frameCount) * Math.PI * 2; // one full loop over the sequence
  const c = (dim - 1) / 2;

  const orbitR = dim * 0.18;
  const cx = c + orbitR * Math.cos(t);
  const cz = c + orbitR * Math.sin(t);
  const cy = c + dim * 0.08 * Math.sin(t * 2);
  const r = dim * 0.22 * (1 + 0.15 * Math.sin(t * 3)); // undulating radius

  for (let x = 0; x < dim; x++) {
    const dx = x - cx;
    const swirlX = Math.sin(x * 0.3 + t);
    for (let y = 0; y < dim; y++) {
      const dy = y - cy;
      for (let z = 0; z < dim; z++) {
        const dz = z - cz;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        let v = 1 - dist / r;
        v += 0.15 * swirlX * Math.sin(z * 0.3 - t);
        v = clamp01(v);
        // Hard floor -> exact-0 background -> sparse grid.
        values[(x * dim + y) * dim + z] = v < 0.06 ? 0 : Math.fround(v);
      }
    }
  }
  return values;
}

/**
 * Build one animated frame as a complete NanoVDB FLOAT grid image (flat u32) —
 * the `makeFrame(t): Uint32Array` the demo and node scripts share. `t` is the
 * frame index.
 */
export function makeFrame(frameIndex: number, dim = FRAME_DIM, frameCount = FRAME_COUNT): Uint32Array {
  const dense = makeDenseFrame(frameIndex, dim, frameCount);
  return buildFromDense(dense, [dim, dim, dim], {
    gridName: `synthetic_smoke_${String(frameIndex).padStart(4, "0")}`,
    background: 0,
    gridClass: "FogVolume",
  });
}

/** World/index extent of the fixed animation domain: [0, dim) on every axis (voxelSize 1). */
export function domainExtent(dim = FRAME_DIM): number {
  return dim;
}
