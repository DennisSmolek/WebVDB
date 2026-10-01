/**
 * Spike fixture corpus (node-only: reads from disk).
 *
 * - `fixtures/primitives/{sphere,torus,box}_fog_float.nvdb`: native-baked
 *   FogVolume FLOAT grids (the `pnpm fixtures:bake` output). The matching
 *   `_fog_fp8.nvdb` files are the native Fp8 reference.
 * - `fixtures/vdb-samples/*.vdb`: openvdb.org samples, via our TS pipeline
 *   `parseVdb` -> `buildFromVdb` -> `writeNvdb`. The `.nvdb` bytes are what
 *   both converters (TS and native) consume, so they see identical input.
 *   Level sets are built with class "Unknown" (our builder writes only
 *   FogVolume/Unknown). PicoVDB treats every non-fog class as SDF, so this
 *   selects the same converter path a native LevelSet .nvdb would.
 */

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineNumber } from "../../../packages/nanovdb-wgsl/src/cpu/stride-tables.js";
import { NanoVDBFile } from "../../../packages/nanovdb-wgsl/src/nvdb-file.js";
import { buildFromVdb, parseVdb, quantize, writeNvdb } from "../../../packages/vdb-web-tools/src/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(here, "../../..");
export const PRIMITIVES_DIR = path.join(REPO_ROOT, "fixtures/primitives");
export const SAMPLES_DIR = path.join(REPO_ROOT, "fixtures/vdb-samples");

export const PRIMITIVES = ["sphere", "torus", "box"] as const;
export const SAMPLES = ["sphere", "cube", "smoke", "utahteapot"] as const;

export interface Fixture {
  /** e.g. "primitives/sphere_fog" or "vdb-samples/smoke". */
  id: string;
  kind: "fog" | "sdf";
  /** The FLOAT `.nvdb` file bytes both converters consume. */
  nvdbFloat: Uint8Array<ArrayBuffer>;
  /** Grid 0 image (GridData at word 0), FLOAT. */
  image: Uint32Array;
  gridClassId: number;
  voxelSize: number;
  indexBBox: { min: [number, number, number]; max: [number, number, number] };
  /** Native-baked Fp8 `.nvdb` bytes, when the corpus has one (primitives only). */
  nativeFp8?: Uint8Array<ArrayBuffer>;
}

function readBytes(p: string): Uint8Array<ArrayBuffer> {
  const b = readFileSync(p);
  const out = new Uint8Array(new ArrayBuffer(b.byteLength));
  out.set(b);
  return out;
}

export function primitivesAvailable(): boolean {
  return PRIMITIVES.every((n) => existsSync(path.join(PRIMITIVES_DIR, `${n}_fog_float.nvdb`)));
}

export function samplesAvailable(): boolean {
  return SAMPLES.every((n) => existsSync(path.join(SAMPLES_DIR, `${n}.vdb`)));
}

function fromNvdb(id: string, kind: "fog" | "sdf", bytes: Uint8Array<ArrayBuffer>, fp8?: Uint8Array<ArrayBuffer>): Fixture {
  const file = NanoVDBFile.fromArrayBuffer(bytes.buffer);
  const meta = file.grids[0]!;
  const image = file.gridImage(0);
  const gridClassId = image[defineNumber("PNANOVDB_GRID_OFF_GRID_CLASS") >>> 2]!;
  return {
    id, kind, nvdbFloat: bytes, image, gridClassId,
    voxelSize: meta.voxelSize[0], indexBBox: meta.indexBBox, nativeFp8: fp8,
  };
}

export function loadPrimitive(name: (typeof PRIMITIVES)[number]): Fixture {
  const bytes = readBytes(path.join(PRIMITIVES_DIR, `${name}_fog_float.nvdb`));
  const fp8 = readBytes(path.join(PRIMITIVES_DIR, `${name}_fog_fp8.nvdb`));
  return fromNvdb(`primitives/${name}_fog`, "fog", bytes, fp8);
}

export function loadSample(name: (typeof SAMPLES)[number]): Fixture {
  const vdb = readBytes(path.join(SAMPLES_DIR, `${name}.vdb`));
  const grid = parseVdb(vdb.buffer).grids[0]!;
  const isFog = String(grid.metadata["class"] ?? "") === "fog volume";
  const image = buildFromVdb(grid, { gridClass: isFog ? "FogVolume" : "Unknown" });
  const nvdb = new Uint8Array(writeNvdb([image]));
  return fromNvdb(`vdb-samples/${name}`, isFog ? "fog" : "sdf", nvdb);
}

export function loadAll(): Fixture[] {
  const out: Fixture[] = [];
  if (primitivesAvailable()) for (const n of PRIMITIVES) out.push(loadPrimitive(n));
  if (samplesAvailable()) for (const n of SAMPLES) out.push(loadSample(n));
  return out;
}

/** Our TS Fp8 re-encode of the fixture's FLOAT grid (vdb-web-tools `quantize`), as an .nvdb file. */
export function tsFp8Nvdb(f: Fixture): Uint8Array<ArrayBuffer> {
  return new Uint8Array(writeNvdb([quantize(f.image, "fp8")]));
}

/** File-safe stem for a fixture id. */
export function stem(f: Fixture): string {
  return f.id.replace("/", "__");
}
