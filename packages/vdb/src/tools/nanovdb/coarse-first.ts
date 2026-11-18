/**
 * coarse-first.ts — `bakeCoarseFirst` (K4.16): the K4.5 option (a) bake. A
 * small preview grid (or a chain, coarsest first) and then the full grid, in
 * one multi-grid `.nvdb` (docs/design/K4-5-coarse-first.md):
 *
 *     [ <name>.lod48 , <name> ]
 *
 * Previews are `resampleGrid(source, { resolution, filter, quantize })` over
 * the source's active cell box, so they draw where the full grid draws.
 * Readers tell them apart by name: a preview is `<name>.lod<resolution>`
 * (`parseNvdbLodName`), the full grid keeps the source name, so a native
 * reader that selects by name (`io::readGrid(file, "density")`) gets the
 * full grid. A reader that takes grid 0 gets the coarsest preview (K4.17).
 */

import { nvdbLodName } from "../nvdb-stream.js";
import type { VdbFile, VdbGrid } from "../vdb/types.js";
import { GRID_NAME_MAX, GRID_OFF_GRID_NAME } from "./bytes.js";
import { buildFromVdb } from "./build-from-vdb.js";
import { collapseGrid, quantize, type QuantizeMode } from "./quantize.js";
import { resampleGrid, type ResampleFilter } from "./resample.js";
import { writeNvdb } from "./write-nvdb.js";

export interface BakeCoarseFirstOptions {
  /**
   * Preview resolution(s): voxels on the longest side of the source's active
   * box (`resampleGrid`'s `resolution`). Several give a chain, written
   * coarsest first. Default 48 (the K4.5 recommendation).
   */
  previewResolution?: number | number[];
  /** Preview filter. Default `'box'` (conserves mass; K4.19 decides between box and point). */
  filter?: ResampleFilter;
  /** Preview encoding. Default `'fpn'`. */
  quantize?: QuantizeMode | "none";
  /** FpN tolerance for the previews (and a quantized full grid). Default `quantize`'s (0.01). */
  tolerance?: number;
  /**
   * Full-grid encoding. Default: the image as given for a NanoVDB source, and
   * `quantize`'s value for a parsed `.vdb` source (built with `buildFromVdb`
   * first). Quantizing needs a FLOAT image.
   */
  fullQuantize?: QuantizeMode | "none";
  /**
   * Per-grid codec. Default `'zip'` (K4.18): the preview is ~12 KiB inside the
   * file whatever the host does; `'none'` relies on HTTP compression (a raw
   * preview is ~300 KiB, the 270 KB upper node).
   */
  codec?: "none" | "zip";
  /** zlib level for `codec: 'zip'`. Default 9. */
  zipLevel?: number;
  /** Which float grid of a `VdbFile` to bake. Default: the first float grid. */
  sourceGridName?: string;
  /**
   * Collapse uniform regions into tiles (K4.6). Default: on for the previews
   * (`resampleGrid`'s default) and off for the full grid (an image keeps its
   * topology; a `.vdb` grid is built as `nanovdb_convert` builds it). A value
   * given applies to every grid.
   */
  collapse?: boolean;
}

function imageName(image: Uint32Array): string {
  const bytes = new Uint8Array(image.buffer, image.byteOffset + GRID_OFF_GRID_NAME, GRID_NAME_MAX);
  const end = bytes.indexOf(0);
  return new TextDecoder().decode(bytes.subarray(0, end < 0 ? bytes.length : end));
}

function pickGrid(file: VdbFile, name: string | undefined): VdbGrid {
  const grid = file.grids.find((g) => g.valueType === "float" && (name === undefined || g.name === name));
  if (!grid) {
    throw new Error(
      `bakeCoarseFirst: no float grid${name === undefined ? "" : ` named "${name}"`} in the file ` +
        `(grids: ${file.grids.map((g) => `${g.name}:${g.valueType}`).join(", ") || "none"})`,
    );
  }
  return grid;
}

/**
 * Bakes a coarse-first `.nvdb`: previews at `previewResolution` (coarsest
 * first), then the full grid, in one segment, as file bytes.
 */
export function bakeCoarseFirst(source: Uint32Array | VdbGrid | VdbFile, options: BakeCoarseFirstOptions = {}): Uint8Array {
  const resolutions = [options.previewResolution ?? 48].flat();
  if (resolutions.length === 0) throw new Error("bakeCoarseFirst: previewResolution is empty");
  for (const r of resolutions) {
    if (!Number.isInteger(r) || r < 1) throw new Error(`bakeCoarseFirst: previewResolution must be positive integers, got ${r}`);
  }
  if (new Set(resolutions).size !== resolutions.length) throw new Error("bakeCoarseFirst: previewResolution has duplicates");
  resolutions.sort((a, b) => a - b);
  const previewQuantize = options.quantize ?? "fpn";

  // The full grid.
  let full: Uint32Array;
  let resampleSource: Uint32Array | VdbGrid;
  if (source instanceof Uint32Array) {
    full = source;
    resampleSource = source;
    const q = options.fullQuantize ?? "none";
    if (q !== "none") full = quantize(source, q, options.tolerance, { collapse: options.collapse === true });
    else if (options.collapse === true) full = collapseGrid(source).image;
  } else {
    const grid = "grids" in source ? pickGrid(source, options.sourceGridName) : source;
    resampleSource = grid;
    full = buildFromVdb(grid, { collapse: options.collapse === true });
    const q = options.fullQuantize ?? previewQuantize;
    if (q !== "none") full = quantize(full, q, options.tolerance);
  }
  const name = imageName(full);

  const previews = resolutions.map(
    (resolution) =>
      resampleGrid(resampleSource, {
        resolution,
        filter: options.filter ?? "box",
        ...(previewQuantize === "none" ? {} : { quantize: previewQuantize }),
        ...(options.tolerance === undefined ? {} : { tolerance: options.tolerance }),
        gridName: nvdbLodName(name, resolution),
        ...(options.collapse === undefined ? {} : { collapse: options.collapse }),
      }).image,
  );

  const file = writeNvdb([...previews, full], {
    codec: options.codec ?? "zip",
    ...(options.zipLevel === undefined ? {} : { zipLevel: options.zipLevel }),
  });
  return new Uint8Array(file);
}
