/**
 * Typed, synchronous access to `vendor/stride-tables.json` — the extracted
 * `PNANOVDB_*` `#define`s and per-grid-type layout constants from
 * `vendor/upstream/PNanoVDB.h`. The CPU reference traversal (`read-value.ts`)
 * reads every byte offset from here rather than hardcoding them, so it stays
 * in lockstep with whatever ABI the vendored header is pinned to.
 *
 * Only bit-math constants that are structural (not per-grid-type layout —
 * e.g. the 8/16/32-voxel dim masks used by the coord-to-offset functions)
 * are literal constants in read-value.ts; everything else comes from here.
 */
import { readFileSync } from "node:fs";

const strideTablesUrl = new URL("../../vendor/stride-tables.json", import.meta.url);

/** Per-grid-type layout constants (see PNanoVDB.h's `pnanovdb_grid_type_constants`). */
export interface GridTypeConstants {
  root_off_background: number;
  root_off_min: number;
  root_off_max: number;
  root_off_ave: number;
  root_off_stddev: number;
  root_size: number;
  value_stride_bits: number;
  table_stride: number;
  root_tile_off_value: number;
  root_tile_size: number;
  upper_off_min: number;
  upper_off_max: number;
  upper_off_ave: number;
  upper_off_stddev: number;
  upper_off_table: number;
  upper_size: number;
  lower_off_min: number;
  lower_off_max: number;
  lower_off_ave: number;
  lower_off_stddev: number;
  lower_off_table: number;
  lower_size: number;
  leaf_off_min: number;
  leaf_off_max: number;
  leaf_off_ave: number;
  leaf_off_stddev: number;
  leaf_off_table: number;
  leaf_size: number;
}

interface StrideTables {
  defines: Record<string, number | string>;
  gridTypes: Record<string, number>;
  gridTypeConstants: Record<string, GridTypeConstants>;
}

const strideTables = JSON.parse(readFileSync(strideTablesUrl, "utf8")) as StrideTables;

/** Raw `#define NAME value` map (numbers, or hex strings like the magic numbers). */
export const defines = strideTables.defines;

/** `PNANOVDB_GRID_TYPE_*` name -> numeric id (e.g. `FLOAT` -> 1). */
export const gridTypeIds = strideTables.gridTypes;

/** Grid type name -> its full layout constants block. */
export const gridTypeConstants = strideTables.gridTypeConstants;

/** Reads a numeric `#define` from stride-tables.json's `defines` map. */
export function defineNumber(name: string): number {
  const value = defines[name];
  if (typeof value !== "number") {
    throw new Error(`stride-tables.json: define "${name}" is not a number`);
  }
  return value;
}

/**
 * Reads a `#define` as a BigInt — used for the 64-bit magic number
 * constants, which are recorded as hex strings (e.g. `"0x314244566f6e614e"`)
 * because they overflow a JS `number`.
 */
export function defineBigInt(name: string): bigint {
  const value = defines[name];
  if (value === undefined) {
    throw new Error(`stride-tables.json: define "${name}" is missing`);
  }
  return BigInt(value);
}

/** Layout constants for one of the three grid types this module supports. */
export function gridTypeConstantsFor(gridTypeName: "FLOAT" | "FP8" | "FPN"): GridTypeConstants {
  const constants = gridTypeConstants[gridTypeName];
  if (!constants) {
    throw new Error(`stride-tables.json: no gridTypeConstants for "${gridTypeName}"`);
  }
  return constants;
}
