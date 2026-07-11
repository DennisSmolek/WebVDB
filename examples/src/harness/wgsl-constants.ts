/**
 * Runtime extraction of PNanoVDB layout constants directly from the fetched
 * vendored WGSL source text (`nanovdb-wgsl/pnanovdb.wgsl?raw`), instead of
 * re-reading `vendor/stride-tables.json` through
 * `packages/nanovdb-wgsl/src/cpu/stride-tables.ts` — that module loads the
 * JSON via `node:fs.readFileSync` at import time, which has no browser
 * equivalent, so it cannot run on this page (see the harness README note in
 * main.ts). Parsing the exact same WGSL text the GPU shader compiles from
 * has a nice side effect beyond working around that: it makes this CPU
 * reference's constants provably in sync with whatever the vendored module
 * actually says at runtime, rather than a second, independently-maintained
 * copy of the numbers that could silently drift.
 *
 * Only the `const NAME: u32 = N u;` scalar defines and the
 * `pnanovdb_grid_type_constants` table are needed — see
 * `packages/nanovdb-wgsl/vendor/pnanovdb.wgsl` for both.
 */

/** Per-grid-type layout constants (`pnanovdb_grid_type_constants_t` in the WGSL). */
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

/** Structural (non-per-grid-type) byte offsets the CPU descent needs. */
export interface WgslScalarConstants {
  PNANOVDB_GRID_SIZE: number;
  PNANOVDB_GRID_OFF_GRID_TYPE: number;
  PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT: number;
  PNANOVDB_ROOT_OFF_TABLE_SIZE: number;
  PNANOVDB_ROOT_TILE_OFF_KEY: number;
  PNANOVDB_ROOT_TILE_OFF_CHILD: number;
  PNANOVDB_ROOT_TILE_OFF_STATE: number;
  PNANOVDB_UPPER_OFF_VALUE_MASK: number;
  PNANOVDB_UPPER_OFF_CHILD_MASK: number;
  PNANOVDB_LOWER_OFF_VALUE_MASK: number;
  PNANOVDB_LOWER_OFF_CHILD_MASK: number;
  PNANOVDB_LEAF_OFF_VALUE_MASK: number;
  PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS: number;
  PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM: number;
  PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM: number;
  PNANOVDB_GRID_TYPE_FLOAT: number;
  PNANOVDB_GRID_TYPE_FP8: number;
  PNANOVDB_GRID_TYPE_FPN: number;
}

export interface ParsedWgslConstants {
  scalars: WgslScalarConstants;
  /** `pnanovdb_grid_type_constants[]`, indexed by numeric grid type id. */
  gridTypeConstants: GridTypeConstants[];
}

function extractU32(src: string, name: string): number {
  const re = new RegExp(`const\\s+${name}\\s*:\\s*u32\\s*=\\s*(\\d+)\\s*u\\s*;`);
  const m = re.exec(src);
  if (!m) {
    throw new Error(
      `wgsl-constants: could not find "const ${name}: u32 = ...u;" in the vendored pnanovdb.wgsl text`,
    );
  }
  return Number(m[1]);
}

// Field order of `struct pnanovdb_grid_type_constants_t` in pnanovdb.wgsl —
// must match the WGSL struct declaration exactly (this is what makes each
// `pnanovdb_grid_type_constants_t(28u, 32u, ...)` row positionally decodable).
const GRID_TYPE_CONSTANTS_FIELDS = [
  "root_off_background",
  "root_off_min",
  "root_off_max",
  "root_off_ave",
  "root_off_stddev",
  "root_size",
  "value_stride_bits",
  "table_stride",
  "root_tile_off_value",
  "root_tile_size",
  "upper_off_min",
  "upper_off_max",
  "upper_off_ave",
  "upper_off_stddev",
  "upper_off_table",
  "upper_size",
  "lower_off_min",
  "lower_off_max",
  "lower_off_ave",
  "lower_off_stddev",
  "lower_off_table",
  "lower_size",
  "leaf_off_min",
  "leaf_off_max",
  "leaf_off_ave",
  "leaf_off_stddev",
  "leaf_off_table",
  "leaf_size",
] as const satisfies readonly (keyof GridTypeConstants)[];

const ROW_RE = /pnanovdb_grid_type_constants_t\(([^)]*)\)/g;

function parseGridTypeConstantsTable(src: string): GridTypeConstants[] {
  const rows: GridTypeConstants[] = [];
  for (const m of src.matchAll(ROW_RE)) {
    const raw = m[1];
    if (!raw) continue;
    const nums = raw.split(",").map((s) => Number.parseInt(s.trim().replace(/u$/, ""), 10));
    if (nums.length !== GRID_TYPE_CONSTANTS_FIELDS.length) {
      throw new Error(
        `wgsl-constants: a pnanovdb_grid_type_constants_t(...) row has ${nums.length} fields, ` +
          `expected ${GRID_TYPE_CONSTANTS_FIELDS.length} — struct layout drifted from this parser`,
      );
    }
    const rec = {} as GridTypeConstants;
    GRID_TYPE_CONSTANTS_FIELDS.forEach((field, i) => {
      rec[field] = nums[i]!;
    });
    rows.push(rec);
  }
  if (rows.length === 0) {
    throw new Error(
      "wgsl-constants: found no pnanovdb_grid_type_constants_t(...) rows in the vendored pnanovdb.wgsl text",
    );
  }
  return rows;
}

/** Parses every constant the CPU reference (`cpu-reference.ts`) needs out of the vendored WGSL source text. */
export function parseWgslConstants(src: string): ParsedWgslConstants {
  const scalars: WgslScalarConstants = {
    PNANOVDB_GRID_SIZE: extractU32(src, "PNANOVDB_GRID_SIZE"),
    PNANOVDB_GRID_OFF_GRID_TYPE: extractU32(src, "PNANOVDB_GRID_OFF_GRID_TYPE"),
    PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT: extractU32(src, "PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT"),
    PNANOVDB_ROOT_OFF_TABLE_SIZE: extractU32(src, "PNANOVDB_ROOT_OFF_TABLE_SIZE"),
    PNANOVDB_ROOT_TILE_OFF_KEY: extractU32(src, "PNANOVDB_ROOT_TILE_OFF_KEY"),
    PNANOVDB_ROOT_TILE_OFF_CHILD: extractU32(src, "PNANOVDB_ROOT_TILE_OFF_CHILD"),
    PNANOVDB_ROOT_TILE_OFF_STATE: extractU32(src, "PNANOVDB_ROOT_TILE_OFF_STATE"),
    PNANOVDB_UPPER_OFF_VALUE_MASK: extractU32(src, "PNANOVDB_UPPER_OFF_VALUE_MASK"),
    PNANOVDB_UPPER_OFF_CHILD_MASK: extractU32(src, "PNANOVDB_UPPER_OFF_CHILD_MASK"),
    PNANOVDB_LOWER_OFF_VALUE_MASK: extractU32(src, "PNANOVDB_LOWER_OFF_VALUE_MASK"),
    PNANOVDB_LOWER_OFF_CHILD_MASK: extractU32(src, "PNANOVDB_LOWER_OFF_CHILD_MASK"),
    PNANOVDB_LEAF_OFF_VALUE_MASK: extractU32(src, "PNANOVDB_LEAF_OFF_VALUE_MASK"),
    PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS: extractU32(
      src,
      "PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS",
    ),
    PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM: extractU32(src, "PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM"),
    PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM: extractU32(src, "PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM"),
    PNANOVDB_GRID_TYPE_FLOAT: extractU32(src, "PNANOVDB_GRID_TYPE_FLOAT"),
    PNANOVDB_GRID_TYPE_FP8: extractU32(src, "PNANOVDB_GRID_TYPE_FP8"),
    PNANOVDB_GRID_TYPE_FPN: extractU32(src, "PNANOVDB_GRID_TYPE_FPN"),
  };

  const gridTypeConstants = parseGridTypeConstantsTable(src);

  return { scalars, gridTypeConstants };
}
