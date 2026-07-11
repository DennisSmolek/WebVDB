// Parses the vendored PNanoVDB.h into structured JSON: integer #defines,
// grid-type ids, and the per-grid-type stride/offset constants table that
// upstream validates with pnanovdb_validate_strides.h. Phase 2's WGSL
// baked-const codegen consumes the JSON so shader constants are provably
// in sync with the C header (docs/PLAN.md Phase 0/2).

/** Integer #defines: `#define PNANOVDB_X 123` / `0x...UL`. Hex values stay strings (may exceed 2^53). */
function parseDefines(text) {
  const defines = {};
  const re = /^#define\s+(PNANOVDB_[A-Z0-9_]+)\s+(0x[0-9a-fA-F]+|\d+)(?:u|l|ul|UL|U|L)*\s*(?:\/\/.*)?$/gm;
  for (const m of text.matchAll(re)) {
    const [, name, value] = m;
    defines[name] = value.startsWith("0x") ? value : Number(value);
  }
  return defines;
}

/** Grid-type ids from PNANOVDB_GRID_TYPE_<NAME> defines (CAP excluded — it's a capacity, not a type). */
function parseGridTypes(defines) {
  const gridTypes = {};
  for (const [name, value] of Object.entries(defines)) {
    const m = /^PNANOVDB_GRID_TYPE_([A-Z0-9_]+)$/.exec(name);
    if (m && m[1] !== "CAP" && typeof value === "number") {
      gridTypes[m[1]] = value;
    }
  }
  return gridTypes;
}

/** Field names of pnanovdb_grid_type_constants_t, in declaration order. */
function parseConstantsFields(text) {
  const structMatch = text.match(
    /struct pnanovdb_grid_type_constants_t\s*\{([\s\S]*?)\};/,
  );
  if (!structMatch) throw new Error("pnanovdb_grid_type_constants_t struct not found");
  return [...structMatch[1].matchAll(/pnanovdb_uint32_t\s+(\w+)\s*;/g)].map((m) => m[1]);
}

/** Rows of the pnanovdb_grid_type_constants[] initializer, as arrays of ints. */
function parseConstantsRows(text) {
  const tableMatch = text.match(
    /pnanovdb_grid_type_constants\[PNANOVDB_GRID_TYPE_CAP\]\s*=\s*\{([\s\S]*?)\n\};/,
  );
  if (!tableMatch) throw new Error("pnanovdb_grid_type_constants table not found");
  const rows = [];
  for (const m of tableMatch[1].matchAll(/\{([^{}]*)\}/g)) {
    const values = m[1]
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map(Number);
    if (values.some(Number.isNaN)) {
      throw new Error(`non-numeric entry in constants row: {${m[1]}}`);
    }
    rows.push(values);
  }
  return rows;
}

/** Single-line auxiliary arrays like pnanovdb_grid_type_value_strides_bits[...] = { ... }; */
function parseAuxArrays(text) {
  const aux = {};
  const re = /pnanovdb_(grid_type_[a-z_]+)\[PNANOVDB_GRID_TYPE_CAP\]\s*=\s*\{([^}]*)\}\s*;/g;
  for (const m of text.matchAll(re)) {
    const [, name, body] = m;
    if (name === "grid_type_constants") continue; // handled structurally above
    aux[name] = body
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => !Number.isNaN(n));
  }
  return aux;
}

/**
 * @param {string} headerText — contents of PNanoVDB.h
 * @param {object} [meta] — provenance (upstream commit, sha256, …), copied into the output
 */
export function extractStrideTables(headerText, meta = {}) {
  const defines = parseDefines(headerText);
  const gridTypes = parseGridTypes(defines);
  const fields = parseConstantsFields(headerText);
  const rows = parseConstantsRows(headerText);

  for (const [i, row] of rows.entries()) {
    if (row.length !== fields.length) {
      throw new Error(
        `constants row ${i} has ${row.length} values, struct has ${fields.length} fields`,
      );
    }
  }

  // Key rows by grid-type name where one exists (retired/padding ids keep a numeric key).
  const idToName = Object.fromEntries(
    Object.entries(gridTypes).map(([name, id]) => [id, name]),
  );
  const gridTypeConstants = {};
  for (const [i, row] of rows.entries()) {
    const key = idToName[i] ?? `TYPE_${i}`;
    gridTypeConstants[key] = Object.fromEntries(fields.map((f, j) => [f, row[j]]));
  }

  const version = [
    defines["PNANOVDB_MAJOR_VERSION_NUMBER"],
    defines["PNANOVDB_MINOR_VERSION_NUMBER"],
    defines["PNANOVDB_PATCH_VERSION_NUMBER"],
  ].join(".");

  return {
    $meta: { ...meta, abi: version, generatedBy: "scripts/extract-stride-tables.mjs" },
    defines,
    gridTypes,
    constantsFields: fields,
    gridTypeConstants,
    auxArrays: parseAuxArrays(headerText),
  };
}
