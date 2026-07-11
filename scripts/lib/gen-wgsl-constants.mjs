export function generateWgslConstants(tables) {
  const lines = [];

  // Header
  const abi = tables.$meta.abi;
  lines.push("// GENERATED FILE — do not edit by hand.");
  lines.push(`// Source: packages/nanovdb-wgsl/vendor/stride-tables.json (NanoVDB ABI ${abi})`);
  lines.push("// Regenerate: node scripts/gen-wgsl-constants.mjs");
  lines.push("// Guarded by: packages/nanovdb-wgsl/test/wgsl-constants.test.ts");
  lines.push("");

  // Defines section
  lines.push("// ---- defines ----");
  for (const [name, value] of Object.entries(tables.defines)) {
    // Skip string values (64-bit hex)
    if (typeof value === "number") {
      lines.push(`const ${name}: u32 = ${value}u;`);
    }
  }

  // Grid type constants section
  lines.push("");
  lines.push("// ---- per-grid-type constants: FLOAT, FP8, FPN ----");

  const gridTypesToEmit = ["FLOAT", "FP8", "FPN"];
  for (const typeName of gridTypesToEmit) {
    const typeId = tables.gridTypes[typeName];
    lines.push(`// ${typeName} (grid type ${typeId})`);

    const constants = tables.gridTypeConstants[typeName];
    for (const fieldName of tables.constantsFields) {
      const value = constants[fieldName];
      const constName = `PNANOVDB_${typeName}_${fieldName.toUpperCase()}`;
      lines.push(`const ${constName}: u32 = ${value}u;`);
    }
  }

  // Aux arrays section
  lines.push("");
  lines.push("// ---- aux stride arrays (indexed by grid type id) ----");

  for (const [key, arr] of Object.entries(tables.auxArrays)) {
    const fnName = `pnanovdb_${key}`;
    lines.push(`fn ${fnName}(grid_type: u32) -> u32 {`);
    lines.push("    switch grid_type {");

    for (let i = 0; i < arr.length; i++) {
      lines.push(`        case ${i}u: { return ${arr[i]}u; }`);
    }

    lines.push("        default: { return 0u; }");
    lines.push("    }");
    lines.push("}");
  }

  lines.push("");
  return lines.join("\n");
}
