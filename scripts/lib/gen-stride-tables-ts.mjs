// Generates a browser-safe TS mirror of vendor/stride-tables.json — the
// `packages/nanovdb-wgsl/src/cpu/stride-tables.generated.ts` module.
//
// `stride-tables.json` itself can only be loaded with `node:fs`
// (`stride-tables.ts` used to do exactly that), which has no browser
// equivalent, so pages could never import `nanovdb-wgsl`'s pure-TS CPU
// reference (`read-value.ts`/`sample-trilinear.ts`). Baking the same data
// into an ordinary ES module (a `const ... as const` object literal) makes it
// importable anywhere ESM works, no fetch/fs required — mirroring the
// existing `pnanovdb-constants.generated.wgsl` bake for the GPU side.

export function generateStrideTablesTs(tables) {
  const abi = tables.$meta.abi;
  const json = JSON.stringify(tables, null, 2);
  return `// GENERATED FILE — do not edit by hand.
// Source: packages/nanovdb-wgsl/vendor/stride-tables.json (NanoVDB ABI ${abi})
// Regenerate: node scripts/extract-stride-tables.mjs
// Guarded by: packages/nanovdb-wgsl/test/stride-tables-generated.test.ts

/**
 * Typed, browser-safe mirror of \`../../vendor/stride-tables.json\` — see
 * \`./stride-tables.ts\` for the accessor surface (defineNumber/defineBigInt/
 * gridTypeConstantsFor/gridTypeConstantsForId/etc.) built on top of this data.
 * Unlike the JSON file, importing this module never touches \`node:fs\`, so it
 * works in bundler/browser contexts (Vite pages, etc.) as well as Node.
 */
export const strideTables = ${json} as const;
`;
}
