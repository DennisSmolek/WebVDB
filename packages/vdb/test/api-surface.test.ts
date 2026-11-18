/**
 * The public API surface (D32). The package's subpaths freeze at the first npm publish, so:
 *
 * 1. Each subpath's exact export list (values and types) is a committed snapshot under
 *    `test/api-surface/`. Adding, removing or renaming an export fails here until the snapshot is
 *    updated on purpose (`pnpm vitest run packages/vdb/test/api-surface.test.ts -u` at the root, then review
 *    the diff with the change).
 * 2. Every export follows the D32 naming rule (checked on the runtime values and the type names).
 * 3. A name exported from two subpaths is one declaration (one `VDBAtlasFilter`, one `VDBVec3`).
 *
 * Internals (`src/{core,tools,three}/internal.ts`) are not package exports and are not snapshotted.
 */
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const ts = require("typescript") as typeof import("typescript");

const SUBPATHS = {
  ".": { dir: "core", load: () => import("../src/core/index.js") },
  "./tools": { dir: "tools", load: () => import("../src/tools/index.js") },
  "./three": { dir: "three", load: () => import("../src/three/index.js") },
} as const;
type Subpath = keyof typeof SUBPATHS;

interface Entry {
  name: string;
  kind: "value" | "type";
  symbol: import("typescript").Symbol;
}

/** Every export of each public index, from the TypeScript checker (values and types). */
function surface(): Record<Subpath, Entry[]> {
  const files = Object.values(SUBPATHS).map((s) => fileURLToPath(new URL(`../src/${s.dir}/index.ts`, import.meta.url)));
  const program = ts.createProgram(files, {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    noEmit: true,
    skipLibCheck: true,
    types: [],
  });
  const checker = program.getTypeChecker();
  const out = {} as Record<Subpath, Entry[]>;
  (Object.keys(SUBPATHS) as Subpath[]).forEach((sub, i) => {
    const sf = program.getSourceFile(files[i]!)!;
    out[sub] = checker
      .getExportsOfModule(checker.getSymbolAtLocation(sf)!)
      .map((s) => {
        const symbol = s.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(s) : s;
        return { name: s.name, kind: symbol.flags & ts.SymbolFlags.Value ? ("value" as const) : ("type" as const), symbol };
      })
      .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  });
  return out;
}

const api = surface();

/** D32: the naming rule, as patterns. */
const D32 = {
  /** Classes, enum-like objects and types: `VDB*` (library) or a format prefix in PascalCase acronym style. */
  pascal: /^(VDB|Vdb|Nvdb|Wvdb)[A-Z][A-Za-z0-9]*$/,
  /** Constants. */
  constant: /^(VDB|NVDB|WVDB)_[A-Z0-9_]+$/,
  /** Functions and other values: `vdb*` (library), `nvdb*` / `wvdb*` (one format). */
  camel: /^(vdb|nvdb|wvdb)[A-Z0-9][A-Za-z0-9]*$/,
  /** The format-verb form: a verb (and qualifier), then the format (`parseVdb`, `writeNvdb`, `readWvdbHeader`). */
  formatVerb: /^[a-z]+(?:[A-Z][a-z]+)*(?:Vdb|Nvdb|Wvdb)(?:[A-Z][A-Za-z0-9]*)?$/,
  /** Its option and result types (`ReadWvdbOptions`, `BuildFromVdbOptions`, `WriteWvdbResult`). */
  formatVerbType: /^[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:Vdb|Nvdb|Wvdb)[A-Z][A-Za-z0-9]*$/,
  /** Spellings D32 retired: mixed-case format names and a person's name. */
  retired: /NanoVDB|NanoVdb|nanoVdb|mjurczyk/i,
};
/** The device helpers keep the names the other pmndrs tools use. */
const PMNDRS_HELPERS = new Set(["getDevice", "getGPUTexture"]);

describe("public API surface (D32)", () => {
  for (const sub of Object.keys(SUBPATHS) as Subpath[]) {
    it(`${sub}: the export list matches the committed snapshot`, async () => {
      const text = api[sub].map((e) => `${e.kind} ${e.name}`).join("\n") + "\n";
      await expect(text).toMatchFileSnapshot(`./api-surface/${SUBPATHS[sub].dir}.txt`);
    });

    it(`${sub}: the runtime exports are exactly the checker's values`, async () => {
      const mod = (await SUBPATHS[sub].load()) as Record<string, unknown>;
      const values = api[sub].filter((e) => e.kind === "value").map((e) => e.name);
      expect(Object.keys(mod).sort()).toEqual(values);
    });

    it(`${sub}: every runtime export follows D32`, async () => {
      const mod = (await SUBPATHS[sub].load()) as Record<string, unknown>;
      const bad: string[] = [];
      for (const [name, value] of Object.entries(mod)) {
        if (D32.retired.test(name)) bad.push(`${name}: retired spelling`);
        else if (PMNDRS_HELPERS.has(name) && sub === "./three") continue;
        else if (typeof value === "function" && /^[A-Z]/.test(name)) {
          if (!D32.pascal.test(name)) bad.push(`${name}: class`);
        } else if (typeof value === "function") {
          if (!D32.camel.test(name) && !D32.formatVerb.test(name)) bad.push(`${name}: function`);
        } else if (/^[A-Z0-9_]+$/.test(name)) {
          if (!D32.constant.test(name)) bad.push(`${name}: constant`);
        } else if (!D32.pascal.test(name) && !D32.camel.test(name)) {
          bad.push(`${name}: value`);
        }
      }
      expect(bad).toEqual([]);
    });

    it(`${sub}: every exported type follows D32`, () => {
      const bad = api[sub]
        .filter((e) => e.kind === "type")
        .map((e) => e.name)
        .filter((n) => D32.retired.test(n) || !(D32.pascal.test(n) || D32.formatVerbType.test(n)));
      expect(bad).toEqual([]);
    });
  }

  it("a name exported from two subpaths is one declaration", () => {
    const seen = new Map<string, { sub: Subpath; symbol: import("typescript").Symbol }>();
    const clashes: string[] = [];
    for (const sub of Object.keys(SUBPATHS) as Subpath[]) {
      for (const e of api[sub]) {
        const first = seen.get(e.name);
        if (first && first.symbol !== e.symbol) clashes.push(`${e.name} (${first.sub} and ${sub})`);
        if (!first) seen.set(e.name, { sub, symbol: e.symbol });
      }
    }
    expect(clashes).toEqual([]);
  });

  it("the internals re-export the public API (tests import one module)", async () => {
    for (const sub of Object.keys(SUBPATHS) as Subpath[]) {
      const pub = (await SUBPATHS[sub].load()) as Record<string, unknown>;
      const internal = (await import(`../src/${SUBPATHS[sub].dir}/internal.ts`)) as Record<string, unknown>;
      for (const [name, value] of Object.entries(pub)) expect(internal[name], `${sub} ${name}`).toBe(value);
    }
  });
});
