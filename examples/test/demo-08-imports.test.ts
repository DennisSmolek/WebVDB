import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// K1.7: demo 08 is the engine-free reference, so its whole module graph must
// be free of three.js. Walk the graph from the entry the page loads, resolving
// relative imports and the workspace aliases (the ones vite.config.ts uses),
// and collect every bare package it reaches.

const repo = fileURLToPath(new URL("../../", import.meta.url));
const entry = path.join(repo, "examples/src/demos/08-webgpu-smoke/main.ts");

const ALIASES: Record<string, string> = {
  "@pmndrs/vdb": "packages/vdb/src/core/index.ts",
  "@pmndrs/vdb/tools": "packages/vdb/src/tools/index.ts",
  "@pmndrs/vdb/three": "packages/vdb/src/three/index.ts",
};

/** Static and dynamic import/export specifiers (same shapes as packages/vdb/test/boundaries.test.ts). */
function specifiers(source: string): string[] {
  const noComments = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[\w*${}\s,]+?\s+from\s+)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["'`]([^"'`$]+)["'`]\s*\)/g,
    /\brequire\s*\(\s*["'`]([^"'`$]+)["'`]\s*\)/g,
  ];
  const found: string[] = [];
  for (const re of patterns) for (const m of noComments.matchAll(re)) found.push(m[1]!.split("?")[0]!);
  return found;
}

/** Resolve a relative specifier the way TS "bundler" resolution does for this repo (`./x.js` -> `./x.ts`). */
function resolveRelative(from: string, spec: string): string {
  const base = path.resolve(path.dirname(from), spec);
  const candidates = [base, base.replace(/\.js$/, ".ts"), `${base}.ts`, path.join(base, "index.ts")];
  // `?url` assets (a committed binary the page fetches) are leaves of the graph
  const hit = candidates.find((c) => existsSync(c) && /\.(ts|js|mjs|wgsl|wvdb|nvdb)$/.test(c));
  if (!hit) throw new Error(`cannot resolve ${spec} from ${path.relative(repo, from)}`);
  return hit;
}

function moduleGraph(start: string): { files: Set<string>; packages: Set<string> } {
  const files = new Set<string>();
  const packages = new Set<string>();
  const queue = [start];
  while (queue.length > 0) {
    const file = queue.pop()!;
    if (files.has(file)) continue;
    files.add(file);
    if (!/\.(ts|js|mjs)$/.test(file)) continue;
    for (const spec of specifiers(readFileSync(file, "utf8"))) {
      if (spec.startsWith(".")) queue.push(resolveRelative(file, spec));
      else if (ALIASES[spec]) {
        packages.add(spec);
        queue.push(path.join(repo, ALIASES[spec]!));
      } else packages.add(spec);
    }
  }
  return { files, packages };
}

describe("demo 08 import graph (K1.7)", () => {
  const { files, packages } = moduleGraph(entry);
  const rel = [...files].map((f) => path.relative(repo, f));

  it("walks the core and tools sources", () => {
    expect(rel).toContain("packages/vdb/src/core/pool/pool.ts");
    expect(rel).toContain("packages/vdb/src/core/wgsl/generate.ts");
    expect(rel).toContain("packages/vdb/src/tools/vdb/index.ts");
    expect(files.size).toBeGreaterThan(20);
  });

  it("reaches no three.js and no three adapter", () => {
    expect([...packages].filter((p) => p === "three" || p.startsWith("three/") || p === "@pmndrs/vdb/three")).toEqual([]);
    expect(rel.filter((f) => f.startsWith("packages/vdb/src/three/"))).toEqual([]);
  });

  it("depends only on the core, tools and their one runtime dependency", () => {
    expect([...packages].sort()).toEqual(["@pmndrs/vdb", "@pmndrs/vdb/tools", "fflate"]);
  });
});

// K6.9: demo 17 renders volumes with the core frame renderer and no three.js; its module graph (the core's internal
// entry point, the tools, the engine-free half of the gallery shell) must stay free of three.
describe("demo 17 import graph (K6.9)", () => {
  const { files, packages } = moduleGraph(path.join(repo, "examples/src/demos/17-raw-webgpu/main.ts"));
  const rel = [...files].map((f) => path.relative(repo, f));

  it("walks the core frame renderer, the level-set pass and the gallery UI", () => {
    expect(rel).toContain("packages/vdb/src/core/frame/renderer.ts");
    expect(rel).toContain("packages/vdb/src/core/surface/level-set-pass.ts");
    expect(rel).toContain("packages/vdb/src/core/source/sparse.ts");
    expect(rel).toContain("examples/src/demos/showcase-ui.ts");
  });

  it("reaches no three.js, no three adapter and not the three half of the gallery shell", () => {
    expect([...packages].filter((p) => p === "three" || p.startsWith("three/") || p === "@pmndrs/vdb/three")).toEqual([]);
    expect(rel.filter((f) => f.startsWith("packages/vdb/src/three/"))).toEqual([]);
    expect(rel).not.toContain("examples/src/demos/showcase-shell.ts");
  });
});
