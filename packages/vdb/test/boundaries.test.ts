import { readdirSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

// K0.3: the repo has no ESLint, so the import boundaries are enforced here.
//   core  : may import tools; must not import `three` / `three/*` or src/three.
//   tools : must not import `three` / `three/*`, src/core or src/three.
//   three : free to import core and tools.

const srcRoot = fileURLToPath(new URL("../src/", import.meta.url));

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (/\.(ts|tsx|mts|js|mjs)$/.test(name) && !name.endsWith(".d.ts")) out.push(full);
  }
  return out;
}

/** Module specifiers from static imports/exports, dynamic `import()` and `require()`. */
function specifiers(source: string): string[] {
  const noComments = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
  const found: string[] = [];
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[\w*${}\s,]+?\s+from\s+)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["']([^"']+)["']\s*\)/g,
    /\brequire\s*\(\s*["']([^"']+)["']\s*\)/g,
  ];
  for (const re of patterns) for (const m of noComments.matchAll(re)) found.push(m[1]!);
  return found;
}

const isThree = (spec: string) => spec === "three" || spec.startsWith("three/");

/** Self-imports through the package name count as imports of that layer. */
const SELF: Record<string, string> = { "@pmndrs/vdb": "core", "@pmndrs/vdb/tools": "tools", "@pmndrs/vdb/three": "three" };

function violations(layer: "core" | "tools", forbiddenDirs: string[]): string[] {
  const bad: string[] = [];
  for (const file of walk(path.join(srcRoot, layer))) {
    const rel = path.relative(srcRoot, file);
    for (const spec of specifiers(readFileSync(file, "utf8"))) {
      if (isThree(spec)) bad.push(`${rel} imports "${spec}"`);
      const selfLayer = SELF[spec];
      if (selfLayer && forbiddenDirs.includes(selfLayer)) bad.push(`${rel} imports "${spec}"`);
      if (spec.startsWith(".")) {
        const target = path.resolve(path.dirname(file), spec);
        for (const dir of forbiddenDirs) {
          const forbidden = path.join(srcRoot, dir);
          if (target === forbidden || target.startsWith(forbidden + path.sep)) {
            bad.push(`${rel} imports "${spec}" (src/${dir})`);
          }
        }
      }
    }
  }
  return bad;
}

describe("source boundaries (K0.3)", () => {
  it("scans a non-trivial number of files per layer", () => {
    expect(walk(path.join(srcRoot, "core")).length).toBeGreaterThan(0);
    expect(walk(path.join(srcRoot, "tools")).length).toBeGreaterThan(10);
  });

  it("core never imports three or src/three", () => {
    expect(violations("core", ["three"])).toEqual([]);
  });

  it("tools never imports three, src/core or src/three", () => {
    expect(violations("tools", ["core", "three"])).toEqual([]);
  });

  it("the specifier scanner catches the shapes it guards against", () => {
    const sample = [
      'import * as THREE from "three";',
      'import { Vector3 } from "three/webgpu";',
      'export { x } from "../three/grid.js";',
      'const m = await import("three");',
      'import type { A } from "three";',
      '// import "three" in a comment is ignored',
    ].join("\n");
    expect(specifiers(sample)).toEqual(["three", "three/webgpu", "../three/grid.js", "three", "three"]);
  });
});
