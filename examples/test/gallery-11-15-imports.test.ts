import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Gallery demos 11 and 15 are written against the public API (D32): their own modules (the shared gallery shell, and
// demo 15's bake worker, an entry of its own) import only the package's public entry points and three, never a
// package source path or an `internal.ts`. Same walk as demo-10-imports.test.ts.

const repo = fileURLToPath(new URL("../../", import.meta.url));
const PUBLIC = new Set(["@pmndrs/vdb", "@pmndrs/vdb/tools", "@pmndrs/vdb/three"]);

/** Static and dynamic import/export specifiers, plus `new URL("./x.ts", import.meta.url)` worker entries. */
function specifiers(source: string): string[] {
  const noComments = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[\w*${}\s,]+?\s+from\s+)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["'`]([^"'`$]+)["'`]\s*\)/g,
    /\bnew\s+URL\(\s*["'](\.[^"']+)["']\s*,\s*import\.meta\.url\s*\)/g,
  ];
  const found: string[] = [];
  for (const re of patterns) for (const m of noComments.matchAll(re)) found.push(m[1]!.split("?")[0]!);
  return found;
}

function demoGraph(start: string): { files: string[]; bare: string[] } {
  const files = new Set<string>();
  const bare = new Set<string>();
  const queue = [start];
  while (queue.length > 0) {
    const file = queue.pop()!;
    if (files.has(file)) continue;
    files.add(file);
    for (const spec of specifiers(readFileSync(file, "utf8"))) {
      if (!spec.startsWith(".")) {
        bare.add(spec);
        continue;
      }
      const base = path.resolve(path.dirname(file), spec);
      const hit = [base, base.replace(/\.js$/, ".ts"), `${base}.ts`].find((c) => existsSync(c) && c.endsWith(".ts"));
      if (!hit) throw new Error(`cannot resolve ${spec} from ${path.relative(repo, file)}`);
      queue.push(hit);
    }
  }
  return { files: [...files].map((f) => path.relative(repo, f)).sort(), bare: [...bare].sort() };
}

describe.each([
  ["11-sky-clouds", ["examples/src/demos/11-sky-clouds/sky.ts", "examples/src/demos/showcase-shell.ts"]],
  ["15-full-res-vs-dense", ["examples/src/demos/15-full-res-vs-dense/bake-worker.ts", "examples/src/demos/15-full-res-vs-dense/protocol.ts"]],
] as const)("demo %s import graph (public API only)", (dir, expected) => {
  const { files, bare } = demoGraph(path.join(repo, `examples/src/demos/${dir}/main.ts`));

  it("stays inside the demos folder and reaches its own modules", () => {
    for (const f of expected) expect(files).toContain(f);
    expect(files.filter((f) => !f.startsWith("examples/src/demos/"))).toEqual([]);
  });

  it("imports only the public entry points and three", () => {
    const other = bare.filter((s) => !PUBLIC.has(s) && s !== "three" && !s.startsWith("three/"));
    expect(other).toEqual([]);
    expect(bare.filter((s) => s.includes("internal"))).toEqual([]);
  });
});
