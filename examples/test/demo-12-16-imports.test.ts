import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Gallery demos 12 (explosion) and 16 (streaming) are written against the public API (D32), as demo 10: their own
// modules (and the shared shell and demo helpers they reach) import only the package's public entry points and
// three, never a package source path or an `internal.ts`.

const repo = fileURLToPath(new URL("../../", import.meta.url));
const PUBLIC = new Set(["@pmndrs/vdb", "@pmndrs/vdb/tools", "@pmndrs/vdb/three"]);

/** Static and dynamic import/export specifiers (same shapes as demo-10-imports.test.ts). */
function specifiers(source: string): string[] {
  const noComments = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[\w*${}\s,]+?\s+from\s+)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["'`]([^"'`$]+)["'`]\s*\)/g,
  ];
  const found: string[] = [];
  for (const re of patterns) for (const m of noComments.matchAll(re)) found.push(m[1]!.split("?")[0]!);
  return found;
}

/** A demo's own module graph: relative imports only (package entry points are leaves). */
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

for (const [name, entry] of [
  ["12 explosion", "examples/src/demos/12-explosion/main.ts"],
  ["16 streaming", "examples/src/demos/16-streaming/main.ts"],
] as const) {
  describe(`demo ${name} import graph (public API only)`, () => {
    const { files, bare } = demoGraph(path.join(repo, entry));

    it("stays inside the demos folder and uses the gallery shell", () => {
      expect(files).toContain("examples/src/demos/showcase-shell.ts");
      expect(files.filter((f) => !f.startsWith("examples/src/demos/"))).toEqual([]);
    });

    it("imports only the public entry points and three", () => {
      const other = bare.filter((s) => !PUBLIC.has(s) && s !== "three" && !s.startsWith("three/"));
      expect(other).toEqual([]);
      expect(bare.filter((s) => s.includes("internal"))).toEqual([]);
      expect(bare.some((s) => PUBLIC.has(s))).toBe(true);
    });
  });
}
