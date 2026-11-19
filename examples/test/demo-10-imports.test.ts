import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Demo 10 is the froxel renderer showcase written against the public API (D32): its own modules (and the
// shared demo helpers they reach) import only the package's public entry points and three, never a
// package source path or an `internal.ts`, unlike the K3 harness pages.

const repo = fileURLToPath(new URL("../../", import.meta.url));
const entry = path.join(repo, "examples/src/demos/10-frame-renderer/main.ts");
const PUBLIC = new Set(["@pmndrs/vdb", "@pmndrs/vdb/tools", "@pmndrs/vdb/three"]);

/** Static and dynamic import/export specifiers (same shapes as demo-08-imports.test.ts). */
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

/** The demo's own module graph: relative imports only (package entry points are leaves). */
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

describe("demo 10 import graph (public API only)", () => {
  const { files, bare } = demoGraph(entry);

  it("stays inside the demos folder", () => {
    expect(files).toContain("examples/src/demos/10-frame-renderer/scenes.ts");
    expect(files.filter((f) => !f.startsWith("examples/src/demos/"))).toEqual([]);
  });

  it("imports only the public entry points and three", () => {
    const other = bare.filter((s) => !PUBLIC.has(s) && s !== "three" && !s.startsWith("three/"));
    expect(other).toEqual([]);
    expect(bare.filter((s) => s.includes("internal"))).toEqual([]);
  });
});
