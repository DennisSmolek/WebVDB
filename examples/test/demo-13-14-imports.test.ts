import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Gallery demos 13 (level sets) and 14 (points) are showcases written against the public API (D32): their modules,
// demo 14's decode worker and the shared shell they reach import only the package's public entry points and three,
// never a package source path or an `internal.ts`. (Same walk as demo-10-imports.test.ts.)

const repo = fileURLToPath(new URL("../../", import.meta.url));
const PUBLIC = new Set(["@pmndrs/vdb", "@pmndrs/vdb/tools", "@pmndrs/vdb/three"]);

function specifiers(source: string): string[] {
  const noComments = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");
  const patterns = [
    /\b(?:import|export)\s+(?:type\s+)?(?:[\w*${}\s,]+?\s+from\s+)?["']([^"']+)["']/g,
    /\bimport\s*\(\s*["'`]([^"'`$]+)["'`]\s*\)/g,
    // module workers: new Worker(new URL("./x.ts", import.meta.url))
    /\bnew\s+URL\s*\(\s*["']([^"']+\.ts)["']\s*,\s*import\.meta\.url\s*\)/g,
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

const DEMOS = [
  { name: "13", entry: "examples/src/demos/13-level-sets/main.ts", expect: ["examples/src/demos/13-level-sets/sculptures.ts", "examples/src/demos/showcase-shell.ts"] },
  { name: "14", entry: "examples/src/demos/14-points/main.ts", expect: ["examples/src/demos/14-points/decode-worker.ts", "examples/src/demos/showcase-shell.ts"] },
];

for (const demo of DEMOS) {
  describe(`demo ${demo.name} import graph (public API only)`, () => {
    const { files, bare } = demoGraph(path.join(repo, demo.entry));

    it("stays inside the demos folder", () => {
      for (const f of demo.expect) expect(files).toContain(f);
      expect(files.filter((f) => !f.startsWith("examples/src/demos/"))).toEqual([]);
    });

    it("imports only the public entry points and three", () => {
      const other = bare.filter((s) => !PUBLIC.has(s) && s !== "three" && !s.startsWith("three/"));
      expect(other).toEqual([]);
      expect(bare.filter((s) => s.includes("internal"))).toEqual([]);
    });
  });
}
