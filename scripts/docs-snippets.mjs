/**
 * Keeps the docs site's code blocks equal to the type-checked samples in docs/snippets/*.ts
 * (the mechanism of @pmndrs/bvh's tools/docs-snippets.ts).
 *
 *     node scripts/docs-snippets.mjs           # copy every region into the MDX pages
 *     node scripts/docs-snippets.mjs --check   # exit 1 if a page differs, or a ts block has no sample
 *
 * A sample is a region of a snippet file:
 *
 *     // #region name
 *     ...
 *     // #endregion
 *
 * A name may have several regions in one file (imports at the top, the body in a function); they join with a
 * blank line, each dedented on its own. In a page, the code block right after `{/* snippet: name *\/}` is that
 * sample. Every `ts` code block in a page must have such a marker, so no sample goes unchecked.
 * `pnpm docs:check` then type-checks docs/snippets against the package source (docs/snippets/tsconfig.json).
 */
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
export const SITE_DIR = join(ROOT, "docs", "site");
export const SNIPPETS_DIR = join(ROOT, "docs", "snippets");

function dedent(lines) {
  const indents = lines.filter((l) => l.trim() !== "").map((l) => l.length - l.trimStart().length);
  const cut = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => l.slice(cut));
}

function trimBlankEdges(lines) {
  let start = 0;
  let end = lines.length;
  while (start < end && lines[start].trim() === "") start++;
  while (end > start && lines[end - 1].trim() === "") end--;
  return lines.slice(start, end);
}

/** Paths of every .ts file under `dir`, relative to it. One file per sample keeps each one's imports its own. */
function tsFiles(dir, prefix = "") {
  const files = [];
  for (const entry of readdirSync(join(dir, prefix), { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) files.push(...tsFiles(dir, rel));
    else if (entry.name.endsWith(".ts")) files.push(rel);
  }
  return files.sort();
}

/** Every sample in docs/snippets, by name. Throws on an unclosed region or a name used in two files. */
export function collectSamples(dir = SNIPPETS_DIR) {
  const samples = new Map();
  for (const entry of tsFiles(dir)) {
    const file = join(dir, entry);
    const parts = new Map();
    let open = null;
    for (const [at, line] of readFileSync(file, "utf8").split("\n").entries()) {
      const start = /^\s*\/\/ #region (\S+)\s*$/.exec(line);
      const end = /^\s*\/\/ #endregion\s*$/.test(line);
      if (start) {
        if (open) throw new Error(`${entry}:${at + 1}: region ${start[1]} opens inside region ${open.name}`);
        open = { name: start[1], lines: [] };
      } else if (end) {
        if (!open) throw new Error(`${entry}:${at + 1}: #endregion without a region`);
        const list = parts.get(open.name) ?? [];
        list.push(trimBlankEdges(dedent(open.lines)));
        parts.set(open.name, list);
        open = null;
      } else if (open) {
        open.lines.push(line);
      }
    }
    if (open) throw new Error(`${entry}: region ${open.name} is never closed`);
    for (const [name, list] of parts) {
      const other = samples.get(name);
      if (other) throw new Error(`sample ${name} is in both ${relative(dir, other.file)} and ${entry}`);
      samples.set(name, { name, file, code: list.map((l) => l.join("\n")).join("\n\n") });
    }
  }
  return samples;
}

/** Every .mdx page under docs/site. */
export function docPages(dir = SITE_DIR) {
  const pages = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) pages.push(...docPages(path));
    else if (entry.name.endsWith(".mdx")) pages.push(path);
  }
  return pages.sort();
}

const MARKER = /^\{\/\* snippet: (\S+) \*\/\}\s*$/;
const FENCE = /^```(\S*)/;

/** The page with every marked code block replaced by its sample, and what was wrong. */
export function syncPage(page, text, samples) {
  const lines = text.split("\n");
  const out = [];
  const problems = [];
  const used = [];
  let pending = null;
  for (let at = 0; at < lines.length; at++) {
    const line = lines[at];
    const marker = MARKER.exec(line);
    if (marker) {
      pending = marker[1];
      out.push(line);
      continue;
    }
    const fence = FENCE.exec(line);
    if (!fence) {
      out.push(line);
      continue;
    }
    let close = at + 1;
    while (close < lines.length && !/^```\s*$/.test(lines[close])) close++;
    if (close >= lines.length) {
      problems.push({ page, line: at + 1, message: "a code block is never closed" });
      out.push(...lines.slice(at));
      break;
    }
    const body = lines.slice(at + 1, close).join("\n");
    const language = fence[1];
    if (pending !== null) {
      const sample = samples.get(pending);
      if (!sample) {
        problems.push({ page, line: at + 1, message: `no sample named ${pending} in docs/snippets` });
        out.push(...lines.slice(at, close + 1));
      } else {
        used.push(pending);
        if (body !== sample.code) problems.push({ page, line: at + 1, message: `the code block differs from sample ${pending} (run pnpm docs:snippets)` });
        out.push(line, sample.code, lines[close]);
      }
    } else {
      if (["ts", "typescript", "js", "tsx"].includes(language)) {
        problems.push({ page, line: at + 1, message: `a ${language} code block has no {/* snippet: name */} marker` });
      }
      out.push(...lines.slice(at, close + 1));
    }
    pending = null;
    at = close;
  }
  return { text: out.join("\n"), problems, used };
}

/** Syncs (or, with `check`, only checks) every page. Returns the problems left and the sample counts. */
export function syncDocs(check) {
  const samples = collectSamples();
  const problems = [];
  const used = new Set();
  let blocks = 0;
  for (const page of docPages()) {
    const text = readFileSync(page, "utf8");
    const result = syncPage(relative(ROOT, page), text, samples);
    result.used.forEach((n) => used.add(n));
    blocks += result.used.length;
    if (check) {
      problems.push(...result.problems);
    } else {
      problems.push(...result.problems.filter((p) => !p.message.startsWith("the code block differs")));
      if (result.text !== text) writeFileSync(page, result.text);
    }
  }
  return { problems, unused: [...samples.keys()].filter((n) => !used.has(n)), blocks, samples: samples.size };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const check = process.argv.includes("--check");
  const { problems, unused, blocks } = syncDocs(check);
  for (const p of problems) console.error(`${p.page}:${p.line}: ${p.message}`);
  for (const name of unused) console.error(`sample ${name} is not used by any page`);
  if (problems.length || unused.length) process.exit(1);
  console.log(check ? `docs: ${blocks} code blocks match their samples` : `docs: ${blocks} code blocks copied from docs/snippets`);
}
