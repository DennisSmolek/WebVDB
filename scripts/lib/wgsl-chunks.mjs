// Declaration-table extraction for the vendored PNanoVDB WGSL fork (K1.4a,
// docs/design/K1-core.md §3.3).
//
// `extractWgslChunks(source)` parses the UNMODIFIED vendored `pnanovdb.wgsl`
// into its top-level declarations (fn, struct, const, alias), in file order,
// with the names of the other declarations each one references.
// `generateChunksTs(...)` renders that table as
// `packages/vdb/src/core/wgsl/pnanovdb.chunks.generated.ts`.
//
// Both are pure and deterministic: the sync test
// (packages/vdb/test/core/wgsl-chunks.test.ts) re-runs them on the vendored
// file and compares the result with the committed module byte for byte.
//
// Chunk text is the declaration with `//` and `/* */` comments removed,
// trailing whitespace trimmed and blank lines dropped. Nothing else changes;
// the generation transforms (buffer access, type table, compile-out, rename)
// are applied later by core/wgsl/generate.ts and logged in vendor/VENDOR.md.

import { createHash } from "node:crypto";

const KEYWORDS = new Set(["fn", "struct", "const", "alias"]);
const IDENT = /[A-Za-z_][A-Za-z0-9_]*/y;

/** Remove `//` line comments and (nestable) block comments; keeps newlines. */
export function stripWgslComments(source) {
  let out = "";
  let i = 0;
  while (i < source.length) {
    const c = source[i];
    const n = source[i + 1];
    if (c === "/" && n === "/") {
      while (i < source.length && source[i] !== "\n") i++;
    } else if (c === "/" && n === "*") {
      let depth = 1;
      i += 2;
      while (i < source.length && depth > 0) {
        if (source[i] === "/" && source[i + 1] === "*") {
          depth++;
          i += 2;
        } else if (source[i] === "*" && source[i + 1] === "/") {
          depth--;
          i += 2;
        } else {
          if (source[i] === "\n") out += "\n";
          i++;
        }
      }
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

function tidy(text) {
  return text
    .split("\n")
    .map((line) => line.replace(/\s+$/, ""))
    .filter((line) => line.length > 0)
    .join("\n");
}

function lineOf(source, index) {
  let line = 1;
  for (let i = 0; i < index; i++) if (source[i] === "\n") line++;
  return line;
}

/**
 * Top-level declarations of a WGSL source, in file order:
 * `{ name, kind, deps, text }[]`. Throws on anything it does not understand,
 * so a vendored edit that adds a new kind of declaration fails loudly.
 */
export function extractWgslChunks(source) {
  const code = stripWgslComments(source.replace(/\r\n/g, "\n"));
  const decls = [];
  let i = 0;
  const skipSpace = () => {
    while (i < code.length && /\s/.test(code[i])) i++;
  };
  const ident = () => {
    IDENT.lastIndex = i;
    const m = IDENT.exec(code);
    if (!m) throw new Error(`extractWgslChunks: expected an identifier at line ${lineOf(code, i)}`);
    i += m[0].length;
    return m[0];
  };

  for (;;) {
    skipSpace();
    if (i >= code.length) break;
    const start = i;
    const kind = ident();
    if (!KEYWORDS.has(kind)) {
      throw new Error(`extractWgslChunks: unsupported top-level token "${kind}" at line ${lineOf(code, start)}`);
    }
    skipSpace();
    const name = ident();

    if (kind === "fn" || kind === "struct") {
      // Up to the body's opening brace, then to its matching close.
      while (i < code.length && code[i] !== "{") i++;
      let depth = 0;
      for (; i < code.length; i++) {
        if (code[i] === "{") depth++;
        else if (code[i] === "}" && --depth === 0) {
          i++;
          break;
        }
      }
      if (depth !== 0) throw new Error(`extractWgslChunks: unbalanced braces in ${name}`);
      if (kind === "struct") {
        // Optional trailing `;` (allowed after a struct in WGSL).
        let j = i;
        while (j < code.length && /\s/.test(code[j])) j++;
        if (code[j] === ";") i = j + 1;
      }
    } else {
      // const / alias: to the `;` at bracket depth 0.
      let depth = 0;
      for (; i < code.length; i++) {
        const c = code[i];
        if (c === "(" || c === "[" || c === "{") depth++;
        else if (c === ")" || c === "]" || c === "}") depth--;
        else if (c === ";" && depth === 0) {
          i++;
          break;
        }
      }
    }
    decls.push({ name, kind, text: tidy(code.slice(start, i)) });
  }

  const order = new Map(decls.map((d, index) => [d.name, index]));
  if (order.size !== decls.length) {
    const seen = new Set();
    for (const d of decls) {
      if (seen.has(d.name)) throw new Error(`extractWgslChunks: "${d.name}" is declared twice`);
      seen.add(d.name);
    }
  }
  return decls.map((d) => {
    const deps = new Set();
    for (const m of d.text.matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) {
      if (m[0] !== d.name && order.has(m[0])) deps.add(m[0]);
    }
    return {
      name: d.name,
      kind: d.kind,
      deps: [...deps].sort((a, b) => order.get(a) - order.get(b)),
      text: d.text,
    };
  });
}

/** The leading `//` comment block of the vendored file (its licence header), up to the first blank comment-free line. */
export function licenseHeader(source) {
  const lines = [];
  for (const line of source.replace(/\r\n/g, "\n").split("\n")) {
    if (!line.startsWith("//")) break;
    if (/^\/\/ NOTE:/.test(line)) break;
    lines.push(line);
  }
  while (lines.length > 0 && lines[lines.length - 1].trim() === "//") lines.pop();
  return lines;
}

export function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

/**
 * Render the chunk table as a TypeScript module.
 * `meta`: `{ pinnedCommit }` (from vendor/VENDOR.md).
 */
export function generateChunksTs(source, meta) {
  const chunks = extractWgslChunks(source);
  const header = licenseHeader(source);
  const rows = chunks
    .map((c) => `  ${JSON.stringify({ name: c.name, kind: c.kind, deps: c.deps, text: c.text })},`)
    .join("\n");
  return `// GENERATED FILE: do not edit by hand.
// Source: packages/vdb/vendor/pnanovdb.wgsl (sha256 ${sha256(source)})
// Upstream: emcfarlane/webgpu-nanovdb @ ${meta.pinnedCommit}, plus the in-tree diff log in vendor/VENDOR.md
// Regenerate: node scripts/extract-wgsl-chunks.mjs
// Guarded by: packages/vdb/test/core/wgsl-chunks.test.ts
//
// The text below is the vendored file's declarations, comments removed. It
// keeps the vendored file's licence:
//
${header.map((line) => line.replace(/\s+$/, "")).join("\n")}

/** One top-level declaration of the vendored \`pnanovdb.wgsl\`. */
export interface PnanovdbChunk {
  /** Declared name (\`pnanovdb_*\` / \`PNANOVDB_*\`). */
  readonly name: string;
  readonly kind: "fn" | "struct" | "const" | "alias";
  /** Names of other chunks this one references, in file order. */
  readonly deps: readonly string[];
  /** Declaration text, comments removed. */
  readonly text: string;
}

/** SHA-256 of the vendored file this table was extracted from. */
export const PNANOVDB_SOURCE_SHA256 = ${JSON.stringify(sha256(source))};

/** Every top-level declaration, in file order (${chunks.length} declarations). */
export const PNANOVDB_CHUNKS: readonly PnanovdbChunk[] = [
${rows}
];
`;
}

/** Pinned upstream commit from vendor/VENDOR.md's Pin table. */
export function pinnedCommitFromVendorDoc(vendorDoc) {
  const m = /\|\s*Pinned commit\s*\|\s*`([0-9a-f]{40})`/.exec(vendorDoc);
  if (!m) throw new Error("VENDOR.md: no pinned commit row");
  return m[1];
}
