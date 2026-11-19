#!/usr/bin/env node
// The hosted examples site's asset set (RELEASING.md section 4).
//
//   node scripts/examples-assets.mjs --out examples/dist/fixtures     copy the set from fixtures/ (or --src)
//   node scripts/examples-assets.mjs --from-tar T --sha256 H --out D  unpack a prebuilt tarball (CI), verify it
//   node scripts/examples-assets.mjs --pack [--pack-dir D]            write examples-assets-<hash>.tar (+ the
//                                                                     release name examples-assets.tar)
//   node scripts/examples-assets.mjs --check                          validate only (files exist, limits)
//   ... --verify-dir examples/dist                                    also fail if that whole dir is >= the cap
//
// The set is examples/assets.manifest.json: explicit file lists (derived from the demos' fixtureUrl() calls).
// A group with `licensingRows` ships only when every named row of docs/assets-licensing.md is `ok`; otherwise
// it is skipped and the demos fall back to their procedural skies and props, as with `?gallery=0`.
// Fails when a listed file is missing, a file is >= maxFileMB, or the set (or --verify-dir) is >= maxTotalMB.

import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  linkSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
  createReadStream,
  openSync,
  writeSync,
  closeSync,
  readSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const MANIFEST_PATH = path.join(repo, "examples/assets.manifest.json");
const LICENSING_PATHS = [path.join(repo, "docs/dev/assets-licensing.md"), path.join(repo, "docs/assets-licensing.md")];
const MB = 1024 * 1024;

export function readManifest(file = MANIFEST_PATH) {
  return JSON.parse(readFileSync(file, "utf8"));
}

/** `{ "sky-midday-HDR.jpg": "pending", ... }` from the table rows of docs/assets-licensing.md (first backticked token, last cell). */
export function parseLicensingTable(markdown) {
  const rows = {};
  for (const line of markdown.split("\n")) {
    if (!line.startsWith("|")) continue;
    const cells = line.split("|").slice(1, -1).map((c) => c.trim());
    if (cells.length < 6) continue;
    const key = /`([^`]+)`/.exec(cells[0])?.[1];
    if (key) rows[key] = cells[cells.length - 1].toLowerCase();
  }
  return rows;
}

export function readLicensing() {
  const file = LICENSING_PATHS.find((p) => existsSync(p));
  return file ? parseLicensingTable(readFileSync(file, "utf8")) : {};
}

/** The groups that ship: every group without `licensingRows`, and gallery groups whose rows are all `ok`. */
export function shippedGroups(manifest, licensing) {
  const shipped = [];
  const skipped = [];
  for (const g of manifest.groups) {
    const rows = g.licensingRows ?? [];
    const notOk = rows.filter((r) => licensing[r] !== "ok");
    if (notOk.length) skipped.push({ group: g, why: `docs/assets-licensing.md: ${notOk.map((r) => `${r} is ${licensing[r] ?? "missing"}`).join(", ")}` });
    else shipped.push(g);
  }
  return { shipped, skipped };
}

function globToRegExp(name) {
  return new RegExp(`^${name.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`);
}

/** The group's file paths (relative to fixtures/), globs expanded against `src`; throws on a missing file or a wrong glob count. */
export function groupFiles(group, src) {
  const out = [...group.files];
  for (const { pattern, count } of group.globs ?? []) {
    const dir = path.posix.dirname(pattern);
    const re = globToRegExp(path.posix.basename(pattern));
    const found = existsSync(path.join(src, dir)) ? readdirSync(path.join(src, dir)).filter((n) => re.test(n)).sort() : [];
    if (found.length !== count) throw new Error(`${group.id}: ${pattern} matches ${found.length} files, expected ${count}`);
    for (const n of found) out.push(`${dir}/${n}`);
  }
  return out;
}

/** Resolves the whole set: `{ files: [{ rel, abs, size, group }], skipped, totalBytes }`. Throws on missing files and size limits. */
export function resolveSet({ manifest = readManifest(), src = path.join(repo, "fixtures"), licensing = readLicensing() } = {}) {
  const { shipped, skipped } = shippedGroups(manifest, licensing);
  const files = [];
  const problems = [];
  const seen = new Set();
  for (const g of shipped) {
    for (const rel of groupFiles(g, src)) {
      if (seen.has(rel)) continue;
      seen.add(rel);
      const abs = path.join(src, rel);
      if (!existsSync(abs)) {
        problems.push(`missing: ${rel} (group ${g.id})`);
        continue;
      }
      files.push({ rel, abs, size: statSync(abs).size, group: g.id });
    }
  }
  const totalBytes = files.reduce((s, f) => s + f.size, 0);
  const maxFile = manifest.limits.maxFileMB * MB;
  for (const f of files) if (f.size >= maxFile) problems.push(`${f.rel} is ${(f.size / MB).toFixed(1)} MB, limit ${manifest.limits.maxFileMB} MB`);
  if (totalBytes >= manifest.limits.maxTotalMB * MB) problems.push(`asset set is ${(totalBytes / MB).toFixed(0)} MB, limit ${manifest.limits.maxTotalMB} MB`);
  if (problems.length) throw new Error(`examples assets:\n  ${problems.join("\n  ")}`);
  files.sort((a, b) => (a.rel < b.rel ? -1 : 1));
  return { files, skipped, totalBytes };
}

export function dirBytes(dir) {
  let total = 0;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    total += e.isDirectory() ? dirBytes(p) : statSync(p).size;
  }
  return total;
}

function copySet(files, out) {
  for (const f of files) {
    const dest = path.join(out, f.rel);
    mkdirSync(path.dirname(dest), { recursive: true });
    copyFileSync(f.abs, dest);
  }
}

// --- a deterministic ustar writer (sorted names, mtime 0, uid/gid 0) -------------------------------------------

function tarHeader(name, size) {
  const h = Buffer.alloc(512);
  let prefix = "";
  let base = name;
  if (Buffer.byteLength(name) > 100) {
    const i = name.lastIndexOf("/", 155);
    if (i < 0 || Buffer.byteLength(name) - i - 1 > 100) throw new Error(`tar: path too long: ${name}`);
    prefix = name.slice(0, i);
    base = name.slice(i + 1);
  }
  h.write(base, 0, "utf8");
  h.write("0000644\0", 100);
  h.write("0000000\0", 108);
  h.write("0000000\0", 116);
  h.write(`${size.toString(8).padStart(11, "0")}\0`, 124);
  h.write(`${"0".padStart(11, "0")}\0`, 136);
  h.write("        ", 148); // checksum placeholder (spaces)
  h.write("0", 156);
  h.write("ustar\0", 257);
  h.write("00", 263);
  h.write(prefix, 345, "utf8");
  let sum = 0;
  for (const b of h) sum += b;
  h.write(`${sum.toString(8).padStart(6, "0")}\0 `, 148);
  return h;
}

export function writeTar(files, outFile) {
  const fd = openSync(outFile, "w");
  const hash = createHash("sha256");
  const put = (buf) => {
    writeSync(fd, buf);
    hash.update(buf);
  };
  try {
    for (const f of files) {
      put(tarHeader(f.rel, f.size));
      const rd = openSync(f.abs, "r");
      const chunk = Buffer.alloc(4 * MB);
      let left = f.size;
      while (left > 0) {
        const n = readSync(rd, chunk, 0, Math.min(chunk.length, left), null);
        if (n <= 0) throw new Error(`short read: ${f.abs}`);
        put(chunk.subarray(0, n));
        left -= n;
      }
      closeSync(rd);
      const pad = (512 - (f.size % 512)) % 512;
      if (pad) put(Buffer.alloc(pad));
    }
    put(Buffer.alloc(1024));
  } finally {
    closeSync(fd);
  }
  return hash.digest("hex");
}

export function sha256File(file) {
  return new Promise((resolve, reject) => {
    const h = createHash("sha256");
    createReadStream(file).on("data", (d) => h.update(d)).on("end", () => resolve(h.digest("hex"))).on("error", reject);
  });
}

// --- the attribution page rows (also used by examples/vite.config.ts) -------------------------------------------

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** HTML `<li>` entries for every shipped group: title, author, source, licence (linked) and the "modified" note. */
export function renderAttribution(manifest = readManifest(), licensing = readLicensing()) {
  const { shipped } = shippedGroups(manifest, licensing);
  return shipped
    .map((g) => {
      const demos = (g.usedBy ?? []).map((n) => String(n).padStart(2, "0")).join(", ");
      const source = g.source ? `<a href="${esc(g.source)}" rel="noopener">${esc(g.source)}</a>` : "";
      const licence = g.licenceUrl ? `<a href="${esc(g.licenceUrl)}" rel="license noopener">${esc(g.licence)}</a>` : esc(g.licence);
      return (
        `<li data-asset-group="${esc(g.id)}"><strong>${esc(g.title)}</strong>` +
        `<dl><dt>Author</dt><dd>${esc(g.author)}</dd><dt>Source</dt><dd>${source}</dd><dt>Licence</dt><dd>${licence}</dd>` +
        (g.modified ? `<dt>Changes</dt><dd>${esc(g.modified)}</dd>` : "") +
        `<dt>Used by demos</dt><dd>${esc(demos)}</dd></dl></li>`
      );
    })
    .join("\n");
}

// --- cli ---------------------------------------------------------------------------------------------------------

function arg(name, argv) {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : undefined;
}

async function main(argv) {
  const manifest = readManifest();
  const licensing = readLicensing();
  const out = arg("--out", argv);
  const verifyDir = arg("--verify-dir", argv);
  const fromTar = arg("--from-tar", argv);
  const src = arg("--src", argv) ? path.resolve(arg("--src", argv)) : path.join(repo, "fixtures");

  if (fromTar) {
    if (!out) throw new Error("--from-tar needs --out <dir>");
    const want = arg("--sha256", argv) ?? manifest.release.sha256;
    if (!want) throw new Error("no sha256: pass --sha256 or set release.sha256 in examples/assets.manifest.json");
    const got = await sha256File(fromTar);
    if (got !== want) throw new Error(`examples-assets tarball sha256 mismatch:\n  got  ${got}\n  want ${want}`);
    rmSync(out, { recursive: true, force: true });
    mkdirSync(out, { recursive: true });
    const r = spawnSync("tar", ["-xf", path.resolve(fromTar), "-C", out], { stdio: "inherit" });
    if (r.status !== 0) throw new Error("tar -xf failed");
    // The unpacked set must be exactly what the manifest lists (gallery groups with a non-ok row are not required).
    const set = resolveSet({ manifest, src: out, licensing });
    console.log(`examples assets: unpacked ${set.files.length} files, ${(set.totalBytes / MB).toFixed(1)} MB, sha256 ${got.slice(0, 12)}`);
  } else {
    const set = resolveSet({ manifest, src, licensing });
    console.log(`examples assets: ${set.files.length} files, ${(set.totalBytes / MB).toFixed(1)} MB`);
    for (const s of set.skipped) console.log(`  skipped group ${s.group.id}: ${s.why}`);
    if (argv.includes("--pack")) {
      const dir = path.resolve(arg("--pack-dir", argv) ?? path.join(repo, "artifacts/examples-assets"));
      rmSync(dir, { recursive: true, force: true });
      mkdirSync(dir, { recursive: true });
      const tmp = path.join(dir, "examples-assets.tar");
      const sha = writeTar(set.files, tmp);
      const named = path.join(dir, `examples-assets-${sha.slice(0, 12)}.tar`);
      try {
        linkSync(tmp, named);
      } catch {
        copyFileSync(tmp, named);
      }
      writeFileSync(path.join(dir, "SHA256"), `${sha}  examples-assets.tar\n`);
      console.log(`packed ${path.relative(repo, tmp)} (${(statSync(tmp).size / MB).toFixed(1)} MB) and ${path.basename(named)}`);
      console.log(`sha256 ${sha}`);
      console.log(`Owner: put that sha256 in examples/assets.manifest.json (release.sha256), then upload ${path.relative(repo, tmp)} (see RELEASING.md section 4).`);
    } else if (out) {
      rmSync(out, { recursive: true, force: true });
      copySet(set.files, out);
      console.log(`copied to ${path.relative(repo, path.resolve(out))}`);
    }
  }

  if (verifyDir) {
    const total = dirBytes(verifyDir);
    console.log(`examples site: ${(total / MB).toFixed(1)} MB in ${path.relative(repo, path.resolve(verifyDir))}`);
    if (total >= manifest.limits.maxTotalMB * MB) throw new Error(`examples site is ${(total / MB).toFixed(0)} MB, limit ${manifest.limits.maxTotalMB} MB (GitHub Pages: 1 GB)`);
    const big = [];
    (function walk(d) {
      for (const e of readdirSync(d, { withFileTypes: true })) {
        const p = path.join(d, e.name);
        if (e.isDirectory()) walk(p);
        else if (statSync(p).size >= manifest.limits.maxFileMB * MB) big.push(p);
      }
    })(verifyDir);
    if (big.length) throw new Error(`files >= ${manifest.limits.maxFileMB} MB in the site:\n  ${big.join("\n  ")}`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(String(err instanceof Error ? err.message : err));
    process.exit(1);
  });
}
