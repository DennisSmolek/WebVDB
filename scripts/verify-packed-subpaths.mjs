// Verifies the packed @pmndrs/vdb tarball resolves all of its public subpaths
// from a clean ESM consumer, the way an npm user would import it.
//
// Assumes `pnpm build` has already produced packages/vdb/dist. Steps:
//   1. npm pack packages/vdb into a temp dir
//   2. install that tarball plus three@0.187.0 into a fresh consumer project
//   3. node -e: import each subpath and check import.meta.resolve points into
//      node_modules/@pmndrs/vdb/dist/, and that the .wgsl export is a real file
// Exits non-zero with a message on any failure. The temp dir is always removed.

import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = path.join(repoRoot, "packages", "vdb");
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const THREE_VERSION = "0.187.0";

class VerifyError extends Error {}

function run(cmd, args, cwd) {
  try {
    return execFileSync(cmd, args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
  } catch (err) {
    const detail = `${err.stderr ?? ""}${err.stdout ?? ""}`.trim();
    throw new VerifyError(`\`${cmd} ${args.join(" ")}\` failed in ${cwd}\n${detail}`);
  }
}

// Runs in the consumer project. Each check exits with a message on failure.
const CONSUMER_CHECK = `
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const fail = (msg) => { console.error("FAIL: " + msg); process.exit(1); };
const distRoot = "node_modules/@pmndrs/vdb/dist/";

for (const [spec, sub] of [
  ["@pmndrs/vdb", "core"],
  ["@pmndrs/vdb/tools", "tools"],
  ["@pmndrs/vdb/three", "three"],
]) {
  try {
    await import(spec);
  } catch (err) {
    fail(spec + " failed to import: " + (err && err.message));
  }
  const url = import.meta.resolve(spec);
  const file = fileURLToPath(url).split("\\\\").join("/");
  if (!file.includes(distRoot + sub + "/")) fail(spec + " resolved to " + file + ", expected inside " + distRoot + sub + "/");
  console.log("ok " + spec + " -> " + file);
}

const wgslUrl = import.meta.resolve("@pmndrs/vdb/pnanovdb.wgsl");
const wgslFile = fileURLToPath(wgslUrl);
if (!existsSync(wgslFile)) fail("@pmndrs/vdb/pnanovdb.wgsl resolved to " + wgslFile + " which does not exist");
console.log("ok @pmndrs/vdb/pnanovdb.wgsl -> " + wgslFile);
`;

function verify() {
  if (!existsSync(path.join(pkgDir, "dist", "core", "index.js"))) {
    throw new VerifyError("packages/vdb/dist is missing. Run `pnpm build` first.");
  }

  const work = mkdtempSync(path.join(tmpdir(), "vdb-packed-subpaths-"));
  try {
    const packDir = path.join(work, "pack");
    const consumer = path.join(work, "consumer");
    mkdirSync(packDir);
    mkdirSync(consumer);

    const packOut = run(npm, ["pack", "--json", "--pack-destination", packDir, "--loglevel=error"], pkgDir);
    let tarballName;
    try {
      tarballName = JSON.parse(packOut)[0].filename;
    } catch {
      throw new VerifyError(`could not read npm pack --json output:\n${packOut}`);
    }
    const tarball = path.join(packDir, tarballName);
    if (!existsSync(tarball)) throw new VerifyError(`npm pack did not produce ${tarball}`);

    writeFileSync(
      path.join(consumer, "package.json"),
      JSON.stringify({ name: "vdb-packed-consumer", private: true, type: "module" }, null, 2) + "\n",
    );
    run(
      npm,
      ["install", "--no-audit", "--no-fund", "--loglevel=error", tarball, `three@${THREE_VERSION}`],
      consumer,
    );

    const installed = path.join(consumer, "node_modules", "@pmndrs", "vdb", "package.json");
    if (!existsSync(installed)) throw new VerifyError("npm install did not put @pmndrs/vdb in node_modules");

    try {
      const out = execFileSync(process.execPath, ["--input-type=module", "-e", CONSUMER_CHECK], {
        cwd: consumer,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      });
      process.stdout.write(out);
    } catch (err) {
      const detail = `${err.stderr ?? ""}${err.stdout ?? ""}`.trim();
      throw new VerifyError(`consumer import check failed:\n${detail}`);
    }
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

try {
  verify();
  console.log("verify:packed-subpaths passed");
} catch (err) {
  console.error(`verify:packed-subpaths FAILED: ${err instanceof VerifyError ? err.message : err?.stack ?? err}`);
  process.exitCode = 1;
}
