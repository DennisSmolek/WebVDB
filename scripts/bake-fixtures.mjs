#!/usr/bin/env node
// Bakes the primitive fixtures (fog sphere/torus/box × float/fp8/fpn →
// .nvdb + JSON sidecars) into fixtures/primitives/.
//
// Two paths, tried in order:
//   1. docker  — the reproducible route: docker/fixture-bake image
//   2. local   — g++ against a sparse clone of the pinned NanoVDB headers
//                (NanoVDB is header-only, so no OpenVDB build is needed)
// Force one with BAKE_MODE=docker or BAKE_MODE=local.

import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { exists } from "./lib/download.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OPENVDB_COMMIT = "a532de5526ef791280b6483a872336a811a68542"; // = vendored PNanoVDB.h pin
const OUT = path.join(ROOT, "fixtures");
const MODE = process.env.BAKE_MODE ?? "auto";

const run = (cmd, args, opts = {}) =>
  spawnSync(cmd, args, { stdio: "inherit", cwd: ROOT, ...opts });
const have = (cmd) => spawnSync(cmd, ["--version"], { stdio: "ignore" }).status === 0;

function bakeDocker() {
  console.error("• baking via docker (docker/fixture-bake)");
  if (run("docker", ["build", "-t", "webvdb-fixture-bake", "docker/fixture-bake"]).status !== 0)
    return false;
  return run("docker", ["run", "--rm", "-v", `${OUT}:/out`, "webvdb-fixture-bake"]).status === 0;
}

async function bakeLocal() {
  console.error("• baking locally with g++ (NanoVDB headers are header-only)");
  const clone = path.join(OUT, "downloads", "openvdb-sparse");
  if (!(await exists(path.join(clone, "nanovdb", "nanovdb", "NanoVDB.h")))) {
    await mkdir(path.dirname(clone), { recursive: true });
    console.error(`  cloning pinned NanoVDB headers (${OPENVDB_COMMIT.slice(0, 8)})…`);
    const steps = [
      ["git", ["init", "-q", clone]],
      ["git", ["-C", clone, "remote", "add", "origin", "https://github.com/AcademySoftwareFoundation/openvdb.git"]],
      ["git", ["-C", clone, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", OPENVDB_COMMIT]],
      ["git", ["-C", clone, "sparse-checkout", "set", "nanovdb/nanovdb"]],
      ["git", ["-C", clone, "checkout", "-q", "FETCH_HEAD"]],
    ];
    for (const [cmd, args] of steps) {
      if (run(cmd, args).status !== 0) return false;
    }
  }
  const bin = path.join(OUT, "downloads", "bake_primitives");
  const gpp = run("g++", [
    "-std=c++17",
    "-O2",
    "-I",
    path.join(clone, "nanovdb"),
    path.join(ROOT, "docker", "fixture-bake", "bake_primitives.cpp"),
    "-o",
    bin,
  ]);
  if (gpp.status !== 0) return false;
  return run(bin, [path.join(OUT, "primitives")]).status === 0;
}

let ok = false;
if (MODE === "docker") ok = bakeDocker();
else if (MODE === "local") ok = await bakeLocal();
else if (have("docker") && spawnSync("docker", ["info"], { stdio: "ignore" }).status === 0)
  ok = bakeDocker();
else if (have("g++")) ok = await bakeLocal();
else console.error("✗ neither a running docker daemon nor g++ found — install one and re-run.");

if (!ok) process.exit(1);
console.error("✓ primitives baked → fixtures/primitives/ (.nvdb git-ignored, sidecars committed)");
