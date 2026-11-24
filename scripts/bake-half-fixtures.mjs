#!/usr/bin/env node
// Bakes the NanoVDB Half fixtures (K7.H) with native NanoVDB from OpenVDB
// v14.0.0 (the first release with GridType::Half implemented, PR 2335):
//   fixtures/primitives/{sphere,torus,box}_fog_half.{nvdb,sidecar.json}
//     (createNanoGrid<FloatGrid, Half> of the `pnpm fixtures:bake` float grids)
//   fixtures/half/half_edge.{nvdb,sidecar.json} (rounding / subnormal / overflow cases)
// via docker/fixture-bake/bake_half.cpp. The .nvdb files are git-ignored; the
// sidecars are committed and guarded by packages/vdb/test/tools/half-native.test.ts.
//
// NanoVDB is header-only, so this needs only a C++17 compiler (c++, g++ or
// clang++) and the v14.0.0 source tarball (fetched once into fixtures/downloads/).
// Override the headers with NANOVDB_V14_INCLUDE=<dir containing nanovdb/NanoVDB.h>.

import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { download, exists } from "./lib/download.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const TAG = "v14.0.0"; // ecae2bda85744412b0cd47e1b8e1e4e76beb7b88
const DOWNLOADS = path.join(ROOT, "fixtures", "downloads");
const PRIMITIVES = path.join(ROOT, "fixtures", "primitives");
const HALF = path.join(ROOT, "fixtures", "half");

const run = (cmd, args) => spawnSync(cmd, args, { stdio: "inherit", cwd: ROOT });
const have = (cmd) => spawnSync(cmd, ["--version"], { stdio: "ignore" }).status === 0;

async function headers() {
  if (process.env.NANOVDB_V14_INCLUDE) return path.resolve(process.env.NANOVDB_V14_INCLUDE);
  const dir = path.join(DOWNLOADS, "openvdb-14.0.0");
  const include = path.join(dir, "nanovdb");
  if (await exists(path.join(include, "nanovdb", "NanoVDB.h"))) return include;
  const tarball = path.join(DOWNLOADS, "openvdb-14.0.0.tar.gz");
  await download(`https://codeload.github.com/AcademySoftwareFoundation/openvdb/tar.gz/refs/tags/${TAG}`, tarball);
  await mkdir(DOWNLOADS, { recursive: true });
  // Only the header-only NanoVDB tree; --no-same-owner and no absolute paths (GNU and BSD tar).
  if (run("tar", ["-xzf", tarball, "-C", DOWNLOADS, "--no-same-owner", "openvdb-14.0.0/nanovdb/nanovdb"]).status !== 0) {
    throw new Error(`could not extract ${tarball}`);
  }
  return include;
}

for (const name of ["sphere_fog", "torus_fog", "box_fog"]) {
  if (!(await exists(path.join(PRIMITIVES, `${name}_float.nvdb`)))) {
    console.error(`✗ fixtures/primitives/${name}_float.nvdb is missing: run \`pnpm fixtures:bake\` first.`);
    process.exit(1);
  }
}

const cxx = ["c++", "g++", "clang++"].find(have);
if (!cxx) {
  console.error("✗ no C++17 compiler (c++, g++ or clang++) found.");
  process.exit(1);
}
const include = await headers();
const bin = path.join(DOWNLOADS, "bake_half");
await mkdir(DOWNLOADS, { recursive: true });
const build = run(cxx, [
  "-std=c++17",
  "-O2",
  "-ffp-contract=off", // as bake_primitives: sidecars must not depend on the compiler's FMA choices
  "-I",
  include,
  path.join(ROOT, "docker", "fixture-bake", "bake_half.cpp"),
  "-o",
  bin,
]);
if (build.status !== 0) process.exit(1);
if (run(bin, [PRIMITIVES, HALF]).status !== 0) process.exit(1);
console.error("✓ Half fixtures baked → fixtures/primitives/*_half.*, fixtures/half/");
