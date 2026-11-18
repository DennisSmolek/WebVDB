#!/usr/bin/env node
// Convert one EmberGen .vdb sequence to FP8 NanoVDB with the pinned Docker
// image. The manifest is written atomically only after every frame validates.

import { spawnSync } from "node:child_process";
import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { NvdbFile } from "../packages/vdb/dist/tools/nvdb-file.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures");
const EMBERGEN = path.join(FIXTURES, "embergen");
const IMAGE = process.env.FIXTURE_BAKE_IMAGE ?? "webvdb-fixture-bake";
const FPS = Number(process.env.SEQUENCE_FPS ?? "24");
const GRID = process.env.SEQUENCE_GRID;
const START_INDEX = Number(process.env.SEQUENCE_START_INDEX ?? "0");
const TEMPORAL = process.env.SEQUENCE_TEMPORAL === "1";

//* File discovery =============================================================

async function filesWithExtension(dir, extension) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...(await filesWithExtension(fullPath, extension)));
    else if (entry.isFile() && entry.name.toLowerCase().endsWith(extension)) files.push(fullPath);
  }
  return files.sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
}

async function discoverPack() {
  for (const entry of (await readdir(EMBERGEN, { withFileTypes: true })).sort((a, b) =>
    a.name.localeCompare(b.name, "en", { numeric: true }),
  )) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(EMBERGEN, entry.name);
    if ((await filesWithExtension(dir, ".vdb")).length > 0) return dir;
  }
  throw new Error(`no .vdb pack found under ${path.relative(ROOT, EMBERGEN)}/`);
}

function dockerCommand() {
  const candidates = [
    process.env.DOCKER_BIN,
    "docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
  ].filter(Boolean);

  for (const command of candidates) {
    if (spawnSync(command, ["info"], { stdio: "ignore" }).status === 0) return command;
  }
  throw new Error("a running Docker daemon is required (set DOCKER_BIN when docker is not on PATH)");
}

//* Validation =================================================================

async function validateNvdb(file) {
  const buffer = await readFile(file);
  if (buffer.byteLength === 0) throw new Error(`empty conversion output: ${file}`);
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  const parsed = NvdbFile.fromArrayBuffer(arrayBuffer);
  if (parsed.grids.length === 0) throw new Error(`no NanoVDB grids in ${file}`);
  return parsed.grids[0];
}

//* Conversion =================================================================

if (!Number.isFinite(FPS) || FPS <= 0) throw new Error(`SEQUENCE_FPS must be positive, got ${FPS}`);
if (!Number.isInteger(START_INDEX) || START_INDEX < 0)
  throw new Error(`SEQUENCE_START_INDEX must be a non-negative integer, got ${START_INDEX}`);
if (!TEMPORAL)
  throw new Error(
    "refusing to create an animation manifest without SEQUENCE_TEMPORAL=1; " +
      "independent VDB variants such as the Cloud pack are not time sequences",
  );

const requestedPack = process.argv[2];
const pack = requestedPack ? path.resolve(ROOT, requestedPack) : await discoverPack();
const relativePack = path.relative(FIXTURES, pack);
if (relativePack.startsWith("..") || path.isAbsolute(relativePack))
  throw new Error(`pack must be inside ${path.relative(ROOT, FIXTURES)}/`);

const discoveredFrames = await filesWithExtension(pack, ".vdb");
const sourceFrames = discoveredFrames.slice(START_INDEX);
if (sourceFrames.length === 0) throw new Error(`no .vdb frames found in ${path.relative(ROOT, pack)}/`);
const temporalIndices = sourceFrames.map((frame) => {
  const match = path.basename(frame, path.extname(frame)).match(/(\d+)(?!.*\d)/);
  return match ? Number(match[1]) : null;
});
if (
  temporalIndices.some((index) => index === null) ||
  new Set(temporalIndices).size !== temporalIndices.length
)
  throw new Error(
    "source filenames must contain distinct trailing frame numbers; " +
      "this pack appears to contain variants rather than temporal frames",
  );
for (let index = 1; index < temporalIndices.length; index++) {
  if (temporalIndices[index] <= temporalIndices[index - 1])
    throw new Error("source frame numbers must increase monotonically");
}

const docker = dockerCommand();
const outputDir = path.join(pack, "baked-fp8");
await mkdir(outputDir, { recursive: true });

const manifestFrames = [];
for (let index = 0; index < sourceFrames.length; index++) {
  const source = sourceFrames[index];
  const relativeSource = path.relative(pack, source);
  const outputName = `${String(index).padStart(4, "0")}-${path.basename(source, path.extname(source))}.nvdb`;
  const output = path.join(outputDir, outputName);
  const sourceInContainer = `/out/${path.relative(FIXTURES, source)}`;
  const outputInContainer = `/out/${path.relative(FIXTURES, output)}`;

  const existing = await stat(output).catch(() => null);
  if (!existing || existing.size === 0) {
    console.error(`  [${index + 1}/${sourceFrames.length}] ${relativeSource} → ${path.relative(pack, output)}`);
    const args = ["run", "--rm", "-v", `${FIXTURES}:/out`, IMAGE, "nanovdb_convert", "--fp8"];
    if (GRID) args.push("--grid", GRID);
    args.push(sourceInContainer, outputInContainer);
    const result = spawnSync(docker, args, { cwd: ROOT, stdio: "inherit" });
    if (result.status !== 0) throw new Error(`nanovdb_convert failed for ${relativeSource}`);
  }

  await validateNvdb(output);
  manifestFrames.push(path.relative(pack, output).split(path.sep).join("/"));
}

const manifest = {
  kind: "temporal-sequence",
  fps: FPS,
  frames: manifestFrames,
  source: {
    grid: GRID ?? null,
    startIndex: START_INDEX,
    frameNumbers: temporalIndices,
  },
};
const manifestPath = path.join(pack, "manifest.json");
const pendingPath = `${manifestPath}.tmp`;
await writeFile(pendingPath, `${JSON.stringify(manifest, null, 2)}\n`);
await rename(pendingPath, manifestPath);

const manifestIndex = [];
for (const entry of await readdir(EMBERGEN, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const candidate = path.join(EMBERGEN, entry.name, "manifest.json");
  if (await stat(candidate).catch(() => null))
    manifestIndex.push(`${entry.name}/manifest.json`);
}
manifestIndex.sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
const indexPath = path.join(EMBERGEN, "index.json");
await writeFile(indexPath, `${JSON.stringify({ manifests: manifestIndex }, null, 2)}\n`);

console.error(
  `✓ sequence: ${path.relative(ROOT, manifestPath)} (${manifestFrames.length} validated FP8 frames at ${FPS} fps)`,
);
