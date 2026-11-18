#!/usr/bin/env node
// Compare pinned native nanovdb_convert output against the pure-TypeScript
// parseVdb -> buildFromVdb -> quantize pipeline for the four sample models.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { rmSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { NanoVDBFile, readValue } from "../packages/vdb/dist/tools/index.js";
import { readLeafQuantum } from "../packages/vdb/dist/tools/cpu/read-value.js";
import {
  buildFromVdb,
  parseVdb,
  quantize,
  writeNvdb,
} from "../packages/vdb/dist/tools/index.js";
import { createParityAccumulator, overallPassed } from "./lib/parity-report.mjs";
import {
  activeCoordinates,
  analyzeNanoVdbTopology,
  classifyNanoVdbByteDifferences,
  compareCoordinate,
  compareNanoVdbTopology,
} from "./lib/nanovdb-topology.mjs";
import { VDB_SAMPLES, VDB_SAMPLES_COMMIT } from "./lib/vdb-samples.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = path.join(ROOT, "fixtures");
const SOURCES = path.join(FIXTURES, "vdb-samples");
const OUTPUT = path.join(FIXTURES, "parity");
const IMAGE = process.env.FIXTURE_BAKE_IMAGE ?? "webvdb-fixture-bake";
const SAMPLES = Object.keys(VDB_SAMPLES).map((name) => path.basename(name, ".vdb"));

//* Native conversion ==========================================================

function dockerCommand() {
  for (const command of [
    process.env.DOCKER_BIN,
    "docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
  ].filter(Boolean)) {
    if (spawnSync(command, ["info"], { stdio: "ignore" }).status === 0) return command;
  }
  throw new Error("a running Docker daemon is required for native parity");
}

function normalizeNative(docker, sample) {
  const input = `/out/vdb-samples/${sample}.vdb`;
  const output = `/out/parity/${sample}.normalized.vdb`;
  const result = spawnSync(
    docker,
    ["run", "--rm", "-v", `${FIXTURES}:/out`, IMAGE, "normalize_vdb", input, output],
    { cwd: ROOT, stdio: "inherit" },
  );
  if (result.status !== 0) throw new Error(`native VDB normalization failed for ${sample}`);
}

function convertNative(docker, sample, mode) {
  const input = `/out/parity/${sample}.normalized.vdb`;
  const output = `/out/parity/${sample}.native.${mode}.nvdb`;
  rmSync(path.join(OUTPUT, `${sample}.native.${mode}.nvdb`), { force: true });
  const args = mode === "fp8" ? ["--fp8", input, output] : [input, output];
  const result = spawnSync(
    docker,
    ["run", "--rm", "-v", `${FIXTURES}:/out`, IMAGE, "nanovdb_convert", ...args],
    { cwd: ROOT, stdio: "inherit" },
  );
  if (result.status !== 0) throw new Error(`native ${mode} conversion failed for ${sample}`);
}

async function loadGridImage(file) {
  const bytes = await readFile(file);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return NanoVDBFile.fromArrayBuffer(buffer).gridImage(0);
}

//* Strict topology + active-coordinate union =================================

function addComparison(accumulator, coordinate, nativeImage, typescriptImage, numericBound) {
  const native = readValue(nativeImage, coordinate);
  const typescript = readValue(typescriptImage, coordinate);
  accumulator.add({ coordinate, native, typescript, numericBound });
}

function fp8NumericBound(coordinate, nativeImage, typescriptImage) {
  const nativeQuantum = readLeafQuantum(nativeImage, coordinate);
  const typescriptQuantum = readLeafQuantum(typescriptImage, coordinate);
  const native = readValue(nativeImage, coordinate);
  const typescript = readValue(typescriptImage, coordinate);
  const float32Roundoff =
    Math.max(Math.abs(native.value), Math.abs(typescript.value), 1) * 2 ** -22;
  return (nativeQuantum + typescriptQuantum) / 2 + float32Roundoff;
}

function compareActiveUnion(
  accumulator,
  mode,
  nativeImage,
  typescriptImage,
  nativeTopology,
  typescriptTopology,
) {
  const nativeIterator = activeCoordinates(nativeTopology);
  const typescriptIterator = activeCoordinates(typescriptTopology);
  let nativeNext = nativeIterator.next();
  let typescriptNext = typescriptIterator.next();
  let unionCoordinates = 0;
  let nativeOnly = 0;
  let typescriptOnly = 0;

  while (!nativeNext.done || !typescriptNext.done) {
    let coordinate;
    if (nativeNext.done) {
      coordinate = typescriptNext.value;
      typescriptOnly++;
      typescriptNext = typescriptIterator.next();
    } else if (typescriptNext.done) {
      coordinate = nativeNext.value;
      nativeOnly++;
      nativeNext = nativeIterator.next();
    } else {
      const ordering = compareCoordinate(nativeNext.value, typescriptNext.value);
      if (ordering === 0) {
        coordinate = nativeNext.value;
        nativeNext = nativeIterator.next();
        typescriptNext = typescriptIterator.next();
      } else if (ordering < 0) {
        coordinate = nativeNext.value;
        nativeOnly++;
        nativeNext = nativeIterator.next();
      } else {
        coordinate = typescriptNext.value;
        typescriptOnly++;
        typescriptNext = typescriptIterator.next();
      }
    }
    unionCoordinates++;
    addComparison(
      accumulator,
      coordinate,
      nativeImage,
      typescriptImage,
      mode === "float" ? 0 : fp8NumericBound(coordinate, nativeImage, typescriptImage),
    );
  }

  return {
    nativeActiveCoordinates: nativeTopology.activeVoxelCount,
    typescriptActiveCoordinates: typescriptTopology.activeVoxelCount,
    unionCoordinates,
    nativeOnly,
    typescriptOnly,
  };
}

function strictReport(sample, mode, nativeImage, typescriptImage) {
  const nativeTopology = analyzeNanoVdbTopology(nativeImage);
  const typescriptTopology = analyzeNanoVdbTopology(typescriptImage);
  const topology = compareNanoVdbTopology(nativeTopology, typescriptTopology);
  const accumulator = createParityAccumulator(sample, mode, []);
  const activeUnion = compareActiveUnion(
    accumulator,
    mode,
    nativeImage,
    typescriptImage,
    nativeTopology,
    typescriptTopology,
  );
  const report = accumulator.finish();
  const byteEvidence = classifyNanoVdbByteDifferences(
    nativeImage,
    typescriptImage,
    nativeTopology,
    typescriptTopology,
    { mode, semanticPassed: report.passed, topologyComparison: topology },
  );
  report.topology = {
    ...topology,
    nodeCounts: nativeTopology.nodeCounts,
    tileCounts: nativeTopology.tileCounts,
  };
  report.activeUnion = activeUnion;
  report.byteDifferences = byteEvidence.classifications;
  report.byteDifferenceCount = byteEvidence.differingBytes;
  if (byteEvidence.differingOffsetSha256)
    report.differingOffsetSha256 = byteEvidence.differingOffsetSha256;
  report.unclassifiedByteDifferences = byteEvidence.unclassifiedByteDifferences;
  report.classifiedSemanticDifferences = [];
  report.passed =
    report.passed &&
    topology.passed &&
    activeUnion.nativeOnly === 0 &&
    activeUnion.typescriptOnly === 0 &&
    byteEvidence.unclassifiedByteDifferences.length === 0;
  return report;
}

async function runSample(docker, sample) {
  const sampleName = `${sample}.vdb`;
  const expected = VDB_SAMPLES[sampleName];
  if (!expected) throw new Error(`missing pinned source metadata for ${sampleName}`);
  const originalSource = await readFile(path.join(SOURCES, sampleName));
  const sourceSha256 = createHash("sha256").update(originalSource).digest("hex");
  if (originalSource.byteLength !== expected.bytes || sourceSha256 !== expected.sha256)
    throw new Error(
      `${sampleName} does not match the pinned fixture revision: ` +
        `${originalSource.byteLength} bytes / sha256 ${sourceSha256}`,
    );

  normalizeNative(docker, sample);
  const sourceBytes = await readFile(path.join(OUTPUT, `${sample}.normalized.vdb`));
  const sourceBuffer = sourceBytes.buffer.slice(
    sourceBytes.byteOffset,
    sourceBytes.byteOffset + sourceBytes.byteLength,
  );
  const parsed = parseVdb(sourceBuffer);
  const sourceGrid = parsed.grids.find((grid) => grid.gridType === "Tree_float_5_4_3") ?? parsed.grids[0];
  if (!sourceGrid) throw new Error(`${sample}.vdb contains no grid`);

  const typescriptFloat = buildFromVdb(sourceGrid);
  const typescriptFp8 = quantize(typescriptFloat, "fp8");
  await writeFile(path.join(OUTPUT, `${sample}.typescript.float.nvdb`), Buffer.from(writeNvdb([typescriptFloat])));
  await writeFile(path.join(OUTPUT, `${sample}.typescript.fp8.nvdb`), Buffer.from(writeNvdb([typescriptFp8])));

  convertNative(docker, sample, "float");
  convertNative(docker, sample, "fp8");
  const nativeFloat = await loadGridImage(path.join(OUTPUT, `${sample}.native.float.nvdb`));
  const nativeFp8 = await loadGridImage(path.join(OUTPUT, `${sample}.native.fp8.nvdb`));

  return {
    source: {
      file: `fixtures/vdb-samples/${sampleName}`,
      bytes: originalSource.byteLength,
      sha256: sourceSha256,
      commit: VDB_SAMPLES_COMMIT,
    },
    reports: [
      strictReport(sample, "float", nativeFloat, typescriptFloat),
      strictReport(sample, "fp8", nativeFp8, typescriptFp8),
    ],
  };
}

//* Run ========================================================================

await mkdir(OUTPUT, { recursive: true });
const docker = dockerCommand();
const reports = [];
const sources = {};
for (const sample of SAMPLES) {
  console.error(`• parity: ${sample}`);
  const result = await runSample(docker, sample);
  sources[sample] = result.source;
  reports.push(...result.reports);
}

const result = {
  generatedAt: new Date().toISOString(),
  command: "pnpm parity:native",
  sources,
  samples: reports,
  passed: overallPassed(reports),
};
const reportPath = path.join(OUTPUT, "report.json");
await writeFile(reportPath, `${JSON.stringify(result, null, 2)}\n`);
console.error(`✓ report: ${path.relative(ROOT, reportPath)} (${result.passed ? "PASS" : "FAIL"})`);
if (!result.passed) process.exitCode = 1;
