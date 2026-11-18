import { describe, expect, it } from "vitest";
import {
  activeCoordinates,
  analyzeNanoVdbTopology,
  classifyNanoVdbByteDifferences,
  compareNanoVdbTopology,
} from "../../../scripts/lib/nanovdb-topology.mjs";
import { buildFromDense } from "../src/index.js";

function fixtureImage() {
  const values = new Float32Array(16 * 16 * 16);
  values[(1 * 16 + 2) * 16 + 3] = 1;
  values[(12 * 16 + 11) * 16 + 10] = 2;
  return buildFromDense(values, [16, 16, 16], { gridName: "strict-parity-test" });
}

describe("strict NanoVDB topology evidence", () => {
  it("enumerates active coordinates from the NanoVDB leaf masks", () => {
    const topology = analyzeNanoVdbTopology(fixtureImage());
    expect(topology.failures).toEqual([]);
    expect([...activeCoordinates(topology)]).toEqual([
      [1, 2, 3],
      [12, 11, 10],
    ]);
  });

  it("rejects a native-only or TypeScript-only active mask bit", () => {
    const original = fixtureImage();
    const changed = original.slice();
    const originalTopology = analyzeNanoVdbTopology(original);
    const firstLeaf = originalTopology.leaves[0];
    const view = new DataView(changed.buffer, changed.byteOffset, changed.byteLength);
    const maskOffset = firstLeaf.address + 16;
    view.setUint32(maskOffset, view.getUint32(maskOffset, true) ^ 1, true);

    const changedTopology = analyzeNanoVdbTopology(changed);
    expect(compareNanoVdbTopology(originalTopology, changedTopology).passed).toBe(false);
    expect([...activeCoordinates(changedTopology)]).not.toEqual([
      ...activeCoordinates(originalTopology),
    ]);
  });

  it("classifies only exact allowed fields and rejects unknown offsets", () => {
    const original = fixtureImage();
    const topology = analyzeNanoVdbTopology(original);

    const checksumChanged = original.slice();
    new Uint8Array(checksumChanged.buffer)[8] ^= 1;
    const checksumTopology = analyzeNanoVdbTopology(checksumChanged);
    const checksumEvidence = classifyNanoVdbByteDifferences(
      original,
      checksumChanged,
      topology,
      checksumTopology,
      {
        mode: "float",
        semanticPassed: true,
        topologyComparison: compareNanoVdbTopology(topology, checksumTopology),
      },
    );
    expect(checksumEvidence.classifications.map((item) => item.classification)).toEqual([
      "grid-checksum",
    ]);
    expect(checksumEvidence.unclassifiedByteDifferences).toEqual([]);

    const unknownChanged = original.slice();
    new Uint8Array(unknownChanged.buffer)[40] ^= 1;
    const unknownTopology = analyzeNanoVdbTopology(unknownChanged);
    const unknownEvidence = classifyNanoVdbByteDifferences(
      original,
      unknownChanged,
      topology,
      unknownTopology,
      {
        mode: "float",
        semanticPassed: true,
        topologyComparison: compareNanoVdbTopology(topology, unknownTopology),
      },
    );
    expect(unknownEvidence.unclassifiedByteDifferences).toHaveLength(1);
  });

  it("maps inactive leaf payload bytes through the proven inactive mask", () => {
    const original = fixtureImage();
    const changed = original.slice();
    const topology = analyzeNanoVdbTopology(original);
    const leaf = topology.leaves[0];
    const inactiveIndex = [...Array(512).keys()].find(
      (index) => ((leaf.valueMask[index >>> 5] >>> (index & 31)) & 1) === 0,
    );
    expect(inactiveIndex).toBeDefined();
    new Uint8Array(changed.buffer)[leaf.address + 96 + inactiveIndex * 4] ^= 1;

    const changedTopology = analyzeNanoVdbTopology(changed);
    const evidence = classifyNanoVdbByteDifferences(
      original,
      changed,
      topology,
      changedTopology,
      {
        mode: "float",
        semanticPassed: true,
        topologyComparison: compareNanoVdbTopology(topology, changedTopology),
      },
    );
    expect(evidence.classifications.map((item) => item.classification)).toEqual([
      "inactive-leaf-value",
    ]);
    expect(evidence.unclassifiedByteDifferences).toEqual([]);
  });
});
