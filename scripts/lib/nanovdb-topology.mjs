import { createHash } from "node:crypto";
import {
  FLOAT_LAYOUT,
  FP8_LAYOUT,
  GRID_OFF_GRID_CLASS,
  GRID_OFF_GRID_TYPE,
  GRID_SIZE,
  LEAF_OFF_VALUE_MASK,
  LEAF_TABLE_COUNT,
  LOWER_OFF_CHILD_MASK,
  LOWER_TABLE_COUNT,
  NODE_OFF_VALUE_MASK,
  ROOT_OFF_TABLE_SIZE,
  ROOT_TILE_OFF_CHILD,
  ROOT_TILE_OFF_KEY,
  ROOT_TILE_OFF_STATE,
  TREE_OFF_NODE_COUNT_LEAF,
  TREE_OFF_NODE_COUNT_LOWER,
  TREE_OFF_NODE_COUNT_UPPER,
  TREE_OFF_NODE_OFFSET_LEAF,
  TREE_OFF_NODE_OFFSET_LOWER,
  TREE_OFF_NODE_OFFSET_ROOT,
  TREE_OFF_NODE_OFFSET_UPPER,
  TREE_OFF_TILE_COUNT_LOWER,
  TREE_OFF_TILE_COUNT_ROOT,
  TREE_OFF_TILE_COUNT_UPPER,
  TREE_OFF_VOXEL_COUNT,
  UPPER_OFF_CHILD_MASK,
  UPPER_TABLE_COUNT,
} from "../../packages/vdb/dist/tools/nanovdb/bytes.js";

const GRID_TYPE_FLOAT = 1;
const GRID_TYPE_FP8 = 14;
const ROOT_SIZE = 64;
const ROOT_TILE_SIZE = 32;
const ROOT_TILE_VALUE_OFFSET = 20;
const ROOT_STATS_START = 32;
const ROOT_STATS_END = 48;

function activeBit(mask, index) {
  return ((mask[index >>> 5] >>> (index & 31)) & 1) !== 0;
}

function coordinateKey(coordinate) {
  return `${coordinate[0]},${coordinate[1]},${coordinate[2]}`;
}

function compareCoordinates(a, b) {
  if (a[0] !== b[0]) return a[0] - b[0];
  if (a[1] !== b[1]) return a[1] - b[1];
  return a[2] - b[2];
}

function readCoordinate(view, offset) {
  return [
    view.getInt32(offset, true),
    view.getInt32(offset + 4, true),
    view.getInt32(offset + 8, true),
  ];
}

function readMask(view, offset, bitCount) {
  const words = new Uint32Array(bitCount >>> 5);
  for (let index = 0; index < words.length; index++)
    words[index] = view.getUint32(offset + index * 4, true);
  return words;
}

function maskHex(mask) {
  return Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength).toString("hex");
}

function signExtend20(value) {
  const masked = value & 0x000f_ffff;
  return masked & 0x0008_0000 ? masked - 0x0010_0000 : masked;
}

function rootOriginFromKey(key) {
  const low = Number(key & 0xffff_ffffn) >>> 0;
  const high = Number((key >> 32n) & 0xffff_ffffn) >>> 0;
  const z = signExtend20(low);
  const y = signExtend20(((low >>> 21) | ((high & 0x1ff) << 11)) >>> 0);
  const x = signExtend20(high >>> 10);
  return [x * 4096, y * 4096, z * 4096];
}

function childOrigin(parentOrigin, slot, axisBits, childSize) {
  const axisMask = (1 << axisBits) - 1;
  return [
    parentOrigin[0] + ((slot >>> (axisBits * 2)) & axisMask) * childSize,
    parentOrigin[1] + ((slot >>> axisBits) & axisMask) * childSize,
    parentOrigin[2] + (slot & axisMask) * childSize,
  ];
}

function maskActiveCount(mask) {
  let count = 0;
  for (const original of mask) {
    let word = original;
    while (word !== 0) {
      word &= word - 1;
      count++;
    }
  }
  return count;
}

function activeNonChildCount(valueMask, childMask) {
  let count = 0;
  for (let index = 0; index < valueMask.length; index++) {
    let word = valueMask[index] & ~childMask[index];
    while (word !== 0) {
      word &= word - 1;
      count++;
    }
  }
  return count;
}

function assertRange(address, size, start, end, label) {
  if (address < start || address + size > end || (address - start) % size !== 0)
    throw new Error(`${label} address ${address} is outside its declared node section`);
}

function canonicalNode(node) {
  return `${coordinateKey(node.origin)}:${maskHex(node.valueMask)}:${maskHex(node.childMask ?? new Uint32Array())}`;
}

function canonicalTopology(topology) {
  const lines = [
    JSON.stringify({
      gridType: topology.gridType,
      gridClass: topology.gridClass,
      voxelCount: topology.voxelCount,
      nodeCounts: topology.nodeCounts,
      tileCounts: topology.tileCounts,
    }),
  ];
  for (const tile of topology.rootTiles)
    lines.push(`root:${tile.key.toString(16)}:${coordinateKey(tile.origin)}:${tile.state}:${tile.hasChild}`);
  for (const node of topology.upperNodes) lines.push(`upper:${canonicalNode(node)}`);
  for (const node of topology.lowerNodes) lines.push(`lower:${canonicalNode(node)}`);
  for (const node of topology.leaves) lines.push(`leaf:${canonicalNode(node)}`);
  return lines;
}

export function analyzeNanoVdbTopology(image) {
  const view = new DataView(image.buffer, image.byteOffset, image.byteLength);
  const gridType = view.getUint32(GRID_OFF_GRID_TYPE, true);
  const layout =
    gridType === GRID_TYPE_FLOAT
      ? FLOAT_LAYOUT
      : gridType === GRID_TYPE_FP8
        ? FP8_LAYOUT
        : undefined;
  if (!layout) throw new Error(`strict parity only supports FLOAT/Fp8, got grid type ${gridType}`);

  const tree = GRID_SIZE;
  const leafStart = tree + Number(view.getBigUint64(tree + TREE_OFF_NODE_OFFSET_LEAF, true));
  const lowerStart = tree + Number(view.getBigUint64(tree + TREE_OFF_NODE_OFFSET_LOWER, true));
  const upperStart = tree + Number(view.getBigUint64(tree + TREE_OFF_NODE_OFFSET_UPPER, true));
  const rootStart = tree + Number(view.getBigUint64(tree + TREE_OFF_NODE_OFFSET_ROOT, true));
  const nodeCounts = {
    leaf: view.getUint32(tree + TREE_OFF_NODE_COUNT_LEAF, true),
    lower: view.getUint32(tree + TREE_OFF_NODE_COUNT_LOWER, true),
    upper: view.getUint32(tree + TREE_OFF_NODE_COUNT_UPPER, true),
  };
  const tileCounts = {
    lower: view.getUint32(tree + TREE_OFF_TILE_COUNT_LOWER, true),
    upper: view.getUint32(tree + TREE_OFF_TILE_COUNT_UPPER, true),
    root: view.getUint32(tree + TREE_OFF_TILE_COUNT_ROOT, true),
  };
  const rootTableSize = view.getUint32(rootStart + ROOT_OFF_TABLE_SIZE, true);
  const rootTiles = [];
  const upperByAddress = new Map();
  const lowerByAddress = new Map();
  const leafByAddress = new Map();
  const failures = [];

  const visitLeaf = (address, origin) => {
    assertRange(address, layout.leafSize, leafStart, image.byteLength, "leaf");
    const existing = leafByAddress.get(address);
    if (existing) {
      if (coordinateKey(existing.origin) !== coordinateKey(origin))
        failures.push(`leaf ${address} reached with conflicting origins`);
      return;
    }
    leafByAddress.set(address, {
      address,
      origin,
      valueMask: readMask(view, address + LEAF_OFF_VALUE_MASK, LEAF_TABLE_COUNT),
    });
  };

  const visitLower = (address, origin) => {
    assertRange(address, layout.lowerSize, lowerStart, leafStart, "lower");
    const existing = lowerByAddress.get(address);
    if (existing) {
      if (coordinateKey(existing.origin) !== coordinateKey(origin))
        failures.push(`lower ${address} reached with conflicting origins`);
      return;
    }
    const valueMask = readMask(view, address + NODE_OFF_VALUE_MASK, LOWER_TABLE_COUNT);
    const childMask = readMask(view, address + LOWER_OFF_CHILD_MASK, LOWER_TABLE_COUNT);
    const node = { address, origin, valueMask, childMask };
    lowerByAddress.set(address, node);
    for (let slot = 0; slot < LOWER_TABLE_COUNT; slot++) {
      if (!activeBit(childMask, slot)) continue;
      const relative = Number(
        view.getBigInt64(address + layout.lowerOffTable + slot * layout.tableStride, true),
      );
      visitLeaf(address + relative, childOrigin(origin, slot, 4, 8));
    }
  };

  const visitUpper = (address, origin) => {
    assertRange(address, layout.upperSize, upperStart, lowerStart, "upper");
    const existing = upperByAddress.get(address);
    if (existing) {
      if (coordinateKey(existing.origin) !== coordinateKey(origin))
        failures.push(`upper ${address} reached with conflicting origins`);
      return;
    }
    const valueMask = readMask(view, address + NODE_OFF_VALUE_MASK, UPPER_TABLE_COUNT);
    const childMask = readMask(view, address + UPPER_OFF_CHILD_MASK, UPPER_TABLE_COUNT);
    const node = { address, origin, valueMask, childMask };
    upperByAddress.set(address, node);
    for (let slot = 0; slot < UPPER_TABLE_COUNT; slot++) {
      if (!activeBit(childMask, slot)) continue;
      const relative = Number(
        view.getBigInt64(address + layout.upperOffTable + slot * layout.tableStride, true),
      );
      visitLower(address + relative, childOrigin(origin, slot, 5, 128));
    }
  };

  for (let index = 0; index < rootTableSize; index++) {
    const address = rootStart + layout.rootSize + index * layout.rootTileSize;
    const key = view.getBigUint64(address + ROOT_TILE_OFF_KEY, true);
    const childOffset = Number(view.getBigInt64(address + ROOT_TILE_OFF_CHILD, true));
    const state = view.getUint32(address + ROOT_TILE_OFF_STATE, true);
    const origin = rootOriginFromKey(key);
    const tile = { address, key, origin, state, hasChild: childOffset !== 0 };
    rootTiles.push(tile);
    if (tile.hasChild) visitUpper(rootStart + childOffset, origin);
  }

  const upperNodes = [...upperByAddress.values()].sort((a, b) =>
    compareCoordinates(a.origin, b.origin),
  );
  const lowerNodes = [...lowerByAddress.values()].sort((a, b) =>
    compareCoordinates(a.origin, b.origin),
  );
  const leaves = [...leafByAddress.values()].sort((a, b) =>
    compareCoordinates(a.origin, b.origin),
  );
  rootTiles.sort((a, b) => compareCoordinates(a.origin, b.origin));

  if (upperNodes.length !== nodeCounts.upper)
    failures.push(`reachable upper nodes ${upperNodes.length} != declared ${nodeCounts.upper}`);
  if (lowerNodes.length !== nodeCounts.lower)
    failures.push(`reachable lower nodes ${lowerNodes.length} != declared ${nodeCounts.lower}`);
  if (leaves.length !== nodeCounts.leaf)
    failures.push(`reachable leaves ${leaves.length} != declared ${nodeCounts.leaf}`);

  const activeTileCount =
    rootTiles.filter((tile) => !tile.hasChild && tile.state !== 0).length +
    upperNodes.reduce((sum, node) => sum + activeNonChildCount(node.valueMask, node.childMask), 0) +
    lowerNodes.reduce((sum, node) => sum + activeNonChildCount(node.valueMask, node.childMask), 0);
  if (activeTileCount !== 0 || Object.values(tileCounts).some((count) => count !== 0))
    failures.push("active internal/root tiles remain after normalization");

  const topology = {
    image,
    view,
    layout,
    gridType,
    gridClass: view.getUint32(GRID_OFF_GRID_CLASS, true),
    voxelCount: Number(view.getBigUint64(tree + TREE_OFF_VOXEL_COUNT, true)),
    nodeCounts,
    tileCounts,
    activeTileCount,
    rootStart,
    upperStart,
    lowerStart,
    leafStart,
    rootTiles,
    upperNodes,
    lowerNodes,
    leaves,
    upperByAddress,
    lowerByAddress,
    leafByAddress,
    failures,
  };
  const canonical = canonicalTopology(topology);
  topology.canonical = canonical;
  topology.digest = createHash("sha256").update(canonical.join("\n")).digest("hex");
  topology.activeVoxelCount = leaves.reduce(
    (sum, leaf) => sum + maskActiveCount(leaf.valueMask),
    0,
  );
  if (topology.activeVoxelCount !== topology.voxelCount)
    failures.push(
      `active leaf coordinates ${topology.activeVoxelCount} != declared voxel count ${topology.voxelCount}`,
    );
  return topology;
}

export function compareNanoVdbTopology(nativeTopology, typescriptTopology) {
  const nativeLines = nativeTopology.canonical;
  const typescriptLines = typescriptTopology.canonical;
  let firstMismatch;
  const lineCount = Math.max(nativeLines.length, typescriptLines.length);
  for (let index = 0; index < lineCount; index++) {
    if (nativeLines[index] === typescriptLines[index]) continue;
    firstMismatch = {
      index,
      native: nativeLines[index] ?? null,
      typescript: typescriptLines[index] ?? null,
    };
    break;
  }
  const physicalOrderMatches =
    ["upperNodes", "lowerNodes", "leaves"].every((name) => {
      const nativeNodes = nativeTopology[name];
      const typescriptNodes = typescriptTopology[name];
      return (
        nativeNodes.length === typescriptNodes.length &&
        nativeNodes.every(
          (node, index) =>
            coordinateKey(node.origin) === coordinateKey(typescriptNodes[index].origin) &&
            node.address === typescriptNodes[index].address,
        )
      );
    }) &&
    nativeTopology.rootTiles.every(
      (tile, index) =>
        coordinateKey(tile.origin) ===
          coordinateKey(typescriptTopology.rootTiles[index]?.origin ?? []) &&
        tile.address === typescriptTopology.rootTiles[index]?.address,
    );
  return {
    passed:
      nativeTopology.failures.length === 0 &&
      typescriptTopology.failures.length === 0 &&
      firstMismatch === undefined,
    nativeDigest: nativeTopology.digest,
    typescriptDigest: typescriptTopology.digest,
    physicalOrderMatches,
    ...(firstMismatch ? { firstMismatch } : {}),
    nativeFailures: nativeTopology.failures,
    typescriptFailures: typescriptTopology.failures,
  };
}

export function* activeCoordinates(topology) {
  for (const leaf of topology.leaves) {
    for (let index = 0; index < LEAF_TABLE_COUNT; index++) {
      if (!activeBit(leaf.valueMask, index)) continue;
      yield [
        leaf.origin[0] + ((index >>> 6) & 7),
        leaf.origin[1] + ((index >>> 3) & 7),
        leaf.origin[2] + (index & 7),
      ];
    }
  }
}

export function compareCoordinate(a, b) {
  return compareCoordinates(a, b);
}

function nodeAtOffset(offset, nodesByAddress, start, size) {
  const index = Math.floor((offset - start) / size);
  if (index < 0) return undefined;
  const address = start + index * size;
  return nodesByAddress.get(address);
}

function fieldForOffset(topology, offset, mode) {
  if (offset >= 8 && offset < 16) return "grid-checksum";
  if (offset >= topology.rootStart + ROOT_STATS_START && offset < topology.rootStart + ROOT_STATS_END)
    return "derived-root-statistic";

  for (const tile of topology.rootTiles) {
    if (
      tile.hasChild &&
      offset >= tile.address + ROOT_TILE_VALUE_OFFSET &&
      offset < tile.address + ROOT_TILE_VALUE_OFFSET + 4
    )
      return "unused-root-child-value";
  }

  const internalField = (nodesByAddress, start, size, tableOffset, statsStart, statsEnd) => {
    const node = nodeAtOffset(offset, nodesByAddress, start, size);
    if (!node) return undefined;
    const local = offset - node.address;
    if (local >= statsStart && local < statsEnd) return "derived-internal-statistic";
    if (local < tableOffset) return undefined;
    const slot = Math.floor((local - tableOffset) / topology.layout.tableStride);
    if (slot < 0 || slot >= node.valueMask.length * 32) return undefined;
    if (activeBit(node.childMask, slot)) return undefined;
    if (activeBit(node.valueMask, slot)) return undefined;
    return "inactive-internal-value";
  };
  const upperField = internalField(
    topology.upperByAddress,
    topology.upperStart,
    topology.layout.upperSize,
    topology.layout.upperOffTable,
    topology.layout.upperOffMin,
    topology.layout.upperOffStdDev + 4,
  );
  if (upperField) return upperField;
  const lowerField = internalField(
    topology.lowerByAddress,
    topology.lowerStart,
    topology.layout.lowerSize,
    topology.layout.lowerOffTable,
    topology.layout.lowerOffMin,
    topology.layout.lowerOffStdDev + 4,
  );
  if (lowerField) return lowerField;

  const leaf = nodeAtOffset(
    offset,
    topology.leafByAddress,
    topology.leafStart,
    topology.layout.leafSize,
  );
  if (!leaf) return undefined;
  const local = offset - leaf.address;
  if (mode === "float") {
    if (local >= topology.layout.leafOffMin && local < topology.layout.leafOffStdDev + 4)
      return "derived-leaf-statistic";
    if (local < topology.layout.leafOffTable) return undefined;
    const slot = Math.floor((local - topology.layout.leafOffTable) / 4);
    if (slot < 0 || slot >= LEAF_TABLE_COUNT || activeBit(leaf.valueMask, slot)) return undefined;
    return "inactive-leaf-value";
  }

  if (local >= 80 && local < 88) return "fp8-quantization-parameter";
  if (local >= 88 && local < 96) return "derived-leaf-statistic";
  if (local < topology.layout.leafOffTable) return undefined;
  const slot = local - topology.layout.leafOffTable;
  if (slot < 0 || slot >= LEAF_TABLE_COUNT) return undefined;
  return activeBit(leaf.valueMask, slot) ? "fp8-active-code" : "inactive-leaf-value";
}

export function classifyNanoVdbByteDifferences(
  nativeImage,
  typescriptImage,
  nativeTopology,
  typescriptTopology,
  { mode, semanticPassed, topologyComparison },
) {
  const nativeBytes = new Uint8Array(
    nativeImage.buffer,
    nativeImage.byteOffset,
    nativeImage.byteLength,
  );
  const typescriptBytes = new Uint8Array(
    typescriptImage.buffer,
    typescriptImage.byteOffset,
    typescriptImage.byteLength,
  );
  const classifications = new Map();
  const unclassifiedOffsets = [];
  let unclassifiedCount = 0;
  let differingBytes = 0;

  const record = (field, offset) => {
    let entry = classifications.get(field);
    if (!entry) {
      entry = { classification: field, differingBytes: 0, firstOffsets: [] };
      classifications.set(field, entry);
    }
    entry.differingBytes++;
    if (entry.firstOffsets.length < 32) entry.firstOffsets.push(offset);
  };

  if (
    nativeBytes.byteLength !== typescriptBytes.byteLength ||
    !topologyComparison.passed ||
    !topologyComparison.physicalOrderMatches
  ) {
    return {
      differingBytes: Math.max(nativeBytes.byteLength, typescriptBytes.byteLength),
      classifications: [],
      unclassifiedByteDifferences: [
        {
          classification: "unclassified-structural-byte-difference",
          reason: "grid lengths, canonical topology, or physical node order differ",
        },
      ],
    };
  }

  const offsetHash = createHash("sha256");
  for (let offset = 0; offset < nativeBytes.byteLength; offset++) {
    if (nativeBytes[offset] === typescriptBytes[offset]) continue;
    differingBytes++;
    const encodedOffset = Buffer.allocUnsafe(8);
    encodedOffset.writeBigUInt64LE(BigInt(offset));
    offsetHash.update(encodedOffset);
    const nativeField = fieldForOffset(nativeTopology, offset, mode);
    const typescriptField = fieldForOffset(typescriptTopology, offset, mode);
    const field = nativeField === typescriptField ? nativeField : undefined;
    const semanticField = field === "fp8-active-code" || field === "fp8-quantization-parameter";
    if (!field || (semanticField && !semanticPassed)) {
      unclassifiedCount++;
      if (unclassifiedOffsets.length < 64) unclassifiedOffsets.push(offset);
      continue;
    }
    record(field, offset);
  }

  return {
    differingBytes,
    differingOffsetSha256: offsetHash.digest("hex"),
    classifications: [...classifications.values()],
    unclassifiedByteDifferences:
      unclassifiedOffsets.length === 0
        ? []
        : [
            {
              classification: "unclassified-structural-byte-difference",
              differingBytes: unclassifiedCount,
              firstOffsets: unclassifiedOffsets,
            },
          ],
  };
}
