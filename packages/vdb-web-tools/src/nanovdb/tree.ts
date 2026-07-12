/**
 * tree.ts — dense array -> sparse NanoVDB node hierarchy (topology, masks,
 * per-node stats and bbox). Pure data; no bytes are written here. `serialize.ts`
 * turns the {@link BuiltTree} into a laid-out grid image.
 *
 * ## Topology policy (T3 decision, v1)
 *
 * A leaf (8^3 block) is materialised iff the block contains at least one active
 * voxel. Every materialised leaf stores the *source value* for all 512 of its
 * slots (active and inactive), so any voxel inside a materialised leaf round-
 * trips exactly. A block with no active voxel collapses to an inactive
 * background tile in its parent lower node — so empty space stays sparse.
 *
 * This builder deliberately does NOT collapse uniform *active* regions into
 * active tiles (native `nanovdb_convert` does). Consequences:
 *  - correctness is unaffected — `readValue` returns identical values;
 *  - every active voxel lives in a leaf, so stats aggregate as a plain sum of
 *    leaf accumulators (no tile-weighting needed);
 *  - the image is larger than native's for big uniform interiors. Acceptable
 *    for the desktop-scale grids v1 targets; active-tile compression is a
 *    documented follow-up alongside quantization.
 *
 * ## Active predicate
 *
 * Default (no `activeThreshold`): a voxel is active iff `value !== background`
 * (exact). With `activeThreshold` given: active iff
 * `|value - background| > activeThreshold`. Note threshold 0 is equivalent to
 * the exact rule. Under a non-zero threshold, a block whose every voxel is
 * inactive collapses to background even if some of those voxels differ from the
 * background value — those sub-threshold values are not preserved. This is
 * documented and intended (topology follows activity).
 */

import {
  leafCoordToOffset,
  lowerCoordToOffset,
  upperCoordToOffset,
  LEAF_TABLE_COUNT,
} from "./bytes.js";
import { StatsAccumulator } from "./stats.js";

export interface Leaf {
  /** 8-aligned index-space min corner. */
  origin: [number, number, number];
  /** All 512 voxel values in leaf-offset order. */
  values: Float32Array;
  /** 512-bit active mask as 16 u32 words. */
  valueMask: Uint32Array;
  stats: StatsAccumulator;
  memIndex: number;
}

export interface LowerNode {
  /** 128-aligned index-space min corner. */
  origin: [number, number, number];
  /** slot (0..4095) -> child leaf. Slots not present are background tiles. */
  children: Map<number, Leaf>;
  stats: StatsAccumulator;
  memIndex: number;
}

export interface UpperNode {
  /** 4096-aligned index-space min corner. */
  origin: [number, number, number];
  /** slot (0..32767) -> child lower node. */
  children: Map<number, LowerNode>;
  stats: StatsAccumulator;
  memIndex: number;
}

export interface RootTile {
  key: bigint;
  child: UpperNode;
}

export interface BuiltTree {
  uppers: UpperNode[];
  lowers: LowerNode[];
  leaves: Leaf[];
  rootTiles: RootTile[];
  /** Global active-value stats + tight index bbox (== root stats/bbox). */
  root: StatsAccumulator;
  activeVoxelCount: number;
}

export interface BuildTreeOptions {
  origin: [number, number, number];
  background: number;
  activeThreshold: number | undefined;
}

function floorDiv(a: number, b: number): number {
  return Math.floor(a / b);
}

/** Deterministic `"x,y,z"` origin key used to dedupe/lookup nodes by coord. */
function leafKey(lx: number, ly: number, lz: number): string {
  return `${lx},${ly},${lz}`;
}

/** Aligns coord down to a power-of-two block size (handles negatives). */
function alignDown(v: number, block: number): number {
  return floorDiv(v, block) * block;
}

function makeIsActive(background: number, threshold: number | undefined): (v: number) => boolean {
  if (threshold === undefined) return (v) => v !== background;
  return (v) => Math.abs(v - background) > threshold;
}

/**
 * Builds the sparse hierarchy for a dense `values` array of shape
 * `[nx,ny,nz]` (x-major: idx = (x*ny + y)*nz + z), placed with its min corner at
 * index-space `origin`.
 */
export function buildTree(
  values: Float32Array,
  dims: [number, number, number],
  opts: BuildTreeOptions,
): BuiltTree {
  const [nx, ny, nz] = dims;
  const [ox, oy, oz] = opts.origin;
  const isActive = makeIsActive(opts.background, opts.activeThreshold);

  const denseIndex = (x: number, y: number, z: number): number => (x * ny + y) * nz + z;

  // -- Pass 1: discover leaves (blocks with >=1 active voxel) ---------------
  const leafMap = new Map<string, Leaf>();

  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      const base = (x * ny + y) * nz;
      for (let z = 0; z < nz; z++) {
        if (!isActive(values[base + z]!)) continue;
        const ix = ox + x;
        const iy = oy + y;
        const iz = oz + z;
        const lx = alignDown(ix, 8);
        const ly = alignDown(iy, 8);
        const lz = alignDown(iz, 8);
        const k = leafKey(lx, ly, lz);
        if (!leafMap.has(k)) {
          leafMap.set(k, {
            origin: [lx, ly, lz],
            values: new Float32Array(LEAF_TABLE_COUNT),
            valueMask: new Uint32Array(16),
            stats: new StatsAccumulator(),
            memIndex: -1,
          });
        }
      }
    }
  }

  // -- Pass 2: fill each leaf's 512 values + mask + stats -------------------
  for (const leaf of leafMap.values()) {
    const [lx, ly, lz] = leaf.origin;
    for (let dx = 0; dx < 8; dx++) {
      const ix = lx + dx;
      const sx = ix - ox; // dense x
      for (let dy = 0; dy < 8; dy++) {
        const iy = ly + dy;
        const sy = iy - oy;
        for (let dz = 0; dz < 8; dz++) {
          const iz = lz + dz;
          const sz = iz - oz;
          const n = leafCoordToOffset(ix, iy, iz);
          const inRange = sx >= 0 && sx < nx && sy >= 0 && sy < ny && sz >= 0 && sz < nz;
          const v = inRange ? values[denseIndex(sx, sy, sz)]! : opts.background;
          leaf.values[n] = v;
          if (inRange && isActive(v)) {
            leaf.valueMask[n >>> 5]! |= 1 << (n & 31);
            leaf.stats.addVoxel(v, ix, iy, iz);
          }
        }
      }
    }
  }

  return assembleTree([...leafMap.values()]);
}

/**
 * Builds the sparse hierarchy directly from a set of already-materialised leaves
 * (each carrying its 8-aligned origin, all 512 values, and a 512-bit value mask).
 * This is the memory-frugal path used by `buildFromVdb` and `quantize`: it never
 * allocates a dense array, so a 7M-voxel grid streams leaf-by-leaf. Per-leaf
 * stats/bbox are (re)computed from the mask + values here.
 */
export function buildTreeFromLeaves(source: Iterable<LeafInput>): BuiltTree {
  const leaves: Leaf[] = [];
  for (const src of source) {
    const [lx, ly, lz] = src.origin;
    const stats = new StatsAccumulator();
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
      const active = (src.valueMask[n >>> 5]! >>> (n & 31)) & 1;
      if (active) {
        const x = lx + ((n >> 6) & 7);
        const y = ly + ((n >> 3) & 7);
        const z = lz + (n & 7);
        stats.addVoxel(src.values[n]!, x, y, z);
      }
    }
    leaves.push({
      origin: [lx, ly, lz],
      values: src.values,
      valueMask: src.valueMask,
      stats,
      memIndex: -1,
    });
  }
  return assembleTree(leaves);
}

/** Minimal shape a leaf source must provide to `buildTreeFromLeaves`. */
export interface LeafInput {
  origin: [number, number, number];
  /** All 512 voxel values in leaf-offset order (active and inactive). */
  values: Float32Array;
  /** 512-bit active mask as 16 u32 words. */
  valueMask: Uint32Array;
}

/**
 * Assembles lower/upper/root nodes (topology, ordering, stats aggregation) from
 * a flat list of leaves. Shared by the dense (`buildTree`) and leaf-iterator
 * (`buildTreeFromLeaves`) build paths.
 */
function assembleTree(leafList: Leaf[]): BuiltTree {
  // -- Assemble lower/upper nodes from leaves ------------------------------
  const upperMap = new Map<string, UpperNode>();
  const lowerMap = new Map<string, LowerNode>();
  const nodeKey = leafKey;

  for (const leaf of leafList) {
    const [lx, ly, lz] = leaf.origin;
    const ux = alignDown(lx, 4096);
    const uy = alignDown(ly, 4096);
    const uz = alignDown(lz, 4096);
    const wx = alignDown(lx, 128);
    const wy = alignDown(ly, 128);
    const wz = alignDown(lz, 128);

    const uk = nodeKey(ux, uy, uz);
    let upper = upperMap.get(uk);
    if (!upper) {
      upper = {
        origin: [ux, uy, uz],
        children: new Map(),
        stats: new StatsAccumulator(),
        memIndex: -1,
      };
      upperMap.set(uk, upper);
    }

    const lk = nodeKey(wx, wy, wz);
    let lower = lowerMap.get(lk);
    if (!lower) {
      lower = {
        origin: [wx, wy, wz],
        children: new Map(),
        stats: new StatsAccumulator(),
        memIndex: -1,
      };
      lowerMap.set(lk, lower);
      const upperSlot = upperCoordToOffset(wx, wy, wz);
      upper.children.set(upperSlot, lower);
    }

    const lowerSlot = lowerCoordToOffset(lx, ly, lz);
    lower.children.set(lowerSlot, leaf);
  }

  // -- Deterministic breadth-first ordering + memIndex assignment ----------
  const byOrigin = (a: { origin: [number, number, number] }, b: { origin: [number, number, number] }) =>
    a.origin[0] - b.origin[0] || a.origin[1] - b.origin[1] || a.origin[2] - b.origin[2];

  const uppers = [...upperMap.values()].sort(byOrigin);
  uppers.forEach((u, i) => (u.memIndex = i));

  // Lowers grouped by parent upper (in upper order), then by slot.
  const lowers: LowerNode[] = [];
  for (const upper of uppers) {
    const slots = [...upper.children.keys()].sort((a, b) => a - b);
    for (const s of slots) lowers.push(upper.children.get(s)!);
  }
  lowers.forEach((l, i) => (l.memIndex = i));

  // Leaves grouped by parent lower (in lower order), then by slot.
  const leaves: Leaf[] = [];
  for (const lower of lowers) {
    const slots = [...lower.children.keys()].sort((a, b) => a - b);
    for (const s of slots) leaves.push(lower.children.get(s)!);
  }
  leaves.forEach((l, i) => (l.memIndex = i));

  // -- Aggregate stats bottom-up -------------------------------------------
  for (const lower of lowers) {
    for (const leaf of lower.children.values()) lower.stats.merge(leaf.stats);
  }
  for (const upper of uppers) {
    for (const lower of upper.children.values()) upper.stats.merge(lower.stats);
  }
  const root = new StatsAccumulator();
  for (const upper of uppers) root.merge(upper.stats);

  // -- Root tiles (one per upper node) -------------------------------------
  const rootTiles: RootTile[] = uppers.map((u) => ({
    key: keyForOrigin(u.origin),
    child: u,
  }));

  return {
    uppers,
    lowers,
    leaves,
    rootTiles,
    root,
    activeVoxelCount: root.count,
  };
}

import { coordToKey } from "./bytes.js";
function keyForOrigin(origin: [number, number, number]): bigint {
  return coordToKey(origin[0], origin[1], origin[2]);
}
