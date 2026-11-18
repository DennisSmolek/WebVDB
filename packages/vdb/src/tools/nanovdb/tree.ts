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
 *  - the image is larger than native's for big uniform interiors. Acceptable
 *    for the desktop-scale grids v1 targets; active-tile compression is a
 *    documented follow-up (K4.6).
 *
 * ## Active tiles passed in (K4.1)
 *
 * The leaf-iterator path ({@link buildTreeFromLeaves}) also accepts active
 * tiles that already exist in the source (a parsed `.vdb`, or a NanoVDB image
 * being re-encoded): lower-node tiles (8^3), upper-node tiles (128^3) and root
 * tiles (4096^3). They are kept as tiles, never expanded into leaves. Stats
 * follow NanoVDB `tools/GridStats.h`: a tile adds its value to its parent's
 * min/max/ave/stddev with weight `ChildT::NUM_VALUES` (512, 4096 or 32768 — the
 * child node's *slot* count, not its voxel count) and grows the parent's bbox
 * by the tile's full extent. The active-voxel count, in contrast, counts each
 * tile as `dim^3` voxels (CreateNanoGrid's `mVoxelCount`).
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
import { StatsAccumulator, Vec3StatsAccumulator } from "./stats.js";

/** A tile (or background) value: a scalar, or `[x, y, z]` for Vec3f grids. */
export type TileValue = number | readonly [number, number, number];

/** Values per voxel: 1 (scalar grids) or 3 (Vec3f, xyz interleaved). */
export type Components = 1 | 3;

export interface Leaf {
  /** 8-aligned index-space min corner. */
  origin: [number, number, number];
  /** All 512 voxel values in leaf-offset order (`components` scalars each). */
  values: Float32Array;
  /** 512-bit active mask as 16 u32 words. */
  valueMask: Uint32Array;
  stats: StatsAccumulator;
  memIndex: number;
}

export interface LowerNode {
  /** 128-aligned index-space min corner. */
  origin: [number, number, number];
  /** slot (0..4095) -> child leaf. */
  children: Map<number, Leaf>;
  /** slot (0..4095) -> value of an active 8^3 tile. Other slots are background tiles. */
  tiles: Map<number, TileValue>;
  stats: StatsAccumulator;
  memIndex: number;
}

export interface UpperNode {
  /** 4096-aligned index-space min corner. */
  origin: [number, number, number];
  /** slot (0..32767) -> child lower node. */
  children: Map<number, LowerNode>;
  /** slot (0..32767) -> value of an active 128^3 tile. Other slots are background tiles. */
  tiles: Map<number, TileValue>;
  stats: StatsAccumulator;
  memIndex: number;
}

/** One root-table entry: a child upper node, or an active 4096^3 tile. */
export type RootTile =
  | { key: bigint; origin: [number, number, number]; child: UpperNode }
  | { key: bigint; origin: [number, number, number]; child: null; value: TileValue };

export interface BuiltTree {
  uppers: UpperNode[];
  lowers: LowerNode[];
  leaves: Leaf[];
  /** Root table, sorted by origin (x, then y, then z), children and tiles interleaved. */
  rootTiles: RootTile[];
  /** Active tile counts per level (TreeData `mTileCount`). */
  tileCounts: { lower: number; upper: number; root: number };
  /** Global active-value stats + tight index bbox (== root stats/bbox). */
  root: StatsAccumulator;
  /** Active voxels in leaves plus `dim^3` per active tile. */
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
export function buildTreeFromLeaves(
  source: Iterable<LeafInput>,
  tiles: Iterable<TileInput> = [],
  components: Components = 1,
): BuiltTree {
  const leaves: Leaf[] = [];
  for (const src of source) {
    const [lx, ly, lz] = src.origin;
    if (src.values.length !== LEAF_TABLE_COUNT * components) {
      throw new Error(
        `buildFromLeaves: leaf at [${lx}, ${ly}, ${lz}] has ${src.values.length} values, ` +
          `expected ${LEAF_TABLE_COUNT * components}`,
      );
    }
    const stats = components === 3 ? new Vec3StatsAccumulator() : new StatsAccumulator();
    for (let n = 0; n < LEAF_TABLE_COUNT; n++) {
      const active = (src.valueMask[n >>> 5]! >>> (n & 31)) & 1;
      if (active) {
        const x = lx + ((n >> 6) & 7);
        const y = ly + ((n >> 3) & 7);
        const z = lz + (n & 7);
        if (stats instanceof Vec3StatsAccumulator) {
          const v = src.values;
          stats.addVoxelVec(v[n * 3]!, v[n * 3 + 1]!, v[n * 3 + 2]!, x, y, z);
        } else {
          stats.addVoxel(src.values[n]!, x, y, z);
        }
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
  return assembleTree(leaves, tiles, components);
}

/** Minimal shape a leaf source must provide to `buildTreeFromLeaves`. */
export interface LeafInput {
  origin: [number, number, number];
  /** All 512 voxel values in leaf-offset order (active and inactive); xyz interleaved for Vec3f. */
  values: Float32Array;
  /** 512-bit active mask as 16 u32 words. */
  valueMask: Uint32Array;
}

/**
 * An active tile to place above leaf level: `dim^3` voxels, all active, all
 * `value`. `dim` 8 is a lower-node tile, 128 an upper-node tile, 4096 a root
 * tile; `origin` must be a multiple of `dim`.
 */
export interface TileInput {
  origin: [number, number, number];
  dim: number;
  value: TileValue;
}

/** NanoVDB GridStats weight for a tile of edge `dim`: the child node type's slot count. */
const TILE_STATS_WEIGHT: Record<number, number> = { 8: 512, 128: 4096, 4096: 32768 };

/**
 * Assembles lower/upper/root nodes (topology, ordering, stats aggregation) from
 * a flat list of leaves. Shared by the dense (`buildTree`) and leaf-iterator
 * (`buildTreeFromLeaves`) build paths.
 */
function assembleTree(leafList: Leaf[], tileList: Iterable<TileInput> = [], components: Components = 1): BuiltTree {
  const newStats = (): StatsAccumulator => (components === 3 ? new Vec3StatsAccumulator() : new StatsAccumulator());
  // -- Assemble lower/upper nodes from leaves ------------------------------
  const upperMap = new Map<string, UpperNode>();
  const lowerMap = new Map<string, LowerNode>();
  const nodeKey = leafKey;

  const upperAt = (x: number, y: number, z: number): UpperNode => {
    const ux = alignDown(x, 4096);
    const uy = alignDown(y, 4096);
    const uz = alignDown(z, 4096);
    const uk = nodeKey(ux, uy, uz);
    let upper = upperMap.get(uk);
    if (!upper) {
      upper = {
        origin: [ux, uy, uz],
        children: new Map(),
        tiles: new Map(),
        stats: newStats(),
        memIndex: -1,
      };
      upperMap.set(uk, upper);
    }
    return upper;
  };

  const lowerAt = (x: number, y: number, z: number): LowerNode => {
    const wx = alignDown(x, 128);
    const wy = alignDown(y, 128);
    const wz = alignDown(z, 128);
    const lk = nodeKey(wx, wy, wz);
    let lower = lowerMap.get(lk);
    if (!lower) {
      lower = {
        origin: [wx, wy, wz],
        children: new Map(),
        tiles: new Map(),
        stats: newStats(),
        memIndex: -1,
      };
      lowerMap.set(lk, lower);
      upperAt(x, y, z).children.set(upperCoordToOffset(wx, wy, wz), lower);
    }
    return lower;
  };

  for (const leaf of leafList) {
    const [lx, ly, lz] = leaf.origin;
    lowerAt(lx, ly, lz).children.set(lowerCoordToOffset(lx, ly, lz), leaf);
  }

  // -- Place active tiles --------------------------------------------------
  const rootTileMap = new Map<string, { origin: [number, number, number]; value: TileValue }>();
  const tileCounts = { lower: 0, upper: 0, root: 0 };
  let tileVoxels = 0;
  for (const tile of tileList) {
    const [tx, ty, tz] = tile.origin;
    const { dim, value } = tile;
    if ((typeof value === "number") !== (components === 1)) {
      throw new Error(`buildFromLeaves: tile value ${JSON.stringify(value)} does not have ${components} component(s)`);
    }
    if (TILE_STATS_WEIGHT[dim] === undefined) {
      throw new Error(`buildFromLeaves: tile dim ${dim} must be 8, 128 or 4096`);
    }
    if (alignDown(tx, dim) !== tx || alignDown(ty, dim) !== ty || alignDown(tz, dim) !== tz) {
      throw new Error(`buildFromLeaves: tile origin [${tx}, ${ty}, ${tz}] is not ${dim}-aligned`);
    }
    const duplicate = (): never => {
      throw new Error(`buildFromLeaves: duplicate ${dim}^3 tile at [${tx}, ${ty}, ${tz}]`);
    };
    if (dim === 8) {
      const lower = lowerAt(tx, ty, tz);
      const slot = lowerCoordToOffset(tx, ty, tz);
      if (lower.tiles.has(slot)) duplicate();
      lower.tiles.set(slot, value);
      tileCounts.lower++;
    } else if (dim === 128) {
      const upper = upperAt(tx, ty, tz);
      const slot = upperCoordToOffset(tx, ty, tz);
      if (upper.tiles.has(slot)) duplicate();
      upper.tiles.set(slot, value);
      tileCounts.upper++;
    } else {
      const k = nodeKey(tx, ty, tz);
      if (rootTileMap.has(k)) duplicate();
      rootTileMap.set(k, { origin: [tx, ty, tz], value });
      tileCounts.root++;
    }
    tileVoxels += dim * dim * dim;
  }

  // A slot holds a child or a tile, never both.
  const overlap = (where: string, origin: readonly number[]): never => {
    throw new Error(`buildFromLeaves: active tile overlaps child data in ${where} at [${origin.join(", ")}]`);
  };
  for (const lower of lowerMap.values()) {
    for (const slot of lower.tiles.keys()) if (lower.children.has(slot)) overlap("a lower node", lower.origin);
  }
  for (const upper of upperMap.values()) {
    for (const slot of upper.tiles.keys()) if (upper.children.has(slot)) overlap("an upper node", upper.origin);
  }
  for (const [key, tile] of rootTileMap) if (upperMap.has(key)) overlap("the root", tile.origin);

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
  // Tiles first, then children: the order NanoVDB GridStats accumulates in.
  for (const lower of lowers) {
    addTileStats(lower.stats, lower.tiles, lower.origin, 4, 8);
    for (const leaf of lower.children.values()) lower.stats.merge(leaf.stats);
  }
  for (const upper of uppers) {
    addTileStats(upper.stats, upper.tiles, upper.origin, 5, 128);
    for (const lower of upper.children.values()) upper.stats.merge(lower.stats);
  }

  // -- Root table: children and tiles, sorted by origin --------------------
  // (The order OpenVDB's root map iterates in, which CreateNanoGrid copies.)
  const rootTiles: RootTile[] = [
    ...uppers.map((u): RootTile => ({ key: keyForOrigin(u.origin), origin: u.origin, child: u })),
    ...[...rootTileMap.values()].map(
      (t): RootTile => ({ key: keyForOrigin(t.origin), origin: t.origin, child: null, value: t.value }),
    ),
  ].sort(byOrigin);

  const root = newStats();
  for (const entry of rootTiles) {
    if (entry.child) {
      root.merge(entry.child.stats);
    } else {
      const [x, y, z] = entry.origin;
      addTileValue(root, entry.value, TILE_STATS_WEIGHT[4096]!, entry.origin, [x + 4095, y + 4095, z + 4095]);
    }
  }

  let leafVoxels = 0;
  for (const leaf of leaves) leafVoxels += leaf.stats.count;

  return {
    uppers,
    lowers,
    leaves,
    rootTiles,
    tileCounts,
    root,
    activeVoxelCount: leafVoxels + tileVoxels,
  };
}

/** Adds an internal node's active tiles to its stats, in slot order (GridStats weighting). */
function addTileStats(
  stats: StatsAccumulator,
  tiles: Map<number, TileValue>,
  origin: readonly [number, number, number],
  log2dim: number,
  childDim: number,
): void {
  if (tiles.size === 0) return;
  const weight = TILE_STATS_WEIGHT[childDim]!;
  const mask = (1 << log2dim) - 1;
  const slots = [...tiles.keys()].sort((a, b) => a - b);
  for (const n of slots) {
    const x = origin[0] + ((n >> (2 * log2dim)) & mask) * childDim;
    const y = origin[1] + ((n >> log2dim) & mask) * childDim;
    const z = origin[2] + (n & mask) * childDim;
    addTileValue(stats, tiles.get(n)!, weight, [x, y, z], [x + childDim - 1, y + childDim - 1, z + childDim - 1]);
  }
}

/** Adds a scalar or vector tile to its accumulator. */
function addTileValue(
  stats: StatsAccumulator,
  value: TileValue,
  weight: number,
  bboxMin: readonly [number, number, number],
  bboxMax: readonly [number, number, number],
): void {
  if (typeof value === "number") stats.addTile(value, weight, bboxMin, bboxMax);
  else (stats as Vec3StatsAccumulator).addTileVec(value, weight, bboxMin, bboxMax);
}

import { coordToKey } from "./bytes.js";
function keyForOrigin(origin: [number, number, number]): bigint {
  return coordToKey(origin[0], origin[1], origin[2]);
}
