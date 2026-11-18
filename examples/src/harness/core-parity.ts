/**
 * K1.6 core parity harness (docs/design/K1-core.md §7): the K1 regression gate.
 *
 * One `VDBGridPool` (started at 4 KiB with `maxGrids: 2`, so the buffer and the
 * table both grow) holds the nine primitive fixtures, a throwaway grid that is
 * removed (a dead id in the middle of the table) and `torus_fog_fp8` a second
 * time with an instance transform. For every generated-module variant of the
 * matrix below the page dispatches compute kernels that call only the public
 * `vdb*` API and compares the results with the CPU twins in
 * `@pmndrs/vdb/tools`. Results go to `window.__CORE_PARITY__` for
 * `e2e/core-parity.spec.ts`.
 *
 * Kernel shape: one `queryMain` entry takes `IN = 12` floats per query
 * `[op, grid, a0..a9]` and writes `OUT = 16` floats (ops below); a second entry
 * `walkMain` runs the canonical HDDA walk (`HddaWalkOptions` doc in
 * tools/cpu/march.ts) per ray. Integers travel as f32 (exact below 2^24).
 */
import { VDBGridPool, VDBError, checkModule, vdbModuleEntries, vdbModuleWgsl } from "@pmndrs/vdb";
import type { VDBGridInfo, VDBGridTypeName, VDBModuleHelper, VDBModuleOptions } from "@pmndrs/vdb";
import {
  NanoVDBFile,
  cellBoundTrilinear,
  getDim,
  hddaWalk,
  indexBBox,
  isActive,
  leafVoxels,
  nodeMax,
  nodeStats,
  probeCoords,
  rayClip,
  rayClipSupport,
  readValueF32,
  sampleNearest,
  sampleTrilinear,
  treeNodes,
} from "@pmndrs/vdb/tools";
import type { HddaSegment, TreeNode, Vec3 } from "@pmndrs/vdb/tools";

// ---------------------------------------------------------------------------
// Tunables (counts are per grid per variant unless noted) and tolerances.

const READ_RANDOM = 220; // random integer coords, bbox dilated by 4 (plus the sidecar's 73 on grids that have one)
const SAMPLE_RANDOM = 100; // random index-space points (plus 30 edge points near the bbox faces)
const WORLD_POINTS = 16; // world positions for vdbWorldToIndex / vdbIndexToWorld / world->sample
const RAYS = 14; // index-space rays for vdbRayClip and the HDDA walk
const NODE_LEAVES = 6; // leaves probed for vdbNodeMax at dims 8 / 128 / 4096 / 2^20 (plus tiles and far voxels)

/** Read / nodeMax / sample values: f32 results of the same op order. SwiftShader is bit-exact; other GPUs may differ by 1 ulp. */
const VALUE_ULP = 1;
/** Trilinear: the twin lerps in f64 over f32 taps, the GPU in f32. Absolute error vs max(1, |ref|). */
const SAMPLE_TOL = 2e-6;
/** World/index affines: f32 record rows and f32 dot products vs the f64 info() matrices; relative to the term-magnitude scale. */
const AFFINE_REL = 2e-6;
/** Ray parameter agreement (clip and walk intervals), relative to max(1, |t|). */
const T_REL = 1e-4;

const NAMES = ["box_fog", "sphere_fog", "torus_fog"].flatMap((p) => ["float", "fp8", "fpn"].map((t) => `${p}_${t}`));
const INSTANCE_SOURCE = NAMES.indexOf("torus_fog_fp8"); // the fixture added a second time
const SLOT = { group: 1, firstBinding: 3 } as const;
const BUFFER_EXPR = "vdbPool.value";
const IN = 12;
const OUT = 16;
const WALK_IN = 8;
const MAX_SEGS = 160;
const SEG = 9;
const WALK_OUT = 1 + MAX_SEGS * SEG;

// ---------------------------------------------------------------------------
// Result types (shared with the e2e spec).

interface CatStats {
  total: number;
  failed: number;
  maxDelta: number;
  maxUlp: number;
}

interface VariantResult {
  id: string;
  options: string;
  wgslBytes: number;
  compileErrors: string[];
  cats: Record<string, CatStats>;
  failures: string[];
  ms: number;
}

interface CoreParity {
  done: boolean;
  error?: string;
  adapter?: string;
  isSwiftShader?: boolean;
  pool?: {
    byteLength: number;
    gridCapacity: number;
    versions: number[];
    gridCount: number;
    ids: number[];
    deadId: number;
    instanceId: number;
    offsets: number[];
  };
  expected?: { variants: string[] };
  variants: VariantResult[];
  totals: Record<string, CatStats>;
  counts: { read: number; sample: number; xform: number; rays: number; cells: number; nodeMax: number };
  poolChecks: { name: string; ok: boolean; detail?: string }[];
  validationError?: string;
  missing?: string[];
  totalMs: number;
}

declare global {
  interface Window {
    __CORE_PARITY__?: CoreParity;
  }
}

// ---------------------------------------------------------------------------
// Small utilities.

const root = document.getElementById("demo-root")!;
function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const fbuf = new Float32Array(1);
const ibuf = new Int32Array(fbuf.buffer);
function ord(x: number): number {
  fbuf[0] = x;
  const i = ibuf[0]!;
  return i < 0 ? -2147483648 - i : i;
}
function ulps(a: number, b: number): number {
  // NaN never matches, even NaN vs NaN: the twins never produce NaN on the fixtures, so any NaN is a bug.
  if (Number.isNaN(a) || Number.isNaN(b)) return Infinity;
  if (a === b) return 0;
  return Math.abs(ord(a) - ord(b));
}
const fr = Math.fround;

// ---------------------------------------------------------------------------
// Stats collection.

class Collector {
  cats: Record<string, CatStats> = {};
  failures: string[] = [];
  private perCat: Record<string, number> = {};
  constructor(readonly variant: string) {}

  /** Records one comparison. `delta`/`ulp` feed the max trackers; `msg` is built lazily for failures. */
  rec(cat: string, ok: boolean, msg: () => string, delta = 0, ulp = 0): void {
    const s = (this.cats[cat] ??= { total: 0, failed: 0, maxDelta: 0, maxUlp: 0 });
    s.total++;
    if (Number.isFinite(delta)) s.maxDelta = Math.max(s.maxDelta, delta);
    if (Number.isFinite(ulp)) s.maxUlp = Math.max(s.maxUlp, ulp);
    if (!ok) {
      s.failed++;
      const n = (this.perCat[cat] = (this.perCat[cat] ?? 0) + 1);
      if (n <= 3) this.failures.push(`${this.variant} [${cat}] ${msg()}`);
    }
  }

  /** Exact-integer or flag comparison. */
  eq(cat: string, got: number, want: number, what: () => string): void {
    this.rec(cat, got === want, () => `${what()}: gpu ${got} want ${want}`, Math.abs(got - want));
  }

  /** f32 value comparison within VALUE_ULP. */
  val(cat: string, got: number, want: number, what: () => string): void {
    const u = ulps(got, want);
    this.rec(cat, u <= VALUE_ULP, () => `${what()}: gpu ${got} want ${want} (${u} ulp)`, Math.abs(got - want), u);
  }
}

// ---------------------------------------------------------------------------
// Grids under test.

interface Sidecar {
  samples: { ijk: Vec3; value: number; active: boolean }[];
}

interface ReadRow {
  ijk: Vec3;
  value: number;
  active: boolean;
  sidecar: { value: number; active: boolean } | null;
}

interface NodeProbe {
  ijk: Vec3;
  dim: number;
  want: number;
  kind: "leaf" | "lower" | "upper" | "root" | "tile" | "far";
}

interface Ray {
  o: Vec3;
  d: Vec3;
}

interface Grid {
  id: number;
  name: string;
  image: Uint32Array;
  info: VDBGridInfo;
  sidecar: Sidecar | null;
  bbox: { min: Vec3; max: Vec3 };
  readRows: ReadRow[];
  rays: Ray[];
  worldPoints: Vec3[];
  worldDirs: Vec3[];
  nodeProbes?: NodeProbe[];
}

function readRows(image: Uint32Array, seed: number, sidecar: Sidecar | null): ReadRow[] {
  const { min, max } = indexBBox(image);
  const rows: ReadRow[] = probeCoords({ seed: BigInt(seed), count: READ_RANDOM, bboxMin: min, bboxMax: max, dilate: 4 }).map((ijk) => ({
    ijk,
    value: readValueF32(image, ijk),
    active: isActive(image, ijk),
    sidecar: null,
  }));
  for (const s of sidecar?.samples ?? []) {
    rows.push({ ijk: s.ijk, value: readValueF32(image, s.ijk), active: isActive(image, s.ijk), sidecar: { value: s.value, active: s.active } });
  }
  return rows;
}

function makeRays(image: Uint32Array, seed: number): Ray[] {
  const r = rng(seed);
  const { min, max } = indexBBox(image);
  const c = [0, 1, 2].map((a) => (min[a]! + max[a]! + 1) / 2);
  const ext = Math.max(...[0, 1, 2].map((a) => max[a]! - min[a]! + 1));
  const rays: Ray[] = [];
  for (let k = 0; k < RAYS; k++) {
    const inside = k % 4 === 0;
    const miss = k % 7 === 6;
    const o = [0, 1, 2].map((a) => fr(inside ? min[a]! + r() * (max[a]! - min[a]! + 1) : c[a]! + (r() - 0.5) * 2 * ext)) as Vec3;
    const shift = miss ? 3 * ext : 0;
    const target = [0, 1, 2].map((a) => min[a]! + r() * (max[a]! - min[a]! + 1) + (a === 0 ? shift : 0));
    let d = [0, 1, 2].map((a) => target[a]! - o[a]! + (r() - 0.5) * 1e-3);
    const len = Math.hypot(d[0]!, d[1]!, d[2]!);
    // No axis-parallel rays (K1.4b open issue 3): every component is non-zero.
    d = d.map((x) => fr(x / len));
    rays.push({ o, d: d as Vec3 });
  }
  return rays;
}

function nodeProbes(image: Uint32Array): NodeProbe[] {
  const nodes: TreeNode[] = treeNodes(image);
  const out: NodeProbe[] = [];
  const leaves = nodes.map((n, i) => ({ n, i })).filter((x) => x.n.level === 0);
  for (let k = 0; k < NODE_LEAVES && leaves.length > 0; k++) {
    const { n: leaf, i } = leaves[Math.floor(((k + 0.5) * leaves.length) / NODE_LEAVES)]!;
    const { active } = leafVoxels(image, leaf.address);
    const v = active.indexOf(1);
    if (v < 0) continue;
    const ijk: Vec3 = [leaf.origin[0] + (v >> 6), leaf.origin[1] + ((v >> 3) & 7), leaf.origin[2] + (v & 7)];
    const lowerIdx = nodes[i]!.parent;
    const upperIdx = nodes[lowerIdx]!.parent;
    out.push({ ijk, dim: 8, want: nodeStats(image, 0, leaf.address).max, kind: "leaf" });
    out.push({ ijk, dim: 128, want: nodeStats(image, 1, nodes[lowerIdx]!.address).max, kind: "lower" });
    out.push({ ijk, dim: 4096, want: nodeStats(image, 2, nodes[upperIdx]!.address).max, kind: "upper" });
    out.push({ ijk, dim: 1 << 20, want: nodeStats(image, 3, nodes[0]!.address).max, kind: "root" });
  }
  let tiles = 0;
  for (const n of nodes) {
    for (const t of n.activeTiles) {
      if (tiles >= 4 || t.dim === 4096) continue;
      out.push({ ijk: t.origin, dim: t.dim, want: fr(t.value), kind: "tile" });
      out.push({ ijk: t.origin, dim: 8, want: fr(t.value), kind: "tile" });
      tiles++;
    }
  }
  for (const ijk of [[100000, 0, 0], [-70000, 5, 9], [3, -90000, 12345]] as Vec3[]) {
    out.push({ ijk, dim: 8, want: nodeMax(image, ijk, 8), kind: "far" });
  }
  return out;
}

// ---------------------------------------------------------------------------
// The variant matrix.

type Test = "read" | "meta" | "dead" | "sample" | "xform" | "world" | "clip" | "nodemax" | "walk" | "helpers" | "fast";

interface Spec {
  id: string;
  profile: "read" | "sample" | "march";
  types: VDBGridTypeName[];
  mode: "slot" | "buffer";
  isActive?: boolean;
  sampler?: "trilinear" | "nearest";
  outside?: "background" | "zero";
  helpers?: VDBModuleHelper[];
  tests: Test[];
}

const ALL: VDBGridTypeName[] = ["float", "fp8", "fpn"];
const BASE: Test[] = ["read", "meta", "dead"];
const SAMPLE: Test[] = [...BASE, "sample"];
const SAMPLE_X: Test[] = [...SAMPLE, "xform", "world", "clip"];
const MARCH: Test[] = [...SAMPLE, "nodemax", "walk"];
const MARCH_X: Test[] = [...MARCH, "xform", "world", "clip"];

const SPECS: Spec[] = [
  { id: "read/float", profile: "read", types: ["float"], mode: "slot", tests: BASE },
  { id: "read/fp8", profile: "read", types: ["fp8"], mode: "slot", tests: BASE },
  { id: "read/fpn", profile: "read", types: ["fpn"], mode: "slot", tests: BASE },
  { id: "read/all", profile: "read", types: ALL, mode: "slot", tests: BASE },
  { id: "read/all/isActive=false", profile: "read", types: ALL, mode: "slot", isActive: false, tests: BASE },
  { id: "read/all/buffer", profile: "read", types: ALL, mode: "buffer", tests: BASE },
  { id: "sample/float", profile: "sample", types: ["float"], mode: "slot", tests: SAMPLE },
  { id: "sample/fp8", profile: "sample", types: ["fp8"], mode: "slot", tests: SAMPLE },
  { id: "sample/fpn", profile: "sample", types: ["fpn"], mode: "slot", tests: SAMPLE },
  { id: "sample/all", profile: "sample", types: ALL, mode: "slot", tests: SAMPLE_X },
  { id: "sample/all/nearest", profile: "sample", types: ALL, mode: "slot", sampler: "nearest", tests: SAMPLE },
  { id: "sample/all/outside=zero", profile: "sample", types: ALL, mode: "slot", outside: "zero", tests: SAMPLE },
  { id: "sample/all/outside=zero,nearest", profile: "sample", types: ALL, mode: "slot", outside: "zero", sampler: "nearest", tests: SAMPLE },
  { id: "march/float", profile: "march", types: ["float"], mode: "slot", tests: MARCH },
  { id: "march/fp8", profile: "march", types: ["fp8"], mode: "slot", tests: MARCH },
  { id: "march/fpn", profile: "march", types: ["fpn"], mode: "slot", tests: MARCH },
  { id: "march/all", profile: "march", types: ALL, mode: "slot", tests: MARCH_X },
  { id: "march/all/buffer", profile: "march", types: ALL, mode: "buffer", tests: MARCH_X },
  { id: "march/all/outside=zero", profile: "march", types: ALL, mode: "slot", outside: "zero", tests: MARCH },
  // K1.7a opt-in helpers: vdbCellBoundTrilinear / vdbRayClipSupport as query ops, vdbHddaCellEnd as the walk's cell end.
  { id: "march/all/helpers", profile: "march", types: ALL, mode: "slot", helpers: ["cellBoundTrilinear", "hddaCellEnd", "rayClipSupport"], tests: [...MARCH, "clip", "helpers"] },
  { id: "sample/all/nearest,helpers", profile: "sample", types: ALL, mode: "slot", sampler: "nearest", helpers: ["rayClipSupport"], tests: [...SAMPLE, "helpers"] },
  // K3.2d: vdbSampleFast (helper fastTrilinear) is bit-identical to vdbSample, fresh or with a warmed accessor, on
  // points across leaf and lower-node faces; with outside 'zero' too, and in the sparse source's march module.
  { id: "sample/all/fast", profile: "sample", types: ALL, mode: "slot", helpers: ["fastTrilinear"], tests: [...SAMPLE, "fast"] },
  { id: "march/all/outside=zero,fast", profile: "march", types: ALL, mode: "slot", outside: "zero", helpers: ["cellBoundTrilinear", "hddaCellEnd", "nodeAverage", "fastTrilinear"], tests: [...MARCH, "fast"] },
];

function moduleOptions(s: Spec): VDBModuleOptions {
  return {
    ...(s.mode === "slot" ? { slot: SLOT } : { buffer: BUFFER_EXPR }),
    types: s.types,
    profile: s.profile,
    ...(s.isActive === undefined ? {} : { isActive: s.isActive }),
    ...(s.sampler === undefined ? {} : { sampler: s.sampler }),
    ...(s.outside === undefined ? {} : { outside: s.outside }),
    ...(s.helpers === undefined ? {} : { helpers: s.helpers }),
  };
}

// ---------------------------------------------------------------------------
// WGSL kernels (public API only).

function kernel(s: Spec, module: string): string {
  const rank = s.profile === "read" ? 0 : s.profile === "sample" ? 1 : 2;
  const withActive = s.isActive !== false;
  const host =
    s.mode === "buffer"
      ? `struct vdbPoolStruct {
    value: array<u32>,
}
@group(${SLOT.group}) @binding(${SLOT.firstBinding}) var<storage, read> vdbPool: vdbPoolStruct;
`
      : "";
  const cases: string[] = [];
  cases.push(`        case 0u: {
            qOut[o] = vdbReadValue(&a, ijk);
            ${withActive ? "qOut[o + 1u] = select(0.0, 1.0, vdbIsActive(&a, ijk));" : ""}
            qOut[o + 2u] = select(0.0, 1.0, a.ok);
        }`);
  cases.push(`        case 1u: {
            let mn = vdbIndexMin(g);
            let mx = vdbIndexMax(g);
            qOut[o] = f32(vdbGridType(g));
            qOut[o + 1u] = select(0.0, 1.0, a.ok);
            qOut[o + 2u] = f32(mn.x);
            qOut[o + 3u] = f32(mn.y);
            qOut[o + 4u] = f32(mn.z);
            qOut[o + 5u] = f32(mx.x);
            qOut[o + 6u] = f32(mx.y);
            qOut[o + 7u] = f32(mx.z);
            qOut[o + 8u] = f32(vdbGridCount());
            ${rank >= 2 ? "qOut[o + 9u] = select(0.0, 1.0, vdbHasStats(g));\n            qOut[o + 10u] = select(0.0, vdbValueMax(g), vdbHasStats(g));" : ""}
        }`);
  if (rank >= 1) {
    cases.push(`        case 2u: {
            qOut[o] = vdbSample(&a, p);
        }`);
    cases.push(`        case 3u: {
            let w = vdbWorldToIndex(g, p);
            let x = vdbIndexToWorld(g, p);
            let wd = vdbWorldToIndexDir(g, d);
            let xd = vdbIndexToWorldDir(g, d);
            qOut[o] = w.x; qOut[o + 1u] = w.y; qOut[o + 2u] = w.z;
            qOut[o + 3u] = x.x; qOut[o + 4u] = x.y; qOut[o + 5u] = x.z;
            qOut[o + 6u] = wd.x; qOut[o + 7u] = wd.y; qOut[o + 8u] = wd.z;
            qOut[o + 9u] = xd.x; qOut[o + 10u] = xd.y; qOut[o + 11u] = xd.z;
        }`);
    cases.push(`        case 4u: {
            var t = vec2f(qv(i, 8u), qv(i, 9u));
            let hit = vdbRayClip(g, p, d, &t);
            qOut[o] = select(0.0, 1.0, hit);
            qOut[o + 1u] = t.x;
            qOut[o + 2u] = t.y;
        }`);
    cases.push(`        case 6u: {
            let w = vdbWorldToIndex(g, p);
            qOut[o] = vdbSample(&a, w);
            qOut[o + 1u] = w.x; qOut[o + 2u] = w.y; qOut[o + 3u] = w.z;
        }`);
  }
  if (rank >= 2) {
    cases.push(`        case 5u: {
            qOut[o] = vdbNodeMax(&a, ijk, u32(qv(i, 5u)));
            qOut[o + 1u] = f32(vdbGetDim(&a, ijk));
        }`);
  }
  const helpers = new Set(s.helpers ?? []);
  if (helpers.has("cellBoundTrilinear")) {
    cases.push(`        case 7u: {
            qOut[o] = vdbCellBoundTrilinear(&a, ijk, u32(qv(i, 5u)));
        }`);
  }
  if (helpers.has("fastTrilinear")) {
    // vdbSample with a fresh accessor; vdbSampleFast fresh, after vdbSample warmed the accessor at p, and after a
    // warm-up at p + d (another leaf / lower node: the cached-node paths). One call site each (SwiftShader compiles
    // every inlined tree walk slowly): k 0 vdbSample fresh -> 0, 1 fast warm -> 2, 2 fast fresh -> 1, 3 vdbSample at
    // p + d fresh -> 4, 4 fast warmed elsewhere -> 3.
    cases.push(`        case 9u: {
            var f = a;
            for (var k = 0u; k < 5u; k++) {
                if k == 2u || k == 3u {
                    f = a;
                }
                if k == 0u || k == 3u {
                    qOut[o + select(0u, 4u, k == 3u)] = vdbSample(&f, select(p, p + d, k == 3u));
                } else {
                    qOut[o + select(select(1u, 2u, k == 1u), 3u, k == 4u)] = vdbSampleFast(&f, p);
                }
            }
        }`);
  }
  if (helpers.has("rayClipSupport")) {
    cases.push(`        case 8u: {
            var t = vec2f(qv(i, 8u), qv(i, 9u));
            let hit = vdbRayClipSupport(g, p, d, &t);
            qOut[o] = select(0.0, 1.0, hit);
            qOut[o + 1u] = t.x;
            qOut[o + 2u] = t.y;
        }`);
  }
  const cellEnd = helpers.has("hddaCellEnd") ? "vdbHddaCellEnd(&h)" : "min(min(h.next.x, h.next.y), min(h.next.z, h.tmax))";
  const walk =
    rank >= 2
      ? `
@compute @workgroup_size(8)
fn walkMain(@builtin(global_invocation_id) gid: vec3u) {
    if gid.x >= arrayLength(&qIn) / ${WALK_IN}u {
        return;
    }
    let r = gid.x * ${WALK_IN}u;
    let o = vec3f(qIn[r], qIn[r + 1u], qIn[r + 2u]);
    let g = u32(qIn[r + 3u]);
    let d = vec3f(qIn[r + 4u], qIn[r + 5u], qIn[r + 6u]);
    let leafDim = u32(qIn[r + 7u]);
    let base = gid.x * ${WALK_OUT}u;
    var n = 0u;
    var t = vec2f(0.0, 3.4e38);
    if vdbRayClip(g, o, d, &t) {
        var a = vdbAccessorInit(g);
        var h: VdbHdda;
        vdbHddaInit(&h, o, t.x, d, t.y, max(vdbGetDim(&a, vec3i(floor(o + d * t.x))), leafDim));
        var updates = 0u;
        for (var i = 0u; i < 4096u; i++) {
            let dimA = vdbGetDim(&a, h.voxel);
            let want = max(dimA, leafDim);
            if want != u32(h.dim) && updates < 2u {
                vdbHddaUpdate(&h, o, d, want);
                updates++;
                continue;
            }
            updates = 0u;
            var act = true;
            if dimA >= u32(h.dim) {
                act = vdbIsActive(&a, h.voxel);
            }
            let m = vdbNodeMax(&a, h.voxel, u32(h.dim));
            if n < ${MAX_SEGS}u {
                let s = base + 1u + n * ${SEG}u;
                qOut[s] = f32(h.dim);
                qOut[s + 1u] = f32(dimA);
                qOut[s + 2u] = h.tmin;
                qOut[s + 3u] = ${cellEnd};
                qOut[s + 4u] = select(0.0, 1.0, act);
                qOut[s + 5u] = m;
                qOut[s + 6u] = f32(h.voxel.x);
                qOut[s + 7u] = f32(h.voxel.y);
                qOut[s + 8u] = f32(h.voxel.z);
            }
            n++;
            if !vdbHddaStep(&h) {
                break;
            }
        }
    }
    qOut[base] = f32(n);
}
`
      : "";
  return `${module}
${host}@group(0) @binding(0) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(1) var<storage, read> qIn: array<f32>;

fn qv(i: u32, k: u32) -> f32 {
    return qIn[i * ${IN}u + k];
}

@compute @workgroup_size(32)
fn queryMain(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if i >= arrayLength(&qIn) / ${IN}u {
        return;
    }
    let op = u32(qv(i, 0u));
    let g = u32(qv(i, 1u));
    let p = vec3f(qv(i, 2u), qv(i, 3u), qv(i, 4u));
    let d = vec3f(qv(i, 5u), qv(i, 6u), qv(i, 7u));
    let ijk = vec3i(p);
    let o = i * ${OUT}u;
    var a = vdbAccessorInit(g);
    switch op {
${cases.join("\n")}
        default: {}
    }
}
${walk}`;
}

// ---------------------------------------------------------------------------
// GPU plumbing.

interface Gpu {
  device: GPUDevice;
  layout0: GPUBindGroupLayout;
  pipelineLayout: GPUPipelineLayout;
  poolGroup: GPUBindGroup;
}

async function dispatch(gpu: Gpu, pipeline: GPUComputePipeline, input: Float32Array, outFloats: number, invocations: number, workgroup: number): Promise<Float32Array> {
  const { device } = gpu;
  const inBuf = device.createBuffer({ size: Math.max(16, input.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, input.buffer as ArrayBuffer, input.byteOffset, input.byteLength);
  const outBuf = device.createBuffer({ size: outFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: outFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(
    0,
    device.createBindGroup({
      layout: gpu.layout0,
      entries: [
        { binding: 0, resource: { buffer: outBuf } },
        { binding: 1, resource: { buffer: inBuf, size: Math.max(16, input.byteLength) } },
      ],
    }),
  );
  pass.setBindGroup(1, gpu.poolGroup);
  pass.dispatchWorkgroups(Math.ceil(invocations / workgroup));
  pass.end();
  enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outFloats * 4);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const res = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  inBuf.destroy();
  outBuf.destroy();
  readBuf.destroy();
  return res;
}

// ---------------------------------------------------------------------------
// Query batches.

class Batch {
  readonly inp: number[] = [];
  readonly verify: ((out: Float32Array, base: number) => void)[] = [];
  add(op: number, g: number, args: number[], verify: (out: Float32Array, base: number) => void): void {
    const row = new Array<number>(IN).fill(0);
    row[0] = op;
    row[1] = g;
    for (let k = 0; k < args.length; k++) row[2 + k] = args[k]!;
    this.inp.push(...row);
    this.verify.push(verify);
  }
  get count(): number {
    return this.verify.length;
  }
}

/** Index-space sample points: random inside the dilated bbox plus edge points around the bbox faces. */
function samplePoints(grid: Grid, seed: number): Vec3[] {
  const r = rng(seed);
  const { min, max } = grid.bbox;
  const pts: Vec3[] = [];
  for (let k = 0; k < SAMPLE_RANDOM; k++) {
    pts.push([0, 1, 2].map((a) => fr(min[a]! - 3 + r() * (max[a]! - min[a]! + 6))) as Vec3);
  }
  const offs = [-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5];
  for (let a = 0; a < 3; a++) {
    for (const side of [0, 1]) {
      for (const off of offs) {
        const p = [0, 1, 2].map((b) => fr(min[b]! + r() * (max[b]! - min[b]!))) as Vec3;
        p[a] = side === 0 ? min[a]! + off : max[a]! + off;
        pts.push(p);
      }
    }
  }
  return pts;
}

function buildBatch(spec: Spec, grids: Grid[], deadIds: number[], gridCount: number, c: Collector): Batch {
  const b = new Batch();
  const compiled = new Set<VDBGridTypeName>(spec.types);
  const has = (t: Test) => spec.tests.includes(t);
  const active = spec.isActive !== false;
  const tri = spec.sampler !== "nearest";
  const zero = spec.outside === "zero";

  for (const g of grids) {
    const ok = compiled.has(g.info.type);
    const tag = (extra: string) => () => `grid ${g.id} (${g.name}) ${extra}`;

    if (has("read")) {
      for (const row of g.readRows) {
        b.add(0, g.id, row.ijk, (out, o) => {
          const wantV = ok ? row.value : 0;
          const wantA = ok && active ? row.active : false;
          if (ok) {
            c.val("read-value", out[o]!, wantV, tag(`ijk ${row.ijk}`));
            if (active) c.eq("read-active", out[o + 1]!, wantA ? 1 : 0, tag(`isActive ${row.ijk}`));
            if (row.sidecar) {
              c.val("sidecar-value", out[o]!, fr(row.sidecar.value), tag(`sidecar ${row.ijk}`));
              if (active) c.eq("sidecar-active", out[o + 1]!, row.sidecar.active ? 1 : 0, tag(`sidecar isActive ${row.ijk}`));
            }
          } else {
            c.eq("negative", out[o]! === 0 && out[o + 1]! === 0 && out[o + 2]! === 0 ? 1 : 0, 1, tag(`type ${g.info.type} not compiled: read ${row.ijk} -> ${out[o]},${out[o + 1]},${out[o + 2]}`));
          }
        });
      }
    }

    if (has("meta")) {
      b.add(1, g.id, [], (out, o) => {
        const bb = g.info.indexBBox;
        c.eq("record", out[o]!, g.info.gridTypeId, tag("grid type"));
        c.eq("record", out[o + 1]!, ok ? 1 : 0, tag("accessor ok"));
        for (let a = 0; a < 3; a++) {
          c.eq("record", out[o + 2 + a]!, bb.min[a]!, tag(`bbox min[${a}]`));
          c.eq("record", out[o + 5 + a]!, bb.max[a]!, tag(`bbox max[${a}]`));
        }
        c.eq("record", out[o + 8]!, gridCount, tag("gridCount"));
        if (spec.profile === "march") {
          c.eq("record", out[o + 9]!, g.info.hasStats ? 1 : 0, tag("hasStats"));
          if (g.info.hasStats) c.val("record", out[o + 10]!, fr(g.info.valueMax), tag("valueMax"));
        }
      });
    }

    if (has("sample")) {
      const pts = samplePoints(g, 5000 + g.id);
      const opt = { outside: zero ? ("zero" as const) : ("background" as const) };
      for (const p of pts) {
        b.add(2, g.id, p, (out, o) => {
          const got = out[o]!;
          if (!ok) {
            c.eq("negative", got === 0 ? 1 : 0, 1, tag(`type ${g.info.type} not compiled: sample ${p} -> ${got}`));
            return;
          }
          const want = tri ? sampleTrilinear(g.image, p, opt) : sampleNearest(g.image, p, opt);
          const d = Math.abs(got - want);
          if (tri) c.rec("sample", d <= SAMPLE_TOL * Math.max(1, Math.abs(want)), () => `${tag(`trilinear ${p}`)()}: gpu ${got} want ${want}`, d);
          else c.val("sample", got, fr(want), tag(`nearest ${p}`));
        });
      }
    }

    if (has("fast") && ok) {
      // Random points plus points on leaf (8k + 7 + f) and lower-node (128k + 127 + f) faces and their corners.
      const r = rng(9000 + g.id);
      const pts = samplePoints(g, 7000 + g.id);
      const { min, max } = g.bbox;
      for (let k = 0; k < 96; k++) {
        const p = [0, 1, 2].map((a) => min[a]! - 2 + r() * (max[a]! - min[a]! + 4)) as Vec3;
        const face = k % 3 === 0 ? 8 : 128;
        for (let a = 0; a < 3; a++) {
          if (r() < 0.6) p[a] = Math.floor(p[a]! / face) * face + face - 1 + r();
        }
        pts.push(p.map(fr) as Vec3);
      }
      const bits = (v: number) => new Uint32Array(new Float32Array([v]).buffer)[0]!;
      for (const p of pts) {
        const d = [0, 1, 2].map(() => fr(Math.round((r() * 2 - 1) * 12))) as Vec3;
        b.add(9, g.id, [...p, ...d], (out, o) => {
          const want = bits(out[o]!);
          for (const [name, k] of [["fresh", 1], ["warm", 2], ["warm elsewhere", 3]] as const) {
            c.rec("fast-sample", bits(out[o + k]!) === want, () => `${tag(`vdbSampleFast ${name} at ${p} (warm-up offset ${d})`)()}: ${out[o + k]} vs vdbSample ${out[o]}`);
          }
        });
      }
    }

    if (has("xform")) {
      const w2i = g.info.worldToIndex;
      const i2w = g.info.indexToWorld;
      const apply = (m: Float64Array, p: Vec3, dir: boolean) =>
        [0, 1, 2].map((r) => {
          const terms = [m[r * 4]! * p[0], m[r * 4 + 1]! * p[1], m[r * 4 + 2]! * p[2]];
          const t = dir ? 0 : m[r * 4 + 3]!;
          const value = terms[0]! + terms[1]! + terms[2]! + t;
          const scale = Math.max(1, Math.abs(terms[0]!) + Math.abs(terms[1]!) + Math.abs(terms[2]!) + Math.abs(t));
          return { value, scale };
        });
      g.worldPoints.forEach((p, k) => {
        const dir = g.worldDirs[k]!;
        b.add(3, g.id, [...p, ...dir], (out, o) => {
          const sets: [string, number, ReturnType<typeof apply>][] = [
            ["worldToIndex", 0, apply(w2i, p, false)],
            ["indexToWorld", 3, apply(i2w, p, false)],
            ["worldToIndexDir", 6, apply(w2i, dir, true)],
            ["indexToWorldDir", 9, apply(i2w, dir, true)],
          ];
          for (const [name, off, want] of sets) {
            for (let a = 0; a < 3; a++) {
              const got = out[o + off + a]!;
              const err = Math.abs(got - want[a]!.value) / want[a]!.scale;
              c.rec("xform", err <= AFFINE_REL, () => `${tag(`${name}[${a}] at ${p}`)()}: gpu ${got} want ${want[a]!.value} (rel ${err.toExponential(2)})`, err);
            }
          }
        });
      });
    }

    if (has("world") && ok) {
      const opt = { outside: zero ? ("zero" as const) : ("background" as const) };
      for (const p of g.worldPoints.slice(0, 8)) {
        b.add(6, g.id, p, (out, o) => {
          // The GPU's own index position is the oracle's input: this checks the shader composes the two calls consistently.
          const idx: Vec3 = [out[o + 1]!, out[o + 2]!, out[o + 3]!];
          const want = tri ? sampleTrilinear(g.image, idx, opt) : sampleNearest(g.image, idx, opt);
          const d = Math.abs(out[o]! - want);
          c.rec("world-sample", d <= SAMPLE_TOL * Math.max(1, Math.abs(want)), () => `${tag(`world ${p} -> idx ${idx}`)()}: gpu ${out[o]} want ${want}`, d);
        });
      }
    }

    if (has("clip")) {
      for (const ray of g.rays) {
        b.add(4, g.id, [...ray.o, ...ray.d, 0, 3.4e38], (out, o) => {
          const want = rayClip(g.image, ray.o, ray.d, [0, 3.4e38]);
          const hit = out[o]! === 1;
          c.eq("clip-hit", hit ? 1 : 0, want.hit ? 1 : 0, tag(`ray ${ray.o} + t*${ray.d} hit`));
          if (hit && want.hit) {
            for (const [name, got, w] of [["t0", out[o + 1]!, want.t0], ["t1", out[o + 2]!, want.t1]] as const) {
              const rel = Math.abs(got - w) / Math.max(1, Math.abs(w));
              c.rec("clip-t", rel <= T_REL, () => `${tag(`ray ${ray.o} ${name}`)()}: gpu ${got} want ${w}`, rel);
            }
          }
        });
      }
    }

    if (has("helpers") && spec.helpers?.includes("cellBoundTrilinear")) {
      // Every node probe's voxel, as a leaf cell (8) and a voxel (1): the 2x2x2 block reaches into neighbour cells.
      for (const probe of g.nodeProbes ?? []) {
        if (probe.kind === "far") continue;
        for (const dim of [8, 1]) {
          b.add(7, g.id, [...probe.ijk, dim], (out, o) => {
            if (!ok) {
              c.eq("negative", out[o]! === 0 ? 1 : 0, 1, tag(`type ${g.info.type} not compiled: cellBound -> ${out[o]}`));
              return;
            }
            c.val("cellbound", out[o]!, cellBoundTrilinear(g.image, probe.ijk, dim), tag(`cellBoundTrilinear ${probe.ijk} dim ${dim}`));
          });
        }
      }
    }

    if (has("helpers") && spec.helpers?.includes("rayClipSupport")) {
      const sampler = tri ? ("trilinear" as const) : ("nearest" as const);
      for (const ray of g.rays) {
        b.add(8, g.id, [...ray.o, ...ray.d, 0, 3.4e38], (out, o) => {
          const want = rayClipSupport(g.image, ray.o, ray.d, [0, 3.4e38], { sampler });
          const hit = out[o]! === 1;
          c.eq("clip-support-hit", hit ? 1 : 0, want.hit ? 1 : 0, tag(`support ray ${ray.o} + t*${ray.d} hit`));
          if (hit && want.hit) {
            for (const [name, got, w] of [["t0", out[o + 1]!, want.t0], ["t1", out[o + 2]!, want.t1]] as const) {
              const rel = Math.abs(got - w) / Math.max(1, Math.abs(w));
              c.rec("clip-support-t", rel <= T_REL, () => `${tag(`support ray ${ray.o} ${name}`)()}: gpu ${got} want ${w}`, rel);
            }
          }
        });
      }
    }

    if (has("nodemax")) {
      for (const probe of g.nodeProbes ?? []) {
        b.add(5, g.id, [...probe.ijk, probe.dim], (out, o) => {
          if (!ok) {
            c.eq("negative", out[o]! === 0 ? 1 : 0, 1, tag(`type ${g.info.type} not compiled: nodeMax -> ${out[o]}`));
            return;
          }
          c.val(`nodemax-${probe.kind}`, out[o]!, probe.want, tag(`nodeMax ${probe.ijk} dim ${probe.dim}`));
          c.eq("nodemax-dim", out[o + 1]!, getDim(g.image, probe.ijk), tag(`getDim ${probe.ijk}`));
        });
      }
    }
  }

  if (has("dead")) {
    // A removed id (in the middle of the table) and an id past every table: documented to read as nothing.
    for (const id of [...deadIds, 100000]) {
      const tag = (extra: string) => () => `dead id ${id} ${extra}`;
      const zeros = (cat: string, out: Float32Array, o: number, from: number, to: number, what: string) => {
        for (let k = from; k < to; k++) c.eq(cat, out[o + k]!, 0, tag(`${what}[${k}]`));
      };
      b.add(1, id, [], (out, o) => {
        c.eq("dead", out[o]!, 0, tag("grid type"));
        c.eq("dead", out[o + 1]!, 0, tag("accessor ok"));
        zeros("dead", out, o, 2, 5, "bbox min");
        for (let a = 5; a < 8; a++) c.eq("dead", out[o + a]!, -1, tag(`bbox max[${a - 5}]`));
        c.eq("dead", out[o + 8]!, gridCount, tag("gridCount"));
        if (spec.profile === "march") zeros("dead", out, o, 9, 11, "stats");
      });
      b.add(0, id, [0, 0, 0], (out, o) => zeros("dead", out, o, 0, 3, "read"));
      b.add(0, id, [60, -3, 17], (out, o) => zeros("dead", out, o, 0, 3, "read"));
      if (spec.profile !== "read") {
        b.add(2, id, [1.5, 2.5, 3.5], (out, o) => zeros("dead", out, o, 0, 1, "sample"));
        b.add(3, id, [1, 2, 3, 4, 5, 6], (out, o) => zeros("dead", out, o, 0, 12, "xform"));
        b.add(4, id, [0.5, 0.5, -10, 0, 0, 1, 0, 1e30], (out, o) => c.eq("dead", out[o]!, 0, tag("rayClip hit")));
        b.add(6, id, [1, 2, 3], (out, o) => zeros("dead", out, o, 0, 4, "world sample"));
      }
      if (spec.profile === "march") {
        b.add(5, id, [0, 0, 0, 8], (out, o) => {
          c.eq("dead", out[o]!, 0, tag("nodeMax"));
          c.eq("dead", out[o + 1]!, 4096, tag("getDim"));
        });
      }
      if (spec.helpers?.includes("cellBoundTrilinear")) b.add(7, id, [0, 0, 0, 8], (out, o) => c.eq("dead", out[o]!, 0, tag("cellBoundTrilinear")));
      if (spec.helpers?.includes("rayClipSupport")) b.add(8, id, [0.5, 0.5, -10, 0, 0, 1, 0, 1e30], (out, o) => c.eq("dead", out[o]!, 0, tag("rayClipSupport hit")));
    }
  }
  return b;
}

// ---------------------------------------------------------------------------
// HDDA walk comparison.

interface Cell {
  dim: number;
  dimA: number;
  voxel: Vec3;
  t0: number;
  t1: number;
  active: boolean;
  max: number;
}

const near = (a: number, b: number, rel: number) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));

function merged(cells: { t0: number; t1: number; active: boolean }[], active: boolean): [number, number][] {
  const out: [number, number][] = [];
  for (const cell of cells.filter((x) => x.active === active).sort((a, b) => a.t0 - b.t0)) {
    const last = out[out.length - 1];
    if (last && cell.t0 <= last[1] + T_REL * Math.max(1, Math.abs(cell.t0))) last[1] = Math.max(last[1], cell.t1);
    else out.push([cell.t0, cell.t1]);
  }
  return out;
}

function compareWalk(c: Collector, g: Grid, rayIx: number, ray: Ray, gpu: Cell[], truncated: boolean, cpu: Omit<HddaSegment, "level" | "max" | "skip">[], counters: { rays: number; cells: number; sequenceExact: number }): void {
  const tag = `grid ${g.id} (${g.name}) ray ${rayIx} (o ${ray.o} d ${ray.d})`;
  counters.rays++;
  c.rec("walk-bounded", !truncated, () => `${tag}: more than ${MAX_SEGS} cells`);
  if (truncated) return;
  const sig = (x: { dim: number; voxel: Vec3; active: boolean }) => `${x.dim}:${x.voxel}:${x.active}`;
  if (gpu.length === cpu.length && gpu.every((x, k) => sig(x) === sig(cpu[k]!))) counters.sequenceExact++;
  const eps = T_REL / 10;
  const gCells = gpu.filter((x) => x.t1 - x.t0 > eps * Math.max(1, Math.abs(x.t1)));
  const cCells = cpu.filter((x) => x.t1 - x.t0 > eps * Math.max(1, Math.abs(x.t1)));
  counters.cells += gCells.length;

  // Covered extent.
  if (gCells.length === 0 || cCells.length === 0) {
    c.eq("walk-extent", gCells.length > 0 ? 1 : 0, cCells.length > 0 ? 1 : 0, () => `${tag}: whether any cell covers the ray (gpu ${gCells.length} cells, cpu ${cCells.length})`);
    return;
  }
  const gA = Math.min(...gCells.map((x) => x.t0));
  const gB = Math.max(...gCells.map((x) => x.t1));
  const cA = Math.min(...cCells.map((x) => x.t0));
  const cB = Math.max(...cCells.map((x) => x.t1));
  const dExt = Math.max(Math.abs(gA - cA) / Math.max(1, Math.abs(cA)), Math.abs(gB - cB) / Math.max(1, Math.abs(cB)));
  c.rec("walk-extent", dExt <= T_REL, () => `${tag}: gpu [${gA}, ${gB}] cpu [${cA}, ${cB}]`, dExt);

  // Covered intervals per class (active / inactive).
  for (const cls of [true, false]) {
    const gm = merged(gCells, cls);
    const cm = merged(cCells, cls);
    let dev = 0;
    let same = gm.length === cm.length;
    if (same) {
      for (let k = 0; k < gm.length; k++) {
        dev = Math.max(dev, Math.abs(gm[k]![0] - cm[k]![0]) / Math.max(1, Math.abs(cm[k]![0])), Math.abs(gm[k]![1] - cm[k]![1]) / Math.max(1, Math.abs(cm[k]![1])));
      }
      same = dev <= T_REL;
    }
    c.rec(`walk-intervals-${cls ? "active" : "inactive"}`, same, () => `${tag}: gpu ${JSON.stringify(gm)} cpu ${JSON.stringify(cm)}`, dev);
  }

  // Per-cell bounds: every GPU cell, classified by the twin at the cell's own corner.
  for (const cell of gCells) {
    const mid = (cell.t0 + cell.t1) / 2;
    const p = [0, 1, 2].map((a) => ray.o[a]! + ray.d[a]! * mid);
    const inside = [0, 1, 2].every((a) => p[a]! >= cell.voxel[a]! - 1e-3 && p[a]! <= cell.voxel[a]! + cell.dim + 1e-3);
    const aligned = cell.voxel.every((v) => (((v % cell.dim) + cell.dim) % cell.dim) === 0);
    c.rec("walk-cell-geometry", inside && aligned, () => `${tag}: cell ${cell.dim} at ${cell.voxel} does not contain its midpoint ${p} (aligned ${aligned})`);
    const a = getDim(g.image, cell.voxel);
    const wantActive = a >= cell.dim ? isActive(g.image, cell.voxel) : true;
    const wantMax = nodeMax(g.image, cell.voxel, cell.dim);
    c.eq("walk-cell-dim", cell.dimA, a, () => `${tag}: getDim at ${cell.voxel}`);
    c.eq("walk-cell-active", cell.active ? 1 : 0, wantActive ? 1 : 0, () => `${tag}: active at ${cell.voxel} dim ${cell.dim}`);
    c.val("walk-cell-max", cell.max, wantMax, () => `${tag}: vdbNodeMax at ${cell.voxel} dim ${cell.dim}`);
  }
}

// ---------------------------------------------------------------------------
// Running one variant.

async function runVariant(gpu: Gpu, spec: Spec, grids: Grid[], deadIds: number[], gridCount: number, walkCounters: { rays: number; cells: number; sequenceExact: number }): Promise<VariantResult> {
  const t0 = performance.now();
  const c = new Collector(spec.id);
  const options = moduleOptions(spec);
  const module = vdbModuleWgsl(options);
  const code = kernel(spec, module);
  const compileErrors: string[] = [];
  const { device } = gpu;
  device.pushErrorScope("validation");
  const sm = device.createShaderModule({ code, label: spec.id });
  const info = await sm.getCompilationInfo();
  for (const m of info.messages) if (m.type === "error") compileErrors.push(`${m.lineNum}:${m.linePos} ${m.message}`);
  const result = (): VariantResult => ({ id: spec.id, options: JSON.stringify({ ...options, buffer: options.buffer }), wgslBytes: module.length, compileErrors, cats: c.cats, failures: c.failures, ms: performance.now() - t0 });
  if (compileErrors.length > 0) {
    await device.popErrorScope();
    return result();
  }
  const makePipeline = (entryPoint: string) => device.createComputePipelineAsync({ layout: gpu.pipelineLayout, compute: { module: sm, entryPoint } });
  const wantWalk = spec.tests.includes("walk");
  const [queryPipe, walkPipe] = await Promise.all([makePipeline("queryMain"), wantWalk ? makePipeline("walkMain") : Promise.resolve(null)]);

  const batch = buildBatch(spec, grids, deadIds, gridCount, c);
  const out = await dispatch(gpu, queryPipe, new Float32Array(batch.inp), batch.count * OUT, batch.count, 32);
  batch.verify.forEach((v, k) => v(out, k * OUT));

  if (walkPipe) {
    const compiled = new Set<VDBGridTypeName>(spec.types);
    const walkGrids = grids.filter((g) => compiled.has(g.info.type));
    const rows: { g: Grid; ray: Ray; ix: number }[] = [];
    for (const g of walkGrids) g.rays.forEach((ray, ix) => rows.push({ g, ray, ix }));
    const inp = new Float32Array(rows.length * WALK_IN);
    rows.forEach(({ g, ray }, k) => inp.set([...ray.o, g.id, ...ray.d, 8], k * WALK_IN));
    const walk = await dispatch(gpu, walkPipe, inp, rows.length * WALK_OUT, rows.length, 8);
    rows.forEach(({ g, ray, ix }, k) => {
      const base = k * WALK_OUT;
      const n = walk[base]!;
      const cells: Cell[] = [];
      for (let s = 0; s < Math.min(n, MAX_SEGS); s++) {
        const q = base + 1 + s * SEG;
        cells.push({ dim: walk[q]!, dimA: walk[q + 1]!, t0: walk[q + 2]!, t1: walk[q + 3]!, active: walk[q + 4] === 1, max: walk[q + 5]!, voxel: [walk[q + 6]!, walk[q + 7]!, walk[q + 8]!] });
      }
      const cpu = hddaWalk(g.image, ray.o, ray.d, { leafDim: 8 });
      compareWalk(c, g, ix, ray, cells, n > MAX_SEGS, cpu, walkCounters);
    });
  }
  const scope = await device.popErrorScope();
  if (scope) compileErrors.push(`validation: ${scope.message}`);
  return result();
}

// ---------------------------------------------------------------------------
// Pool construction.

function rotationScaleTranslation(): number[] {
  // Column-major 4x4: scale 0.75 * rotation of 30 degrees about the unit axis (1, 2, 2) / 3, translation (40, -25, 17).
  const s = 0.75;
  const th = (30 * Math.PI) / 180;
  const [x, y, z] = [1 / 3, 2 / 3, 2 / 3];
  const cs = Math.cos(th);
  const sn = Math.sin(th);
  const t = 1 - cs;
  const r = [
    [cs + x * x * t, x * y * t - z * sn, x * z * t + y * sn],
    [y * x * t + z * sn, cs + y * y * t, y * z * t - x * sn],
    [z * x * t - y * sn, z * y * t + x * sn, cs + z * z * t],
  ];
  const m = new Array<number>(16).fill(0);
  for (let col = 0; col < 3; col++) for (let row = 0; row < 3; row++) m[col * 4 + row] = s * r[row]![col]!;
  m[12] = 40;
  m[13] = -25;
  m[14] = 17;
  m[15] = 1;
  return m;
}

async function run(): Promise<CoreParity> {
  const t00 = performance.now();
  const result: CoreParity = {
    done: false,
    variants: [],
    totals: {},
    counts: { read: 0, sample: 0, xform: 0, rays: 0, cells: 0, nodeMax: 0 },
    poolChecks: [],
    totalMs: 0,
  };
  if (!navigator.gpu) return { ...result, done: true, error: "WebGPU unavailable (navigator.gpu undefined)." };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...result, done: true, error: "No WebGPU adapter." };
  const ai = (adapter as unknown as { info?: Record<string, string> }).info ?? {};
  result.adapter = [ai.vendor, ai.architecture, ai.device, ai.description].filter(Boolean).join(" / ") || "unknown";
  result.isSwiftShader = /swiftshader/i.test(result.adapter) || /swiftshader/i.test(ai.description ?? "");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  const uncaptured: string[] = [];
  device.addEventListener("uncapturederror", (e) => uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const loaded = await Promise.all(
    NAMES.map(async (name) => {
      const [nv, sc] = await Promise.all([fetch(`/fixtures/primitives/${name}.nvdb`), fetch(`/fixtures/primitives/${name}.sidecar.json`)]);
      return {
        image: nv.ok ? NanoVDBFile.fromArrayBuffer(await nv.arrayBuffer()).gridImage(0) : null,
        sidecar: sc.ok ? ((await sc.json()) as Sidecar) : null,
      };
    }),
  );
  const missing = NAMES.filter((_, k) => loaded[k]!.image === null || loaded[k]!.sidecar === null);
  if (missing.length > 0) return { ...result, done: true, missing };

  device.pushErrorScope("validation");
  const pool = await VDBGridPool.create(device, { initialBytes: 4096, maxGrids: 2, label: "core parity" });
  const versions: number[] = [pool.version];
  const ids: number[] = [];
  const sources: { image: Uint32Array; sidecar: Sidecar | null; name: string }[] = [];
  for (let k = 0; k < NAMES.length; k++) {
    ids.push(pool.add(loaded[k]!.image!, { label: NAMES[k]! }));
    sources.push({ image: loaded[k]!.image!, sidecar: loaded[k]!.sidecar, name: NAMES[k]! });
    pool.flush();
    versions.push(pool.version);
  }
  const deadId = pool.add(loaded[0]!.image!, { label: "throwaway" });
  pool.flush();
  versions.push(pool.version);
  const instanceId = pool.add(loaded[INSTANCE_SOURCE]!.image!, { label: "torus_fog_fp8 instance" });
  pool.flush();
  versions.push(pool.version);
  pool.setTransform(instanceId, rotationScaleTranslation());
  pool.remove(deadId);
  pool.flush();
  pool.flush(); // the second update lets a real device retire the last buffer
  versions.push(pool.version);
  ids.push(instanceId);
  sources.push({ image: loaded[INSTANCE_SOURCE]!.image!, sidecar: loaded[INSTANCE_SOURCE]!.sidecar, name: "torus_fog_fp8_instance" });
  const gridCount = Math.max(...ids) + 1;

  const grids: Grid[] = ids.map((id, k) => {
    const src = sources[k]!;
    const info = pool.info(id);
    const image = src.image;
    const bbox = indexBBox(image);
    const r = rng(9000 + id);
    const wb = info.worldBBox;
    const worldPoints = Array.from({ length: WORLD_POINTS }, () => [0, 1, 2].map((a) => fr(wb.min[a]! - 5 + r() * (wb.max[a]! - wb.min[a]! + 10))) as Vec3);
    const worldDirs = Array.from({ length: WORLD_POINTS }, () => [0, 1, 2].map(() => fr(r() * 2 - 1)) as Vec3);
    return {
      id,
      name: src.name,
      image,
      info,
      sidecar: src.sidecar,
      bbox,
      readRows: readRows(image, 100 + id, src.sidecar),
      rays: makeRays(image, 7000 + id),
      worldPoints,
      worldDirs,
    };
  });
  // Instance record checks (K1.3 contract seen from the harness): the transform moved the grid, the image is untouched.
  const inst = grids[grids.length - 1]!;
  const check = (name: string, ok: boolean, detail?: string) => result.poolChecks.push({ name, ok, ...(detail === undefined ? {} : { detail }) });
  check("instance has a transform", inst.info.instanceTransform);
  check("instance image equals its source fixture", inst.image === sources[INSTANCE_SOURCE]!.image);
  check(
    "instance world bbox differs from the source's",
    JSON.stringify(inst.info.worldBBox) !== JSON.stringify(grids[INSTANCE_SOURCE]!.info.worldBBox),
  );
  check("buffer and table both grew", versions[versions.length - 1]! >= 4 && pool.gridCapacity >= 16, `version ${pool.version}, capacity ${pool.gridCapacity}`);
  check("all images at non-zero 32 B aligned offsets", grids.every((g) => g.info.imageOffset! > 0 && g.info.imageOffset! % 32 === 0));
  check("removed id has no info", (() => { try { pool.info(deadId); return false; } catch (e) { return e instanceof VDBError && e.code === "VDB_UNKNOWN_GRID"; } })());
  // checkModule: the all-types module covers this pool; a float-only module is refused naming fp8.
  check("checkModule accepts the all-types module", (() => { try { checkModule(pool, moduleOptions(SPECS.find((s) => s.id === "read/all")!)); return true; } catch { return false; } })());
  check("checkModule refuses a float-only module", (() => { try { checkModule(pool, moduleOptions(SPECS[0]!)); return false; } catch (e) { return e instanceof VDBError && e.code === "VDB_TYPE_NOT_COMPILED"; } })());

  result.pool = {
    byteLength: pool.byteLength,
    gridCapacity: pool.gridCapacity,
    versions,
    gridCount,
    ids,
    deadId,
    instanceId,
    offsets: grids.map((g) => g.info.imageOffset!),
  };
  result.expected = { variants: SPECS.map((s) => s.id) };

  // Pool bind group, shared by every variant (slot at group 1, binding 3; buffer mode binds the same buffer).
  const layout0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const layout1 = device.createBindGroupLayout({ entries: vdbModuleEntries({ slot: SLOT, visibility: GPUShaderStage.COMPUTE }) });
  const gpu: Gpu = {
    device,
    layout0,
    pipelineLayout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
    poolGroup: device.createBindGroup({ layout: layout1, entries: [pool.bindGroupEntry(SLOT)] }),
  };

  // CPU-side oracle tables the march variants need.
  const probeCache = new Map<Uint32Array, NodeProbe[]>();
  for (const g of grids) {
    if (!probeCache.has(g.image)) probeCache.set(g.image, nodeProbes(g.image));
    g.nodeProbes = probeCache.get(g.image)!;
  }
  // The twin's own stat accessors must agree with nodeMax (self-consistency of the oracle).
  for (const g of grids) {
    for (const p of g.nodeProbes ?? []) {
      if (p.kind !== "tile" && p.kind !== "far") {
        const twin = nodeMax(g.image, p.ijk, p.dim);
        if (twin !== p.want) result.poolChecks.push({ name: `oracle: nodeMax twin == nodeStats at grid ${g.id} ${p.kind}`, ok: false, detail: `${twin} vs ${p.want}` });
      }
    }
  }

  const walkCounters = { rays: 0, cells: 0, sequenceExact: 0 };
  for (const spec of SPECS) {
    const r = await runVariant(gpu, spec, grids, [deadId], gridCount, walkCounters);
    result.variants.push(r);
    const failed = Object.values(r.cats).reduce((n, s) => n + s.failed, 0) + r.compileErrors.length;
    log(`${spec.id}: ${failed === 0 ? "ok" : `${failed} FAILED`} (${Math.round(r.ms)} ms)`, failed === 0 ? "ok" : "fail");
    for (const [cat, s] of Object.entries(r.cats)) {
      const t = (result.totals[cat] ??= { total: 0, failed: 0, maxDelta: 0, maxUlp: 0 });
      t.total += s.total;
      t.failed += s.failed;
      t.maxDelta = Math.max(t.maxDelta, s.maxDelta);
      t.maxUlp = Math.max(t.maxUlp, s.maxUlp);
    }
  }
  const total = (cat: string) => result.totals[cat]?.total ?? 0;
  result.counts = {
    read: total("read-value"),
    sample: total("sample"),
    xform: total("xform"),
    rays: walkCounters.rays,
    cells: walkCounters.cells,
    nodeMax: ["leaf", "lower", "upper", "root", "tile", "far"].reduce((n, k) => n + total(`nodemax-${k}`), 0),
  };
  result.poolChecks.push({ name: `walk cell sequences identical to the twin on ${walkCounters.rays} rays (informational)`, ok: true, detail: `${walkCounters.sequenceExact}/${walkCounters.rays}` });

  const scope = await device.popErrorScope();
  const validationError = [scope?.message, ...uncaptured].filter(Boolean).join("; ");
  pool.dispose();
  device.destroy();
  return { ...result, done: true, ...(validationError ? { validationError } : {}), totalMs: performance.now() - t00 };
}

root.textContent = "";
run()
  .catch((e: unknown): CoreParity => ({
    done: true,
    error: e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e),
    variants: [],
    totals: {},
    counts: { read: 0, sample: 0, xform: 0, rays: 0, cells: 0, nodeMax: 0 },
    poolChecks: [],
    totalMs: 0,
  }))
  .then((r) => {
    window.__CORE_PARITY__ = r;
    if (r.error) log(`error: ${r.error}`, "fail");
    else if (r.missing) log(`fixtures missing: ${r.missing.join(", ")} (pnpm fixtures:bake)`, "fail");
    else {
      log(`done in ${Math.round(r.totalMs)} ms on ${r.adapter ?? "?"}`);
      for (const [cat, s] of Object.entries(r.totals)) log(`${cat}: ${s.total - s.failed}/${s.total} (max delta ${s.maxDelta.toExponential(2)}, max ulp ${s.maxUlp})`, s.failed === 0 ? "ok" : "fail");
      for (const v of r.variants) for (const f of v.failures) log(f, "fail");
    }
  });
