// K1.4b GPU parity probe, run by scripts/check-wgsl-variants.mjs after the
// compile pass. Small by design: the committed regression gate is K1.6's
// e2e/core-parity.spec.ts (examples/src/harness/core-parity.ts), which covers
// everything below (walk by covered intervals, nodeMax, dead ids, outside:'zero')
// across the whole variant matrix. This script stays as a quick local check.
//
// One VDBGridPool (the real class, from packages/vdb/dist, in the page) holds:
//   id 0 sphere_fog_float, id 1 removed (dead), id 2 torus_fog_fp8,
//   id 3 torus_fog_fpn, id 4 an 8^3 block of 1.0 with background 3 (built by tools).
// Probes, each compared with the tools CPU twin in Node:
//   walk      hddaWalk (leafDim 8) for rays on ids 0, 2, 3: same cells (dim,
//             level, voxel, active), max exact, t within 1e-5 relative
//   nodeMax   vdbNodeMax at active voxels of the Fp grids (leaf stats fix), dims 8/128/4096/2^20
//   dead      ids 1 (removed) and 99 (out of range): no ray hit, type 0, empty bbox, zero transforms
//   zero      vdbSample with outside 'zero', trilinear and nearest, on id 4 vs vdbSampleTrilinear/vdbSampleNearest

import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const FIXTURES = ["sphere_fog_float", null, "torus_fog_fp8", "torus_fog_fpn"];
const MAX_SEGS = 192;
const SEG = 9;
const SLOT = { group: 1, firstBinding: 0 };

export function marchProbeAvailable() {
  return FIXTURES.every((n) => n === null || existsSync(path.join(ROOT, `fixtures/primitives/${n}.nvdb`)));
}

/** Deterministic PRNG (mulberry32), as in test/tools/cpu-march.test.ts. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds the probe inputs in Node. Returns what the page needs and what Node compares against. */
export async function buildMarchProbe() {
  const core = await import(path.join(ROOT, "packages/vdb/dist/core/index.js"));
  const tools = await import(path.join(ROOT, "packages/vdb/dist/tools/internal.js"));
  const images = FIXTURES.map((n) => {
    if (n === null) return null;
    const buf = readFileSync(path.join(ROOT, `fixtures/primitives/${n}.nvdb`));
    return tools.NvdbFile.fromArrayBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).gridImage(0);
  });
  images.push(tools.vdbBuildFromDense(new Float32Array(512).fill(1), [8, 8, 8], { background: 3 }));

  // Rays: index space (identity transforms are not needed: the walk works on index-space rays).
  const rays = [];
  const r = rng(4242);
  for (const g of [0, 2, 3]) {
    const img = images[g];
    const { min, max } = tools.indexBBox(img);
    const c = [0, 1, 2].map((a) => (min[a] + max[a] + 1) / 2);
    const ext = Math.max(...[0, 1, 2].map((a) => max[a] - min[a] + 1));
    for (let k = 0; k < 48; k++) {
      const inside = k % 4 === 0;
      const o = [0, 1, 2].map((a) => Math.fround(inside ? min[a] + r() * (max[a] - min[a] + 1) : c[a] + (r() - 0.5) * 2 * ext));
      const target = [0, 1, 2].map((a) => min[a] + r() * (max[a] - min[a] + 1));
      let d = [0, 1, 2].map((a) => target[a] - o[a] + (r() - 0.5) * 1e-3);
      const len = Math.hypot(...d);
      d = d.map((x) => Math.fround(x / len));
      rays.push({ g, o, d });
    }
  }
  const walks = rays.map(({ g, o, d }) => tools.hddaWalk(images[g], o, d, { leafDim: 8 }));

  // nodeMax probes at active voxels of the Fp grids.
  const nodeProbes = [];
  for (const g of [2, 3]) {
    const img = images[g];
    let k = 0;
    for (const node of tools.treeNodes(img)) {
      if (node.level !== 0 || k++ % 7 !== 0) continue;
      const { active } = tools.leafVoxels(img, node.address);
      const n = active.indexOf(1);
      if (n < 0) continue;
      const ijk = [node.origin[0] + (n >> 6), node.origin[1] + ((n >> 3) & 7), node.origin[2] + (n & 7)];
      for (const dim of [8, 128, 4096, 1 << 20]) nodeProbes.push({ g, ijk, dim, want: tools.nodeMax(img, ijk, dim) });
    }
  }

  // outside: 'zero' samples on id 4 along x (and a few off-axis).
  const samplePoints = [];
  for (let x = -2; x <= 10; x += 0.1) samplePoints.push([Math.fround(x), 3.25, 4.5]);
  for (let k = 0; k < 64; k++) samplePoints.push([-2 + 12 * r(), -2 + 12 * r(), -2 + 12 * r()].map(Math.fround));
  const zero = { outside: "zero" };
  const sampleWant = {
    trilinear: samplePoints.map((p) => tools.vdbSampleTrilinear(images[4], p, zero)),
    nearest: samplePoints.map((p) => tools.vdbSampleNearest(images[4], p, zero)),
  };

  const types = ["float", "fp8", "fpn"];
  const marchModule = core.vdbModuleWgsl({ slot: SLOT, types, profile: "march" });
  const zeroModules = {
    trilinear: core.vdbModuleWgsl({ slot: SLOT, types, profile: "sample", outside: "zero" }),
    nearest: core.vdbModuleWgsl({ slot: SLOT, types, profile: "sample", outside: "zero", sampler: "nearest" }),
  };

  const page = {
    images: images.map((img, i) => (img === null ? null : `/probe-image/${i}`)),
    removeId: 1,
    rays: rays.flatMap(({ g, o, d }) => [...o, g, ...d, 8]),
    rayCount: rays.length,
    nodeProbes: nodeProbes.flatMap((p) => [...p.ijk, p.g, p.dim]),
    nodeCount: nodeProbes.length,
    samplePoints: samplePoints.flatMap((p) => [...p, 4]),
    sampleCount: samplePoints.length,
    marchCode: marchKernel(marchModule),
    zeroCodes: { trilinear: sampleKernel(zeroModules.trilinear), nearest: sampleKernel(zeroModules.nearest) },
    maxSegs: MAX_SEGS,
    seg: SEG,
    slot: SLOT,
  };
  const imageBytes = images.map((img) => (img === null ? null : Buffer.from(img.buffer, img.byteOffset, img.byteLength)));
  return { page, imageBytes, walks, nodeProbes, sampleWant };
}

function marchKernel(module) {
  return `${module}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

const MAX_SEGS: u32 = ${MAX_SEGS}u;
const SEG: u32 = ${SEG}u;

fn v3(i: u32) -> vec3f {
    return vec3f(inp[i], inp[i + 1u], inp[i + 2u]);
}

@compute @workgroup_size(1)
fn walkMain(@builtin(global_invocation_id) gid: vec3u) {
    let r = gid.x * 8u;
    let o = v3(r);
    let g = u32(inp[r + 3u]);
    let d = v3(r + 4u);
    let leafDim = u32(inp[r + 7u]);
    let base = gid.x * (1u + MAX_SEGS * SEG);
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
            if n < MAX_SEGS {
                let s = base + 1u + n * SEG;
                out[s] = f32(h.dim);
                out[s + 1u] = f32(dimA);
                out[s + 2u] = h.tmin;
                out[s + 3u] = min(min(h.next.x, h.next.y), min(h.next.z, h.tmax));
                out[s + 4u] = select(0.0, 1.0, act);
                out[s + 5u] = m;
                out[s + 6u] = f32(h.voxel.x);
                out[s + 7u] = f32(h.voxel.y);
                out[s + 8u] = f32(h.voxel.z);
            }
            n++;
            if !vdbHddaStep(&h) {
                break;
            }
        }
    }
    out[base] = f32(n);
}

@compute @workgroup_size(1)
fn nodeMaxMain(@builtin(global_invocation_id) gid: vec3u) {
    let p = gid.x * 5u;
    let ijk = vec3i(i32(inp[p]), i32(inp[p + 1u]), i32(inp[p + 2u]));
    var a = vdbAccessorInit(u32(inp[p + 3u]));
    out[gid.x] = vdbNodeMax(&a, ijk, u32(inp[p + 4u]));
}

@compute @workgroup_size(1)
fn deadMain(@builtin(global_invocation_id) gid: vec3u) {
    let g = select(99u, 1u, gid.x == 0u);
    var t = vec2f(0.0, 1.0e30);
    let hit = vdbRayClip(g, vec3f(0.5, 0.5, -10.0), vec3f(0.0, 0.0, 1.0), &t);
    var a = vdbAccessorInit(g);
    let o = gid.x * 12u;
    out[o] = select(0.0, 1.0, hit);
    out[o + 1u] = f32(vdbGridType(g));
    out[o + 2u] = select(0.0, 1.0, a.ok);
    out[o + 3u] = f32(vdbIndexMin(g).x);
    out[o + 4u] = f32(vdbIndexMax(g).x);
    let w = vdbWorldToIndex(g, vec3f(1.0, 2.0, 3.0));
    out[o + 5u] = w.x + w.y + w.z;
    out[o + 6u] = vdbReadValue(&a, vec3i(0));
    out[o + 7u] = f32(vdbGetDim(&a, vec3i(0)));
    out[o + 8u] = vdbNodeMax(&a, vec3i(0), 8u);
    out[o + 9u] = select(0.0, 1.0, vdbHasStats(g));
    out[o + 10u] = vdbValueMax(g);
    out[o + 11u] = f32(vdbGridCount());
}
`;
}

function sampleKernel(module) {
  return `${module}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

@compute @workgroup_size(1)
fn sampleMain(@builtin(global_invocation_id) gid: vec3u) {
    let p = gid.x * 4u;
    var a = vdbAccessorInit(u32(inp[p + 3u]));
    out[gid.x] = vdbSample(&a, vec3f(inp[p], inp[p + 1u], inp[p + 2u]));
}
`;
}

/**
 * Runs in the page (serialized by Playwright). `dist` is the URL prefix of
 * packages/vdb/dist; images come from the server at `probe.images[i]`.
 */
export async function runMarchProbeInPage({ probe, dist }) {
  const { VDBGridPool } = await import(`${dist}/core/index.js`);
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  const errors = [];
  device.pushErrorScope("validation");
  const pool = await VDBGridPool.create(device, { initialBytes: 4096 });
  for (const url of probe.images) pool.add(new Uint32Array(await (await fetch(url ?? probe.images[2])).arrayBuffer()));
  pool.remove(probe.removeId);
  pool.flush();

  const layout0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const layout1 = device.createBindGroupLayout({
    entries: [{ binding: probe.slot.firstBinding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] });
  const poolGroup = device.createBindGroup({ layout: layout1, entries: [pool.bindGroupEntry(probe.slot)] });

  async function run(code, entryPoint, input, outFloats, count) {
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    for (const m of info.messages) if (m.type === "error") errors.push(`${entryPoint}: ${m.lineNum}:${m.linePos} ${m.message}`);
    const pipeline = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } });
    const inBuf = device.createBuffer({ size: Math.max(16, input.length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, new Float32Array(input));
    const outBuf = device.createBuffer({ size: outFloats * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: outFloats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: layout0, entries: [{ binding: 0, resource: { buffer: outBuf } }, { binding: 1, resource: { buffer: inBuf } }] }));
    pass.setBindGroup(1, poolGroup);
    pass.dispatchWorkgroups(count);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outFloats * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const res = Array.from(new Float32Array(readBuf.getMappedRange()));
    readBuf.unmap();
    return res;
  }

  const walk = await run(probe.marchCode, "walkMain", probe.rays, probe.rayCount * (1 + probe.maxSegs * probe.seg), probe.rayCount);
  const nodeMax = await run(probe.marchCode, "nodeMaxMain", probe.nodeProbes, probe.nodeCount, probe.nodeCount);
  const dead = await run(probe.marchCode, "deadMain", [0], 24, 2);
  const trilinear = await run(probe.zeroCodes.trilinear, "sampleMain", probe.samplePoints, probe.sampleCount, probe.sampleCount);
  const nearest = await run(probe.zeroCodes.nearest, "sampleMain", probe.samplePoints, probe.sampleCount, probe.sampleCount);
  const scope = await device.popErrorScope();
  if (scope) errors.push(`validation: ${scope.message}`);
  pool.dispose();
  return { errors, walk, nodeMax, dead, trilinear, nearest };
}

/** Compares the page's results with the CPU twins. Returns failure lines (empty = pass) and a summary. */
export function compareMarchProbe(expected, got) {
  const fails = [...got.errors];
  const relT = (a, b) => Math.abs(a - b) <= 1e-5 * Math.max(1, Math.abs(b));
  const sameMax = (a, b) => a === b || Math.abs(a - b) <= 4 * 2 ** -23 * Math.max(1, Math.abs(b));
  let cells = 0;
  let tMax = 0;
  let rayMismatch = 0;
  const stride = 1 + MAX_SEGS * SEG;
  expected.walks.forEach((segs, ray) => {
    const base = ray * stride;
    const n = got.walk[base];
    const why = [];
    if (n !== segs.length) why.push(`count ${n} vs ${segs.length}`);
    for (let k = 0; k < Math.min(n, segs.length, MAX_SEGS) && why.length === 0; k++) {
      const s = base + 1 + k * SEG;
      const w = segs[k];
      const g = got.walk.slice(s, s + SEG);
      const level = g[1] === 1 ? 0 : g[1] === 8 ? 1 : g[1] === 128 ? 2 : 3;
      if (g[0] !== w.dim || level !== w.level || g[6] !== w.voxel[0] || g[7] !== w.voxel[1] || g[8] !== w.voxel[2]) why.push(`cell ${k}: gpu dim ${g[0]} at ${g.slice(6)} vs cpu ${w.dim} at ${w.voxel}`);
      else if ((g[4] === 1) !== w.active) why.push(`cell ${k}: active`);
      else if (!sameMax(g[5], w.max)) why.push(`cell ${k}: max ${g[5]} vs ${w.max}`);
      else if (!relT(g[2], w.t0) || !relT(g[3], w.t1)) why.push(`cell ${k}: t ${g[2]},${g[3]} vs ${w.t0},${w.t1}`);
      tMax = Math.max(tMax, Math.abs(g[2] - w.t0) / Math.max(1, Math.abs(w.t0)), Math.abs(g[3] - w.t1) / Math.max(1, Math.abs(w.t1)));
      cells++;
    }
    if (why.length > 0) {
      rayMismatch++;
      fails.push(`walk ray ${ray}: ${why[0]}`);
    }
  });
  let nodeBad = 0;
  expected.nodeProbes.forEach((p, i) => {
    if (!sameMax(got.nodeMax[i], p.want)) {
      nodeBad++;
      if (nodeBad <= 5) fails.push(`nodeMax g${p.g} ${p.ijk} dim ${p.dim}: gpu ${got.nodeMax[i]} cpu ${p.want}`);
    }
  });
  for (const [k, id] of [[0, 1], [1, 99]]) {
    const d = got.dead.slice(k * 12, k * 12 + 12);
    // hit, type, ok, min.x, max.x, w2i sum, read, dim, nodeMax, hasStats, valueMax, gridCount
    const want = [0, 0, 0, 0, -1, 0, 0, 4096, 0, 0, 0, 5];
    if (d.some((v, i) => v !== want[i])) fails.push(`dead id ${id}: got ${d.join(",")} want ${want.join(",")}`);
  }
  let sampleBad = 0;
  for (const kind of ["trilinear", "nearest"]) {
    expected.sampleWant[kind].forEach((w, i) => {
      if (Math.abs(got[kind][i] - w) > 1e-5) {
        sampleBad++;
        if (sampleBad <= 5) fails.push(`zero ${kind} sample ${i}: gpu ${got[kind][i]} cpu ${w}`);
      }
    });
  }
  const summary =
    `march probe: ${expected.walks.length} rays (${rayMismatch} mismatched), ${cells} cells compared, max |dt|/max(1,t) ${tMax.toExponential(2)}; ` +
    `${expected.nodeProbes.length} nodeMax probes (${nodeBad} bad); dead ids 1 and 99 checked; ` +
    `${expected.sampleWant.trilinear.length * 2} outside-zero samples (${sampleBad} bad)`;
  return { fails, summary };
}
