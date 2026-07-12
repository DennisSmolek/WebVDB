/**
 * Demo 03 — gpu-read: read a NanoVDB volume on the GPU with WebGPU.
 *
 * Goal (docs/PLAN.md Phase 3, docs/SPEC.md §5 row 03 — a *stated main goal*,
 * marked "educational"): a single, self-contained file that a newcomer can
 * read top to bottom and come away understanding how NanoVDB volumes get
 * from disk to a GPU shader and back. Everything you need is in this one
 * file; there is no framework underneath it, on purpose — this is raw
 * WebGPU (`navigator.gpu` directly), not three.js/TSL. Demos 01 and 02 show
 * the TSL binding (`wgslFn` + a `ptr<storage, …>` parameter) if that's what
 * you're after; here we skip that layer so there is one fewer thing between
 * you and the bytes.
 *
 * The five sections below are the five things you need to know:
 *
 *   1. What a NanoVDB grid IS — a single flat buffer, no pointers.
 *   2. What a `.nvdb` FILE adds around that grid, and how we get the grid
 *      back out of it (using the `NanoVDBFile` loader from `nanovdb-wgsl`
 *      — we teach its API here, we do not reimplement it).
 *   3. How those bytes reach the GPU: unchanged, in one copy.
 *   4. How a WGSL function walks the grid's 4-level tree to read one voxel.
 *   5. Wiring it into an actual compute dispatch, and checking the answers.
 */
import { NanoVDBFile } from "nanovdb-wgsl";

// ---------------------------------------------------------------------------
// Types shared with the e2e harness (window.__DEMO03__) and the page table.
// ---------------------------------------------------------------------------

interface Sample {
  ijk: [number, number, number];
  value: number;
  active: boolean;
}

interface Sidecar {
  grid: { name: string };
  samples: Sample[];
}

interface Demo03Result {
  total: number;
  matched: number;
  error?: string;
}

declare global {
  interface Window {
    __DEMO03__?: Demo03Result;
  }
}

const FIXTURE_NVDB = "/fixtures/primitives/sphere_fog_float.nvdb";
const FIXTURE_SIDECAR = "/fixtures/primitives/sphere_fog_float.sidecar.json";
const SAMPLE_COUNT = 8; // "read ~8 voxels" — the first 8 rows of the sidecar
const VALUE_EPS = 1e-5;

const root = document.querySelector<HTMLDivElement>("#demo-root");
if (!root) throw new Error("missing #demo-root element");

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  if (cls) p.className = cls;
  p.textContent = text;
  root!.appendChild(p);
}

// =============================================================================
// 1. What IS a NanoVDB grid?
// =============================================================================
//
// It is one contiguous, pointer-free block of memory — an immutable snapshot
// of a sparse volume where every "this points to that" relationship is a
// plain byte offset, not a pointer. Compact ASCII map of the layout (full
// version: docs/FEASIBILITY.md §3; authoritative byte offsets:
// packages/nanovdb-wgsl/vendor/upstream/PNanoVDB.h):
//
//   byte 0                                                          byte N
//   +----------+----------+------------------+-----------+-----------+------+
//   | GridData | TreeData |  RootData + tiles | 32^3 uppers | 16^3 lowers | 8^3
//   |  672 B   |   64 B   |  (background val,  |  (internal | (internal   | leaves
//   |          |          |   one tile per     |   nodes)   |   nodes)    | (voxel
//   |          |          |   populated 4096^3 |            |             |  data)
//   |          |          |   region)          |            |             |
//   +----------+----------+------------------+-----------+-----------+------+
//
// Fixed 4-level tree, always: root -> 32^3 "upper" internal node -> 16^3
// "lower" internal node -> 8^3 "leaf" node (512 voxels). No recursion, no
// variable depth — every voxel lookup is the same four steps. That
// uniformity is *why* this format works as a GPU shader: there is no tree
// shape to branch on, just four fixed-size hops.
//
// Two more things that make the format GPU-shaped:
//   - Everything is 32-byte aligned, so the whole blob can be uploaded to a
//     GPUBuffer at offset 0 with zero repacking (see section 3).
//   - Each node carries an "active" bitmask (one bit per voxel/child slot:
//     is this slot populated, or does it fall back to a constant value?)
//     alongside a "child" bitmask (does this slot hold a child pointer, or a
//     constant tile value?). Both are stored as plain u32 words, so reading
//     them on the GPU never needs 64-bit arithmetic — see section 4.

// =============================================================================
// 2. What does a `.nvdb` FILE add around that grid?
// =============================================================================
//
// The tree above is the "grid image" — what actually gets bound to the GPU.
// A `.nvdb` file wraps zero or more grid images in a small amount of
// bookkeeping so a program can find them without already knowing their
// size: a 16-byte FileHeader (magic + version + grid count + compression
// codec), then for each grid a 176-byte FileMetaData record (name, type,
// bounding boxes, voxel size, and critically the *decompressed* byte size
// of the grid image) followed by the grid's name string and its
// (optionally zlib-compressed) bytes.
//
// None of that parsing happens in this file — that's the job of
// `NanoVDBFile` (packages/nanovdb-wgsl/src/nvdb-file.ts, exported as
// `nanovdb-wgsl`). This demo is about the traversal, not the container
// format, so we just call it:
//
//     const file = await NanoVDBFile.fromURL(url);
//     const gridImage = file.gridImage(0); // Uint32Array, ready for the GPU
//
// `gridImage(0)` decompresses (if needed) and hands back exactly the flat
// buffer described in section 1 — GridData at byte 0, TreeData right after
// it, and so on. If you need the full loader API (multiple grids, blind
// data, other codecs), that lives in the same package; we only touch the
// two calls above.

// =============================================================================
// 3. Getting those bytes onto the GPU — zero transformation
// =============================================================================
//
// Because the grid image is already a flat array of 32-bit words with every
// cross-reference expressed as a byte offset, "uploading" it is not a
// conversion step at all — it's a `memcpy`. We create a `storage` buffer
// sized to match and copy the `Uint32Array` straight in:
//
//     const gridBuffer = device.createBuffer({
//       size: gridImage.byteLength,
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//     });
//     device.queue.writeBuffer(gridBuffer, 0, gridImage);
//
// That's it. Nothing on the GPU side "loads" this into some other runtime
// representation — the bytes we just copied *are* the tree the shader will
// walk in section 4. Contrast this with, say, a PNG texture upload, which
// decodes/repacks pixel formats; here the wire format and the in-shader
// format are identical by design.
//
// The actual buffer creation for this demo happens in `run()` below,
// alongside the two small buffers holding our voxel coordinates and results.

// =============================================================================
// 4. Reading one voxel in WGSL: root -> upper -> lower -> leaf
// =============================================================================
//
// This is the whole algorithm, hand-written, restricted on purpose to FLOAT
// grids (the fixture used below is one). It is a direct transliteration of
// the audited reference in `examples/src/demos/01-hello-nvdb/traversal.ts`
// and the vendored, generated module `packages/nanovdb-wgsl/vendor/
// pnanovdb.wgsl` (imported as `nanovdb-wgsl/pnanovdb.wgsl?raw` in demos
// 01/02), restricted here to one grid type and stripped of caching, HDDA
// raymarching, trilinear sampling and quantized (Fp4/8/16/FpN) decoding —
// none of which a first read needs. For the real thing, import the vendored
// module; this copy exists so you can read one function and see the whole
// idea at once.
//
// Two habits of the code below are worth flagging before you read it:
//
//   - WGSL has no 64-bit integer type. NanoVDB's on-disk fields that are
//     logically 64 bits (the magic number, a root tile's key, a child's
//     byte offset) are read as TWO adjacent u32 words, conventionally
//     called `lo` (low 32 bits) and `hi` (high 32 bits). This fixture is
//     small enough that every offset used below fits in `lo`; `hi` is only
//     consulted where correctness demands it (the magic check, the key
//     comparison). Grids near/above 4 GiB would need `hi` carried through
//     the address math too — see the vendored module for that.
//   - `active` is a *reserved keyword* in WGSL. The boolean this function
//     returns is therefore called `act`.
//
// The function returns `vec2<f32>(value, activeFlag)`: `activeFlag` is 1.0
// (active voxel), 0.0 (inactive — background/constant region), or negative
// as a sanity-check sentinel (-1 bad magic, -2 wrong grid type) so a caller
// can surface a real error instead of a silently wrong zero.
const NVDB_READ_FLOAT_WGSL = /* wgsl */ `
fn nvdb_read_float(grid: ptr<storage, array<u32>, read>, ijk: vec3<i32>) -> vec2<f32> {

  // ---- 0. Sanity check: is this actually a NanoVDB grid, and is it FLOAT? --
  // The magic number is the ASCII bytes "NanoVDB1" read as a little-endian
  // u64, i.e. two u32 words: low = "Nano" (0x6f6e614e), high = "VDB1"
  // (0x31424456). Reading two words instead of one 64-bit value is exactly
  // the "no i64 in WGSL" habit described above.
  let magic_lo: u32 = (*grid)[0u];
  let magic_hi: u32 = (*grid)[1u];
  if (magic_lo != 0x6f6e614eu || magic_hi != 0x31424456u) {
    return vec2<f32>(0.0, -1.0); // sentinel: this buffer is not a NanoVDB grid
  }
  // GridData.mGridType lives at byte 636 (PNANOVDB_GRID_OFF_GRID_TYPE);
  // "1" is PNANOVDB_GRID_TYPE_FLOAT. Byte offset -> word index is always
  // "shift right by 2" (divide by 4, the word size) because every field
  // here is 4-byte aligned.
  let grid_type: u32 = (*grid)[636u >> 2u];
  if (grid_type != 1u) {
    return vec2<f32>(0.0, -2.0); // sentinel: not a FLOAT grid (this fn only reads FLOAT)
  }

  // ---- 1. Tree -> root node address ---------------------------------------
  // GridData is a fixed 672 bytes, so TreeData always starts right after it.
  // TreeData.mNodeOffset[3] (the offset to the ROOT node) is a u64 at
  // byte 24 within TreeData; again we only need the low word here.
  let tree_addr: u32 = 672u; // PNANOVDB_GRID_SIZE
  let root_off: u32 = (*grid)[(tree_addr + 24u) >> 2u];
  let root_addr: u32 = tree_addr + root_off;

  // ---- 2. Turn a voxel coordinate into the root's 64-bit tile key ---------
  // The root organizes space into 4096^3 tiles (4096 = 32 upper * 16 lower *
  // 8 leaf voxels per side). A tile's key packs the tile-space coordinates
  // (ijk >> 12, since 2^12 = 4096) into 21 bits per axis. u32(ijk.x) >> 12u
  // reinterprets the sign bits of a negative i32 as unsigned first (so the
  // shift is a plain logical shift, matching how the key was built on the
  // CPU) and then divides by 4096. The 63-bit key is then split across two
  // u32 words the same way NanoVDB's own (non-native-64) fallback path does.
  let iu: u32 = u32(ijk.x) >> 12u;
  let ju: u32 = u32(ijk.y) >> 12u;
  let ku: u32 = u32(ijk.z) >> 12u;
  let key_lo: u32 = ku | (ju << 21u);
  let key_hi: u32 = (iu << 10u) | (ju >> 11u);

  // ---- 3. Root: linear-scan the tile table for our key --------------------
  // The root has no bitmask shortcut — it just lists its populated tiles and
  // we scan them (root tiles are typically few; this is the one place the
  // traversal isn't O(1), and it's fine because there aren't many tiles).
  // RootData.mTableSize is a u32 at root+24; the tile table starts at
  // root+64 (the FLOAT root's header size); each tile is 32 bytes: key (u64
  // @0), child byte-offset (i64 @8), active state (u32 @16), value (f32 @20).
  let tile_count: u32 = (*grid)[(root_addr + 24u) >> 2u];
  let tile0: u32 = root_addr + 64u;
  var tile_addr: u32 = 0u; // 0 is never a valid tile address, so it doubles as "not found"
  for (var i: u32 = 0u; i < tile_count; i = i + 1u) {
    let cand: u32 = tile0 + i * 32u;
    let k_lo: u32 = (*grid)[(cand + 0u) >> 2u];
    let k_hi: u32 = (*grid)[(cand + 4u) >> 2u];
    if (k_lo == key_lo && k_hi == key_hi) {
      tile_addr = cand;
      break;
    }
  }

  // ---- 4a. No matching tile: this voxel is outside every populated region -
  // "Background" is the constant value the whole sparse volume implicitly
  // holds everywhere nothing was ever written (for a fog volume, this is
  // usually 0 — empty space).
  if (tile_addr == 0u) {
    let bg: f32 = bitcast<f32>((*grid)[(root_addr + 28u) >> 2u]);
    return vec2<f32>(bg, 0.0); // inactive
  }

  // ---- 4b. Tile found, but it has no child: the whole 4096^3 region is one
  // constant value (this is what makes sparse volumes sparse — most of a
  // shape's bounding box is one or two of these constant tiles, not
  // millions of individually-stored voxels).
  let child_lo: u32 = (*grid)[(tile_addr + 8u) >> 2u];
  let child_hi: u32 = (*grid)[(tile_addr + 12u) >> 2u];
  if (child_lo == 0u && child_hi == 0u) {
    let v: f32 = bitcast<f32>((*grid)[(tile_addr + 20u) >> 2u]);
    let state: u32 = (*grid)[(tile_addr + 16u) >> 2u];
    return vec2<f32>(v, select(0.0, 1.0, state != 0u));
  }

  // ---- 5. Descend into the upper internal node (32 voxels per side) -------
  // The tile DOES have a child, so we follow it: the child offset is
  // relative to the ROOT node's address (not the tile's), giving the upper
  // node's address. Every internal/leaf node in NanoVDB works this same
  // way — "my child's address" is "my own address plus a stored offset".
  let upper_addr: u32 = root_addr + child_lo;
  // Which of the node's 32*32*32 = 32768 slots does our voxel fall into?
  // Mask off the low 12 bits of each axis (position within this 4096^3
  // region), then divide by 128 (2^7) because each slot spans 128 voxels
  // per side at this level, and pack the three 5-bit results into one
  // linear index (>>10, >>5, >>0 place x/y/z in non-overlapping bit ranges).
  let un: u32 =
    (((u32(ijk.x) & 4095u) >> 7u) << 10u) +
    (((u32(ijk.y) & 4095u) >> 7u) << 5u) +
    ((u32(ijk.z) & 4095u) >> 7u);
  // The child mask is one bit per slot: is there a lower-node child here,
  // or just a constant tile value? It's stored as an array of u32 words
  // (32 bits at a time), so "bit un" lives in word "un / 32" at bit "un % 32".
  let ucm: u32 = (*grid)[(upper_addr + 4128u + 4u * (un >> 5u)) >> 2u];
  if (((ucm >> (un & 31u)) & 1u) == 0u) {
    // No child: this slot's value comes straight from the upper node's own
    // value table, and its active flag comes from a SEPARATE mask (the
    // value mask) using the exact same bit-indexing trick.
    let v: f32 = bitcast<f32>((*grid)[(upper_addr + 8256u + 8u * un) >> 2u]);
    let uvm: u32 = (*grid)[(upper_addr + 32u + 4u * (un >> 5u)) >> 2u];
    return vec2<f32>(v, select(0.0, 1.0, ((uvm >> (un & 31u)) & 1u) != 0u));
  }

  // ---- 6. Descend into the lower internal node (16 voxels per side) -------
  // Same shape as the upper step, one level finer: the upper node's table
  // entry at slot un holds the lower node's child offset (relative to the
  // UPPER node's own address, not the grid's), and the coordinate math
  // shrinks from "128 voxels per slot" to "8 voxels per slot".
  let ut_addr: u32 = upper_addr + 8256u + 8u * un;
  let lower_off: u32 = (*grid)[ut_addr >> 2u];
  let lower_addr: u32 = upper_addr + lower_off;
  let ln: u32 =
    (((u32(ijk.x) & 127u) >> 3u) << 8u) +
    (((u32(ijk.y) & 127u) >> 3u) << 4u) +
    ((u32(ijk.z) & 127u) >> 3u);
  let lcm: u32 = (*grid)[(lower_addr + 544u + 4u * (ln >> 5u)) >> 2u];
  if (((lcm >> (ln & 31u)) & 1u) == 0u) {
    let v: f32 = bitcast<f32>((*grid)[(lower_addr + 1088u + 8u * ln) >> 2u]);
    let lvm: u32 = (*grid)[(lower_addr + 32u + 4u * (ln >> 5u)) >> 2u];
    return vec2<f32>(v, select(0.0, 1.0, ((lvm >> (ln & 31u)) & 1u) != 0u));
  }

  // ---- 7. Descend into the leaf node (8 voxels per side, 512 voxels) ------
  // The bottom of the tree: an actual per-voxel value table, no more child
  // pointers. leaf_n (0..511) is the same "mask off low bits, pack into
  // one index" trick, one level finer still (1 voxel per slot now).
  let lt_addr: u32 = lower_addr + 1088u + 8u * ln;
  let leaf_off: u32 = (*grid)[lt_addr >> 2u];
  let leaf_addr: u32 = lower_addr + leaf_off;
  let leaf_n: u32 =
    ((u32(ijk.x) & 7u) << 6u) +
    ((u32(ijk.y) & 7u) << 3u) +
    (u32(ijk.z) & 7u);
  // The leaf's own active mask, same bit trick as every level above.
  let lvm: u32 = (*grid)[(leaf_addr + 16u + 4u * (leaf_n >> 5u)) >> 2u];
  let act: f32 = select(0.0, 1.0, ((lvm >> (leaf_n & 31u)) & 1u) != 0u);
  // FLOAT leaves store one f32 per voxel, packed contiguously starting at
  // byte offset 96 within the leaf: byte = 96 + (32 bits * leaf_n) / 8.
  // (Quantized grids replace this one line with a decode — a multiply-add
  // against a per-leaf min/quantum pair — everything above it is unchanged;
  // see the vendored module if that's what brought you here.)
  let val_addr: u32 = leaf_addr + 96u + ((32u * leaf_n) >> 3u);
  let v: f32 = bitcast<f32>((*grid)[val_addr >> 2u]);
  return vec2<f32>(v, act);
}
`;

// =============================================================================
// 5. Wiring it up: a tiny raw-WebGPU compute dispatch, and grading the answers
// =============================================================================
//
// The rest of this file is plumbing: get a GPUDevice, upload the grid image
// and a handful of voxel coordinates as storage buffers, run one compute
// pass that calls `nvdb_read_float` once per coordinate, read the two
// results back, and compare them against the sidecar's native ground truth.

// The compute shader is `nvdb_read_float` (section 4) plus a thin entry
// point: unpack this thread's `ijk` from a flat i32 buffer, call the
// function, write its `vec2<f32>(value, active)` into two output buffers.
const DISPATCH_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> grid_buffer: array<u32>;
@group(0) @binding(1) var<storage, read> coord_buffer: array<i32>;
@group(0) @binding(2) var<storage, read_write> out_value: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_active: array<u32>;

@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&out_value)) {
    return; // guard: workgroup size (8) happens to match our sample count exactly
  }
  let ijk = vec3<i32>(coord_buffer[i * 3u], coord_buffer[i * 3u + 1u], coord_buffer[i * 3u + 2u]);
  let result = nvdb_read_float(&grid_buffer, ijk);
  out_value[i] = result.x;
  out_active[i] = select(0u, 1u, result.y > 0.5);
}
`;

function alignTo4(n: number): number {
  return Math.ceil(n / 4) * 4;
}

/** A storage buffer we only ever write from JS and read from the shader. */
function makeInputStorage(device: GPUDevice, data: Uint32Array | Int32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: Math.max(4, alignTo4(data.byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data.slice());
  return buf;
}

/** A storage buffer the shader writes into, sized for a later readback copy. */
function makeOutputStorage(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, alignTo4(byteLength)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function makeReadback(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: Math.max(4, alignTo4(byteLength)),
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
}

async function run(): Promise<void> {
  root!.textContent = "";
  log("Demo 03 — gpu-read: reading a NanoVDB volume on the GPU", "h");

  if (!navigator.gpu) {
    const msg = "WebGPU unavailable (navigator.gpu undefined) — needs a secure context + adapter.";
    log(msg, "fail");
    window.__DEMO03__ = { total: 0, matched: 0, error: msg };
    return;
  }

  // --- Section 2 in practice: load the .nvdb file and pull out the grid image
  const [file, sidecar] = await Promise.all([
    NanoVDBFile.fromURL(FIXTURE_NVDB),
    fetch(FIXTURE_SIDECAR).then((r) => {
      if (!r.ok) throw new Error(`sidecar fetch failed: ${r.status}`);
      return r.json() as Promise<Sidecar>;
    }),
  ]);
  const gridImage = file.gridImage(0); // Uint32Array — the flat grid image from section 1
  const gridMeta = file.grids[0]!;
  const samples = sidecar.samples.slice(0, SAMPLE_COUNT);
  const total = samples.length;
  log(
    `Loaded grid "${gridMeta.name}" (${gridMeta.gridType}): ${gridImage.length} u32 words ` +
      `(${(gridImage.byteLength / 1024 / 1024).toFixed(2)} MiB). Reading ${total} voxels.`,
  );

  // --- Device-first bootstrap: request our own adapter/device so we can ask
  // for a large-enough maxStorageBufferBindingSize before creating buffers
  // (the default 128 MiB limit is already plenty for this fixture, but real
  // grids can be bigger — see demo 01 for the "raise the limit" version).
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    const msg = "No WebGPU adapter (headless without SwiftShader?).";
    log(msg, "fail");
    window.__DEMO03__ = { total, matched: 0, error: msg };
    return;
  }
  const device = await adapter.requestDevice();

  // --- Section 3 in practice: the grid's raw bytes, unchanged, on the GPU.
  const gridBuffer = makeInputStorage(device, gridImage);
  log(`Uploaded the grid image as a ${gridImage.byteLength}-byte storage buffer — no repacking.`, "ok");

  // Flatten the 8 voxel coordinates we're probing into one i32 buffer
  // (x0,y0,z0, x1,y1,z1, …) — the simplest layout a compute shader can index.
  const coordData = new Int32Array(total * 3);
  for (let i = 0; i < total; i++) {
    const [x, y, z] = samples[i]!.ijk;
    coordData[i * 3 + 0] = x;
    coordData[i * 3 + 1] = y;
    coordData[i * 3 + 2] = z;
  }
  const coordBuffer = makeInputStorage(device, coordData);
  const valueOut = makeOutputStorage(device, total * 4);
  const activeOut = makeOutputStorage(device, total * 4);
  const valueReadback = makeReadback(device, total * 4);
  const activeReadback = makeReadback(device, total * 4);

  // --- Compile the shader (section 4's function + this section's entry
  // point) and check for compile errors before we bother dispatching.
  const shaderModule = device.createShaderModule({ code: `${NVDB_READ_FLOAT_WGSL}\n${DISPATCH_WGSL}` });
  const info = await shaderModule.getCompilationInfo();
  const compileErrors = info.messages.filter((m) => m.type === "error");
  if (compileErrors.length > 0) {
    const detail = compileErrors.map((m) => `line ${m.lineNum}:${m.linePos}: ${m.message}`).join("\n");
    const msg = `WGSL compilation failed:\n${detail}`;
    log(msg, "fail");
    window.__DEMO03__ = { total, matched: 0, error: msg };
    return;
  }

  // A single bind group covering the shader's four bindings; `layout:
  // "auto"` is fine here because there is exactly one pipeline (the trap
  // documented in the Phase 2 harness — auto-derived layouts aren't safe to
  // *share* across multiple pipelines — doesn't apply with only one).
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gridBuffer } },
      { binding: 1, resource: { buffer: coordBuffer } },
      { binding: 2, resource: { buffer: valueOut } },
      { binding: 3, resource: { buffer: activeOut } },
    ],
  });

  // --- The dispatch itself: one workgroup of 8 threads, one thread per voxel.
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(valueOut, 0, valueReadback, 0, valueReadback.size);
  encoder.copyBufferToBuffer(activeOut, 0, activeReadback, 0, activeReadback.size);
  device.queue.submit([encoder.finish()]);

  await Promise.all([valueReadback.mapAsync(GPUMapMode.READ), activeReadback.mapAsync(GPUMapMode.READ)]);
  const gotValues = new Float32Array(valueReadback.getMappedRange().slice(0));
  const gotActive = new Uint32Array(activeReadback.getMappedRange().slice(0));
  valueReadback.unmap();
  activeReadback.unmap();

  // --- Grade the answers against the sidecar's native ground truth (values
  // computed by the real, C++ NanoVDB library reading the same fixture).
  let matched = 0;
  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr><th>#</th><th>ijk</th><th>native value</th><th>GPU value</th>" +
    "<th>act(native/GPU)</th><th></th></tr></thead>";
  const tbody = document.createElement("tbody");
  for (let i = 0; i < total; i++) {
    const s = samples[i]!;
    const gotValue = gotValues[i]!;
    const gotIsActive = gotActive[i] !== 0;
    const ok = Math.abs(gotValue - s.value) <= VALUE_EPS && gotIsActive === s.active;
    if (ok) matched++;
    const tr = document.createElement("tr");
    tr.className = ok ? "row-ok" : "row-fail";
    tr.innerHTML =
      `<td>${i}</td><td>[${s.ijk.join(", ")}]</td>` +
      `<td>${s.value.toFixed(6)}</td><td>${gotValue.toFixed(6)}</td>` +
      `<td>${s.active ? "1" : "0"}/${gotIsActive ? "1" : "0"}</td>` +
      `<td>${ok ? "✓" : "✗"}</td>`;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  root!.appendChild(table);

  window.__DEMO03__ = { total, matched };
  log(
    matched === total
      ? `All ${total} voxels match native ground truth.`
      : `${matched}/${total} voxels matched native ground truth.`,
    matched === total ? "ok" : "fail",
  );

  gridBuffer.destroy();
  coordBuffer.destroy();
  valueOut.destroy();
  activeOut.destroy();
  valueReadback.destroy();
  activeReadback.destroy();
}

run().catch((err: unknown) => {
  const message = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
  log(`ERROR: ${message}`, "fail");
  const prev = window.__DEMO03__;
  window.__DEMO03__ = {
    total: prev?.total ?? 0,
    matched: prev?.matched ?? 0,
    error: message,
  };
});
