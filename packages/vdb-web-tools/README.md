# vdb-web-tools

The CPU half of [WebVDB](../../README.md): parse `.vdb` containers, build
NanoVDB grid images (from a dense array or straight from a parsed `.vdb`),
quantize to Fp8/FpN, apply affine transforms, inspect memory layout, and
write `.nvdb` files — all **pure TypeScript, zero runtime dependencies,
zero WASM** ([decision D3](../../docs/DECISIONS.md#d3--cpu-half-pure-typescript-first-wasm-as-targeted-escalation)).
WASM stays an optional escalation rung, adopted only on demonstrated need
— nothing in this package requires it.

## Install

```sh
npm install vdb-web-tools
```

No peer/runtime dependencies to bring along.

## Quickstart: the demo 07 pipeline

The full authoring round-trip — generate/obtain voxels, build a NanoVDB
grid, optionally quantize, write a real `.nvdb` file — trimmed from
[`examples/src/demos/07-builder`](../../examples/src/demos/07-builder):

```ts
import { buildFromDenseDetailed, quantize, writeNvdb } from "vdb-web-tools";

const dim = 32;
const values = new Float32Array(dim * dim * dim); // fill with your density field
// ... write nonzero values into `values` at your active voxels ...

const built = buildFromDenseDetailed(values, [dim, dim, dim], {
  gridName: "my_grid",
  background: 0,
});
// built: { image: Uint32Array, voxelCount, indexBBox, worldBBox, voxelSize, nodeCounts }

const fp8Image = quantize(built.image, "fp8"); // or "fpn"; per-leaf min/quantum, native-exact rounding

const fileBuffer = writeNvdb([fp8Image]); // ArrayBuffer — a real NanoVDB2 file, ready to save/upload
```

`fileBuffer` round-trips through `nanovdb-wgsl`'s `NanoVDBFile.fromArrayBuffer`
byte-for-byte (that loader is this package's read path — see its README).

## Quickstart: parse a real `.vdb` and build NanoVDB from it

The drop-in-a-file pipeline, trimmed from
[`examples/src/demos/06-explorer`](../../examples/src/demos/06-explorer):

```ts
import { parseVdb, buildFromVdbDetailed, inspect } from "vdb-web-tools";

const buffer = await fetch("/fixtures/vdb-samples/utahteapot.vdb").then((r) => r.arrayBuffer());
const vdbFile = parseVdb(buffer); // { fileVersion, grids: VdbGrid[] }

const built = buildFromVdbDetailed(vdbFile.grids[0]!); // streams leaves, no dense allocation
const report = inspect(built.image);
console.log(report.nodeCounts, report.memoryBreakdown);
```

`buildFromVdb`/`buildFromVdbDetailed` stream the parser's leaves straight
into the NanoVDB serializer (proven on a 7M-voxel teapot with no dense
array), carrying the `.vdb` file's uniform scale+translate transform into
the NanoVDB `Map`.

## API surface

| Function | What it does |
|---|---|
| `parseVdb(buffer)` | Reads a `.vdb` container: FloatGrid, 5-4-3 tree, uncompressed or zlib. Throws a clear error for blosc. |
| `buildFromDense` / `buildFromDenseDetailed` | Dense `Float32Array` -> NanoVDB FLOAT grid image |
| `buildFromLeavesDetailed` | Lower-level: build from an already-leaf-shaped source (what `buildFromVdb` and `quantize` are built on) |
| `buildFromVdb` / `buildFromVdbDetailed` | Parsed `.vdb` grid -> NanoVDB FLOAT grid image, leaf-streamed |
| `quantize` / `quantizeDetailed` | FLOAT grid image -> Fp8/FpN (per-leaf min/quantum; FpN bit-width picked per leaf) |
| `transform` | Metadata-only affine edit (uniform scale + translate) on a copy; throws with a GPU-resample pointer for rotation/shear ([D6](../../docs/DECISIONS.md#d6--browser-first-companion-service-is-an-interim-crutch-to-eliminate)) |
| `inspect` | Node counts + memory-by-section breakdown, summing to the grid's total byte size |
| `writeNvdb` | Serializes one or more grid images into a `.nvdb` file `ArrayBuffer` (NanoVDB2 layout) |
| `readNvdb` | Stub — reading `.nvdb` is `nanovdb-wgsl`'s `NanoVDBFile` (kept in that package since it's shared with the GPU path) |

## Supported `.vdb` subset

| | Support |
|---|---|
| Grid type | `FloatGrid` |
| Tree configuration | 5-4-3 (OpenVDB's default) |
| Compression | none, zlib |
| Compression | blosc — throws a clear "not supported" error |
| Half-float leaf storage | yes |
| Transform | uniform scale + translate (affine `Map`); non-uniform/rotated throws on `transform()`, parses fine on `parseVdb()`/`buildFromVdb()` |

## Native parity

Every function above is validated byte/value-wise against real
`nanovdb_convert`/OpenVDB output on the project's fixture corpus (four
openvdb.org samples + primitives). The one anchor that needs a native
machine to close out (this package's numbers already match at every
tested coordinate) is tracked in
[`docs/handoffs/PHASE-5.md`](../../docs/handoffs/PHASE-5.md#native-machine-checklist-the-deferred-anchor)
— see that doc for the full parity checklist and known semantic edge
cases (root background vs. near-field tile fill, active-tile compression,
CRC).

## Packaging note (for contributors)

Like the other two WebVDB packages, this one's `"."` export has a
`"development"` condition (`./src/index.ts`) for in-workspace resolution
and `"import"`/`"types"` conditions pointing at the built `dist/` for
published consumers.
