# Divergences

Where we do something differently from a source, and why. Columns: what the source does, what we do, the kind of
reason (platform / perf / licence / simplification / bug-workaround / extension), and when to revisit. Seeded
from v1 on 2026-11-18; add new rows at the bottom.

| ID | Date | Source | What we do differently | Kind | Revisit when |
|---|---|---|---|---|---|
| V1 | 2026-07-11 | emcfarlane `pnanovdb.wgsl` @265e8d8 | Extended in place: Fp4/Fp8/Fp16/FpN decoders, typed dispatch, world↔index map, trilinear sampling, stats readers. Upstream lines are untouched; the diff log is in `vendor/VENDOR.md`. | extension | K1.4 splits the module into profiles |
| V2 | 2026-07-11 | `PNanoVDB.h` 32-bit address mode | 64-bit child offsets are read as their low 32 bits, as upstream does in 32-bit mode. | platform | A grid over 4 GiB |
| V3 | 2026-07-11 | NanoVDB `createNanoGrid` | Our TS serializer writes the "checksum disabled" sentinel, not a CRC. No reader in this repo checks it. | simplification | A consumer validates checksums |
| V4 | 2026-07-11 | NanoVDB `createNanoGrid` | Uniform active regions are written as leaves, not tiles. Values are identical. Since K4.1, quantize keeps source tiles (Fp8 sizes equal native on the primitives); only our own dense builds still expand uniform regions. | simplification | K4.6 |
| V5 | 2026-07-11 | NanoVDB root background | Quantize fills inactive slots with the effective near-field value, not the root background (native fixtures differ: 3 vs 0). | bug-workaround | R3 parity passed 2026-07-12 with this in place; revisit if a native consumer disagrees |
| V6 | 2026-07-11 | OpenVDB `.vdb` | The TS parser rejects vector and point grids, and drops inactive tiles. (Active tiles and Blosc LZ4/zlib are supported since K4.1.) | simplification | K4.2, K4.9 |
| V7 | 2026-07-11 | three.js canvas presentation | The v1 demos render offscreen and blit to a 2D canvas, because the sandbox's SwiftShader drops the Dawn instance on canvas present. | platform | R9 passed on hardware without it (2026-07-12); drop from the default path in K2.5 |
| V8 | 2026-07-11 | three.js `Data3DTexture` + `texture3D()` | Demo 04 samples its atlas from a storage buffer, because the sandbox's Dawn rejects the 3D texture view. | platform | R9 passed on hardware without it (2026-07-12); drop in K2.5 |
| V9 | 2026-11-18 | v1 decision D4 (the library creates the device) | The core takes the host's device (D8). | — | — |
| V10 | 2026-11-18 | Plan v2 D12 (study mjurczyk's renderer, don't port it) | We port it and re-base it on the sparse core (D18). | extension | — |
