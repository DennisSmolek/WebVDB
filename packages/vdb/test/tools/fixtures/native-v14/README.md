# Native v14 fixtures (K7.C)

Small files written by native OpenVDB / NanoVDB **v14.0.0** and by NVIDIA Warp, committed so CI checks that files
in the wild open in `@pmndrs/vdb/tools` (`test/tools/native-v14.test.ts`, `test/tools/nvdb-blosc.test.ts`). They
are procedural shapes made by the programs in `generators/` (ours, MIT); total about 100 KiB.

| File | Writer | What it covers |
|---|---|---|
| `sphere_fog_blosc.nvdb` | NanoVDB 32.11.0 `io::writeGrid(.., Codec::BLOSC)` (`gen_nvdb_blosc.cpp`) | One Float fog sphere, LZ4 Blosc frames from the Blosc OpenVDB v14 vendors (`nanovdb_blosc`) |
| `two_grids_blosc.nvdb` | same, after `mergeGrids` | One segment, two grids (Float + Fp8) |
| `warp_sphere_blosclz.nvdb` | Warp 1.19.0 `Volume.save_to_nvdb(codec="blosc")` (`warp_blosclz.py`; python-blosc 1.11.4, c-blosc 1.21) | BloscLZ frames (typesize 8, byte shuffle), `NanoVDB0` header, empty grid name, FileMetaData `indexBBox`/`voxelCount` left zero, `fileSize` without the 8-byte prefix |
| `half_{none,zip,blosc}.vdb` | OpenVDB 14.0.0 `io::File::write` (`gen_vdb_half.cpp`), file version 225 | Two `HalfGrid`s (`Tree_half_5_4_3`): `fog` (leaf voxels incl. the largest and smallest normal half, an inactive voxel with its own value, an active lower-node tile) and `ls` (a narrow-band level set: inactive values of both signs). Active-mask compression in all three; the zip file marks the grids class Unknown, since OpenVDB never ZIPs a fog volume or level set |
| `nvdb-probes.json`, `warp-probes.json`, `vdb-probes.json` | the generators | Values and active states read with native accessors (Warp: the dense source array), the Warp image's SHA-256 |
| `nvdb-images.json` | `shasum`-style over the generators' raw `.image` dumps | Each native grid image's size and SHA-256, also with header bytes 8–32 zeroed (merging grids rewrites the checksum, index and count) |

## Regenerating

Native tools, built from the `v14.0.0` tag of AcademySoftwareFoundation/openvdb (macOS, Apple clang 21; any
platform works). OpenVDB needs oneTBB (built from source, v2022.1.0) and zlib; Blosc is vendored since v14.

```sh
git clone --depth 1 --branch v14.0.0 https://github.com/AcademySoftwareFoundation/openvdb.git ovdb
# NanoVDB is header-only; its Blosc is one C file.
clang -c -O1 -I ovdb/ext/blosc/openvdb/ext/blosc ovdb/ext/blosc/openvdb/ext/blosc/nanovdb_blosc.c -o nanovdb_blosc.o
clang++ -std=c++17 -O1 -DNANOVDB_USE_BLOSC -I ovdb/nanovdb -I ovdb/ext/blosc/openvdb/ext/blosc \
  generators/gen_nvdb_blosc.cpp nanovdb_blosc.o -o gen_nvdb_blosc
./gen_nvdb_blosc out
# OpenVDB core (cmake: OPENVDB_BUILD_CORE=ON, USE_ZSTD=OFF, a prefix holding oneTBB), then:
clang++ -std=c++17 -O1 -I prefix/include generators/gen_vdb_half.cpp -L prefix/lib -lopenvdb -ltbb \
  -Wl,-rpath,prefix/lib -o gen_vdb_half
./gen_vdb_half out
# Warp (CPU is enough): a venv with warp-lang and blosc.
python -I generators/warp_blosclz.py out
```

The `.nvdb` files are deterministic (byte-identical on rerun); the `.vdb` files carry a random UUID. Native tools
read every file here (`nanovdb_print -l` lists the BLOSC files, the Warp one included).
