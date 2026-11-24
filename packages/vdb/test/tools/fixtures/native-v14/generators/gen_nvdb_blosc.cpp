// Writes small BLOSC (LZ4) .nvdb files with native NanoVDB (OpenVDB v14.0.0, NanoVDB 32.11) and the
// Blosc copy it vendors (ext/blosc, nanovdb_blosc). Header-only NanoVDB plus nanovdb_blosc.c:
//
//   clang++ -std=c++17 -O1 -DNANOVDB_USE_BLOSC -I <openvdb>/nanovdb -I <openvdb>/ext/blosc/openvdb/ext/blosc \
//     gen_nvdb_blosc.cpp <openvdb>/ext/blosc/openvdb/ext/blosc/nanovdb_blosc.c -o gen_nvdb_blosc
//   ./gen_nvdb_blosc <out-dir>
//
// Writes:
//   sphere_fog_blosc.nvdb    one segment, one Float fog sphere, codec BLOSC
//   two_grids_blosc.nvdb     one segment, two grids (a Float fog sphere and an Fp8 one), codec BLOSC
//   <name>.image             each grid's raw image (codec NONE bytes), for hashing (not committed)
//   nvdb-probes.json           values at a few index coordinates, read with a native accessor
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace nanovdb;

static void dumpImage(const std::string& path, const void* data, uint64_t size)
{
    std::ofstream os(path, std::ios::binary);
    os.write(reinterpret_cast<const char*>(data), std::streamsize(size));
}

template<typename GridT>
static void probes(FILE* f, const char* name, const GridT* grid, bool last)
{
    auto acc = grid->getAccessor();
    const int coords[][3] = {{16, 16, 16}, {19, 14, 17}, {11, 16, 20}, {22, 22, 22}, {7, 17, 18}, {36, 16, 16}};
    std::fprintf(f, "  \"%s\": [", name);
    for (int i = 0; i < 6; ++i) {
        Coord ijk(coords[i][0], coords[i][1], coords[i][2]);
        std::fprintf(f, "%s{\"ijk\": [%d, %d, %d], \"value\": %.9g, \"active\": %s}", i ? ", " : "", ijk[0], ijk[1], ijk[2],
                     double(acc.getValue(ijk)), acc.isActive(ijk) ? "true" : "false");
    }
    std::fprintf(f, "]%s\n", last ? "" : ",");
}

int main(int argc, char** argv)
{
    const std::string out = argc > 1 ? argv[1] : ".";
    // A fog sphere of radius 8 voxels centred at (16,16,16): one upper and one lower node, so the image is ~320 KiB and Blosc packs it to a few KiB.
    auto sphere = tools::createFogVolumeSphere<float>(8.0, Vec3d(16.0), 1.0, 3.0, Vec3d(0.0), "density");
    io::writeGrid(out + "/sphere_fog_blosc.nvdb", sphere, io::Codec::BLOSC);
    dumpImage(out + "/sphere_fog_blosc.density.image", sphere.data(), sphere.gridSize());

    auto fp8 = tools::createFogVolumeSphere<Fp8>(6.0, Vec3d(18.0, 16.0, 16.0), 1.0, 3.0, Vec3d(0.0), "density_fp8");
    std::vector<GridHandle<HostBuffer>> handles;
    handles.push_back(tools::createFogVolumeSphere<float>(8.0, Vec3d(16.0), 1.0, 3.0, Vec3d(0.0), "density"));
    handles.push_back(tools::createFogVolumeSphere<Fp8>(6.0, Vec3d(18.0, 16.0, 16.0), 1.0, 3.0, Vec3d(0.0), "density_fp8"));
    auto merged = mergeGrids(handles);
    io::writeGrid(out + "/two_grids_blosc.nvdb", merged, io::Codec::BLOSC);
    dumpImage(out + "/two_grids_blosc.density_fp8.image", fp8.data(), fp8.gridSize());

    FILE* f = std::fopen((out + "/nvdb-probes.json").c_str(), "w");
    std::fprintf(f, "{\n  \"nanovdb\": \"%s\",\n", std::to_string(NANOVDB_MAJOR_VERSION_NUMBER).append(".").append(std::to_string(NANOVDB_MINOR_VERSION_NUMBER)).append(".").append(std::to_string(NANOVDB_PATCH_VERSION_NUMBER)).c_str());
    probes(f, "density", sphere.grid<float>(), false);
    probes(f, "density_fp8", fp8.grid<Fp8>(), true);
    std::fprintf(f, "}\n");
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}
