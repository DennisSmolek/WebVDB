// Bakes the Phase 0 primitive fixtures (docs/PLAN.md): fog-volume
// sphere/torus/box in Float, Fp8, and FpN, each written as a Codec::NONE
// .nvdb plus a JSON sidecar of sampled ground-truth values and tree stats.
// The traversal test suite (Phase 2) replays the sidecar samples against
// the WGSL port; determinism matters more than realism here.
//
// Build: g++ -std=c++17 -O2 -I<openvdb>/nanovdb bake_primitives.cpp -o bake_primitives
// Usage: bake_primitives <output-dir>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

// Deterministic PRNG (splitmix64) so re-bakes produce identical sidecars.
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int32_t range(int32_t lo, int32_t hi) { // inclusive
        return lo + static_cast<int32_t>(next() % static_cast<uint64_t>(hi - lo + 1));
    }
};

static std::string jsonFloat(float v) {
    std::ostringstream os;
    os << std::setprecision(9) << v;
    return os.str();
}

template <typename BuildT>
static void bakeOne(nanovdb::GridHandle<>&& handle,
                    const std::string& typeName,
                    const std::string& baseName,
                    const fs::path& outDir) {
    const auto* grid = handle.template grid<BuildT>();
    if (!grid) throw std::runtime_error(baseName + ": handle has no " + typeName + " grid");

    const std::string stem = baseName + "_" + typeName;
    const fs::path nvdbPath = outDir / (stem + ".nvdb");
    nanovdb::io::writeGrid(nvdbPath.string(), handle, nanovdb::io::Codec::NONE);

    auto acc = grid->getAccessor();
    const auto& bbox = grid->indexBBox();
    const auto& wbbox = grid->worldBBox();
    const auto vs = grid->voxelSize();
    const auto& tree = grid->tree();

    std::ofstream js(outDir / (stem + ".sidecar.json"));
    js << "{\n";
    js << "  \"file\": \"" << stem << ".nvdb\",\n";
    js << "  \"grid\": {\n";
    js << "    \"name\": \"" << grid->gridName() << "\",\n";
    js << "    \"type\": \"" << typeName << "\",\n";
    js << "    \"class\": \"FogVolume\",\n";
    js << "    \"gridByteSize\": " << grid->gridSize() << ",\n";
    js << "    \"activeVoxelCount\": " << grid->activeVoxelCount() << ",\n";
    js << "    \"nodeCounts\": {\"leaf\": " << tree.nodeCount(0)
       << ", \"lower\": " << tree.nodeCount(1)
       << ", \"upper\": " << tree.nodeCount(2) << "},\n";
    js << "    \"indexBBox\": [[" << bbox.min()[0] << ", " << bbox.min()[1] << ", " << bbox.min()[2]
       << "], [" << bbox.max()[0] << ", " << bbox.max()[1] << ", " << bbox.max()[2] << "]],\n";
    js << "    \"worldBBox\": [[" << jsonFloat(float(wbbox.min()[0])) << ", " << jsonFloat(float(wbbox.min()[1]))
       << ", " << jsonFloat(float(wbbox.min()[2])) << "], [" << jsonFloat(float(wbbox.max()[0]))
       << ", " << jsonFloat(float(wbbox.max()[1])) << ", " << jsonFloat(float(wbbox.max()[2])) << "]],\n";
    js << "    \"voxelSize\": [" << jsonFloat(float(vs[0])) << ", " << jsonFloat(float(vs[1]))
       << ", " << jsonFloat(float(vs[2])) << "]\n";
    js << "  },\n";

    // 64 pseudo-random probes over the bbox dilated by 4 voxels (so the set
    // includes inactive/outside coords) + the 8 bbox corners + center.
    js << "  \"samples\": [\n";
    Rng rng(0x57EB0DB000000001ULL); // fixed seed: sidecars must be reproducible
    bool first = true;
    auto emit = [&](const nanovdb::Coord& ijk) {
        const float v = acc.getValue(ijk);
        if (!first) js << ",\n";
        first = false;
        js << "    {\"ijk\": [" << ijk[0] << ", " << ijk[1] << ", " << ijk[2]
           << "], \"value\": " << jsonFloat(v)
           << ", \"active\": " << (acc.isActive(ijk) ? "true" : "false") << "}";
    };
    const auto mn = bbox.min(), mx = bbox.max();
    for (int corner = 0; corner < 8; ++corner) {
        emit(nanovdb::Coord(corner & 1 ? mx[0] : mn[0],
                            corner & 2 ? mx[1] : mn[1],
                            corner & 4 ? mx[2] : mn[2]));
    }
    emit(nanovdb::Coord((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2));
    for (int n = 0; n < 64; ++n) {
        emit(nanovdb::Coord(rng.range(mn[0] - 4, mx[0] + 4),
                            rng.range(mn[1] - 4, mx[1] + 4),
                            rng.range(mn[2] - 4, mx[2] + 4)));
    }
    js << "\n  ]\n}\n";

    std::cout << "  ✓ " << stem << ".nvdb (" << grid->gridSize() / 1024 << " KiB, "
              << grid->activeVoxelCount() << " active voxels) + sidecar\n";
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: bake_primitives <output-dir>\n";
        return 1;
    }
    const fs::path outDir = argv[1];
    fs::create_directories(outDir);
    using nanovdb::Vec3d;
    using namespace nanovdb::tools;

    try {
        std::cout << "sphere (r=50, voxel=1):\n";
        bakeOne<float>(createFogVolumeSphere<float>(50.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "sphere_fog"),
                       "float", "sphere_fog", outDir);
        bakeOne<nanovdb::Fp8>(createFogVolumeSphere<nanovdb::Fp8>(50.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "sphere_fog"),
                              "fp8", "sphere_fog", outDir);
        bakeOne<nanovdb::FpN>(createFogVolumeSphere<nanovdb::FpN>(50.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "sphere_fog"),
                              "fpn", "sphere_fog", outDir);

        std::cout << "torus (R=60, r=25, voxel=1):\n";
        bakeOne<float>(createFogVolumeTorus<float>(60.0, 25.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "torus_fog"),
                       "float", "torus_fog", outDir);
        bakeOne<nanovdb::Fp8>(createFogVolumeTorus<nanovdb::Fp8>(60.0, 25.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "torus_fog"),
                              "fp8", "torus_fog", outDir);
        bakeOne<nanovdb::FpN>(createFogVolumeTorus<nanovdb::FpN>(60.0, 25.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "torus_fog"),
                              "fpn", "torus_fog", outDir);

        std::cout << "box (80×40×60, voxel=1):\n";
        bakeOne<float>(createFogVolumeBox<float>(80.0, 40.0, 60.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "box_fog"),
                       "float", "box_fog", outDir);
        bakeOne<nanovdb::Fp8>(createFogVolumeBox<nanovdb::Fp8>(80.0, 40.0, 60.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "box_fog"),
                              "fp8", "box_fog", outDir);
        bakeOne<nanovdb::FpN>(createFogVolumeBox<nanovdb::FpN>(80.0, 40.0, 60.0, Vec3d(0), 1.0, 3.0, Vec3d(0), "box_fog"),
                              "fpn", "box_fog", outDir);
    } catch (const std::exception& e) {
        std::cerr << "bake failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
