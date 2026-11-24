// Bakes the NanoVDB Half fixtures (K7.H) with native NanoVDB >= 13.2 (built
// against OpenVDB v14.0.0's header-only NanoVDB; scripts/bake-half-fixtures.mjs).
// NanoVDB's Half grid type (GridType::Half = 9, raw IEEE binary16) was a
// placeholder until OpenVDB PR 2335; `nanovdb_convert` has no half flag, so
// Half grids come from `tools::createNanoGrid<SrcGridT, nanovdb::Half>`.
//
//  1. <primitives>/{sphere,torus,box}_fog_float.nvdb (the a532de55 bake,
//     `pnpm fixtures:bake`) -> createNanoGrid<FloatGrid, Half> ->
//     <primitives>/<name>_half.nvdb + <name>_half.sidecar.json (the primitive
//     sidecar schema plus root stats and an FNV-1a hash of every leaf block).
//  2. A small edge-case grid (rounding ties, subnormals, overflow to infinity,
//     -0, a non-half background) from tools::build::Grid<float> ->
//     <half-dir>/half_edge.nvdb + half_edge.sidecar.json (the inputs, the
//     native bits of every probe and every node's stats as raw bits).
//
// The sidecars are committed; the .nvdb files are git-ignored. Our writer
// (`vdbQuantize(…, 'half')`, `VDB_HALF_LEAF_CODEC`) is checked against them by
// packages/vdb/test/tools/half-native.test.ts.
//
// Build: c++ -std=c++17 -O2 -ffp-contract=off -I<openvdb-14.0.0>/nanovdb bake_half.cpp -o bake_half
// Usage: bake_half <primitives-dir> <half-dir>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/HalfLite.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/GridBuilder.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using HalfGrid = nanovdb::NanoGrid<nanovdb::Half>;

// Deterministic PRNG (splitmix64), the same probe sequence as bake_primitives.cpp.
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

static uint32_t floatBits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

static float h2f(nanovdb::Half h) { return nanovdb::math::HalfToFloat(h.raw); }

static uint32_t fnv1a(const uint8_t* p, size_t n, uint32_t h = 0x811c9dc5u) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x01000193u;
    }
    return h;
}

/// Raw stats of one node: min/max binary16 bits, ave/stddev f32 bits.
template <typename DataT>
static std::string statsJson(const DataT* d) {
    std::ostringstream os;
    os << "{\"min\": " << d->mMinimum.raw << ", \"max\": " << d->mMaximum.raw
       << ", \"ave\": " << floatBits(d->mAverage) << ", \"stddev\": " << floatBits(d->mStdDevi) << "}";
    return os.str();
}

static std::string versionString(const HalfGrid* grid) {
    const auto v = grid->version();
    std::ostringstream os;
    os << v.getMajor() << "." << v.getMinor() << "." << v.getPatch();
    return os.str();
}

/// Root stats and a hash of every leaf block (all 1120 bytes, memory order).
static void writeSummary(std::ofstream& js, const HalfGrid* grid) {
    const auto& tree = grid->tree();
    const auto* root = tree.root().data();
    uint32_t leafHash = 0x811c9dc5u, valueHash = 0x811c9dc5u;
    const uint32_t nLeaf = tree.nodeCount(0);
    const auto* leaf0 = tree.getFirstNode<0>();
    for (uint32_t i = 0; i < nLeaf; ++i) {
        const auto* leaf = leaf0 + i;
        const auto* bytes = reinterpret_cast<const uint8_t*>(leaf);
        leafHash = fnv1a(bytes, sizeof(*leaf), leafHash);
        valueHash = fnv1a(bytes + 96, 1024, valueHash);
    }
    js << "  \"native\": {\n";
    js << "    \"nanovdb\": \"" << versionString(grid) << "\",\n";
    js << "    \"source\": \"OpenVDB v14.0.0 tools::createNanoGrid<SrcGridT, nanovdb::Half> (docker/fixture-bake/bake_half.cpp)\",\n";
    js << "    \"gridFlags\": " << grid->data()->mFlags.data() << ",\n";
    js << "    \"background\": " << root->mBackground.raw << ",\n";
    js << "    \"root\": " << statsJson(root) << ",\n";
    js << "    \"leafBlocksFnv1a\": " << leafHash << ",\n";
    js << "    \"leafValuesFnv1a\": " << valueHash << "\n";
    js << "  },\n";
}

static void bakePrimitive(const fs::path& primDir, const std::string& baseName) {
    const fs::path src = primDir / (baseName + "_float.nvdb");
    auto floatHandle = nanovdb::io::readGrid(src.string());
    const auto* floatGrid = floatHandle.grid<float>();
    if (!floatGrid) throw std::runtime_error(src.string() + ": no float grid (run pnpm fixtures:bake first)");

    auto handle = nanovdb::tools::createNanoGrid<nanovdb::NanoGrid<float>, nanovdb::Half>(*floatGrid);
    const auto* grid = handle.grid<nanovdb::Half>();
    if (!grid) throw std::runtime_error(baseName + ": conversion produced no Half grid");

    const std::string stem = baseName + "_half";
    nanovdb::io::writeGrid((primDir / (stem + ".nvdb")).string(), handle, nanovdb::io::Codec::NONE);

    auto acc = grid->getAccessor();
    const auto& bbox = grid->indexBBox();
    const auto& wbbox = grid->worldBBox();
    const auto vs = grid->voxelSize();
    const auto& tree = grid->tree();

    std::ofstream js(primDir / (stem + ".sidecar.json"));
    js << "{\n";
    js << "  \"file\": \"" << stem << ".nvdb\",\n";
    js << "  \"grid\": {\n";
    js << "    \"name\": \"" << grid->gridName() << "\",\n";
    js << "    \"type\": \"half\",\n";
    js << "    \"class\": \"FogVolume\",\n";
    js << "    \"gridByteSize\": " << grid->gridSize() << ",\n";
    js << "    \"activeVoxelCount\": " << grid->activeVoxelCount() << ",\n";
    js << "    \"nodeCounts\": {\"leaf\": " << tree.nodeCount(0) << ", \"lower\": " << tree.nodeCount(1)
       << ", \"upper\": " << tree.nodeCount(2) << "},\n";
    js << "    \"indexBBox\": [[" << bbox.min()[0] << ", " << bbox.min()[1] << ", " << bbox.min()[2] << "], ["
       << bbox.max()[0] << ", " << bbox.max()[1] << ", " << bbox.max()[2] << "]],\n";
    js << "    \"worldBBox\": [[" << jsonFloat(float(wbbox.min()[0])) << ", " << jsonFloat(float(wbbox.min()[1]))
       << ", " << jsonFloat(float(wbbox.min()[2])) << "], [" << jsonFloat(float(wbbox.max()[0])) << ", "
       << jsonFloat(float(wbbox.max()[1])) << ", " << jsonFloat(float(wbbox.max()[2])) << "]],\n";
    js << "    \"voxelSize\": [" << jsonFloat(float(vs[0])) << ", " << jsonFloat(float(vs[1])) << ", "
       << jsonFloat(float(vs[2])) << "]\n";
    js << "  },\n";
    writeSummary(js, grid);

    // The primitive probe set: 8 bbox corners, the centre, 64 seeded probes over the bbox dilated by 4.
    js << "  \"samples\": [\n";
    Rng rng(0x57EB0DB000000001ULL);
    bool first = true;
    auto emit = [&](const nanovdb::Coord& ijk) {
        const nanovdb::Half h = acc.getValue(ijk);
        if (!first) js << ",\n";
        first = false;
        js << "    {\"ijk\": [" << ijk[0] << ", " << ijk[1] << ", " << ijk[2] << "], \"value\": " << jsonFloat(h2f(h))
           << ", \"bits\": " << h.raw << ", \"active\": " << (acc.isActive(ijk) ? "true" : "false") << "}";
    };
    const auto mn = bbox.min(), mx = bbox.max();
    for (int corner = 0; corner < 8; ++corner) {
        emit(nanovdb::Coord(corner & 1 ? mx[0] : mn[0], corner & 2 ? mx[1] : mn[1], corner & 4 ? mx[2] : mn[2]));
    }
    emit(nanovdb::Coord((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2));
    for (int n = 0; n < 64; ++n) {
        const int32_t x = rng.range(mn[0] - 4, mx[0] + 4);
        const int32_t y = rng.range(mn[1] - 4, mx[1] + 4);
        const int32_t z = rng.range(mn[2] - 4, mx[2] + 4);
        emit(nanovdb::Coord(x, y, z));
    }
    js << "\n  ]\n}\n";
    std::cout << "  ✓ " << stem << ".nvdb (" << grid->gridSize() / 1024 << " KiB, " << grid->activeVoxelCount()
              << " active voxels, NanoVDB " << versionString(grid) << ") + sidecar\n";
}

/// Edge-case voxels: { i, j, k, value }. Each leaf groups one kind of case.
struct EdgeVoxel {
    int32_t i, j, k;
    float v;
};

static std::vector<EdgeVoxel> edgeVoxels() {
    const float tie = std::ldexp(1.0f, -25);
    return {
        // Leaf (0,0,0): normal range, rounding and ties.
        {0, 0, 0, 1.0f}, {0, 0, 1, -2.5f}, {0, 0, 2, 0.1f}, {0, 0, 3, 1.0f / 3.0f}, {0, 0, 4, -7.75f},
        {0, 0, 5, 3.14159265f}, {0, 0, 6, 2049.0f}, {0, 0, 7, 2051.0f}, {0, 1, 0, 65504.0f}, {0, 1, 1, 65519.0f},
        {0, 1, 2, 1234.5678f}, {0, 1, 3, -0.0f}, {0, 1, 4, 0.5f}, {7, 7, 7, 0.999f}, {3, 5, 1, -1000.1f},
        // Leaf (8,0,0): subnormals and underflow.
        {8, 0, 0, 1e-5f}, {8, 0, 1, 6e-8f}, {8, 0, 2, tie}, {8, 0, 3, 3.0f * tie}, {8, 0, 4, 1e-30f},
        {8, 0, 5, -1e-6f}, {8, 0, 6, 6.1e-5f}, {8, 0, 7, 5.96e-8f}, {9, 2, 3, 6.103515625e-05f},
        // Leaf (0,8,0): overflow to infinity (its ave is +inf, its stddev NaN).
        {0, 8, 0, 65520.0f}, {0, 8, 1, 1e5f}, {0, 8, 2, 7.0f},
        // Leaf (0,0,8): negative infinity only next to finite values.
        {0, 0, 8, -1e5f}, {1, 1, 9, -3.0f},
        // A leaf in another upper node.
        {4096, -4096, 128, 0.75f}, {4097, -4095, 130, 42.0f},
    };
}

static void bakeEdge(const fs::path& halfDir) {
    const float background = 0.1f; // not a half: the background rounds too
    nanovdb::tools::build::Grid<float> src(background, "half_edge", nanovdb::GridClass::Unknown);
    auto srcAcc = src.getAccessor();
    const auto voxels = edgeVoxels();
    for (const auto& e : voxels) srcAcc.setValue(nanovdb::Coord(e.i, e.j, e.k), e.v);

    auto handle = nanovdb::tools::createNanoGrid<nanovdb::tools::build::Grid<float>, nanovdb::Half>(src);
    const auto* grid = handle.grid<nanovdb::Half>();
    if (!grid) throw std::runtime_error("half_edge: conversion produced no Half grid");
    nanovdb::io::writeGrid((halfDir / "half_edge.nvdb").string(), handle, nanovdb::io::Codec::NONE);

    const auto& tree = grid->tree();
    std::ofstream js(halfDir / "half_edge.sidecar.json");
    js << "{\n";
    js << "  \"file\": \"half_edge.nvdb\",\n";
    js << "  \"grid\": {\"name\": \"half_edge\", \"type\": \"half\", \"class\": \"Unknown\", \"gridByteSize\": "
       << grid->gridSize() << ", \"activeVoxelCount\": " << grid->activeVoxelCount()
       << ", \"nodeCounts\": {\"leaf\": " << tree.nodeCount(0) << ", \"lower\": " << tree.nodeCount(1)
       << ", \"upper\": " << tree.nodeCount(2) << "}, \"backgroundInput\": " << floatBits(background) << "},\n";
    writeSummary(js, grid);

    // Every voxel's f32 input (as bits) and the native binary16 bits.
    auto acc = grid->getAccessor();
    js << "  \"voxels\": [\n";
    for (size_t n = 0; n < voxels.size(); ++n) {
        const auto& e = voxels[n];
        const nanovdb::Coord ijk(e.i, e.j, e.k);
        js << "    {\"ijk\": [" << e.i << ", " << e.j << ", " << e.k << "], \"input\": " << floatBits(e.v)
           << ", \"bits\": " << acc.getValue(ijk).raw << ", \"active\": " << (acc.isActive(ijk) ? "true" : "false") << "}"
           << (n + 1 < voxels.size() ? "," : "") << "\n";
    }
    js << "  ],\n";
    // Inactive probes: inside a leaf, in an empty lower slot, far outside.
    js << "  \"inactive\": [\n";
    const nanovdb::Coord probes[] = {{1, 0, 0}, {6, 6, 6}, {40, 40, 40}, {-500, 3, 9}, {100000, 0, 0}};
    for (size_t n = 0; n < 5; ++n) {
        const auto& p = probes[n];
        js << "    {\"ijk\": [" << p[0] << ", " << p[1] << ", " << p[2] << "], \"bits\": " << acc.getValue(p).raw
           << ", \"active\": " << (acc.isActive(p) ? "true" : "false") << "}" << (n + 1 < 5 ? "," : "") << "\n";
    }
    js << "  ],\n";
    // Every node's stats as raw bits, by origin.
    js << "  \"leaves\": [\n";
    const auto* leaf0 = tree.getFirstNode<0>();
    for (uint32_t i = 0; i < tree.nodeCount(0); ++i) {
        const auto* leaf = leaf0 + i;
        const auto o = leaf->origin();
        const auto* d = leaf->data();
        js << "    {\"origin\": [" << o[0] << ", " << o[1] << ", " << o[2] << "], \"flags\": " << int(d->mFlags)
           << ", \"bboxMin\": [" << d->mBBoxMin[0] << ", " << d->mBBoxMin[1] << ", " << d->mBBoxMin[2]
           << "], \"bboxDif\": [" << int(d->mBBoxDif[0]) << ", " << int(d->mBBoxDif[1]) << ", " << int(d->mBBoxDif[2])
           << "], \"stats\": " << statsJson(d) << "}" << (i + 1 < tree.nodeCount(0) ? "," : "") << "\n";
    }
    js << "  ],\n";
    js << "  \"lowers\": [\n";
    const auto* lower0 = tree.getFirstNode<1>();
    for (uint32_t i = 0; i < tree.nodeCount(1); ++i) {
        const auto* node = lower0 + i;
        const auto o = node->origin();
        js << "    {\"origin\": [" << o[0] << ", " << o[1] << ", " << o[2] << "], \"stats\": " << statsJson(node->data())
           << "}" << (i + 1 < tree.nodeCount(1) ? "," : "") << "\n";
    }
    js << "  ],\n";
    js << "  \"uppers\": [\n";
    const auto* upper0 = tree.getFirstNode<2>();
    for (uint32_t i = 0; i < tree.nodeCount(2); ++i) {
        const auto* node = upper0 + i;
        const auto o = node->origin();
        js << "    {\"origin\": [" << o[0] << ", " << o[1] << ", " << o[2] << "], \"stats\": " << statsJson(node->data())
           << "}" << (i + 1 < tree.nodeCount(2) ? "," : "") << "\n";
    }
    js << "  ]\n}\n";
    std::cout << "  ✓ half_edge.nvdb (" << grid->gridSize() / 1024 << " KiB, " << grid->activeVoxelCount()
              << " active voxels) + sidecar\n";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: bake_half <primitives-dir> <half-dir>\n";
        return 1;
    }
    const fs::path primDir = argv[1];
    const fs::path halfDir = argv[2];
    fs::create_directories(halfDir);
    try {
        std::cout << "Half primitives (createNanoGrid<FloatGrid, Half>):\n";
        for (const char* name : {"sphere_fog", "torus_fog", "box_fog"}) bakePrimitive(primDir, name);
        std::cout << "Half edge cases (createNanoGrid<build::Grid<float>, Half>):\n";
        bakeEdge(halfDir);
    } catch (const std::exception& e) {
        std::cerr << "bake failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
