// Writes small HalfGrid (Tree_half_5_4_3) .vdb files with native OpenVDB v14.0.0 (file version 225).
//
//   clang++ -std=c++17 -O1 -I <prefix>/include gen_vdb_half.cpp -L <prefix>/lib -lopenvdb -ltbb \
//     -Wl,-rpath,<prefix>/lib -o gen_vdb_half
//   ./gen_vdb_half <out-dir>
//
// Writes half_<codec>.vdb for codec none, zip and blosc (each with active-mask compression, the default
// alongside Blosc), each holding two HalfGrids (class Unknown in the zip file: OpenVDB never ZIPs a
// fog volume or level set):
//   "fog"   background 0: a few leaf voxels, an active lower-node tile and an active root-level tile value
//   "ls"    a narrow-band level set sphere (background 3 voxels, interior -3) converted from a FloatGrid,
//           so inactive values take both signs (mask compression with one and two inactive values)
// and vdb-probes.json: values and active states at a few coordinates, read with a native accessor.
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/LevelSetSphere.h>

#include <cstdio>
#include <string>

using namespace openvdb;

static HalfGrid::Ptr makeFog()
{
    auto grid = HalfGrid::create(math::half(0.0f));
    grid->setName("fog");
    grid->setGridClass(GRID_FOG_VOLUME);
    grid->setTransform(math::Transform::createLinearTransform(0.5));
    auto acc = grid->getAccessor();
    acc.setValue(Coord(0, 0, 0), math::half(0.25f));
    acc.setValue(Coord(1, 2, 3), math::half(0.333f));
    acc.setValue(Coord(7, 7, 7), math::half(1.0f));
    acc.setValue(Coord(9, 5, 1), math::half(65504.0f)); // largest half
    acc.setValue(Coord(15, 8, 8), math::half(6.1035156e-05f)); // smallest normal half
    acc.setValue(Coord(3, 3, 3), math::half(0.5f));
    acc.setValueOff(Coord(3, 3, 3), math::half(0.125f)); // an inactive voxel with its own value
    // An active lower-node tile (8^3 voxels, level 1) at (64, 0, 0).
    grid->tree().addTile(1, Coord(64, 0, 0), math::half(0.75f), true);
    return grid;
}

static HalfGrid::Ptr makeLevelSet()
{
    auto ls = tools::createLevelSetSphere<FloatGrid>(/*radius=*/5.0f, Vec3f(16.0f), /*voxelSize=*/1.0f, /*halfWidth=*/3.0f);
    auto grid = HalfGrid::create(math::half(ls->background()));
    grid->setName("ls");
    grid->setGridClass(GRID_LEVEL_SET);
    grid->setTransform(ls->transform().copy());
    // Copy every leaf (values and active mask), then every tile, active or not.
    for (auto leaf = ls->tree().cbeginLeaf(); leaf; ++leaf) {
        auto* out = grid->tree().touchLeaf(leaf->origin());
        for (Index i = 0; i < FloatTree::LeafNodeType::SIZE; ++i) out->setValueOnly(i, math::half(leaf->getValue(i)));
        out->setValueMask(leaf->getValueMask());
    }
    for (auto it = ls->tree().cbeginValueAll(); it; ++it) {
        if (!it.isVoxelValue()) grid->tree().addTile(it.getLevel(), it.getCoord(), math::half(*it), it.isValueOn());
    }
    return grid;
}

int main(int argc, char** argv)
{
    initialize();
    const std::string out = argc > 1 ? argv[1] : ".";
    auto fog = makeFog();
    auto ls = makeLevelSet();
    GridPtrVec grids{fog, ls};

    const struct { const char* name; uint32_t flags; } codecs[] = {
        {"none", io::COMPRESS_ACTIVE_MASK},
        {"zip", io::COMPRESS_ZIP | io::COMPRESS_ACTIVE_MASK},
        {"blosc", io::COMPRESS_BLOSC | io::COMPRESS_ACTIVE_MASK},
    };
    for (const auto& c : codecs) {
        // OpenVDB never ZIPs level sets or fog volumes (Archive::setGridCompression): mark the grids
        // GRID_UNKNOWN in the zip file so its leaf and node arrays really are zlib streams.
        const bool zip = (c.flags & io::COMPRESS_ZIP) != 0;
        fog->setGridClass(zip ? GRID_UNKNOWN : GRID_FOG_VOLUME);
        ls->setGridClass(zip ? GRID_UNKNOWN : GRID_LEVEL_SET);
        io::File file(out + "/half_" + c.name + ".vdb");
        file.setCompression(c.flags);
        file.write(grids);
        file.close();
    }

    FILE* f = std::fopen((out + "/vdb-probes.json").c_str(), "w");
    std::fprintf(f, "{\n  \"openvdb\": \"%s\",\n  \"fileVersion\": %u,\n", OPENVDB_LIBRARY_VERSION_STRING, OPENVDB_FILE_VERSION);
    const Coord fogCoords[] = {{0, 0, 0}, {1, 2, 3}, {7, 7, 7}, {9, 5, 1}, {15, 8, 8}, {3, 3, 3}, {64, 0, 0}, {71, 7, 7}, {72, 0, 0}, {100, 100, 100}};
    const Coord lsCoords[] = {{16, 16, 16}, {21, 16, 16}, {20, 17, 16}, {16, 22, 17}, {16, 16, 9}, {17, 17, 17}, {25, 16, 16}, {-30, 0, 0}};
    auto dump = [&](const char* name, HalfGrid::Ptr g, const Coord* cs, int n, bool last) {
        auto acc = g->getConstAccessor();
        std::fprintf(f, "  \"%s\": {\"background\": %.9g, \"activeVoxelCount\": %llu, \"probes\": [", name,
                     double(float(g->background())), (unsigned long long)g->activeVoxelCount());
        for (int i = 0; i < n; ++i) {
            std::fprintf(f, "%s{\"ijk\": [%d, %d, %d], \"value\": %.9g, \"active\": %s}", i ? ", " : "", cs[i][0], cs[i][1], cs[i][2],
                         double(float(acc.getValue(cs[i]))), acc.isValueOn(cs[i]) ? "true" : "false");
        }
        std::fprintf(f, "]}%s\n", last ? "" : ",");
    };
    dump("fog", fog, fogCoords, 10, false);
    dump("ls", ls, lsCoords, 8, true);
    std::fprintf(f, "}\n");
    std::fclose(f);
    std::printf("wrote %s\n", out.c_str());
    return 0;
}
