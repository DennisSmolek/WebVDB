#include <openvdb/io/Compression.h>
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

#include <iostream>
#include <string>

//* VDB Parity Normalization ===================================================

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: normalize_vdb <input.vdb> <output.vdb>\n";
        return 2;
    }

    openvdb::initialize();

    openvdb::io::File input(argv[1]);
    input.open();
    openvdb::GridPtrVecPtr grids = input.getGrids();
    input.close();

    for (openvdb::GridBase::Ptr& base : *grids) {
        if (!base->isType<openvdb::FloatGrid>()) continue;
        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
        grid->tree().voxelizeActiveTiles();
    }

    openvdb::io::File output(argv[2]);
    output.setCompression(openvdb::io::COMPRESS_ACTIVE_MASK);
    output.write(*grids);
    output.close();

    std::cout << "normalized " << argv[1] << " -> " << argv[2] << '\n';
    return 0;
}
