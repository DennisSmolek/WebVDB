# Reproducible fixture-bake environment (docs/PLAN.md Phase 0).
# Builds native OpenVDB + NanoVDB tools at the SAME pinned commit as the
# vendored PNanoVDB.h, plus the primitive bake program. Per D6 this image
# is also the seed of the interim companion service (convert/export/
# resample endpoints) — browser parity retires it endpoint by endpoint.
#
# Build:  docker build -t webvdb-fixture-bake docker/fixture-bake
# Bake:   docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake
# Tools:  docker run --rm -v "$PWD/fixtures:/out" webvdb-fixture-bake \
#           nanovdb_convert --fp8 /out/wdas/wdas_cloud_quarter.vdb /out/wdas/wdas_cloud_quarter_fp8.nvdb

FROM ubuntu:24.04 AS build
ARG OPENVDB_COMMIT=a532de5526ef791280b6483a872336a811a68542

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates \
    libboost-iostreams-dev libtbb-dev libblosc-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git init openvdb && cd openvdb \
    && git remote add origin https://github.com/AcademySoftwareFoundation/openvdb.git \
    && git fetch --depth 1 origin "${OPENVDB_COMMIT}" \
    && git checkout FETCH_HEAD

RUN cmake -S openvdb -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENVDB_BUILD_CORE=ON \
      -DOPENVDB_BUILD_BINARIES=OFF \
      -DOPENVDB_BUILD_UNITTESTS=OFF \
      -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
      -DUSE_NANOVDB=ON \
      -DNANOVDB_USE_OPENVDB=ON \
      -DNANOVDB_BUILD_TOOLS=ON \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build

# The bake program is NanoVDB-header-only — no OpenVDB link needed.
COPY bake_primitives.cpp /src/
RUN g++ -std=c++17 -O2 -I /src/openvdb/nanovdb /src/bake_primitives.cpp -o /usr/local/bin/bake_primitives

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libboost-iostreams1.83.0 libtbb12 libblosc1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local /usr/local
COPY bake-fixtures.sh /usr/local/bin/bake-fixtures.sh
RUN ldconfig && chmod +x /usr/local/bin/bake-fixtures.sh

VOLUME /out
ENTRYPOINT ["/usr/local/bin/bake-fixtures.sh"]
