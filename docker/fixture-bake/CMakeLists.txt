cmake_minimum_required(VERSION 3.20)
project(webvdb_fixture_tools LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)

add_executable(normalize_vdb normalize_vdb.cpp)
target_compile_features(normalize_vdb PRIVATE cxx_std_17)
target_link_libraries(normalize_vdb PRIVATE OpenVDB::openvdb)
install(TARGETS normalize_vdb RUNTIME DESTINATION bin)
