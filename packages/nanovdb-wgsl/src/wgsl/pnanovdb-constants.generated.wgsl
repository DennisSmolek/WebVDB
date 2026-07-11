// GENERATED FILE — do not edit by hand.
// Source: packages/nanovdb-wgsl/vendor/stride-tables.json (NanoVDB ABI 32.9.1)
// Regenerate: node scripts/gen-wgsl-constants.mjs
// Guarded by: packages/nanovdb-wgsl/test/wgsl-constants.test.ts

// ---- defines ----
const PNANOVDB_FALSE: u32 = 0u;
const PNANOVDB_TRUE: u32 = 1u;
const PNANOVDB_MAJOR_VERSION_NUMBER: u32 = 32u;
const PNANOVDB_MINOR_VERSION_NUMBER: u32 = 9u;
const PNANOVDB_PATCH_VERSION_NUMBER: u32 = 1u;
const PNANOVDB_GRID_TYPE_UNKNOWN: u32 = 0u;
const PNANOVDB_GRID_TYPE_FLOAT: u32 = 1u;
const PNANOVDB_GRID_TYPE_DOUBLE: u32 = 2u;
const PNANOVDB_GRID_TYPE_INT16: u32 = 3u;
const PNANOVDB_GRID_TYPE_INT32: u32 = 4u;
const PNANOVDB_GRID_TYPE_INT64: u32 = 5u;
const PNANOVDB_GRID_TYPE_VEC3F: u32 = 6u;
const PNANOVDB_GRID_TYPE_VEC3D: u32 = 7u;
const PNANOVDB_GRID_TYPE_MASK: u32 = 8u;
const PNANOVDB_GRID_TYPE_HALF: u32 = 9u;
const PNANOVDB_GRID_TYPE_UINT32: u32 = 10u;
const PNANOVDB_GRID_TYPE_BOOLEAN: u32 = 11u;
const PNANOVDB_GRID_TYPE_RGBA8: u32 = 12u;
const PNANOVDB_GRID_TYPE_FP4: u32 = 13u;
const PNANOVDB_GRID_TYPE_FP8: u32 = 14u;
const PNANOVDB_GRID_TYPE_FP16: u32 = 15u;
const PNANOVDB_GRID_TYPE_FPN: u32 = 16u;
const PNANOVDB_GRID_TYPE_VEC4F: u32 = 17u;
const PNANOVDB_GRID_TYPE_VEC4D: u32 = 18u;
const PNANOVDB_GRID_TYPE_INDEX: u32 = 19u;
const PNANOVDB_GRID_TYPE_ONINDEX: u32 = 20u;
const PNANOVDB_GRID_TYPE_POINTINDEX: u32 = 23u;
const PNANOVDB_GRID_TYPE_VEC3U8: u32 = 24u;
const PNANOVDB_GRID_TYPE_VEC3U16: u32 = 25u;
const PNANOVDB_GRID_TYPE_UINT8: u32 = 26u;
const PNANOVDB_GRID_TYPE_END: u32 = 27u;
const PNANOVDB_GRID_TYPE_CAP: u32 = 32u;
const PNANOVDB_GRID_CLASS_UNKNOWN: u32 = 0u;
const PNANOVDB_GRID_CLASS_LEVEL_SET: u32 = 1u;
const PNANOVDB_GRID_CLASS_FOG_VOLUME: u32 = 2u;
const PNANOVDB_GRID_CLASS_STAGGERED: u32 = 3u;
const PNANOVDB_GRID_CLASS_POINT_INDEX: u32 = 4u;
const PNANOVDB_GRID_CLASS_POINT_DATA: u32 = 5u;
const PNANOVDB_GRID_CLASS_TOPOLOGY: u32 = 6u;
const PNANOVDB_GRID_CLASS_VOXEL_VOLUME: u32 = 7u;
const PNANOVDB_GRID_CLASS_INDEX_GRID: u32 = 8u;
const PNANOVDB_GRID_CLASS_TENSOR_GRID: u32 = 9u;
const PNANOVDB_GRID_CLASS_VOXEL_BVH: u32 = 10u;
const PNANOVDB_GRID_CLASS_END: u32 = 11u;
const PNANOVDB_LEAF_TYPE_DEFAULT: u32 = 0u;
const PNANOVDB_LEAF_TYPE_LITE: u32 = 1u;
const PNANOVDB_LEAF_TYPE_FP: u32 = 2u;
const PNANOVDB_LEAF_TYPE_INDEX: u32 = 3u;
const PNANOVDB_LEAF_TYPE_POINTINDEX: u32 = 5u;
const PNANOVDB_MAP_SIZE: u32 = 264u;
const PNANOVDB_MAP_OFF_MATF: u32 = 0u;
const PNANOVDB_MAP_OFF_INVMATF: u32 = 36u;
const PNANOVDB_MAP_OFF_VECF: u32 = 72u;
const PNANOVDB_MAP_OFF_TAPERF: u32 = 84u;
const PNANOVDB_MAP_OFF_MATD: u32 = 88u;
const PNANOVDB_MAP_OFF_INVMATD: u32 = 160u;
const PNANOVDB_MAP_OFF_VECD: u32 = 232u;
const PNANOVDB_MAP_OFF_TAPERD: u32 = 256u;
const PNANOVDB_GRID_SIZE: u32 = 672u;
const PNANOVDB_GRID_OFF_MAGIC: u32 = 0u;
const PNANOVDB_GRID_OFF_CHECKSUM: u32 = 8u;
const PNANOVDB_GRID_OFF_VERSION: u32 = 16u;
const PNANOVDB_GRID_OFF_FLAGS: u32 = 20u;
const PNANOVDB_GRID_OFF_GRID_INDEX: u32 = 24u;
const PNANOVDB_GRID_OFF_GRID_COUNT: u32 = 28u;
const PNANOVDB_GRID_OFF_GRID_SIZE: u32 = 32u;
const PNANOVDB_GRID_OFF_GRID_NAME: u32 = 40u;
const PNANOVDB_GRID_OFF_MAP: u32 = 296u;
const PNANOVDB_GRID_OFF_WORLD_BBOX: u32 = 560u;
const PNANOVDB_GRID_OFF_VOXEL_SIZE: u32 = 608u;
const PNANOVDB_GRID_OFF_GRID_CLASS: u32 = 632u;
const PNANOVDB_GRID_OFF_GRID_TYPE: u32 = 636u;
const PNANOVDB_GRID_OFF_BLIND_METADATA_OFFSET: u32 = 640u;
const PNANOVDB_GRID_OFF_BLIND_METADATA_COUNT: u32 = 648u;
const PNANOVDB_GRID_OFF_DATA0: u32 = 652u;
const PNANOVDB_GRID_OFF_DATA1: u32 = 656u;
const PNANOVDB_GRID_OFF_DATA2: u32 = 664u;
const PNANOVDB_GRIDBLINDMETADATA_CLASS_UNKNOWN: u32 = 0u;
const PNANOVDB_GRIDBLINDMETADATA_CLASS_INDEX_ARRAY: u32 = 1u;
const PNANOVDB_GRIDBLINDMETADATA_CLASS_ATTRIBUTE_ARRAY: u32 = 2u;
const PNANOVDB_GRIDBLINDMETADATA_CLASS_GRID_NAME: u32 = 3u;
const PNANOVDB_GRIDBLINDMETADATA_CLASS_CHANNEL_ARRAY: u32 = 4u;
const PNANOVDB_GRIDBLINDMETADATA_CLASS_END: u32 = 5u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_UNKNOWN: u32 = 0u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_POSITION: u32 = 1u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_COLOR: u32 = 2u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_NORMAL: u32 = 3u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_RADIUS: u32 = 4u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_VELOCITY: u32 = 5u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_ID: u32 = 6u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_WORLD_COORDS: u32 = 7u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_GRID_COORDS: u32 = 8u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_VOXEL_COORDS: u32 = 9u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_LEVEL_SET: u32 = 10u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_FOG_VOLUME: u32 = 11u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_STAGGERED: u32 = 12u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_OPACITY: u32 = 13u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_QUAT: u32 = 14u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_SCALE: u32 = 15u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_SH0: u32 = 16u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_POINT_SHN: u32 = 17u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_LINE_ID: u32 = 18u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_TRIANGLE_ID: u32 = 19u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_GAUSSIAN_ID: u32 = 20u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_RANGE: u32 = 21u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_VOXEL_BVH: u32 = 22u;
const PNANOVDB_GRIDBLINDMETADATA_SEMANTIC_END: u32 = 23u;
const PNANOVDB_GRIDBLINDMETADATA_SIZE: u32 = 288u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_OFFSET: u32 = 0u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_VALUE_COUNT: u32 = 8u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_VALUE_SIZE: u32 = 16u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_SEMANTIC: u32 = 20u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_CLASS: u32 = 24u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_DATA_TYPE: u32 = 28u;
const PNANOVDB_GRIDBLINDMETADATA_OFF_NAME: u32 = 32u;
const PNANOVDB_TREE_SIZE: u32 = 64u;
const PNANOVDB_TREE_OFF_NODE_OFFSET_LEAF: u32 = 0u;
const PNANOVDB_TREE_OFF_NODE_OFFSET_LOWER: u32 = 8u;
const PNANOVDB_TREE_OFF_NODE_OFFSET_UPPER: u32 = 16u;
const PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT: u32 = 24u;
const PNANOVDB_TREE_OFF_NODE_COUNT_LEAF: u32 = 32u;
const PNANOVDB_TREE_OFF_NODE_COUNT_LOWER: u32 = 36u;
const PNANOVDB_TREE_OFF_NODE_COUNT_UPPER: u32 = 40u;
const PNANOVDB_TREE_OFF_TILE_COUNT_LOWER: u32 = 44u;
const PNANOVDB_TREE_OFF_TILE_COUNT_UPPER: u32 = 48u;
const PNANOVDB_TREE_OFF_TILE_COUNT_ROOT: u32 = 52u;
const PNANOVDB_TREE_OFF_VOXEL_COUNT: u32 = 56u;
const PNANOVDB_ROOT_BASE_SIZE: u32 = 28u;
const PNANOVDB_ROOT_OFF_BBOX_MIN: u32 = 0u;
const PNANOVDB_ROOT_OFF_BBOX_MAX: u32 = 12u;
const PNANOVDB_ROOT_OFF_TABLE_SIZE: u32 = 24u;
const PNANOVDB_ROOT_TILE_BASE_SIZE: u32 = 20u;
const PNANOVDB_ROOT_TILE_OFF_KEY: u32 = 0u;
const PNANOVDB_ROOT_TILE_OFF_CHILD: u32 = 8u;
const PNANOVDB_ROOT_TILE_OFF_STATE: u32 = 16u;
const PNANOVDB_UPPER_TABLE_COUNT: u32 = 32768u;
const PNANOVDB_UPPER_BASE_SIZE: u32 = 8224u;
const PNANOVDB_UPPER_OFF_BBOX_MIN: u32 = 0u;
const PNANOVDB_UPPER_OFF_BBOX_MAX: u32 = 12u;
const PNANOVDB_UPPER_OFF_FLAGS: u32 = 24u;
const PNANOVDB_UPPER_OFF_VALUE_MASK: u32 = 32u;
const PNANOVDB_UPPER_OFF_CHILD_MASK: u32 = 4128u;
const PNANOVDB_LOWER_TABLE_COUNT: u32 = 4096u;
const PNANOVDB_LOWER_BASE_SIZE: u32 = 1056u;
const PNANOVDB_LOWER_OFF_BBOX_MIN: u32 = 0u;
const PNANOVDB_LOWER_OFF_BBOX_MAX: u32 = 12u;
const PNANOVDB_LOWER_OFF_FLAGS: u32 = 24u;
const PNANOVDB_LOWER_OFF_VALUE_MASK: u32 = 32u;
const PNANOVDB_LOWER_OFF_CHILD_MASK: u32 = 544u;
const PNANOVDB_LEAF_TABLE_COUNT: u32 = 512u;
const PNANOVDB_LEAF_BASE_SIZE: u32 = 80u;
const PNANOVDB_LEAF_OFF_BBOX_MIN: u32 = 0u;
const PNANOVDB_LEAF_OFF_BBOX_DIF_AND_FLAGS: u32 = 12u;
const PNANOVDB_LEAF_OFF_VALUE_MASK: u32 = 16u;
const PNANOVDB_LEAF_TABLE_NEG_OFF_BBOX_DIF_AND_FLAGS: u32 = 84u;
const PNANOVDB_LEAF_TABLE_NEG_OFF_MINIMUM: u32 = 16u;
const PNANOVDB_LEAF_TABLE_NEG_OFF_QUANTUM: u32 = 12u;

// ---- per-grid-type constants: FLOAT, FP8, FPN ----
// FLOAT (grid type 1)
const PNANOVDB_FLOAT_ROOT_OFF_BACKGROUND: u32 = 28u;
const PNANOVDB_FLOAT_ROOT_OFF_MIN: u32 = 32u;
const PNANOVDB_FLOAT_ROOT_OFF_MAX: u32 = 36u;
const PNANOVDB_FLOAT_ROOT_OFF_AVE: u32 = 40u;
const PNANOVDB_FLOAT_ROOT_OFF_STDDEV: u32 = 44u;
const PNANOVDB_FLOAT_ROOT_SIZE: u32 = 64u;
const PNANOVDB_FLOAT_VALUE_STRIDE_BITS: u32 = 32u;
const PNANOVDB_FLOAT_TABLE_STRIDE: u32 = 8u;
const PNANOVDB_FLOAT_ROOT_TILE_OFF_VALUE: u32 = 20u;
const PNANOVDB_FLOAT_ROOT_TILE_SIZE: u32 = 32u;
const PNANOVDB_FLOAT_UPPER_OFF_MIN: u32 = 8224u;
const PNANOVDB_FLOAT_UPPER_OFF_MAX: u32 = 8228u;
const PNANOVDB_FLOAT_UPPER_OFF_AVE: u32 = 8232u;
const PNANOVDB_FLOAT_UPPER_OFF_STDDEV: u32 = 8236u;
const PNANOVDB_FLOAT_UPPER_OFF_TABLE: u32 = 8256u;
const PNANOVDB_FLOAT_UPPER_SIZE: u32 = 270400u;
const PNANOVDB_FLOAT_LOWER_OFF_MIN: u32 = 1056u;
const PNANOVDB_FLOAT_LOWER_OFF_MAX: u32 = 1060u;
const PNANOVDB_FLOAT_LOWER_OFF_AVE: u32 = 1064u;
const PNANOVDB_FLOAT_LOWER_OFF_STDDEV: u32 = 1068u;
const PNANOVDB_FLOAT_LOWER_OFF_TABLE: u32 = 1088u;
const PNANOVDB_FLOAT_LOWER_SIZE: u32 = 33856u;
const PNANOVDB_FLOAT_LEAF_OFF_MIN: u32 = 80u;
const PNANOVDB_FLOAT_LEAF_OFF_MAX: u32 = 84u;
const PNANOVDB_FLOAT_LEAF_OFF_AVE: u32 = 88u;
const PNANOVDB_FLOAT_LEAF_OFF_STDDEV: u32 = 92u;
const PNANOVDB_FLOAT_LEAF_OFF_TABLE: u32 = 96u;
const PNANOVDB_FLOAT_LEAF_SIZE: u32 = 2144u;
// FP8 (grid type 14)
const PNANOVDB_FP8_ROOT_OFF_BACKGROUND: u32 = 28u;
const PNANOVDB_FP8_ROOT_OFF_MIN: u32 = 32u;
const PNANOVDB_FP8_ROOT_OFF_MAX: u32 = 36u;
const PNANOVDB_FP8_ROOT_OFF_AVE: u32 = 40u;
const PNANOVDB_FP8_ROOT_OFF_STDDEV: u32 = 44u;
const PNANOVDB_FP8_ROOT_SIZE: u32 = 64u;
const PNANOVDB_FP8_VALUE_STRIDE_BITS: u32 = 0u;
const PNANOVDB_FP8_TABLE_STRIDE: u32 = 8u;
const PNANOVDB_FP8_ROOT_TILE_OFF_VALUE: u32 = 20u;
const PNANOVDB_FP8_ROOT_TILE_SIZE: u32 = 32u;
const PNANOVDB_FP8_UPPER_OFF_MIN: u32 = 8224u;
const PNANOVDB_FP8_UPPER_OFF_MAX: u32 = 8228u;
const PNANOVDB_FP8_UPPER_OFF_AVE: u32 = 8232u;
const PNANOVDB_FP8_UPPER_OFF_STDDEV: u32 = 8236u;
const PNANOVDB_FP8_UPPER_OFF_TABLE: u32 = 8256u;
const PNANOVDB_FP8_UPPER_SIZE: u32 = 270400u;
const PNANOVDB_FP8_LOWER_OFF_MIN: u32 = 1056u;
const PNANOVDB_FP8_LOWER_OFF_MAX: u32 = 1060u;
const PNANOVDB_FP8_LOWER_OFF_AVE: u32 = 1064u;
const PNANOVDB_FP8_LOWER_OFF_STDDEV: u32 = 1068u;
const PNANOVDB_FP8_LOWER_OFF_TABLE: u32 = 1088u;
const PNANOVDB_FP8_LOWER_SIZE: u32 = 33856u;
const PNANOVDB_FP8_LEAF_OFF_MIN: u32 = 88u;
const PNANOVDB_FP8_LEAF_OFF_MAX: u32 = 90u;
const PNANOVDB_FP8_LEAF_OFF_AVE: u32 = 92u;
const PNANOVDB_FP8_LEAF_OFF_STDDEV: u32 = 94u;
const PNANOVDB_FP8_LEAF_OFF_TABLE: u32 = 96u;
const PNANOVDB_FP8_LEAF_SIZE: u32 = 608u;
// FPN (grid type 16)
const PNANOVDB_FPN_ROOT_OFF_BACKGROUND: u32 = 28u;
const PNANOVDB_FPN_ROOT_OFF_MIN: u32 = 32u;
const PNANOVDB_FPN_ROOT_OFF_MAX: u32 = 36u;
const PNANOVDB_FPN_ROOT_OFF_AVE: u32 = 40u;
const PNANOVDB_FPN_ROOT_OFF_STDDEV: u32 = 44u;
const PNANOVDB_FPN_ROOT_SIZE: u32 = 64u;
const PNANOVDB_FPN_VALUE_STRIDE_BITS: u32 = 0u;
const PNANOVDB_FPN_TABLE_STRIDE: u32 = 8u;
const PNANOVDB_FPN_ROOT_TILE_OFF_VALUE: u32 = 20u;
const PNANOVDB_FPN_ROOT_TILE_SIZE: u32 = 32u;
const PNANOVDB_FPN_UPPER_OFF_MIN: u32 = 8224u;
const PNANOVDB_FPN_UPPER_OFF_MAX: u32 = 8228u;
const PNANOVDB_FPN_UPPER_OFF_AVE: u32 = 8232u;
const PNANOVDB_FPN_UPPER_OFF_STDDEV: u32 = 8236u;
const PNANOVDB_FPN_UPPER_OFF_TABLE: u32 = 8256u;
const PNANOVDB_FPN_UPPER_SIZE: u32 = 270400u;
const PNANOVDB_FPN_LOWER_OFF_MIN: u32 = 1056u;
const PNANOVDB_FPN_LOWER_OFF_MAX: u32 = 1060u;
const PNANOVDB_FPN_LOWER_OFF_AVE: u32 = 1064u;
const PNANOVDB_FPN_LOWER_OFF_STDDEV: u32 = 1068u;
const PNANOVDB_FPN_LOWER_OFF_TABLE: u32 = 1088u;
const PNANOVDB_FPN_LOWER_SIZE: u32 = 33856u;
const PNANOVDB_FPN_LEAF_OFF_MIN: u32 = 88u;
const PNANOVDB_FPN_LEAF_OFF_MAX: u32 = 90u;
const PNANOVDB_FPN_LEAF_OFF_AVE: u32 = 92u;
const PNANOVDB_FPN_LEAF_OFF_STDDEV: u32 = 94u;
const PNANOVDB_FPN_LEAF_OFF_TABLE: u32 = 96u;
const PNANOVDB_FPN_LEAF_SIZE: u32 = 96u;

// ---- aux stride arrays (indexed by grid type id) ----
fn pnanovdb_grid_type_value_strides_bits(grid_type: u32) -> u32 {
    switch grid_type {
        case 0u: { return 0u; }
        case 1u: { return 32u; }
        case 2u: { return 64u; }
        case 3u: { return 16u; }
        case 4u: { return 32u; }
        case 5u: { return 64u; }
        case 6u: { return 96u; }
        case 7u: { return 192u; }
        case 8u: { return 0u; }
        case 9u: { return 16u; }
        case 10u: { return 32u; }
        case 11u: { return 1u; }
        case 12u: { return 32u; }
        case 13u: { return 4u; }
        case 14u: { return 8u; }
        case 15u: { return 16u; }
        case 16u: { return 0u; }
        case 17u: { return 128u; }
        case 18u: { return 256u; }
        case 19u: { return 0u; }
        case 20u: { return 0u; }
        case 21u: { return 0u; }
        case 22u: { return 0u; }
        case 23u: { return 16u; }
        case 24u: { return 24u; }
        case 25u: { return 48u; }
        case 26u: { return 8u; }
        case 27u: { return 0u; }
        default: { return 0u; }
    }
}
fn pnanovdb_grid_type_table_strides_bits(grid_type: u32) -> u32 {
    switch grid_type {
        case 0u: { return 64u; }
        case 1u: { return 64u; }
        case 2u: { return 64u; }
        case 3u: { return 64u; }
        case 4u: { return 64u; }
        case 5u: { return 64u; }
        case 6u: { return 128u; }
        case 7u: { return 192u; }
        case 8u: { return 64u; }
        case 9u: { return 64u; }
        case 10u: { return 64u; }
        case 11u: { return 64u; }
        case 12u: { return 64u; }
        case 13u: { return 64u; }
        case 14u: { return 64u; }
        case 15u: { return 64u; }
        case 16u: { return 64u; }
        case 17u: { return 128u; }
        case 18u: { return 256u; }
        case 19u: { return 64u; }
        case 20u: { return 64u; }
        case 21u: { return 64u; }
        case 22u: { return 64u; }
        case 23u: { return 64u; }
        case 24u: { return 64u; }
        case 25u: { return 64u; }
        case 26u: { return 64u; }
        case 27u: { return 64u; }
        default: { return 0u; }
    }
}
fn pnanovdb_grid_type_minmax_strides_bits(grid_type: u32) -> u32 {
    switch grid_type {
        case 0u: { return 0u; }
        case 1u: { return 32u; }
        case 2u: { return 64u; }
        case 3u: { return 16u; }
        case 4u: { return 32u; }
        case 5u: { return 64u; }
        case 6u: { return 96u; }
        case 7u: { return 192u; }
        case 8u: { return 8u; }
        case 9u: { return 16u; }
        case 10u: { return 32u; }
        case 11u: { return 8u; }
        case 12u: { return 32u; }
        case 13u: { return 32u; }
        case 14u: { return 32u; }
        case 15u: { return 32u; }
        case 16u: { return 32u; }
        case 17u: { return 128u; }
        case 18u: { return 256u; }
        case 19u: { return 64u; }
        case 20u: { return 64u; }
        case 21u: { return 0u; }
        case 22u: { return 0u; }
        case 23u: { return 64u; }
        case 24u: { return 24u; }
        case 25u: { return 48u; }
        case 26u: { return 8u; }
        case 27u: { return 0u; }
        default: { return 0u; }
    }
}
fn pnanovdb_grid_type_minmax_aligns_bits(grid_type: u32) -> u32 {
    switch grid_type {
        case 0u: { return 0u; }
        case 1u: { return 32u; }
        case 2u: { return 64u; }
        case 3u: { return 16u; }
        case 4u: { return 32u; }
        case 5u: { return 64u; }
        case 6u: { return 32u; }
        case 7u: { return 64u; }
        case 8u: { return 8u; }
        case 9u: { return 16u; }
        case 10u: { return 32u; }
        case 11u: { return 8u; }
        case 12u: { return 32u; }
        case 13u: { return 32u; }
        case 14u: { return 32u; }
        case 15u: { return 32u; }
        case 16u: { return 32u; }
        case 17u: { return 32u; }
        case 18u: { return 64u; }
        case 19u: { return 64u; }
        case 20u: { return 64u; }
        case 21u: { return 0u; }
        case 22u: { return 0u; }
        case 23u: { return 64u; }
        case 24u: { return 8u; }
        case 25u: { return 16u; }
        case 26u: { return 8u; }
        case 27u: { return 0u; }
        default: { return 0u; }
    }
}
fn pnanovdb_grid_type_stat_strides_bits(grid_type: u32) -> u32 {
    switch grid_type {
        case 0u: { return 0u; }
        case 1u: { return 32u; }
        case 2u: { return 64u; }
        case 3u: { return 32u; }
        case 4u: { return 32u; }
        case 5u: { return 64u; }
        case 6u: { return 32u; }
        case 7u: { return 64u; }
        case 8u: { return 8u; }
        case 9u: { return 32u; }
        case 10u: { return 32u; }
        case 11u: { return 8u; }
        case 12u: { return 32u; }
        case 13u: { return 32u; }
        case 14u: { return 32u; }
        case 15u: { return 32u; }
        case 16u: { return 32u; }
        case 17u: { return 32u; }
        case 18u: { return 64u; }
        case 19u: { return 64u; }
        case 20u: { return 64u; }
        case 21u: { return 0u; }
        case 22u: { return 0u; }
        case 23u: { return 64u; }
        case 24u: { return 32u; }
        case 25u: { return 32u; }
        case 26u: { return 32u; }
        case 27u: { return 0u; }
        default: { return 0u; }
    }
}
fn pnanovdb_grid_type_leaf_type(grid_type: u32) -> u32 {
    switch grid_type {
        case 0u: { return 0u; }
        case 1u: { return 0u; }
        case 2u: { return 0u; }
        case 3u: { return 0u; }
        case 4u: { return 0u; }
        case 5u: { return 0u; }
        case 6u: { return 0u; }
        case 7u: { return 0u; }
        case 8u: { return 1u; }
        case 9u: { return 0u; }
        case 10u: { return 0u; }
        case 11u: { return 1u; }
        case 12u: { return 0u; }
        case 13u: { return 2u; }
        case 14u: { return 2u; }
        case 15u: { return 2u; }
        case 16u: { return 2u; }
        case 17u: { return 0u; }
        case 18u: { return 0u; }
        case 19u: { return 3u; }
        case 20u: { return 3u; }
        case 21u: { return 0u; }
        case 22u: { return 0u; }
        case 23u: { return 5u; }
        case 24u: { return 0u; }
        case 25u: { return 0u; }
        case 26u: { return 0u; }
        case 27u: { return 0u; }
        default: { return 0u; }
    }
}
