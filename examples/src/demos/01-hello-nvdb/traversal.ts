/**
 * Hand-written WGSL subset of NanoVDB traversal — FLOAT grids only.
 *
 * This is the load-bearing artefact of the Phase 1 spike (docs/PLAN.md
 * Phase 1, docs/FEASIBILITY.md §9 risk #1): a raw WGSL `fn` that walks a
 * NanoVDB grid image living in a `storage` buffer, bound into three.js TSL
 * via `wgslFn` with a `ptr<storage, array<u32>, read>` parameter.
 *
 * It is a faithful transliteration of the CPU reference descent in
 * `packages/nanovdb-wgsl/src/cpu/read-value.ts` (itself validated 657/657
 * against native NanoVDB sidecars) and the vendored port
 * `packages/nanovdb-wgsl/vendor/pnanovdb.wgsl`, restricted to the FLOAT
 * grid type (id 1). Every magic byte offset is annotated with its
 * `PNANOVDB_*` name from
 * `packages/nanovdb-wgsl/src/wgsl/pnanovdb-constants.generated.wgsl`
 * (equivalently `vendor/stride-tables.json`, ABI 32.9.1).
 *
 * ## Layout constants inlined below (FLOAT grid type)
 *
 *   GridData:   size 672 (PNANOVDB_GRID_SIZE), magic @0, gridType @636.
 *   TreeData:   @+672; root node byte-offset is a u64 @ tree+24
 *               (PNANOVDB_TREE_OFF_NODE_OFFSET_ROOT).
 *   RootData:   background f32 @+28, tileCount u32 @+24
 *               (PNANOVDB_ROOT_OFF_TABLE_SIZE); tiles start at root+64
 *               (FLOAT root_size). Each tile is 32 bytes (FLOAT
 *               root_tile_size): key u64 @0, child i64 @8, state u32 @16,
 *               value f32 @20.
 *   Upper (32^3): value_mask @32, child_mask @4128, table @8256, stride 8.
 *   Lower (16^3): value_mask @32, child_mask @544,  table @1088, stride 8.
 *   Leaf  (8^3):  value_mask @16, table @96, value_stride_bits 32.
 *
 * The single entry point returns `vec2<f32>(value, activeFlag)` where
 * activeFlag is 1.0 (active), 0.0 (inactive), or negative as a sanity-check
 * sentinel (-1 bad magic, -2 wrong grid type) so the JS side can surface a
 * useful error instead of a silent zero.
 */
export const NVDB_PROBE_FLOAT_WGSL = /* wgsl */ `
fn nvdb_probe_float( grid : ptr<storage, array<u32>, read>, ijk : vec3<i32> ) -> vec2<f32> {

  // --- 0. GridData sanity: magic + grid type ------------------------------
  // PNANOVDB_MAGIC_GRID = "NanoVDB1" as a little-endian u64:
  //   low  word = 0x6f6e614e ("Nano"), high word = 0x31424456 ("VDB1").
  let magic_lo : u32 = (*grid)[ 0u ];              // PNANOVDB_GRID_OFF_MAGIC = 0
  let magic_hi : u32 = (*grid)[ 1u ];
  if ( magic_lo != 0x6f6e614eu || magic_hi != 0x31424456u ) {
    return vec2<f32>( 0.0, -1.0 );                 // sentinel: bad magic
  }
  let grid_type : u32 = (*grid)[ 636u >> 2u ];     // PNANOVDB_GRID_OFF_GRID_TYPE = 636
  if ( grid_type != 1u ) {                         // PNANOVDB_GRID_TYPE_FLOAT = 1
    return vec2<f32>( 0.0, -2.0 );                 // sentinel: wrong grid type
  }

  // --- 1. Tree -> root node address ---------------------------------------
  let tree_addr : u32 = 672u;                      // PNANOVDB_GRID_SIZE
  // Root offset is a u64; small scenes fit the low 32 bits (matches the
  // vendor port's pnanovdb_read_int64_as_offset).
  let root_off : u32 = (*grid)[ ( tree_addr + 24u ) >> 2u ]; // TREE_OFF_NODE_OFFSET_ROOT = 24
  let root_addr : u32 = tree_addr + root_off;

  // --- 2. coord -> root key (pnanovdb_coord_to_key, non-native-64 path) ----
  // Same bit layout as vendor pnanovdb.wgsl: split the 64-bit key across two
  // u32 lanes. u32(i32) reinterprets the sign bits, then >>12u is a logical
  // shift, exactly pnanovdb_int32_as_uint32(v) >> 12u.
  let iu : u32 = u32( ijk.x ) >> 12u;
  let ju : u32 = u32( ijk.y ) >> 12u;
  let ku : u32 = u32( ijk.z ) >> 12u;
  let key_lo : u32 = ku | ( ju << 21u );
  let key_hi : u32 = ( iu << 10u ) | ( ju >> 11u );

  // --- 3. Root tile linear scan (pnanovdb_root_find_tile) ------------------
  let tile_count : u32 = (*grid)[ ( root_addr + 24u ) >> 2u ]; // ROOT_OFF_TABLE_SIZE = 24
  let tile0 : u32 = root_addr + 64u;               // FLOAT root_size = 64
  var tile_addr : u32 = 0u;                        // 0 == "not found" (root is never at addr 0)
  for ( var i : u32 = 0u; i < tile_count; i = i + 1u ) {
    let cand : u32 = tile0 + i * 32u;              // FLOAT root_tile_size = 32
    let k_lo : u32 = (*grid)[ ( cand + 0u ) >> 2u ]; // ROOT_TILE_OFF_KEY = 0 (u64)
    let k_hi : u32 = (*grid)[ ( cand + 4u ) >> 2u ];
    if ( k_lo == key_lo && k_hi == key_hi ) {
      tile_addr = cand;
      break;
    }
  }

  // --- 4a. No tile -> background value, inactive --------------------------
  if ( tile_addr == 0u ) {
    let bg : f32 = bitcast<f32>( (*grid)[ ( root_addr + 28u ) >> 2u ] ); // root_off_background = 28
    return vec2<f32>( bg, 0.0 );
  }

  // --- 4b. Tile with no child -> constant tile value ----------------------
  let child_lo : u32 = (*grid)[ ( tile_addr + 8u ) >> 2u ]; // ROOT_TILE_OFF_CHILD = 8 (i64)
  let child_hi : u32 = (*grid)[ ( tile_addr + 12u ) >> 2u ];
  if ( child_lo == 0u && child_hi == 0u ) {
    let v : f32 = bitcast<f32>( (*grid)[ ( tile_addr + 20u ) >> 2u ] ); // root_tile_off_value = 20
    let state : u32 = (*grid)[ ( tile_addr + 16u ) >> 2u ];             // ROOT_TILE_OFF_STATE = 16
    return vec2<f32>( v, select( 0.0, 1.0, state != 0u ) );
  }

  // --- 5. Descend into the upper (32^3) node ------------------------------
  let upper_addr : u32 = root_addr + child_lo;
  // pnanovdb_upper_coord_to_offset
  let un : u32 =
    ( ( ( u32( ijk.x ) & 4095u ) >> 7u ) << 10u ) +
    ( ( ( u32( ijk.y ) & 4095u ) >> 7u ) <<  5u ) +
    ( ( u32( ijk.z ) & 4095u ) >> 7u );
  // upper child mask bit -> is there a lower child, or is this a tile value?
  let ucm : u32 = (*grid)[ ( upper_addr + 4128u + 4u * ( un >> 5u ) ) >> 2u ]; // UPPER_OFF_CHILD_MASK = 4128
  if ( ( ( ucm >> ( un & 31u ) ) & 1u ) == 0u ) {
    // upper tile value + upper value-mask bit
    let v : f32 = bitcast<f32>( (*grid)[ ( upper_addr + 8256u + 8u * un ) >> 2u ] ); // upper_off_table=8256, stride=8
    let uvm : u32 = (*grid)[ ( upper_addr + 32u + 4u * ( un >> 5u ) ) >> 2u ];        // UPPER_OFF_VALUE_MASK = 32
    return vec2<f32>( v, select( 0.0, 1.0, ( ( uvm >> ( un & 31u ) ) & 1u ) != 0u ) );
  }

  // --- 6. Descend into the lower (16^3) node ------------------------------
  let ut_addr : u32 = upper_addr + 8256u + 8u * un; // upper table entry (i64 child offset)
  let lower_off : u32 = (*grid)[ ut_addr >> 2u ];
  let lower_addr : u32 = upper_addr + lower_off;
  // pnanovdb_lower_coord_to_offset
  let ln : u32 =
    ( ( ( u32( ijk.x ) & 127u ) >> 3u ) << 8u ) +
    ( ( ( u32( ijk.y ) & 127u ) >> 3u ) << 4u ) +
    ( ( u32( ijk.z ) & 127u ) >> 3u );
  let lcm : u32 = (*grid)[ ( lower_addr + 544u + 4u * ( ln >> 5u ) ) >> 2u ]; // LOWER_OFF_CHILD_MASK = 544
  if ( ( ( lcm >> ( ln & 31u ) ) & 1u ) == 0u ) {
    // lower tile value + lower value-mask bit
    let v : f32 = bitcast<f32>( (*grid)[ ( lower_addr + 1088u + 8u * ln ) >> 2u ] ); // lower_off_table=1088, stride=8
    let lvm : u32 = (*grid)[ ( lower_addr + 32u + 4u * ( ln >> 5u ) ) >> 2u ];        // LOWER_OFF_VALUE_MASK = 32
    return vec2<f32>( v, select( 0.0, 1.0, ( ( lvm >> ( ln & 31u ) ) & 1u ) != 0u ) );
  }

  // --- 7. Descend into the leaf (8^3) node --------------------------------
  let lt_addr : u32 = lower_addr + 1088u + 8u * ln; // lower table entry (i64 leaf offset)
  let leaf_off : u32 = (*grid)[ lt_addr >> 2u ];
  let leaf_addr : u32 = lower_addr + leaf_off;
  // pnanovdb_leaf_coord_to_offset (0..511)
  let leaf_n : u32 =
    ( ( u32( ijk.x ) & 7u ) << 6u ) +
    ( ( u32( ijk.y ) & 7u ) << 3u ) +
    ( u32( ijk.z ) & 7u );
  let lvm : u32 = (*grid)[ ( leaf_addr + 16u + 4u * ( leaf_n >> 5u ) ) >> 2u ]; // LEAF_OFF_VALUE_MASK = 16
  // NB: "active" is a reserved WGSL keyword, so this flag is named "act".
  let act : f32 = select( 0.0, 1.0, ( ( lvm >> ( leaf_n & 31u ) ) & 1u ) != 0u );
  // FLOAT leaf value table: byte = leaf_off_table(96) + ((value_stride_bits(32) * n) >> 3)
  let val_addr : u32 = leaf_addr + 96u + ( ( 32u * leaf_n ) >> 3u );
  let v : f32 = bitcast<f32>( (*grid)[ val_addr >> 2u ] );
  return vec2<f32>( v, act );
}
`;
