# STATUS

- **Phases 0–4 complete** (CI scope; see docs/handoffs/). Phase 5 wave 1 complete (parser 5a + serializer 5b); **wave 2 in flight**: buildFromVdb + Fp8/FpN quantization via the LeafCodec seam
- **Suite:** 177 unit / 7 e2e (demos 01–04 + gpu-parity + smoke; goldens stable) / clean tsc — all pushed
- **Deferred to a native/hardware machine:** WDAS 60fps + perf table (PHASE-3.md steps); nanovdb_convert byte/value parity for the TS serializer+parser; Docker bake image build; Data3DTexture/texture3D path retest (sandbox Dawn bug, PHASE-4.md)
- **Open decisions:** keep the dependency-free inflate in vdb-web-tools vs add fflate; picovdb license question upstream (PLAN §3.3) still unfiled
- **Blockers:** none
- **Read first on resume:** this file → docs/handoffs/PHASE-4.md
