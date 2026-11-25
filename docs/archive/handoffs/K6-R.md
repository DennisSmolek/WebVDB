# K6.R handoff: the reference renderer

Design: [K6-core-renderer.md](../../dev/design/K6-core-renderer.md) §7.4 (and §3.4 mixing, §3.6 multiple scattering),
D34. Built in parallel with K6.1 (`core/frame/**`, untouched here). Owner context (2026-11-25): mjurczyk's renderer is a
benchmark, not the spec; the reference judges the quality choices (multiple-scattering default, K7.3 edge mode,
mixing of overlapping volumes). Correctness and quality only: no timings.

## What exists

| Piece | Where | Notes |
|---|---|---|
| GPU kernel + host object | `packages/vdb/src/core/reference/{wgsl,renderer}.ts` | `await vdbCreateReferenceRendererGpu(device, { source, width, height })` → `setScene(scene, settings)`, `render(encoder, batches)`, `readback()`, `reset()`, `dispose()`. Internal (`core/internal.ts`), no public export, no api-surface change. |
| Scene, stream, CPU twin, media, analytic cases | `packages/vdb/src/tools/reference/{scene,rng,trace,media,analytic}.ts` | `VDBReferenceScene` (plain data), `vdbRefRng` (PCG), `vdbReferenceRenderCpu`, `vdbReferenceStatistics`, `vdbReference{Dense,Sparse,Constant}MediumCpu`, `vdbRefBeerLambert` / `vdbRefEmissionOnly` / `vdbRefSingleScatter`, `vdbReferenceLightsFromFrame`, `vdbReferenceSigma`. Internal (`tools/internal.ts`). |
| Unit tests | `packages/vdb/test/tools/reference.test.ts` (12), `packages/vdb/test/core/reference.test.ts` (5) | CPU twin vs closed forms + mutations; WGSL names / call sites / patches; uniform layout computed from the WGSL struct text vs the packer. |
| Harness + e2e | `examples/src/harness/k6-reference.{html,ts}`, `e2e/k6-reference.spec.ts` (9) | Raw WebGPU, no three. `?case=analytic|twin|report`. Committed and synthetic data only: no fixtures, no skips. 14 s on SwiftShader. |
| Report | `scripts/k6/reference-report.mjs` | Local only (needs `fixtures/k3/bunny_cloud.wvdb` and `fixtures/k3-reference/75e7345/`). Writes `artifacts/k6-reference/`. |

## The model (what "reference" means here)

A brute-force volumetric path tracer over the renderer's medium, so that differences measure approximation, not
convention:

- **Medium** (three/frame `sampleExtinction` / `sampleAlbedo` / `sampleEmission` at level 0): per volume a local unit
  box placed by `localToWorld`; σt = `sigma` · value(uvw), value = density × mask, 0 at or below `cutoff`;
  `sigma` = densityScale (÷ mean with `normalizeDensity`) × extinction ÷ worldScale (`vdbReferenceSigma`). Albedo =
  base colour × albedo channel. Emission is the renderer's source term σt · Le, Le = colour · (T/T₁)⁴ · intensity.
  Overlapping volumes add their σt (co-located media); a collision picks volume i with probability σᵢ/σ. `shape:
  'sphere'` (the inscribed ball) exists for analytic tests.
- **Transport**: delta tracking (Woodcock) for collisions, ratio tracking (Novák et al. 2014) for every transmittance
  (primary T, light visibility), Russian roulette in ratio tracking below T = 0.1 (survive ½); piecewise majorants
  Σ σᵢ·majorantᵢ over the ray's intervals in each volume, so gaps cost nothing; majorant = the volume's density max
  (`stats.max`) by default. Next-event estimation of every punctual light at every collision; the phase (HG or
  dual-lobe, exact normalisation) is sampled exactly (weight 1); the environment (uniform, ≤ 4 hemispheres, an
  equirect map read nearest) is gathered when a path leaves after ≥ 1 collision. Russian roulette on β from
  `rrStart` (survival min(max β, 1): albedo 1 never terminates early), cap `maxBounces` (default 256; 512 in the
  report). Camera rays through pixel centres or box-filtered.
- **Light units** (`vdbReferenceLightsFromFrame`): the renderer's punctual lights are irradiance **4·colour**
  (`VDB_REFERENCE_DIRECT_LIGHT_SCALE`: 'volume' lighting uses phase/(1/4π) and divides by π), point/spot without
  distance falloff (`inverseSquare` optional); ambient C = uniform radiance C/π; hemisphere sky/ground = S/π, G/π.
  With these, the TSL renderer's single scattering lands within ~15 % of the reference's (below), so the
  conventions agree; a factor 4 would show as −75 %/+300 %.
- **Output**: per pixel (L.rgb, T): L the radiance the media send to the camera (the background behind is *not* in L),
  T the primary transmittance. Composite `L + T·background` (or his sRGB blend: the report does).
- **Accumulation**: rgba32float ping-pong textures hold Σ batch means and Σ squared batch means (one batch = `spp`
  samples per pixel per dispatch), so `vdbReferenceStatistics` gives mean and standard error per pixel and channel.
  Batch records are a dynamic-offset uniform ring (≤ 64 batches per `render()`; one `render()` per submit).
- **Bindings**: group 0 the scene uniform (2,736 B: ≤ 8 volumes, ≤ 16 lights, ≤ 4 hemispheres) + the batch record;
  group 1 accumulation (2 sampled + 2 storage rgba32float) + the equirect map; group 2 the source (`vdbSrc*`, the
  K6 group-2 contract: dense textures or the pool — the kernel's only storage buffer, D10). All volumes of one kernel
  share one source object; volumes are (slot `v`, transform, medium params), so one grid can be instanced.
- **Compile time**: one `vdbRefTrack` call site (the path is a state machine: primary T → segment → light k …), one
  `vdbSrcDensity` / `Mask` / `Albedo` / `Emission` site each (a unit test counts them), so SwiftShader compiles it
  without the K3.7 multi-minute builds (the whole e2e spec, 9 page loads and 9 compiles, ran in 14 s; not a timing).
- **Random stream**: PCG (Jarzynski & Olano 2020), u32 only, identical bits in WGSL and JS; one stream per
  (seed, sample index, pixel); floats = top 24 bits. The twin draws in the same order (documented in `trace.ts`).

## Gates and numbers (SwiftShader, headless Chromium, this machine)

### Analytic cases (GPU kernel; the CPU twin passes the same cases in vitest)

16 × 16 orthographic pixels, 24 batches × 8 spp (single scattering 16 spp), per value |mean − exact| ≤ 4.5 σ +
2e-5, ≤ 0.5 % values over, and the image unbiased (|Σe/√Σσ²| < 4):

| Case | Exact | values over / 1024 | image z |
|---|---|---|---|
| Beer–Lambert, slab (σ 1.3, majorant 2σ) | exp(−σ·chord) | 0 | 0.67 |
| Beer–Lambert, ball | exp(−σ·chord) | 1 | −0.95 |
| Emission only (albedo 0) | Le·(1 − T) | 0 | 2.60 |
| Single scattering, slab (dual-lobe 0.6/−0.3/0.25, RGB albedo, sun) | Simpson quadrature, 4096 panels | 0 | 0.06 |
| Single scattering, ball | quadrature | 0 | −0.34 |
| White furnace (albedo 1, g 0.5, ball τ ≈ 3.8 across, uniform Le) | L + T·Le = Le | 1 | −2.43 |

Fail-before (WGSL patches; the CPU twin has the same names): `ratio-weight` fails all six (image z 7.7–26.6),
`delta-accept` fails single scattering and the furnace (z 6.7–25.9), `phase-norm` and `nee-transmittance` fail single
scattering (z ≥ 10.8), `emission-albedo` fails emission (z −410), `escape-depth` fails the furnace (z −325).

### GPU vs CPU twin (same seeds, same stream)

| Scene | Size × samples | values over 4.5σ | max z | values equal (1e-3 rel) | paths ≥ 2 bounces / longest |
|---|---|---|---|---|---|
| Dense: committed CC0 cloud via `vdbDenseSource`, perspective, sun + point light, ambient + hemisphere, dual-lobe, extinction 8 | 20 × 16 × 16 | 0 / 1280 | 0.30 | 99.5 % | 5.7 % / 9 |
| Sparse: synthetic grids via `vdbSparseSourceGpu` (`lod: 'none'`, clamp01): one grid twice (overlapping, rotated) + one with mask, RGB albedo, emission and a cutoff; spot (inverse square) + sun; equirect map (rotated); box filter | 18 × 14 × 16 | 0 / 1008 | 0 | 100 % | 5.4 % / 15 |

Determinism: two GPU renders with the same seed are bitwise equal. Fail-before: `phase-norm` on the dense twin
→ 14 values over, image z 22.8, 84 % equal.

### Spec runs

- `pnpm typecheck`: exit 0.
- `pnpm test`: exit 0 (86 files passed, 4 skipped; 1602 passed, 243 skipped: fixture skips in this worktree).
- `WEBVDB_E2E_PORT=5296 pnpm exec playwright test e2e/k6-reference.spec.ts`: exit 0, 9 passed, 14.2 s.
  `K6R_LOG=1` prints every case's numbers.

## First report (hardware, report only)

`node scripts/k6/reference-report.mjs --headless --w=160 --h=90 --scale=8 --budget=240 --batches=16384` on the
owner's Mac (installed Chrome 158 headless, adapter "apple metal-3"; not a quiet machine — no timing is claimed, the
seconds below only say how long the run was). His bunny (`fixtures/k3/bunny_cloud.wvdb`, his capture manifest's
camera, box and sun), dual-lobe 0.8 / −0.3 / 0.2 (k3-frame sets it for both lightings), extinction 100 (optical
depth ≈ 100 along the box side at the mean density), albedo 1. Reference 160 × 90, box-filtered pixels, ≤ 512
bounces: 16,384 batches × 1 spp (two bunnies: 13,110 in the 240 s budget); per-pixel noise (mean relative σ of Y over
the volume pixels) 4.3 % with multiple scattering, 1.3 % single. TSL frames at 1280 × 720 (his size) box-downsampled
8× in linear light. **Our TSL frames and his captures give the same numbers to 0.1 % in every row** (the K3 parity),
so the columns below hold for his renderer too.

Columns: luminance bias = Σ Y / Σ Y_ref − 1 over the volume pixels (the aggregate energy error; noise-free to ≪ 1 %);
median relative luminance error per pixel (floor ≈ the reference noise, ~3–4 %); PSNR of the sRGB frame with his
background (dominated by the background; read the first two).

| Group / image | vs reference with multiple scattering: bias | median rel. err | PSNR dB | vs reference single scattering: bias |
|---|---|---|---|---|
| 'volume' lighting, ms off | −96.2 % | 98.1 % | 22.37 | **−14.3 %** (median 15.7 %) |
| 'volume' lighting, ms on (1D diffusion) | −66.9 % | 62.9 % | 27.73 | +640 % |
| 'frame' lighting, ms off | −97.5 % | 98.3 % | 22.16 | −43.4 % (median 31.3 %) |
| 'frame' lighting, diffusion (k3-frame's Step 3 default = his `step3-default` capture) | −66.8 % | 62.5 % | 27.78 | +642 % |
| 'frame' lighting, octaves (4, ½ ½ ½) | −63.7 % | 66.8 % | 27.83 | +712 % |
| 'frame' lighting, two-sided | **+1.5 %** | 28.8 % | 34.63 | +2171 % |
| 'frame' lighting, neural (his v0.26 default) | **−12.1 %** | **21.4 %** | **37.71** | +1866 % |
| two overlapping bunnies, 'frame', diffusion, draw-order 'over' | −67.8 % | 66.1 % | 25.90 | +533 % |
| (reference single scattering itself) | −95.5 % | 95.9 % | 22.57 | 0 |

What it says (first reading; one scene, one light):

1. **Conventions agree.** Single scattering of the TSL path ('volume' lighting, ms off) is 14 % darker than the exact
   single scattering, the residue of its approximations (froxel slices, the lit position, LOD-filtered density,
   32-step shadow march): no factor 4 or π is missing. 'frame' lighting single scattering is 43 % dark (its shadow
   grid at 64³ smooths the bunny's thin features). Both are measurable targets for K6.1/K6.2.
2. **Multiple scattering dominates this scene**: the full solution carries ~22× the single-scattered energy (albedo
   1, τ ≈ 100). 1D diffusion (`true`) recovers only a third of it (−67 %), octaves likewise (−64 %); **two-sided
   diffusion gets the total energy right (+1.5 %)** and **the neural network gets the per-pixel distribution best**
   (median 21 %, PSNR 37.7, energy −12 %). For K6.3's default the comparable candidates in the TSL path today are
   diffusion (clearly worst of the MS modes here), two-sided and neural; `'multi-scale'` (K7.3) is not in the TSL
   path, so it can be ranked only once K6.3 wires it (add the row; follow-up). Read with care: one scene, one sun,
   one phase, thick albedo-1 medium; thinner or absorbing media will rank differently, so the K6.3 decision needs the
   scene set extended (WDAS, smoke, an absorbing medium) before it is made.
3. **Overlap**: the two-bunny frame (draw-order 'over', diffusion) is 68 % dark, the same as one bunny with diffusion,
   so on this scene the MS model, not the mixing, sets the error. Judging `'mixture'` vs `'over'` needs K6.1's core
   frames (both modes) against the same reference, ideally with MS off, where mixing is the main difference.

Artifacts: `artifacts/k6-reference/{volume,frame,two}/` (reference, reference-single, every TSL frame and capture at
160 × 90, `*.err.png` heat maps vs the reference), `report.{json,md}`. The script's `--groups=` reruns some groups
and keeps the others in `report.*`.

## Follow-ups

- **K6.1 lands**: add the core frames to the report (`?backend=core` in k3-frame) and the `froxelMix: 'mixture'` vs
  `'over'` rows on the two-bunnies group; the reference mixes co-located media by construction (σ adds).
- **K6.3**: multi-scale (K7.3) is not in the TSL path; when K6.3 wires it, add `ms=multi-scale` rows (both edge modes,
  σ = 2^k) to the `frame` group. Pick the default on the luminance bias and median error against the reference.
- Lorenz-Mie / Mie RGB phase (sample HG with the table asymmetry, weight by the table ratio; uniform LUT), light-probe
  SH, the renderer's blackbody emission colour (a LUT) and its 'directional' environment-light extraction are not in
  the reference: scenes using them can't be judged yet (the report maps only directional/ambient/hemisphere).
- Per-node majorants (node stats via the sparse source; `vdbSrcMaxIn` is global today, K6.7 follow-up): fewer null
  collisions in sparse, thick grids. Spectral (per-channel σ, hero wavelength) for K6.4.
- The WDAS cloud (sparse, `fixtures/wdas/`) in the report needs a k3-frame scene for it, or a reference-only page.
- FLIP (or SSIM) in the report; a convergence stop (per-pixel relative σ < ε) instead of a time budget.
- The pmndrs pathtracer (P1) becomes a second reference; cross-check it against this one on the analytic cases.

## DIVERGES rows (provisional numbers, for the lead)

| ID | Date | Source | What we do differently | Kind | Revisit when |
|---|---|---|---|---|---|
| V80 | 2026-11-25 | Physical light units (radiometry), as in pathtracers | K6.R: the reference renderer follows the froxel renderer's conventions, not three's or a physical renderer's: punctual lights are irradiance 4 × three colour (the renderer's phase/(1/4π) then ÷π), point and spot lights have no distance falloff, emission is σt·Le. So the report measures approximation error, not convention. | consistency (reference ↔ renderer) | The renderer adopts physical units (then set `VDB_REFERENCE_DIRECT_LIGHT_SCALE` to 1, falloff on) |
| V81 | 2026-11-25 | mjurczyk/openvdb `75e7345` 'volume' lighting (ambient and light probe added unshadowed) | K6.R: the reference treats ambient light as a uniform environment radiance C/π that the medium shadows (physical). Scenes with ambient light show the renderer's unshadowed ambient as error. | reference (physical) | — |
