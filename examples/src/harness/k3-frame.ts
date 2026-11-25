/**
 * K3.1 harness (R1 §6 Step 1): our port of mjurczyk's froxel pipeline with `'volume'` lighting,
 * on the dense `.wvdb` source, rendered offscreen and read back.
 *
 * Scenes (`?scene=`):
 * - `smoke` (default): the committed CC0 `cloud_04_variant_0000.wvdb`, one directional light, a
 *   small froxel grid; the e2e scene (SwiftShader-sized).
 * - his parity scenes, by the names of `scripts/k3/capture-reference.mjs` (`step1-volume-dir-ms-off`,
 *   `step1-volume-dir-ms-on`, `step1-lighting-volume`, `step1-lights-directional`,
 *   `step1-multiple-scattering-off`, `step0-bunny`, `step0-smoke`): his `frame.js` scene rebuilt (his
 *   `bunny_cloud.wvdb` / `smoke.wvdb` from `/fixtures/k3/`, not committed: CC BY-SA), his camera,
 *   volume placement, phase and light; or, when `?manifest=<url>` (default
 *   `/fixtures/k3-reference/75e7345/manifest.json`) has the scenario, the captured camera and volume
 *   matrices. 1280x720, his froxel defaults (512x512x24, 32 sub-steps). `parityExpected` says whether
 *   the Step 1 port can match his capture (see docs/archive/handoffs/K3-1.md: environment and 'frame' lighting).
 *
 * Pipeline: scene → linear HalfFloat RenderTarget (as his canvas frame-buffer target: 4x MSAA for his scenes, his
 * `antialias: true`; none for ours, `?msaa=`) → resolve →
 * readback → sRGB OETF on the CPU (three's formula, NoToneMapping) → RGBA8. Jitter and temporal
 * off: deterministic. Also: a probe pass of the TSL chunks against the CPU twins (`vdbFrame*`,
 * tools/frame) and, with `?twin=1`, the integrate pass against `vdbFrameIntegrate` from the read-back
 * froxel atlas (froxel grid = screen size, so each pixel reads one froxel column unfiltered).
 *
 * Other query: `?w=&h=`, `?froxels=WxHxD`, `?substeps=`, `?shadowSteps=`, `?ms=0|1`, `?mips=0`,
 * `?occluder=1` (an opaque box in front of the volume centre), `?novolume=1`, `?timing=1` (per-pass
 * GPU ms: scene, froxels (inject), integrate; `?timingFrames=300&timingWarmup=60` for medians; provisional per
 * D25), `?file=`, `?debug=1` (raw chunk-probe values in the result).
 *
 * K3.3 ('frame' lighting, R1 §6 Step 3): `?lighting=frame` (his Step 3 scenes set it: `step3-default`,
 * `step3-shadow-cascades-4`, `step3-shadow-grid-high`, `step3-two-bunnies`, `step3-environment-sky-midday`
 * and their `*-dir-isolated` / `*-env-isolated` parity variants), `?source=sparse` (the same texels as a
 * NanoVDB grid, `?lod=` default 'coarse-pyramid'), `?cascades=`, `?shadowRes=`, `?shadowDistance=`,
 * `?second=1` (his second bunny), `?lights=dir|dir-hemi|none` (his sun; + his hemisphere and ambient
 * lights), `?env=synthetic|<UltraHDR url>` (`?envIntensity=`, `?envMode=directional|spherical-harmonics|outward`),
 * `?dist=` (smoke camera distance / diagonal), `?jitter=1`, `?temporal=1`, `?twins=1` (transmittance,
 * environment grid, density and lighting passes vs their CPU twins from read-back targets; the grid
 * chunk probe always runs), `?redraw=N` (N more frames, then the redraw-skip counters),
 * `?temporalCheck=N` (N jittered frames without and with temporal reprojection: convergence).
 *
 * K3.4 (phase functions + multiple scattering, R1 §6 Step 4): `?ms=0|1|two-sided|octaves|neural` (default: the
 * scene's; his default 'neural' for the Step 4 scenes), `?octaves=N&octaveExtinction=&octaveContribution=&octavePhase=`,
 * `?phase=isotropic|henyey-greenstein|dual-lobe|lorenz-mie|lorenz-mie-rgb&phaseG=&phaseBackG=&phaseBackWeight=&phaseGExponent=`
 * (the renderer's 'frame' phase and the volume's 'volume' phase), `?extinction=`; his Step 4 scenes (`step4-*`, as
 * is and `*-dir-isolated`); `?furnace=1`: his white furnace (utils/furnace.js): a uniform environment of radiance
 * L = 0.5 as background and lighting, a white (non-absorbing) volume, no scene light; the result's `furnace` has the
 * 9 × 9 centre mean / L (his readout) and volume-pixel statistics of radiance / L. With `?twins=1` also the K3.4
 * chunk probe (`scatterProbe`) and, with 'neural', the grid mip twin.
 *
 * K3.5 (spectral media + presets, R1 §6 Step 5): `?spectral=1` (the renderer's `spectral`), `?medium=<name>` (his GUI
 * 'Medium' field: getFrameMediumProps on the volume — albedo, Ångström exponent, phase — and the renderer's 'frame'
 * phase; turns spectral on), `?angstrom=` (the volume's Ångström exponent); his Step 5 scenes (`step5-*`, as is and
 * isolated). With `?twins=1` and spectral on, the spectral pass twins (transmittance per slot group, environment
 * grid, density, lighting) and the spectral chunk probe (`spectralProbe`); `?twin=1` takes the spectral integrate twin;
 * `?extraLights=N` adds N directional lights (6 shadow slots fill the 4 spectral slot groups).
 * 'volume' lighting now reads the environment, hemisphere and light-probe terms (the furnace runs on it too).
 * Results: `window.__VDB_K3_FRAME__` (e2e/k3-frame.spec.ts writes PNGs to artifacts/k3/ours/).
 */
import * as THREE from "three";
import { NodeMaterial, PMREMGenerator, WebGPURenderer } from "three/webgpu";
import { VDB_FRAME_ENVIRONMENT, vdbFrameRendererEnvironment } from "../../../packages/vdb/src/core/internal.js";
import * as TSL from "three/tsl";
import { UltraHDRLoader } from "three/examples/jsm/loaders/UltraHDRLoader.js";
import { VDBFrameEmissionLight, VDBFrameRenderer, VDBFrameVolume, VDBGridNode, vdbDenseTextures, vdbFrameBlackbodyEmission, vdbFrameMedium, vdbFrameChunks, vdbSparseSource, type VDBFrameMultipleScattering, type VDBFramePhase, type VDBFrameRedrawStats, type VDBFrameRendererSettings, type VDBThreeVolumeSource } from "../../../packages/vdb/src/three/internal.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { densityTwin, environmentGridTwin, furnaceEnvironment, gridMipTwin, lightingTwin, probeGrid, probeScattering, syntheticEnvironment, transmittanceTwin, type TwinStats } from "./k3-frame-lighting.js";
import {
  wvdbDenseToNvdb,
  readWvdb,
  vdbFrameDistanceFade,
  vdbFrameHenyeyGreenstein,
  vdbFrameIntegrate,
  vdbFrameNoise,
  vdbFramePhaseWeight,
  vdbFrameShadowAbsorption,
  vdbFrameShadowDepth,
  vdbFrameShadowStep,
  vdbFrameShadowTransmittance,
  vdbFrameShadowedLight,
  vdbFrameSliceDepth,
  vdbFrameSrgbBlend,
  vdbSrgbOetf,
  vdbSrgbEotf as vdbSrgbEotfLocal,
  wvdbDenseVolume,
  type VDBColorMatchingFunctions,
  type VDBFrameFroxelSample,
  type VDBFrameVec3,
} from "../../../packages/vdb/src/tools/internal.js";
import { probeSpectral, spectralDensityTwin, spectralEnvironmentGridTwin, spectralIntegrateTwin, spectralLightingTwin, spectralTransmittanceTwin } from "./k3-frame-spectral.js";
import cloudWvdbUrl from "../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;
const { Fn, float, screenCoordinate, vec2, vec3, vec4, If, select } = TSL as unknown as Record<string, N>;

interface SceneDef {
  name: string;
  file: string;
  /** Lights of his `lightSetup` we can render ('dir': his sun; 'none': no scene light). */
  lightSetup: "dir" | "environment" | "none";
  lighting: "volume" | "frame";
  multipleScattering: VDBFrameMultipleScattering;
  /** True when the port can match his capture of this scene. */
  parityExpected: boolean;
  why?: string;
  smoke?: boolean;
  /** K3.3 renderer settings of the scene (his GUI fields). */
  shadowCascades?: number;
  shadowResolution?: number;
  /** His `source: 'twoBunnies'`: a second bunny overlapping the first. */
  twoBunnies?: boolean;
  /** His `environment` field: the UltraHDR map `/fixtures/k3/<name>-HDR.jpg` as scene.environment (intensity 0.4). */
  environment?: string;
  /** K3.4: renderer settings of the scene (his `frame-phase*` and `scatteringOctave*` GUI fields). */
  settings?: Partial<VDBFrameRendererSettings>;
  /** K3.4: his volume phase fields (`phaseG`, ...: 'volume' lighting). */
  volumePhase?: { phaseG?: number; phaseBackG?: number; phaseBackWeight?: number; phaseGExponent?: number };
  /** K3.5: his Spectral folder fields (`spectral`, `medium`, `angstromExponent`). */
  spectral?: boolean;
  medium?: string;
  angstromExponent?: number;
  /**
   * K3.6: his Shadow Casting and Light Emission fields (capture-reference.mjs Step 6): `castShadow` (castShadows on, his
   * sun and props cast and receive, the ShadowMaterial floor shown), `showObstacle` (the sphere, pillars and spheres),
   * `explosion` (his ground explosion frame with its emission light, 'planck' spectrum), the emission light's probes /
   * cascades, and the isolation of the capture (`shadow`: his sun only; `emission`: no scene light, no environment).
   */
  step6?: { castShadow?: boolean; showObstacle?: boolean; castShadowCascades?: number; castShadowCoarse?: boolean; explosion?: boolean; emissionResolution?: number; emissionCascades?: number; shadowOnly?: boolean };
}

// K3.3 (R1 §6 Step 3): his 'frame' lighting scenes. The as-is captures use his 'environment' light setup:
// three-environment-shadows scene lights found in the HDR map + the map without them + a grounded skybox, none of
// which this harness renders, so only the `*-isolated` variants (capture-reference.mjs `isolate: 'frame'`) gate.
const STEP3_WHY_ENV =
  "his 'environment' light setup: three-environment-shadows scene lights extracted from the UltraHDR map, the map without them as scene.environment, and the grounded skybox (not ported; the *-isolated variant gates)";
const step3 = (name: string, extra: Partial<SceneDef>, parityExpected: boolean, why?: string): SceneDef => ({
  name,
  file: "/fixtures/k3/bunny_cloud.wvdb",
  lightSetup: "dir",
  lighting: "frame",
  multipleScattering: true,
  parityExpected,
  ...(why ? { why } : {}),
  ...extra,
});
const STEP3_SCENES: Record<string, SceneDef> = Object.fromEntries(
  [
    step3("step3-default", { lightSetup: "environment" }, false, STEP3_WHY_ENV),
    step3("step3-shadow-cascades-4", { lightSetup: "environment", shadowCascades: 4 }, false, STEP3_WHY_ENV),
    step3("step3-shadow-grid-high", { lightSetup: "environment", shadowResolution: 160 }, false, STEP3_WHY_ENV),
    step3("step3-two-bunnies", { lightSetup: "environment", twoBunnies: true }, false, STEP3_WHY_ENV),
    step3("step3-environment-sky-midday", { lightSetup: "environment", environment: "sky-midday" }, false, STEP3_WHY_ENV),
    step3("step3-default-dir-isolated", {}, true),
    step3("step3-shadow-cascades-4-dir-isolated", { shadowCascades: 4 }, true),
    step3("step3-shadow-grid-high-dir-isolated", { shadowResolution: 160 }, true),
    step3("step3-two-bunnies-dir-isolated", { twoBunnies: true }, true),
    step3("step3-environment-sky-midday-env-isolated", { lightSetup: "none", environment: "sky-midday" }, true),
  ].map((d) => [d.name, d]),
);

// K3.4 (R1 §6 Step 4): his frame-phase-*, multiple-scattering-* and octave* validation scenes (tests/validation-set/
// demo-1-*.json). The as-is ones use his 'environment' light setup (INFO, as Step 3); the `*-dir-isolated` variants
// (capture-reference.mjs `isolate: 'frame'`, his sun only) gate. His default multipleScattering is 'neural'.
const STEP4_WHY_ENV = STEP3_WHY_ENV;
const step4 = (name: string, settings: Partial<VDBFrameRendererSettings>, extra: Partial<SceneDef> = {}): SceneDef[] => [
  step3(`${name}-asis`, { lightSetup: "environment", multipleScattering: "neural", settings, ...extra }, false, STEP4_WHY_ENV),
  step3(`${name}-dir-isolated`, { multipleScattering: "neural", settings, ...extra }, true),
];
const STEP4_SCENES: Record<string, SceneDef> = Object.fromEntries(
  [
    ...step4("step4-default-neural", {}),
    ...step4("step4-multiple-scattering-two-sided", { multipleScattering: "two-sided" }),
    ...step4("step4-multiple-scattering-octaves", { multipleScattering: "octaves" }),
    ...step4("step4-octaves-1", { multipleScattering: "octaves", scatteringOctaves: 1 }),
    ...step4("step4-octaves-8", { multipleScattering: "octaves", scatteringOctaves: 8 }),
    ...step4("step4-octave-extinction-high", { multipleScattering: "octaves", scatteringOctaveExtinction: 1 }),
    ...step4("step4-octave-contribution-high", { multipleScattering: "octaves", scatteringOctaveContribution: 1 }),
    ...step4("step4-octave-phase-low", { multipleScattering: "octaves", scatteringOctavePhase: 0 }),
    ...step4("step4-frame-phase-isotropic", { phase: "isotropic" }),
    ...step4("step4-frame-phase-g-low", { phaseG: -0.5 }),
    ...step4("step4-frame-phase-dual-lobe", { phase: "dual-lobe" }),
    ...step4("step4-frame-phase-dual-lobe-back-g-high", { phase: "dual-lobe", phaseBackG: 0.5 }),
    ...step4("step4-frame-phase-dual-lobe-back-weight-high", { phase: "dual-lobe", phaseBackWeight: 1 }),
    ...step4("step4-frame-phase-lorenz-mie", { phase: "lorenz-mie" }),
    ...step4("step4-frame-phase-lorenz-mie-rgb", { phase: "lorenz-mie-rgb" }),
    ...step4("step4-frame-phase-g-exponent", { phaseGExponent: 1 }),
    // 'volume' lighting with his volume phase field (demo-1-lighting-volume-phase-g-low): `isolate: true`
    ...step4("step4-lighting-volume-phase-g-low", {}, { lighting: "volume", volumePhase: { phaseG: -0.5 } }),
  ]
    .flat()
    .map((d) => [d.name, d]),
);

// K3.5 (R1 §6 Step 5): his Spectral folder (utils/medium.js): `spectral`, each measured `medium` (which turns spectral on
// and sets the albedo, Ångström exponent and the volume and 'frame' phases), an Ångström exponent alone, and 'volume'
// lighting with a medium. He has no validation JPEGs for them. As Step 4: `*-asis` (his 'environment' light setup; INFO)
// and `*-dir-isolated` (his sun only; the gate). The 'volume' lighting environment scenes (`*-env-isolated`: no scene
// light, his sky-midday map kept, `isolate: 'environment'`) gate the 'volume'-lighting environment terms.
const STEP5_MEDIA = ["cloud", "fog", "ice", "soot", "white-smoke", "dark-smoke", "haze", "dust"];
const step5 = (name: string, extra: Partial<SceneDef>): SceneDef[] => [
  step3(`${name}-asis`, { lightSetup: "environment", multipleScattering: "neural", ...extra }, false, STEP4_WHY_ENV),
  step3(`${name}-dir-isolated`, { multipleScattering: "neural", ...extra }, true),
];
const STEP5_SCENES: Record<string, SceneDef> = Object.fromEntries(
  [
    ...step5("step5-spectral", { spectral: true }),
    ...STEP5_MEDIA.flatMap((m) => step5(`step5-medium-${m}`, { medium: m })),
    ...step5("step5-angstrom-air", { spectral: true, angstromExponent: 4.09 }),
    ...step5("step5-lighting-volume-medium-dark-smoke", { lighting: "volume", medium: "dark-smoke" }),
    step3("step5-lighting-volume-sky-midday-env-isolated", { lighting: "volume", multipleScattering: "neural", lightSetup: "none", environment: "sky-midday" }, true),
    step3("step5-lighting-volume-medium-cloud-sky-midday-env-isolated", { lighting: "volume", multipleScattering: "neural", lightSetup: "none", environment: "sky-midday", medium: "cloud" }, true),
  ].map((d) => [d.name, d]),
);

// K3.6 (R1 §6 Step 6): his demo-1 cast-shadow* and explosion-emission* scenes (capture-reference.mjs). `*-asis`: his
// 'environment' light setup (INFO). `*-dir-isolated`: his sun only, the floor and props kept (`isolate: 'shadow'`): the
// gate, with a shadow-only diff on his ShadowMaterial floor. `*-env-isolated`: the explosion lit by nothing, the props lit
// by its emission light only (`isolate: 'emission'`). The explosion frame comes from fixtures/k3/ground_explosion/ (his
// CC0 EmberGen sequence: copy the folder from his clone); the frame is the capture's `pinFrame` (K3.6b; default 30).
// D33: the package's blackbody colour integrates the analytic CIE fit (Wyman et al. 2013); his integrates the CIE
// 1931 table. The explosion scenes inject the exact table through the volume's emissionColor hook when the test-only
// fixture is present (`node scripts/fetch-cie-cmf.mjs` → fixtures/cie/, CC BY-SA 4.0, never committed); without it
// they render with the fit and do not gate.
const STEP6_WHY_NO_CIE =
  "D33: fixtures/cie/CIE_xyz_1931_2deg.csv absent (node scripts/fetch-cie-cmf.mjs), so the explosion's 'planck' colour uses the package's analytic CIE fit (Wyman et al. 2013) while his uses the CIE 1931 table (up to 0.021 per channel)";
const CIE_CSV = "/fixtures/cie/CIE_xyz_1931_2deg.csv";

/** D33: the exact CIE 1931 2° colour matching functions from the test-only fixture, or null when it is absent. */
async function loadCieTable(): Promise<VDBColorMatchingFunctions | null> {
  try {
    const response = await fetch(CIE_CSV);
    if (!response.ok) return null;
    const rows = new Map<number, VDBFrameVec3>();
    for (const line of (await response.text()).trim().split(/\r?\n/)) {
      const [w, x, y, z] = line.split(",").map(Number);
      if ([w, x, y, z].every((v) => Number.isFinite(v))) rows.set(w!, [x!, y!, z!]);
    }
    if (rows.size !== 471) return null; // 360–830 nm, 1 nm
    return (wavelength) => {
      const row = rows.get(wavelength);
      if (!row) throw new Error(`${CIE_CSV}: no row at ${wavelength} nm`);
      return row;
    };
  } catch {
    return null;
  }
}

const STEP6_WHY_EXPLOSION_ASIS = "his 'environment' light setup (scene lights from the HDR map, the map as scene.environment, the grounded skybox; not rendered here)";
const step6 = (name: string, step: NonNullable<SceneDef["step6"]>, extra: Partial<SceneDef> = {}): SceneDef[] => [
  step3(`${name}-asis`, { lightSetup: "environment", multipleScattering: "neural", step6: step, ...extra }, false, step.explosion ? STEP6_WHY_EXPLOSION_ASIS : STEP3_WHY_ENV),
  step3(step.explosion ? `${name}-env-isolated` : `${name}-dir-isolated`, { lightSetup: step.explosion ? "none" : "dir", multipleScattering: "neural", step6: step.explosion ? { ...step, showObstacle: true } : step, ...extra }, true),
];
const STEP6_SCENES: Record<string, SceneDef> = Object.fromEntries(
  [
    ...step6("step6-cast-shadow", { castShadow: true }),
    ...step6("step6-cast-shadow-obstacle", { castShadow: true, showObstacle: true }),
    ...step6("step6-cast-shadow-obstacle-cascades-4", { castShadow: true, showObstacle: true, castShadowCascades: 4 }),
    ...step6("step6-cast-shadow-obstacle-coarse", { castShadow: true, showObstacle: true, castShadowCoarse: true }),
    ...step6("step6-cast-shadow-two-bunnies", { castShadow: true }, { twoBunnies: true }),
    ...step6("step6-explosion-emission", { castShadow: true, explosion: true, emissionResolution: 32 }),
    ...step6("step6-explosion-emission-cascades-4", { castShadow: true, explosion: true, emissionResolution: 32, emissionCascades: 4 }),
    ...step6("step6-explosion-emission-probes-8", { castShadow: true, explosion: true, emissionResolution: 8 }),
    // The shadow-only diff on his ShadowMaterial floor: the volume proxies hidden for the captured frame (isolate 'shadow-only')
    step3("step6-cast-shadow-shadow-only-dir-isolated", { multipleScattering: "neural", step6: { castShadow: true, shadowOnly: true } }, true),
    step3("step6-cast-shadow-obstacle-shadow-only-dir-isolated", { multipleScattering: "neural", step6: { castShadow: true, showObstacle: true, shadowOnly: true } }, true),
    step3("step6-cast-shadow-obstacle-cascades-4-shadow-only-dir-isolated", { multipleScattering: "neural", step6: { castShadow: true, showObstacle: true, castShadowCascades: 4, shadowOnly: true } }, true),
  ].map((d) => [d.name, d]),
);

// His frame.js scene list (capture-reference.mjs SCENARIOS). `multipleScattering` 'neural' (his default) is the
// same as true with 'volume' lighting (the network runs only in 'frame' lighting).
const HIS_SCENES: Record<string, SceneDef> = {
  // The Step 1 parity scenes: his page with `isolate` (capture-reference.mjs): directional light only, no environment
  // light on the volume, no skybox/ground/props, background 0x598eff.
  "step1-volume-dir-ms-off-isolated": { name: "step1-volume-dir-ms-off-isolated", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "dir", lighting: "volume", multipleScattering: false, parityExpected: true },
  "step1-volume-dir-ms-on-isolated": { name: "step1-volume-dir-ms-on-isolated", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "dir", lighting: "volume", multipleScattering: true, parityExpected: true },
  "step1-volume-dir-ms-off": {
    name: "step1-volume-dir-ms-off",
    file: "/fixtures/k3/bunny_cloud.wvdb",
    lightSetup: "dir",
    lighting: "volume",
    multipleScattering: false,
    parityExpected: false,
    why: "his 'dir' setup keeps scene.environment, which lights the volume too (ported in K3.5; this harness scene does not load his map), and shows the skybox and ground",
  },
  "step1-volume-dir-ms-on": {
    name: "step1-volume-dir-ms-on",
    file: "/fixtures/k3/bunny_cloud.wvdb",
    lightSetup: "dir",
    lighting: "volume",
    multipleScattering: true,
    parityExpected: false,
    why: "his 'dir' setup keeps scene.environment, which lights the volume too (ported in K3.5; this harness scene does not load his map), and shows the skybox and ground",
  },
  "step1-lighting-volume": {
    name: "step1-lighting-volume",
    file: "/fixtures/k3/bunny_cloud.wvdb",
    lightSetup: "environment",
    lighting: "volume",
    multipleScattering: true,
    parityExpected: false,
    why: "his 'environment' light setup: scene lights extracted from the HDR map, the map as scene.environment and the grounded skybox; rendered unlit here (the 'volume' environment terms gate in step5-lighting-volume-sky-midday-env-isolated)",
  },
  "step1-lights-directional": { name: "step1-lights-directional", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "dir", lighting: "frame", multipleScattering: true, parityExpected: false, why: "'frame' lighting (K3.3); rendered with 'volume' lighting" },
  "step1-multiple-scattering-off": { name: "step1-multiple-scattering-off", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "environment", lighting: "frame", multipleScattering: false, parityExpected: false, why: "'frame' lighting + environment (K3.3/K3.4)" },
  "step0-bunny": { name: "step0-bunny", file: "/fixtures/k3/bunny_cloud.wvdb", lightSetup: "environment", lighting: "frame", multipleScattering: true, parityExpected: false, why: "'frame' lighting + environment (K3.3/K3.4)" },
  "step0-smoke": { name: "step0-smoke", file: "/fixtures/k3/smoke.wvdb", lightSetup: "environment", lighting: "frame", multipleScattering: true, parityExpected: false, why: "'frame' lighting + environment (K3.3/K3.4)" },
  ...STEP3_SCENES,
  ...STEP4_SCENES,
  ...STEP5_SCENES,
  ...STEP6_SCENES,
};

interface ChunkProbe {
  count: number;
  maxAbs: number;
  maxRel: number;
  over: number;
  firstBad?: { row: number; x: number; channel: number; gpu: number; cpu: number };
}

interface TwinCheck {
  pixels: number;
  covered: number;
  maxAbs: number;
  over: number;
  firstBad?: { x: number; y: number; gpu: number[]; cpu: number[] };
}

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  scene?: string;
  parityExpected?: boolean;
  parityNote?: string;
  source?: "manifest" | "frame.js";
  file?: string;
  volume?: { dimensions: number[]; lod: number; meanDensity: number; mipLevels: number; statsSource: string };
  width?: number;
  height?: number;
  froxels?: number[];
  /** K3.6b: MSAA samples of the render target (his page: 4). */
  msaa?: number;
  /** K3.6b: the volume's densityScale (explosion: frame mean / his sequence mean). */
  densityScale?: number;
  /** K3.6b: the explosion frame loaded (the capture's pinned frame). */
  explosionFrame?: number | null;
  /** D33: the colour matching functions of the explosion's 'planck' hook (the CIE table when fixtures/cie has it). */
  emissionCmf?: "cie-1931-table" | "wyman-2013-fit";
  rgba?: string;
  /** Screen rectangle of the volume box in pixels (top-left origin): x0, y0, x1, y1. */
  volumeRect?: number[];
  inPageIdentical?: boolean;
  nonBackground?: number;
  chunkProbe?: ChunkProbe;
  twin?: TwinCheck;
  timings?: { provisional: true; note: string; sceneMs: number | null; froxelsMs: number | null; fullMs: number | null; integrateMs: number | null; passes?: Record<string, number | null> };
  warnings: string[];
  uncaptured: string[];
  // K3.3 ('frame' lighting)
  lighting?: "volume" | "frame";
  sourceKind?: "dense" | "sparse";
  settings?: Record<string, unknown>;
  grid?: { count: number; resolution: number; gridMin: number[][]; gridVoxel: number[][] };
  gridProbe?: TwinStats;
  transmittanceTwin?: TwinStats & { marched: number };
  lightingTwin?: TwinStats & { lit: number };
  environmentGridTwin?: TwinStats & { marched: number };
  densityTwin?: TwinStats & { covered: number };
  environment?: { mode: string; count: number; directions: number[][]; colors: number[][]; frames: number } | null;
  redraw?: VDBFrameRedrawStats;
  // K3.4
  firstFrameMs?: number;
  scatterProbe?: TwinStats;
  gridMipTwin?: TwinStats & { levels: number };
  furnace?: { radiance: number; centerRatio: number[]; corner: number[]; volumePixels: number; meanRatio: number; minRatio: number; maxRatio: number; meanAbsDeviation: number; p05: number; p95: number; environment: { mode: string; count: number } | null };
  temporalCheck?: { frames: number; diffsOff: number[]; diffsOn: number[]; psnrSingleVsMean: number; psnrTemporalVsMean: number; volumePixels: number };
  // K6.1
  backend?: "tsl" | "core";
  froxelMix?: "mixture" | "over";
  // K3.5
  spectral?: boolean;
  medium?: string | null;
  spectralProbe?: TwinStats;
  spectralGroups?: number;
}

declare global {
  interface Window {
    __VDB_K3_FRAME__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const SCENE = params.get("scene") ?? "smoke";
const def: SceneDef =
  SCENE === "smoke"
    ? { name: "smoke", file: cloudWvdbUrl, lightSetup: "dir", lighting: "volume", multipleScattering: true, parityExpected: false, why: "our own test scene", smoke: true }
    : HIS_SCENES[SCENE]!;
const W = Number(params.get("w") ?? (def?.smoke ? 320 : 1280));
const H = Number(params.get("h") ?? (def?.smoke ? 180 : 720));
const FROXELS = (params.get("froxels") ?? (def?.smoke ? "128x128x24" : "512x512x24")).split("x").map(Number);
const SUBSTEPS = Number(params.get("substeps") ?? 32);
const SHADOW_STEPS = Number(params.get("shadowSteps") ?? 32);
const MIPS = params.get("mips") !== "0";
const TIMING = params.get("timing") === "1";
const TWIN = params.get("twin") === "1";
const OCCLUDER = params.get("occluder") === "1";
const NOVOLUME = params.get("novolume") === "1";
const MANIFEST = params.get("manifest") ?? "/fixtures/k3-reference/75e7345/manifest.json";
// K3.3
const LIGHTING = (params.get("lighting") ?? def?.lighting ?? "volume") as "volume" | "frame";
const SOURCE_KIND = (params.get("source") ?? "dense") as "dense" | "sparse";
const CASCADES = Number(params.get("cascades") ?? def?.shadowCascades ?? 1);
const SHADOW_RES = Number(params.get("shadowRes") ?? def?.shadowResolution ?? 64);
const SHADOW_DISTANCE = Number(params.get("shadowDistance") ?? 512);
const TWO = params.has("second") ? params.get("second") === "1" : !!def?.twoBunnies;
const FURNACE = params.get("furnace") === "1";
const DEFAULTS = params.get("defaults") === "1";
const FURNACE_RADIANCE = 0.5;
const ENV =params.get("env") ?? (FURNACE ? "furnace" : def?.environment ? `/fixtures/k3/${def.environment}-HDR.jpg` : null);
const ENV_INTENSITY = Number(params.get("envIntensity") ?? (ENV === "synthetic" || ENV === "furnace" ? 1.0 : 0.4));
const ENV_MODE = (params.get("envMode") ?? "directional") as "outward" | "directional" | "spherical-harmonics";
const TWINS = params.get("twins") === "1";
const TEMPORAL_CHECK = Number(params.get("temporalCheck") ?? 0);
const REDRAW_FRAMES = Number(params.get("redraw") ?? 0);
const LIGHT_SETUP = (params.get("lights") ?? (FURNACE ? "none" : def?.lightSetup) ?? "dir") as SceneDef["lightSetup"] | "dir-hemi";
// K3.5: his Spectral folder (a medium turns spectral on, as his GUI)
const MEDIUM = params.get("medium") ?? def?.medium ?? null;
const MEDIUM_PROPS = MEDIUM ? vdbFrameMedium(MEDIUM) : null;
const SPECTRAL = params.has("spectral") ? params.get("spectral") === "1" : !!(def?.spectral || MEDIUM_PROPS);
// K3.6
const SHADOW_ONLY = !!def?.step6?.shadowOnly;
// K3.6b: his page's output path. main.jsx makes `new WebGPURenderer({ antialias: !isPhone })` (desktop: 4 samples) with
// outputColorSpace sRGB, so three renders the scene into its frame-buffer target (Renderer._getFrameBufferTarget: HalfFloat,
// `samples: renderer.samples` = 4) and the output pass resolves + sRGB-encodes it to the canvas. His volume proxy draws in
// that MSAA pass (its viewport colour / depth copies read the resolved target), so mesh edges are 4x MSAA. His scenes get 4
// samples here too (`?msaa=0|4` overrides); our own test scenes (smoke, twins) stay at 0.
const MSAA = Number(params.get("msaa") ?? (def?.smoke ? 0 : 4));
const ANGSTROM = params.has("angstrom") ? Number(params.get("angstrom")) : (def?.angstromExponent ?? null);
// K3.4: ?ms=0|1|two-sided|octaves|neural
const parseMs = (v: string): VDBFrameMultipleScattering => (v === "0" || v === "false" ? false : v === "1" || v === "true" ? true : (v as VDBFrameMultipleScattering));
const MS: VDBFrameMultipleScattering = params.has("ms") ? parseMs(params.get("ms")!) : (def?.multipleScattering ?? true);
const num = (k: string): number | undefined => (params.has(k) ? Number(params.get(k)) : undefined);
const PHASE_OVERRIDES: Partial<VDBFrameRendererSettings> = Object.fromEntries(
  (
    [
      ["phase", params.get("phase") as VDBFramePhase | null],
      ["phaseG", num("phaseG")],
      ["phaseBackG", num("phaseBackG")],
      ["phaseBackWeight", num("phaseBackWeight")],
      ["phaseGExponent", num("phaseGExponent")],
      ["scatteringOctaves", num("octaves")],
      ["scatteringOctaveExtinction", num("octaveExtinction")],
      ["scatteringOctaveContribution", num("octaveContribution")],
      ["scatteringOctavePhase", num("octavePhase")],
    ] as [string, unknown][]
  ).filter(([, v]) => v !== null && v !== undefined),
);
// His examples page: scene.background 0x598eff until the environment loads (main.jsx L41).
const BACKGROUND = new THREE.Color(0x598eff);

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, warnings: [], uncaptured: [] };
window.__VDB_K3_FRAME__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/**
 * Readback of a render target as tight rows. three r187 WebGPU returns the rows top first (row 0 is
 * `screenCoordinate.y` 0), padded to 256 bytes; the chunk probe below checks the orientation.
 */
function tightRows<T extends Uint16Array | Float32Array | Uint8Array>(pixels: T, w: number, h: number, channels: number): T {
  const tight = w * channels;
  const stride = h > 1 ? (pixels.length - tight) / (h - 1) : tight;
  if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${pixels.length} elements for ${w}x${h}`);
  const out = new (pixels.constructor as new (n: number) => T)(w * h * channels);
  for (let y = 0; y < h; y++) out.set(pixels.subarray(y * stride, y * stride + tight) as never, y * tight);
  return out;
}

async function readHalf(renderer: WebGPURenderer, rt: THREE.RenderTarget, w: number, h: number): Promise<Float32Array> {
  const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h)) as unknown as Uint16Array;
  const rows = tightRows(raw, w, h, 4);
  const out = new Float32Array(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = THREE.DataUtils.fromHalfFloat(rows[i]!);
  return out;
}

const toByte = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);

/** Linear (top row first) → display RGBA8 as his canvas output pass: NoToneMapping, sRGB OETF, 8-bit. */
function toDisplay(linear: Float32Array): Uint8Array {
  const out = new Uint8Array(linear.length);
  for (let i = 0; i < linear.length; i += 4) {
    out[i] = toByte(vdbSrgbOetf(linear[i]!));
    out[i + 1] = toByte(vdbSrgbOetf(linear[i + 1]!));
    out[i + 2] = toByte(vdbSrgbOetf(linear[i + 2]!));
    out[i + 3] = 255;
  }
  return out;
}

interface ManifestScenario {
  name: string;
  /** K3.6b: the explosion frame the capture paused his sequence at. */
  pinFrame?: number;
  metadata?: {
    camera?: { matrixWorld: number[]; projectionMatrix: number[]; near: number; far: number; fov: number; aspect: number };
    volumes?: { matrixWorld: number[]; visible?: boolean; densityScale?: number; meanDensity?: number }[];
    /** K3.3 captures: scene.environment intensity and rotation. */
    environment?: { intensity: number; rotation: [number, number, number, string] } | null;
    /** K3.6 captures: the meshes outside the volumes, the lights' shadows and the emission light, his animation frame. */
    meshes?: CapturedMesh[];
    lights?: { type: string; intensity: number; color?: number[]; matrixWorld: number[]; castShadow?: boolean; shadow?: { mapSize: number[]; bias: number; normalBias: number; radius: number; camera: { left?: number; right?: number; top?: number; bottom?: number; near: number; far: number } }; resolution?: number; range?: number; cascades?: number; distance?: number; steps?: number }[];
    animations?: { frame: number; playing?: boolean; visibleFrames?: number[] }[];
    shadowMap?: { enabled: boolean; type: number };
  };
}

interface CapturedMesh {
  geometry: { type: string; parameters: Record<string, number> | null };
  matrixWorld: number[];
  castShadow: boolean;
  receiveShadow: boolean;
  material: { type: string; color: number[] | null; opacity: number; transparent: boolean; roughness: number | null; metalness: number | null };
}

/** K3.6: a mesh of his scene from the capture (floor, obstacle sphere, pillars, spheres). */
function meshFromCapture(m: CapturedMesh): THREE.Mesh {
  const g = m.geometry.parameters ?? {};
  const geometry =
    m.geometry.type === "PlaneGeometry"
      ? new THREE.PlaneGeometry(g.width, g.height, g.widthSegments, g.heightSegments)
      : m.geometry.type === "BoxGeometry"
        ? new THREE.BoxGeometry(g.width, g.height, g.depth, g.widthSegments, g.heightSegments, g.depthSegments)
        : m.geometry.type === "SphereGeometry"
          ? new THREE.SphereGeometry(g.radius, g.widthSegments, g.heightSegments, g.phiStart, g.phiLength, g.thetaStart, g.thetaLength)
          : null;
  if (!geometry) throw new Error(`K3.6 harness: captured geometry ${m.geometry.type} is not rebuilt`);
  const color = m.material.color ? new THREE.Color().fromArray(m.material.color) : new THREE.Color(0x000000);
  const material =
    m.material.type === "ShadowMaterial"
      ? new THREE.ShadowMaterial({ color, opacity: m.material.opacity })
      : m.material.type === "MeshStandardMaterial"
        ? new THREE.MeshStandardMaterial({ color, roughness: m.material.roughness ?? 1, metalness: m.material.metalness ?? 0, opacity: m.material.opacity, transparent: m.material.transparent })
        : null;
  if (!material) throw new Error(`K3.6 harness: captured material ${m.material.type} is not rebuilt`);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.matrixAutoUpdate = false;
  mesh.matrix.fromArray(m.matrixWorld);
  mesh.matrixWorld.copy(mesh.matrix);
  mesh.castShadow = m.castShadow;
  mesh.receiveShadow = m.receiveShadow;
  return mesh;
}

/**
 * K3.6: his Step 6 scene around the volumes: castShadows and its settings, the floor / obstacle / props (from the
 * capture, else his frame.js and utils/emission.js values), the sun's shadow (his mapSize 2048, normalBias 0.5 and
 * fitSunShadow camera) and the explosion's emission light (intensity 25, his probes and cascades).
 */
function setupStep6(o: { scene: THREE.Scene; renderer: WebGPURenderer; frameRenderer: VDBFrameRenderer; sun: THREE.DirectionalLight | null; manifest: ManifestScenario | null; worldSize: THREE.Vector3; zUp: boolean }): void {
  const s6 = def.step6!;
  const { scene, renderer, frameRenderer, sun, manifest } = o;
  const captured = manifest?.metadata;
  const groundLevel = -50.0;
  renderer.shadowMap.enabled = captured?.shadowMap?.enabled ?? true;
  if (captured?.shadowMap?.type !== undefined) renderer.shadowMap.type = captured.shadowMap.type as THREE.ShadowMapType;
  Object.assign(frameRenderer.settings, { castShadows: !!s6.castShadow, castShadowCascades: s6.castShadowCascades ?? 1, castShadowCoarse: !!s6.castShadowCoarse });

  // utils/emission.js placeMeshes: the stage scale is the box width (x and the horizontal of y / z)
  const width = Math.max(o.worldSize.x, o.zUp ? o.worldSize.y : o.worldSize.z);
  const floorBox = new THREE.Box3();
  if (captured?.meshes?.length) {
    for (const m of captured.meshes) {
      const mesh = meshFromCapture(m);
      scene.add(mesh);
      if (m.material.type === "ShadowMaterial") floorBox.expandByObject(mesh);
    }
  } else if (s6.castShadow) {
    // utils/emission.js: the floor (ShadowMaterial 0.5, 8 × 8 box widths) at groundLevel, scaled by the box width; frame.js:
    // the obstacle sphere (r 20 at (0, 0, 40)) and the pillars / spheres on a 0.67-width circle. Cast Shadows makes all cast and receive.
    const stage = new THREE.Group();
    stage.position.y = groundLevel;
    stage.scale.setScalar(width);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(8.0, 8.0), new THREE.ShadowMaterial({ opacity: 0.5 }));
    floor.rotation.x = -Math.PI / 2.0;
    stage.add(floor);
    if (s6.showObstacle) {
      const material = new THREE.MeshStandardMaterial({ color: 0x9a9a9a, roughness: 0.8 });
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2.0;
        const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.5, 0.12), material);
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.15, 32, 16), material);
        pillar.position.set(Math.cos(angle) * 0.67, 0.25, Math.sin(angle) * 0.67);
        sphere.position.set(Math.cos(angle + Math.PI / 8.0) * 0.67, 0.15, Math.sin(angle + Math.PI / 8.0) * 0.67);
        stage.add(pillar, sphere);
      }
      const obstacle = new THREE.Mesh(new THREE.SphereGeometry(20.0, 32, 32), new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.0, roughness: 0.5 }));
      obstacle.position.set(0.0, 0.0, 40.0);
      obstacle.castShadow = obstacle.receiveShadow = true;
      scene.add(obstacle);
    }
    stage.traverse((m) => {
      if ((m as THREE.Mesh).isMesh) m.castShadow = m.receiveShadow = true;
    });
    scene.add(stage);
    stage.updateMatrixWorld(true);
    floorBox.setFromObject(floor);
  }

  if (sun && s6.castShadow) {
    const cl = captured?.lights?.find((l) => l.type === "DirectionalLight" && l.shadow);
    sun.castShadow = true;
    sun.shadow.mapSize.setScalar(cl?.shadow?.mapSize[0] ?? 2048);
    sun.shadow.normalBias = cl?.shadow?.normalBias ?? 0.5;
    sun.shadow.bias = cl?.shadow?.bias ?? 0;
    const cam = sun.shadow.camera;
    if (cl?.shadow?.camera.left !== undefined) {
      Object.assign(cam, { left: cl.shadow.camera.left, right: cl.shadow.camera.right, top: cl.shadow.camera.top, bottom: cl.shadow.camera.bottom, near: cl.shadow.camera.near, far: cl.shadow.camera.far });
    } else {
      // frame.js fitSunShadow: the floor's box seen from the sun, near −1000, far 1000
      sun.updateMatrixWorld();
      sun.shadow.updateMatrices(sun);
      const view = new THREE.Box3();
      for (let i = 0; i < 8; i++) {
        view.expandByPoint(new THREE.Vector3(i & 1 ? floorBox.max.x : floorBox.min.x, i & 2 ? floorBox.max.y : floorBox.min.y, i & 4 ? floorBox.max.z : floorBox.min.z).applyMatrix4(cam.matrixWorldInverse));
      }
      Object.assign(cam, { left: view.min.x, right: view.max.x, top: view.max.y, bottom: view.min.y, near: -1000.0, far: 1000.0 });
    }
    cam.updateProjectionMatrix();
  }

  if (s6.explosion) {
    const cl = captured?.lights?.find((l) => l.type === "FrameEmissionLight");
    const light = new VDBFrameEmissionLight({
      intensity: cl?.intensity ?? 25,
      resolution: cl?.resolution ?? s6.emissionResolution ?? 32,
      cascades: cl?.cascades ?? s6.emissionCascades ?? 1,
      range: cl?.range ?? 3,
      distance: cl?.distance ?? 512,
      steps: cl?.steps ?? 24,
    });
    if (cl?.color) light.color.fromArray(cl.color);
    // K6.5: `?backend=core` takes the probes from the core renderer's emission kernels
    light.backend = frameRenderer.backend;
    scene.add(light);
  }
}

async function loadManifest(name: string): Promise<ManifestScenario | null> {
  try {
    const r = await fetch(MANIFEST);
    if (!r.ok) return null;
    const m = (await r.json()) as { scenarios?: ManifestScenario[] };
    return m.scenarios?.find((s) => s.name === name) ?? null;
  } catch {
    return null;
  }
}

async function main(): Promise<void> {
  root.textContent = "";
  if (!def) throw new Error(`unknown ?scene=${SCENE}; known: smoke, ${Object.keys(HIS_SCENES).join(", ")}`);
  result.scene = def.name;
  result.parityExpected = def.parityExpected;
  if (def.why) result.parityNote = def.why;
  const warn = console.warn.bind(console);
  console.warn = (...a: unknown[]) => {
    result.warnings.push(a.map(String).join(" "));
    warn(...a);
  };

  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter");
  const info = adapter.info;
  result.adapter = [info?.vendor, info?.architecture, info?.description].filter(Boolean).join(" ");
  const features: GPUFeatureName[] = TIMING && adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures: features, ...(SOURCE_KIND === "sparse" ? { requiredLimits: VDBGridPool.requiredLimits(adapter) } : {}) });
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  // --- the dense volume (his bytes) ----------------------------------------------------------
  // K3.6: the explosion's frame. K3.6b: the capture pins his sequence (paused at `pinFrame`, capture-reference.mjs
  // EXPLOSION_PIN_FRAME; the manifest's animation frame is checked equal and not playing); an older capture's recorded
  // frame was read after the screenshot of a playing sequence (not the drawn one), so it is refused.
  const manifest = def.smoke ? null : await loadManifest(def.name);
  let explosionFrame: number | null = null;
  if (def.step6?.explosion) {
    const anim = manifest?.metadata?.animations?.[0];
    if (params.has("explosionFrame")) explosionFrame = Number(params.get("explosionFrame"));
    else if (manifest) {
      if (manifest.pinFrame === undefined || anim?.frame !== manifest.pinFrame || anim.playing !== false) {
        throw new Error(`${def.name}: the capture did not pin the explosion frame (pinFrame ${manifest.pinFrame}, animation ${JSON.stringify(anim ? { frame: anim.frame, playing: anim.playing } : null)}); recapture it with scripts/k3/capture-reference.mjs --merge`);
      }
      explosionFrame = manifest.pinFrame;
    } else explosionFrame = 30;
  }
  result.explosionFrame = explosionFrame;
  // D33: his 'planck' colours come from the CIE 1931 table; inject it through the hook when the fixture is present
  let explosionColor: ReturnType<typeof vdbFrameBlackbodyEmission> | undefined;
  if (explosionFrame !== null) {
    const cie = await loadCieTable();
    result.emissionCmf = cie ? "cie-1931-table" : "wyman-2013-fit";
    explosionColor = vdbFrameBlackbodyEmission("planck", 1.39, cie ? { cmf: cie } : {});
    if (!cie && result.parityExpected) {
      result.parityExpected = false;
      result.parityNote = STEP6_WHY_NO_CIE;
    }
  }
  const file = params.get("file") ?? (explosionFrame !== null ? `/fixtures/k3/ground_explosion/ground_explosion_${String(explosionFrame).padStart(4, "0")}.wvdb` : def.file);
  const response = await fetch(file);
  if (!response.ok) throw new Error(`fetch ${file}: ${response.status}${def.smoke ? "" : " (his .wvdb files are not committed: copy them from his clone into fixtures/k3/)"}`);
  const wvdb = readWvdb(new Uint8Array(await response.arrayBuffer()));
  // K3.6 explosion: his emissiveGrid "flames" ÷ emissiveScale 6 (frame.js L25–47)
  const dense = wvdbDenseVolume(wvdb, explosionFrame !== null ? { mipmaps: MIPS, emission: "flames", scales: { flames: 6.0 } } : { mipmaps: MIPS });
  result.file = file;
  result.volume = {
    dimensions: [...dense.dimensions],
    lod: dense.lod,
    meanDensity: dense.meanDensity,
    mipLevels: dense.density.mip?.levels ?? 0,
    statsSource: dense.densityStatsSource,
  };

  // --- renderer and scene -------------------------------------------------------------------
  const renderer = new WebGPURenderer({ device, trackTimestamp: TIMING && features.length > 0 });
  await renderer.init();
  // NOTE The froxel grid is capped by the drawing buffer size (layout.ts getFroxelSize), as on his canvas.
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);

  const frameRenderer = new VDBFrameRenderer();
  // K6.1: `?backend=core` renders through the core frame renderer (internal A/B); `?froxelMix=mixture|over`
  frameRenderer.backend = params.get("backend") === "core" ? "core" : "tsl";
  frameRenderer.froxelMix = params.get("froxelMix") === "over" ? "over" : "mixture";
  // K6.3 `?msEdge=zero|clamp`, `?msLevels=`: the core's 'multi-scale' settings (with `?ms=multi-scale`, backend core)
  if (params.has("msEdge")) frameRenderer.coreSettings.multiScaleEdge = params.get("msEdge") === "clamp" ? "clamp" : "zero";
  if (params.has("msLevels")) frameRenderer.coreSettings.multiScaleLevels = Number(params.get("msLevels"));
  // K6.3 `?occlusion=0`: no mesh → volume shadows in the core (K6.5b; his renderer has none) for core vs TSL A/Bs
  if (params.get("occlusion") === "0") frameRenderer.occlusion = false;
  result.backend = frameRenderer.backend;
  result.froxelMix = frameRenderer.froxelMix;
  Object.assign(frameRenderer.settings, {
    width: FROXELS[0],
    height: FROXELS[1],
    depth: FROXELS[2],
    subSteps: SUBSTEPS,
    jitter: params.get("jitter") === "1",
    temporal: params.get("temporal") === "1",
    lighting: LIGHTING,
    multipleScattering: MS,
    // K3.3: his 'frame' settings for the scene; his demos set the renderer phase to 2HG (utils/phase.js).
    shadowCascades: CASCADES,
    shadowResolution: SHADOW_RES,
    shadowDistance: SHADOW_DISTANCE,
    phase: "dual-lobe",
    phaseG: 0.8,
    phaseBackG: -0.3,
    phaseBackWeight: 0.2,
    phaseGExponent: 0.0,
    environmentLighting: ENV_MODE,
    // K3.4: the scene's GUI fields, then the query
    ...(def.settings ?? {}),
    ...PHASE_OVERRIDES,
  });
  if (params.has("ms")) frameRenderer.settings.multipleScattering = MS;
  // K3.5: his medium onChange also sets the 'frame' phase fields (frame-phase, frame-phaseG, frame-phaseGExponent)
  frameRenderer.settings.spectral = SPECTRAL;
  if (MEDIUM_PROPS) {
    if (!params.has("phase")) frameRenderer.settings.phase = MEDIUM_PROPS.phase;
    if (!params.has("phaseG")) frameRenderer.settings.phaseG = MEDIUM_PROPS.phaseG;
    if (!params.has("phaseGExponent")) frameRenderer.settings.phaseGExponent = MEDIUM_PROPS.phaseGExponent;
  }
  if (MEDIUM && !MEDIUM_PROPS) throw new Error(`unknown ?medium=${MEDIUM}`);
  result.spectral = SPECTRAL;
  result.medium = MEDIUM;
  // K3.4 `?defaults=1`: the renderer's own lighting / temporal / multipleScattering defaults (his set since K3.4)
  if (DEFAULTS) {
    const fresh = new VDBFrameRenderer().settings;
    frameRenderer.settings.lighting = fresh.lighting;
    frameRenderer.settings.temporal = fresh.temporal;
    frameRenderer.settings.multipleScattering = fresh.multipleScattering;
  }
  result.froxels = [...FROXELS];
  result.lighting = LIGHTING;
  result.sourceKind = SOURCE_KIND;
  result.settings = { ...frameRenderer.settings };

  // Dense: his bytes. Sparse: the same texels as a NanoVDB FLOAT grid (wvdbDenseToNvdb, voxel-exact), the K3.2
  // default LOD ('coarse-pyramid', D28), normalized by the dense mean so only the sampler differs (K3.2 A/B).
  let source: VDBThreeVolumeSource;
  if (SOURCE_KIND === "sparse") {
    const grid = wvdbDenseToNvdb(dense);
    const pool = await VDBGridPool.create(device, { label: "k3-frame" });
    const id = pool.add(grid.image);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, { profile: "march", helpers: ["cellBoundTrilinear", "hddaCellEnd", "fastTrilinear"] });
    source = vdbSparseSource(grids, { density: id }, {
      lod: (params.get("lod") ?? "coarse-pyramid") as "coarse-pyramid",
      images: { density: grid.image },
      indexBox: grid.textureIndexBox,
      stats: { meanNonZero: dense.meanDensity, max: dense.maxDensity, activeVoxels: dense.activeTexels },
      ...(dense.bbox ? { bbox: dense.bbox as [[number, number, number], [number, number, number]] } : {}),
    });
  } else {
    source = vdbDenseTextures(dense);
  }
  const bbox = new THREE.Box3(new THREE.Vector3(...(dense.bbox?.[0] ?? [-0.5, -0.5, -0.5])), new THREE.Vector3(...(dense.bbox?.[1] ?? [0.5, 0.5, 0.5])));
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());

  // K3.6b: his explosion is a FrameVolumeAnimation, and normalizeDensity divides by the SEQUENCE's mean density
  // (FrameVolume.getDensityScale: `(owner.sequence ?? owner).meanDensity`, the mean of all 200 frames' non-zero texels),
  // not the frame's own (ours: the source's). The capture records his getDensityScale() of the shown frame (densityScale
  // 1 / sequence mean); densityScale = frame mean × that makes ours divide by the same mean. Without a capture: 1.
  const capturedVolume = manifest?.metadata?.volumes?.find((v) => v.visible !== false);
  const explosionDensityScale = explosionFrame !== null && capturedVolume?.densityScale ? capturedVolume.densityScale * dense.meanDensity : 1.0;
  result.densityScale = explosionDensityScale;

  // His frame.js demo params (L93–116, utils/phase.js phaseParams): dual-lobe 0.8 / -0.3 / 0.2, extinction 100.
  // K3.4: his volume phase fields (`def.volumePhase`), and `?phase*` for 'volume' lighting.
  const volumePhase = { phaseG: 0.8, phaseBackG: -0.3, phaseBackWeight: 0.2, phaseGExponent: 0.0, ...(def.volumePhase ?? {}) };
  if (MEDIUM_PROPS) Object.assign(volumePhase, { phaseG: MEDIUM_PROPS.phaseG, phaseGExponent: MEDIUM_PROPS.phaseGExponent });
  for (const k of ["phaseG", "phaseBackG", "phaseBackWeight", "phaseGExponent"] as const) if (PHASE_OVERRIDES[k] !== undefined) volumePhase[k] = PHASE_OVERRIDES[k];
  const volume = new VDBFrameVolume(source, {
    frameRenderer,
    bbox,
    baseColor: MEDIUM_PROPS ? MEDIUM_PROPS.baseColor : "#ffffff",
    // His frame.js default angstromExponent 0 (grey); a medium sets its own
    angstromExponent: MEDIUM_PROPS ? MEDIUM_PROPS.angstromExponent : (ANGSTROM ?? 0.0),
    absorbance: 1.0,
    densityScale: explosionDensityScale,
    normalizeDensity: true,
    opacity: 1.0,
    extinction: num("extinction") ?? (def.smoke ? 40 : 100),
    phase: (PHASE_OVERRIDES.phase as VDBFramePhase | undefined) ?? MEDIUM_PROPS?.phase ?? "dual-lobe",
    ...volumePhase,
    shadowSteps: SHADOW_STEPS,
    mipmaps: MIPS,
    // K3.6 explosion (frame.js L25–47, his 'planck' values): base colour #555555, 0–3000 K, emissive intensity 15
    ...(explosionFrame !== null ? { baseColor: "#555555", temperatureRange: [0, 3000] as [number, number], emissiveIntensity: 15, emissionColor: explosionColor } : {}),
  });

  const scene = new THREE.Scene();
  scene.background = BACKGROUND.clone();
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 10000.0);
  // (manifest: loaded above)

  if (manifest?.metadata?.camera && manifest.metadata.volumes?.[0]) {
    result.source = "manifest";
    const c = manifest.metadata.camera;
    camera.matrixAutoUpdate = false;
    camera.matrixWorld.fromArray(c.matrixWorld);
    camera.matrix.copy(camera.matrixWorld);
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    camera.near = c.near;
    camera.far = c.far;
    camera.projectionMatrix.fromArray(c.projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    volume.matrixAutoUpdate = false;
    // K3.6: his explosion sequence has one FrameVolume per frame; the shown one is the visible one
    volume.matrix.fromArray((manifest.metadata.volumes.find((v) => v.visible !== false) ?? manifest.metadata.volumes[0]).matrixWorld);
    volume.matrixWorld.copy(volume.matrix);
  } else if (def.smoke) {
    result.source = "frame.js";
    const diag = size.length();
    camera.near = diag * 0.01;
    camera.far = diag * 20;
    camera.position.copy(center).add(new THREE.Vector3(0.9, 0.35, 1.1).normalize().multiplyScalar(diag * Number(params.get("dist") ?? 0.75)));
    camera.lookAt(center);
    camera.updateProjectionMatrix();
  } else {
    // His frame.js: the volume box is centred on the stage at groundLevel + height / 2 (L141–165, L351–364, L453–457);
    // main.jsx: camera (-100, -11, 80) looking at the orbit target (origin), fov 60, near 0.01, far 10000.
    result.source = "frame.js";
    const groundLevel = -50.0;
    const isSmoke = def.file.includes("smoke");
    const scale = isSmoke ? 100.0 / Math.max(size.x, size.y, size.z) : 1.0;
    volume.position.copy(center).negate().multiplyScalar(scale);
    volume.scale.setScalar(scale);
    volume.position.y += groundLevel + (size.y * scale) / 2.0;
    if (explosionFrame !== null) {
      // The explosion (frame.js L384–425, L141–165): Z-up (−90° about x), 100 units, the box centred on the stage at
      // groundLevel + height / 2 (height: the z size)
      const s = 100.0 / Math.max(size.x, size.y, size.z);
      volume.matrixAutoUpdate = false;
      volume.matrix
        .makeTranslation(0, groundLevel + (size.z * s) / 2.0, 0)
        .multiply(new THREE.Matrix4().makeRotationX(-Math.PI / 2.0))
        .multiply(new THREE.Matrix4().makeScale(s, s, s))
        .multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
      volume.matrixWorld.copy(volume.matrix);
    }
    camera.position.set(-100.0, -11.0, 80.0);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }
  if (!NOVOLUME) scene.add(volume);

  // K3.3 his `source: 'twoBunnies'` (frame.js L456–470): a second bunny overlapping the first by about 40% of its
  // width, at (0.6, 0.1, -0.2) × size from the first, turning about y (his interval; 0 here unless the manifest has it).
  const volumes: VDBFrameVolume[] = [volume];
  if (TWO) {
    const second = new VDBFrameVolume(source, {
      frameRenderer,
      bbox,
      extinction: def.smoke ? 40 : 100,
      phase: "dual-lobe",
      shadowSteps: SHADOW_STEPS,
      mipmaps: MIPS,
    });
    const capture = manifest?.metadata?.volumes?.[1];
    if (capture) {
      second.matrixAutoUpdate = false;
      second.matrix.fromArray(capture.matrixWorld);
      second.matrixWorld.copy(second.matrix);
    } else {
      second.position.copy(volume.position).add(new THREE.Vector3(size.x * 0.6, size.y * 0.1, -size.z * 0.2).multiplyScalar(volume.scale.x));
      second.scale.copy(volume.scale);
    }
    if (!NOVOLUME) scene.add(second);
    volumes.push(second);
  }

  let sun: THREE.DirectionalLight | null = null;
  if (LIGHT_SETUP === "dir" || LIGHT_SETUP === "dir-hemi") {
    // His sun: DirectionalLight(0xffffff, 1.0) at (100, 200, 100), target at the origin (frame.js L191–194).
    sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(100.0, 200.0, 100.0);
    scene.add(sun);
    scene.add(sun.target);
  }
  // K3.5 `?extraLights=N`: N more directional lights (shadow slots 1..N; 6 slots fill the 4 spectral slot groups)
  for (let i = 0; i < Number(params.get("extraLights") ?? 0); i++) {
    const a = (i + 1) * 1.3;
    const extra = new THREE.DirectionalLight(new THREE.Color().setHSL((i * 0.17) % 1, 0.6, 0.6), 0.35);
    extra.position.set(Math.cos(a) * 150.0, 80.0 + 30.0 * i, Math.sin(a) * 150.0);
    scene.add(extra);
    scene.add(extra.target);
  }
  if (LIGHT_SETUP === "dir-hemi") {
    // His hemisphere and ambient lights (frame.js L196–200), for the 'frame' lighting twin (sky up / down slots).
    scene.add(new THREE.HemisphereLight(0x88aaff, 0x443322, 0.5));
    scene.add(new THREE.AmbientLight(0xffffff, 0.3));
  }

  // K3.3 environment: his UltraHDR map (owner copies it into fixtures/k3/; intensity 0.4 as his resources.js) or a
  // synthetic sky with a sun. 'directional' environment lights come from a PMREM readback (async), so the first
  // frames render with the 'outward' path; frames are rendered until the lights exist (or 60 tries).
  if (ENV) {
    const envTexture: THREE.Texture =
      ENV === "synthetic"
        ? syntheticEnvironment()
        : ENV === "furnace"
          ? furnaceEnvironment(FURNACE_RADIANCE)
          : await new Promise<THREE.Texture>((resolve, reject) => {
            new UltraHDRLoader().load(ENV, (t) => resolve(t), undefined, (e) => reject(new Error(`UltraHDR ${ENV}: ${String(e)} (his maps are not committed: copy ${ENV.split("/").pop()} from his clone into fixtures/k3/)`)));
          });
    envTexture.mapping = THREE.EquirectangularReflectionMapping;
    scene.environment = envTexture;
    // K6.2b `?envPmrem=1`: scene.environment as three's PMREM (CUBE_UV) atlas, both backends (the core then draws its lat-long)
    if (params.get("envPmrem") === "1") scene.environment = new PMREMGenerator(renderer).fromEquirectangular(envTexture).texture;
    // His furnace: the uniform environment is the background too (the volume must vanish against it).
    if (FURNACE) scene.background = new THREE.Color().setRGB(FURNACE_RADIANCE, FURNACE_RADIANCE, FURNACE_RADIANCE, THREE.LinearSRGBColorSpace);
    const captured = manifest?.metadata?.environment;
    scene.environmentIntensity = params.has("envIntensity") || !captured ? ENV_INTENSITY : captured.intensity;
    if (captured?.rotation) scene.environmentRotation.set(captured.rotation[0], captured.rotation[1], captured.rotation[2], captured.rotation[3] as THREE.EulerOrder);
  }

  // K3.6: the floor, props, sun shadow and emission light of his Step 6 scenes
  if (def.step6) {
    scene.updateMatrixWorld(true);
    const s = explosionFrame !== null ? 100.0 / Math.max(size.x, size.y, size.z) : 1.0;
    setupStep6({ scene, renderer, frameRenderer, sun, manifest, worldSize: size.clone().multiplyScalar(s), zUp: explosionFrame !== null });
    result.settings = { ...frameRenderer.settings };
  }
  const volumeCenter = new THREE.Vector3();
  camera.updateMatrixWorld(true);
  scene.updateMatrixWorld(true);
  volume.updateMatrixWorld(true);
  const boxWorld = new THREE.Box3().copy(bbox).applyMatrix4(volume.matrixWorld);
  boxWorld.getCenter(volumeCenter);

  if (OCCLUDER) {
    // An opaque box between the camera and the volume centre, half way.
    const camPos = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const occ = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: 0xff3366 }));
    occ.position.lerpVectors(camPos, volumeCenter, 0.5);
    occ.scale.setScalar(boxWorld.getSize(new THREE.Vector3()).length() * 0.08);
    occ.lookAt(camPos);
    scene.add(occ);
  }
  scene.updateMatrixWorld(true);

  // Screen rect of the volume box (top-left origin), for the compare mask.
  {
    const vp = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(i & 1 ? bbox.max.x : bbox.min.x, i & 2 ? bbox.max.y : bbox.min.y, i & 4 ? bbox.max.z : bbox.min.z).applyMatrix4(volume.matrixWorld);
      const v = new THREE.Vector4(p.x, p.y, p.z, 1).applyMatrix4(vp);
      const sx = ((v.x / v.w) * 0.5 + 0.5) * W;
      const sy = (1 - ((v.y / v.w) * 0.5 + 0.5)) * H;
      x0 = Math.min(x0, sx);
      x1 = Math.max(x1, sx);
      y0 = Math.min(y0, sy);
      y1 = Math.max(y1, sy);
    }
    result.volumeRect = [Math.max(0, Math.floor(x0)), Math.max(0, Math.floor(y0)), Math.min(W, Math.ceil(x1)), Math.min(H, Math.ceil(y1))];
    // K3.6: the shadows and the lit props are outside the volume's rectangle: compare the whole frame
    if (def.step6) result.volumeRect = [0, 0, W, H];
  }

  const rt = new THREE.RenderTarget(W, H, { type: THREE.HalfFloatType, depthBuffer: true, samples: MSAA });
  result.msaa = MSAA;

  let frames = 0;
  async function frame(): Promise<Float32Array> {
    const t0 = performance.now();
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const out = await readHalf(renderer, rt, W, H);
    // K3.4: the first frame's wall time includes the WGSL / pipeline compile (D25: provisional, SwiftShader)
    if (frames++ === 0) result.firstFrameMs = performance.now() - t0;
    return out;
  }

  // K6.2b: the core reduces the environment on the GPU (no readback): its lights exist in the first frame; read the
  // environment buffer once for the report
  if (ENV && frameRenderer.backend === "core") {
    await frame();
    const core = frameRenderer.core!;
    const env = vdbFrameRendererEnvironment(core);
    const read = (renderer as unknown as { backend: { device: GPUDevice } }).backend.device;
    const buf = read.createBuffer({ size: VDB_FRAME_ENVIRONMENT.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = read.createCommandEncoder();
    enc.copyBufferToBuffer(env.buffer, 0, buf, 0, VDB_FRAME_ENVIRONMENT.size);
    read.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u32 = new Uint32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    const f32 = new Float32Array(u32.buffer);
    const at = (name: string, i: number) => {
      const o = VDB_FRAME_ENVIRONMENT.fields.get(name)!.offset / 4 + i * 4;
      return [f32[o]!, f32[o + 1]!, f32[o + 2]!];
    };
    const count = u32[0]!;
    result.environment = { mode: ["outward", "directional", "spherical-harmonics"][u32[1]!] ?? "outward", count, directions: Array.from({ length: count }, (_, i) => at("direction", i)), colors: Array.from({ length: count }, (_, i) => at("color", i)), frames: 1 };
  }
  // K3.5: 'volume' lighting reads the environment lights too
  if (ENV && ENV_MODE !== "outward" && frameRenderer.backend !== "core") {
    // K3.4 furnace / any environment: wait for the 'directional' lights (async readback).
    let tries = 0;
    while (!frameRenderer.environment && tries++ < 60) {
      await frame();
      await frameRenderer.environmentLights.settled;
    }
    const e = frameRenderer.environment as { mode: string; count: number; directions: number[][]; colors: number[][] } | null;
    result.environment = e ? { mode: e.mode, count: e.count, directions: e.directions.slice(0, e.count).map((d) => [...d]), colors: e.colors.slice(0, e.count).map((c) => [...c]), frames: tries } : null;
    if (!e) throw new Error("the environment lights did not arrive within 60 frames");
  }

  // K3.6: the volume shadow filter goes on the lights in the first frame (the proxy's onBeforeRender, after the floor
  // drew), the receivers build with it in the next, whose first receiver draws the maps; his captures run 6+ frames.
  // shadow-only: the volume proxies hidden for the captured frames (the volumes still cast).
  if (def.step6) {
    for (let i = 0; i < 4; i++) await frame();
    if (SHADOW_ONLY) for (const v of volumes) v.proxy.visible = false;
  }
  const firstLinear = await frame();
  const first = toDisplay(firstLinear);
  const second = toDisplay(await frame());
  result.inPageIdentical = first.length === second.length && first.every((v, i) => v === second[i]);
  const bg = BACKGROUND.clone().convertLinearToSRGB();
  const bgBytes = [bg.r, bg.g, bg.b].map((v) => Math.round(v * 255));
  let nonBackground = 0;
  for (let i = 0; i < first.length; i += 4) {
    if (Math.abs(first[i]! - bgBytes[0]!) + Math.abs(first[i + 1]! - bgBytes[1]!) + Math.abs(first[i + 2]! - bgBytes[2]!) > 6) nonBackground++;
  }
  result.width = W;
  result.height = H;
  result.rgba = toBase64(first);
  result.nonBackground = nonBackground;

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(first), W, H), 0, 0);
  root.append(canvas);
  log(
    `${def.name} (${result.source}): ${file.split("/").pop()} LOD ${dense.lod} ${dense.dimensions.join("x")}, mean ${dense.meanDensity.toFixed(4)}; ${W}x${H}, froxels ${FROXELS.join("x")}; ${nonBackground} volume px; repeat identical: ${result.inPageIdentical}`,
    result.inPageIdentical ? "ok" : "fail",
  );
  if (!def.parityExpected) log(`parity with his capture not expected: ${def.why}`, "warn");

  // --- integrate pass vs its CPU twin (froxel grid = screen: one unfiltered froxel column per pixel) ---
  if (TWIN) {
    const bg = BACKGROUND.clone();
    result.twin = SPECTRAL ? await spectralIntegrateTwin(renderer, frameRenderer, firstLinear, W, H, [bg.r, bg.g, bg.b]) : await integrateTwin(renderer, frameRenderer, firstLinear);
  }
  if (SPECTRAL || params.get("spectralProbe") === "1") {
    result.spectralProbe = await probeSpectral(renderer);
    log(`spectral chunk probe: ${result.spectralProbe.compared} values, max |d| ${result.spectralProbe.maxAbs.toExponential(2)} (max rel ${result.spectralProbe.maxRel.toExponential(2)}), ${result.spectralProbe.over} over`, result.spectralProbe.over === 0 ? "ok" : "fail");
  }

  // --- TSL chunks vs the CPU twins --------------------------------------------------------------
  result.chunkProbe = await probeChunks(renderer);
  log(
    `chunk probe: ${result.chunkProbe.count} values, max |GPU - CPU twin| ${result.chunkProbe.maxAbs.toExponential(2)} (max rel ${result.chunkProbe.maxRel.toExponential(2)}), ${result.chunkProbe.over} over`,
    result.chunkProbe.over === 0 ? "ok" : "fail",
  );
  if (result.twin) {
    log(`integrate twin: ${result.twin.pixels} pixels (${result.twin.covered} with volume), max |GPU - CPU| ${result.twin.maxAbs.toExponential(2)}, ${result.twin.over} over`, result.twin.over === 0 ? "ok" : "fail");
  }

  // --- K3.4: the white furnace (his utils/furnace.js) ------------------------------------------------
  if (FURNACE) {
    // The furnace frame (background L) and the same frame over black: per pixel (sRGB blending, the integrate
    // pass's default) OETF(r_L) = OETF(L)(1 − α) + α OETF(v) and OETF(r_0) = α OETF(v), so α and the volume's
    // own radiance v follow; v / L = 1 is a passed furnace. Pixels with α > 0.5 only (the volume, not its edge).
    const L = FURNACE_RADIANCE;
    scene.background = new THREE.Color(0, 0, 0);
    const black = await frame();
    scene.background = new THREE.Color().setRGB(L, L, L, THREE.LinearSRGBColorSpace);
    const oetf = (x: number) => Math.min(1, Math.max(0, vdbSrgbOetf(Math.max(0, x))));
    const eL = oetf(L);
    const ratios: number[] = [];
    for (let i = 0; i < firstLinear.length; i += 4) {
      const r1 = oetf(firstLinear[i + 1]!);
      const r0 = oetf(black[i + 1]!);
      const alpha = (eL + r0 - r1) / eL;
      if (!(alpha > 0.5)) continue;
      const v = vdbSrgbEotfLocal(r0 / alpha);
      ratios.push(v / L);
    }
    ratios.sort((a, b) => a - b);
    const meanOf = (x: number, y: number, n: number) => {
      const m = [0, 0, 0];
      for (let dy = 0; dy < n; dy++) for (let dx = 0; dx < n; dx++) for (let c = 0; c < 3; c++) m[c]! += firstLinear[((y + dy) * W + x + dx) * 4 + c]! / (n * n);
      return m;
    };
    const center = meanOf(Math.floor(W / 2 - 4.5), Math.floor(H / 2 - 4.5), 9);
    const mean = ratios.reduce((a, b) => a + b, 0) / Math.max(1, ratios.length);
    const env = frameRenderer.environment as { mode: string; count: number } | null;
    result.furnace = {
      radiance: L,
      centerRatio: center.map((v) => v / L),
      corner: meanOf(0, 0, 9),
      volumePixels: ratios.length,
      meanRatio: mean,
      minRatio: ratios[0] ?? NaN,
      maxRatio: ratios[ratios.length - 1] ?? NaN,
      meanAbsDeviation: ratios.reduce((a, b) => a + Math.abs(b - 1), 0) / Math.max(1, ratios.length),
      p05: ratios[Math.floor(ratios.length * 0.05)] ?? NaN,
      p95: ratios[Math.floor(ratios.length * 0.95)] ?? NaN,
      environment: env ? { mode: env.mode, count: env.count } : null,
    };
    log(`furnace (ms ${String(frameRenderer.settings.multipleScattering)}): ${JSON.stringify(result.furnace)}`);
  }

  // --- K3.3 'frame' lighting: grid layout, grid chunks, pass twins, redraw skips, temporal convergence ---------
  if (LIGHTING === "frame") {
    const g = frameRenderer.gridUniforms;
    const count = g.frameGridCount.value as number;
    result.grid = {
      count,
      resolution: g.frameGridResolution.value as number,
      gridMin: (g.frameGridMin.array as THREE.Vector3[]).slice(0, count).map((v) => v.toArray()),
      gridVoxel: (g.frameGridVoxel.array as THREE.Vector3[]).slice(0, count).map((v) => v.toArray()),
    };
    // The last frame drew the passes the twins read (identical frames; the targets hold the last draw).
    if (TWINS && SPECTRAL) {
      const t = await spectralTransmittanceTwin(renderer, frameRenderer);
      result.spectralGroups = t.groups;
      result.transmittanceTwin = t;
      result.lightingTwin = await spectralLightingTwin(renderer, frameRenderer);
      if (SOURCE_KIND === "dense" && volumes.length === 1) result.densityTwin = await spectralDensityTwin(renderer, frameRenderer, volume, dense);
      if (frameRenderer.environmentGridMaterial.frameLightCount.value > 0 && frameRenderer.redrawStats.environmentGrid.drawn > 0) result.environmentGridTwin = await spectralEnvironmentGridTwin(renderer, frameRenderer);
      log(`spectral twins (${t.groups} groups): transmittance ${JSON.stringify(result.transmittanceTwin)}; lighting ${JSON.stringify(result.lightingTwin)}; density ${JSON.stringify(result.densityTwin ?? null)}; environment grid ${JSON.stringify(result.environmentGridTwin ?? null)}`);
    } else if (TWINS) {
      result.transmittanceTwin = await transmittanceTwin(renderer, frameRenderer);
      result.lightingTwin = await lightingTwin(renderer, frameRenderer);
      if (SOURCE_KIND === "dense" && volumes.length === 1) {
        result.densityTwin = await densityTwin(renderer, frameRenderer, volume, dense);
        log(`density twin: ${result.densityTwin.compared} values (${result.densityTwin.covered} froxels), max |d| ${result.densityTwin.maxAbs.toExponential(2)}, ${result.densityTwin.over} over`, result.densityTwin.over === 0 ? "ok" : "fail");
      }
      if (frameRenderer.environmentGridMaterial.frameLightCount.value > 0 && frameRenderer.redrawStats.environmentGrid.drawn > 0) {
        result.environmentGridTwin = await environmentGridTwin(renderer, frameRenderer);
        log(`environment grid twin: ${result.environmentGridTwin.compared} values (${result.environmentGridTwin.marched} marched), max |d| ${result.environmentGridTwin.maxAbs.toExponential(2)}, ${result.environmentGridTwin.over} over`, result.environmentGridTwin.over === 0 ? "ok" : "fail");
      }
      log(
        `transmittance twin: ${result.transmittanceTwin.compared} values (${result.transmittanceTwin.marched} marched voxels), max |d| ${result.transmittanceTwin.maxAbs.toExponential(2)}, ${result.transmittanceTwin.over} over; lighting twin: ${result.lightingTwin.compared} values (${result.lightingTwin.lit} froxels), max |d| ${result.lightingTwin.maxAbs.toExponential(2)}, ${result.lightingTwin.over} over`,
        result.transmittanceTwin.over + result.lightingTwin.over === 0 ? "ok" : "fail",
      );
    }
    if (TWINS) {
      // K3.4: the scattering chunks; the neural input mips
      result.scatterProbe = await probeScattering(renderer);
      log(`scattering probe: ${result.scatterProbe.compared} values, max |d| ${result.scatterProbe.maxAbs.toExponential(2)} (max rel ${result.scatterProbe.maxRel.toExponential(2)}), ${result.scatterProbe.over} over`, result.scatterProbe.over === 0 ? "ok" : "fail");
      if (frameRenderer.transmittanceMaterial.neural) {
        result.gridMipTwin = await gridMipTwin(renderer, frameRenderer);
        log(`grid mip twin: ${result.gridMipTwin.compared} values over ${result.gridMipTwin.levels} levels, max |d| ${result.gridMipTwin.maxAbs.toExponential(2)}, ${result.gridMipTwin.over} over`, result.gridMipTwin.over === 0 ? "ok" : "fail");
      }
    }
    if (SPECTRAL) result.spectralGroups = Math.ceil(Math.max(0, (frameRenderer.transmittanceMaterial.frameShadowLightCount.value as number) - 1) / 2) + 1;
    result.gridProbe = await probeGrid(renderer, frameRenderer);
    log(`grid chunk probe: ${result.gridProbe.compared} values, max |d| ${result.gridProbe.maxAbs.toExponential(2)}, ${result.gridProbe.over} over`, result.gridProbe.over === 0 ? "ok" : "fail");

    for (let f = 0; f < REDRAW_FRAMES; f++) await frame();
    result.redraw = JSON.parse(JSON.stringify(frameRenderer.redrawStats)) as VDBFrameRedrawStats;
    log(`redraws (drawn/skipped): ${JSON.stringify(result.redraw)}`);
  }

  if (TEMPORAL_CHECK > 0) {
    // Jittered frames without and with temporal reprojection (static camera): the temporal frames settle toward
    // the mean of the jittered frames and change less from frame to frame.
    const mask: number[] = [];
    for (let i = 0; i < first.length; i += 4) {
      if (Math.abs(first[i]! - bgBytes[0]!) + Math.abs(first[i + 1]! - bgBytes[1]!) + Math.abs(first[i + 2]! - bgBytes[2]!) > 6) mask.push(i);
    }
    const run = async (temporal: boolean): Promise<Float32Array[]> => {
      frameRenderer.settings.jitter = true;
      frameRenderer.settings.temporal = temporal;
      frameRenderer.jitterIndex = 0;
      const out: Float32Array[] = [];
      for (let k = 0; k < TEMPORAL_CHECK; k++) out.push(await frame());
      return out;
    };
    const diffs = (frames: Float32Array[]) =>
      frames.slice(1).map((f, k) => {
        const a = toDisplay(frames[k]!);
        const b = toDisplay(f);
        let s = 0;
        for (const i of mask) s += Math.abs(a[i]! - b[i]!) + Math.abs(a[i + 1]! - b[i + 1]!) + Math.abs(a[i + 2]! - b[i + 2]!);
        return s / Math.max(1, mask.length * 3);
      });
    const psnr = (a: Uint8Array, b: Uint8Array) => {
      let se = 0;
      for (const i of mask) for (let c = 0; c < 3; c++) se += (a[i + c]! - b[i + c]!) ** 2;
      const mse = se / Math.max(1, mask.length * 3);
      return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
    };
    const off = await run(false);
    const mean = new Float32Array(off[0]!.length);
    for (const f of off) for (let i = 0; i < f.length; i++) mean[i] = mean[i]! + f[i]! / off.length;
    const on = await run(true);
    const meanDisplay = toDisplay(mean);
    result.temporalCheck = {
      frames: TEMPORAL_CHECK,
      diffsOff: diffs(off),
      diffsOn: diffs(on),
      psnrSingleVsMean: psnr(toDisplay(off[0]!), meanDisplay),
      psnrTemporalVsMean: psnr(toDisplay(on[on.length - 1]!), meanDisplay),
      volumePixels: mask.length,
    };
    frameRenderer.settings.jitter = false;
    frameRenderer.settings.temporal = false;
    log(`temporal: ${JSON.stringify(result.temporalCheck)}`);
  }

  // --- per-pass timing (D25: provisional) ---------------------------------------------------------
  if (TIMING && features.length > 0) {
    const resolve = async () => {
      const ms = await renderer.resolveTimestampsAsync("render");
      return typeof ms === "number" ? ms : null;
    };
    const median = (v: (number | null)[]) => {
      const x = v.filter((n): n is number => n !== null).sort((a, b) => a - b);
      return x.length ? x[Math.floor(x.length / 2)]! : null;
    };
    const frames = Number(params.get("timingFrames") ?? 1);
    const warmup = Number(params.get("timingWarmup") ?? 0);
    const samples = { scene: [] as (number | null)[], froxels: [] as (number | null)[], integrate: [] as (number | null)[], full: [] as (number | null)[], grid: [] as (number | null)[] };
    await resolve();
    for (let f = 0; f < warmup + frames; f++) {
      // Scene alone (volume hidden); the froxel (inject) passes alone; scene + integrate over the held atlas; the full frame.
      volume.visible = false;
      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      const sceneMs = await resolve();
      volume.visible = true;
      frameRenderer.renderFroxels(renderer, scene, camera);
      const froxelsMs = await resolve();
      frameRenderer.holdFroxels = true;
      renderer.setRenderTarget(rt);
      renderer.render(scene, camera);
      frameRenderer.holdFroxels = false;
      const heldMs = await resolve();
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      const fullMs = await resolve();
      // K3.3 'frame' lighting: the froxel passes again with the grid passes forced (signatures reset): the cost of a
      // redraw of the shadow density, transmittance and environment grid.
      let gridMs: number | null = null;
      if (LIGHTING === "frame") {
        frameRenderer.shadowDrawnVersion = -1;
        frameRenderer.transmittanceSignature = [];
        frameRenderer.gridMipSignature = [];
        frameRenderer.environmentGridSignature = [];
        frameRenderer.renderFroxels(renderer, scene, camera);
        const redrawMs = await resolve();
        gridMs = redrawMs !== null && froxelsMs !== null ? redrawMs - froxelsMs : null;
      }
      if (f < warmup) continue;
      samples.scene.push(sceneMs);
      samples.froxels.push(froxelsMs);
      samples.integrate.push(heldMs !== null && sceneMs !== null ? heldMs - sceneMs : null);
      samples.full.push(fullMs);
      samples.grid.push(gridMs);
    }
    result.timings = {
      provisional: true,
      note: `D25: not from a quiet machine; median of ${frames} frame(s) after ${warmup} warm-up; never gate on it. full is three's figure for the whole render (nested froxel passes may not be included). 'frame' lighting: froxels is a frame with the grid passes skipped (static scene), gridRedraw the extra cost when they are drawn`,
      sceneMs: median(samples.scene),
      froxelsMs: median(samples.froxels),
      integrateMs: median(samples.integrate),
      fullMs: median(samples.full),
      ...(LIGHTING === "frame" ? { passes: { gridRedrawMs: median(samples.grid) } } : {}),
    };
    log(`timings (provisional, D25): ${JSON.stringify(result.timings)}`, "warn");
  } else {
    result.timings = { provisional: true, note: "timing off (?timing=1 needs timestamp-query); D25", sceneMs: null, froxelsMs: null, fullMs: null, integrateMs: null };
  }
  for (const w of result.warnings) log(`warning: ${w}`, "warn");
  source.dispose();
}

/**
 * The integrate pass against `vdbFrameIntegrate` + `vdbFrameSrgbBlend`. Needs froxels = screen size
 * (each pixel reads its own froxel column at texel centres) and no occluder (empty depth).
 */
async function integrateTwin(renderer: WebGPURenderer, frameRenderer: VDBFrameRenderer, linear: Float32Array): Promise<TwinCheck> {
  const atlasTarget = frameRenderer.atlasTarget!;
  const aw = atlasTarget.width;
  const ah = atlasTarget.height;
  const atlas = await readHalf(renderer, atlasTarget, aw, ah);
  const fu = frameRenderer.froxelUniforms;
  const grid = [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z];
  if (grid[0] !== W || grid[1] !== H) throw new Error(`?twin=1 needs ?froxels=${W}x${H}xD (got ${grid.join("x")})`);
  const columns = fu.froxelColumns.value as number;
  const range: [number, number] = [fu.froxelRange.value.x, fu.froxelRange.value.y];
  const sr = frameRenderer.integrateMaterial.frameSliceRange.value;
  const bg = BACKGROUND.clone();
  const tol = 4e-3;
  let maxAbs = 0;
  let over = 0;
  let covered = 0;
  let firstBad: TwinCheck["firstBad"];
  const step = 3;
  let pixels = 0;
  for (let y = 3; y < H; y += step) {
    for (let x = 3; x < W; x += step) {
      pixels++;
      // Screen pixel (x, y top-left) → bottom-left froxel column (x, H - 1 - y).
      const fy = H - 1 - y;
      const froxels: VDBFrameFroxelSample[] = [];
      for (let s = sr.x; s < sr.y; s++) {
        const tx = (s % columns) * grid[0]! + x;
        const ty = Math.floor(s / columns) * grid[1]! + fy;
        // Atlas rows are bottom-left in the shaders' convention; the readback is top row first.
        const row = ah - 1 - ty;
        const o = (row * aw + tx) * 4;
        froxels.push([atlas[o]!, atlas[o + 1]!, atlas[o + 2]!, atlas[o + 3]!]);
      }
      const r = vdbFrameIntegrate(froxels, [sr.x, sr.y], range, grid[2]!, 1e20);
      const o = (y * W + x) * 4;
      const gpu = [linear[o]!, linear[o + 1]!, linear[o + 2]!];
      const cpu = r ? vdbFrameSrgbBlend([bg.r, bg.g, bg.b], r.radiance, r.alpha) : [bg.r, bg.g, bg.b];
      if (r) covered++;
      for (let c = 0; c < 3; c++) {
        // Half-float output: relative tolerance on top of the absolute one.
        const d = Math.abs(gpu[c]! - cpu[c]!);
        maxAbs = Math.max(maxAbs, d);
        if (d > tol + cpu[c]! * 2e-3) {
          over++;
          firstBad ??= { x, y, gpu, cpu };
        }
      }
    }
  }
  return { pixels, covered, maxAbs, over, ...(firstBad ? { firstBad } : {}) };
}

/** Evaluates the TSL chunks in a fragment pass (RGBA32F, one value per channel) and compares with the CPU twins. */
async function probeChunks(renderer: WebGPURenderer): Promise<ChunkProbe> {
  const PW = 64;
  const PH = 4;
  const C = vdbFrameChunks;
  const froxel = C.createFroxelUniforms();
  const range: [number, number] = [0.5, 300.0];
  froxel.froxelRange.value.set(range[0], range[1]);
  froxel.frameFadeRadius.value = 50.0;
  const jitterOn = 0.3;
  const albedo: [number, number, number] = [0.9, 0.8, 0.7];
  const ms = 0.42;
  const phase: [number, number, number, number] = [0.8, -0.3, 0.2, 0];

  const material = new NodeMaterial();
  material.fragmentNode = Fn(() => {
    const px = screenCoordinate.x.floor();
    const row = screenCoordinate.y.floor().toVar();
    const a = px.add(0.5).div(PW).toVar();
    const out = vec4(0.0).toVar();
    const cosT = a.mul(2.0).sub(1.0).toVar();
    const sinT = float(1.0).sub(cosT.mul(cosT)).max(0.0).sqrt();
    const ray = vec3(0.0, 0.0, 1.0);
    const light = vec3(sinT, 0.0, cosT);
    const phaseU = vec4(...phase);
    const e = a.mul(4.0).toVar();
    const alb = vec3(...albedo);
    const msU = float(ms);
    If(row.lessThan(0.5), () => {
      out.assign(vec4(C.frameHenyeyGreenstein(cosT, float(0.8)), C.frameHenyeyGreenstein(cosT, float(-0.3)), C.frameHenyeyGreenstein(cosT, float(0.0)), C.framePhaseWeight(phaseU, ray, light)));
    })
      .ElseIf(row.lessThan(1.5), () => {
        out.assign(
          vec4(
            C.frameSliceDepth(froxel, a),
            C.frameNoise(float(-1.0), vec2(px.mul(7.0).add(3.0), row.mul(13.0).add(5.0))),
            C.frameNoise(float(jitterOn), vec2(px.mul(7.0).add(3.0), row.mul(13.0).add(5.0))),
            C.frameDistanceFade(froxel, vec3(0.0, 0.0, a.mul(-60.0))),
          ),
        );
      })
      .ElseIf(row.lessThan(2.5), () => {
        const absorption = C.frameShadowAbsorption(e, alb).toVar();
        const s = C.frameShadowStep(e, absorption, msU).toVar();
        const depths = s.mul(3.0).toVar();
        out.assign(vec4(s.y, C.frameShadowDepth(depths, msU), C.frameShadowTransmittance(depths, msU, vec4(0.0)).y, absorption));
      })
      .Else(() => {
        const shadow = vec2(e.negate().exp(), e.mul(-0.5).exp()).toVar();
        const s = C.frameShadowStep(e, C.frameShadowAbsorption(e, alb), float(0.0)).mul(2.0).toVar();
        out.assign(
          vec4(
            C.frameShadowedLight(phaseU, vec3(1.0, 1.0, 0.0), vec4(0.0), ray, light, shadow).x,
            C.frameShadowDepth(s, float(0.0)),
            C.frameShadowTransmittance(s, float(0.0), vec4(0.0)).x,
            select(a.greaterThan(0.5), C.framePhaseColor(phaseU, vec3(1.0, 1.0, 0.0), ray, light).y, float(0.0)),
          ),
        );
      });
    return out;
  })();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  material.vertexNode = vec4(TSL.positionGeometry.xy, 0.0, 1.0);
  quad.frustumCulled = false;
  const probeScene = new THREE.Scene();
  probeScene.add(quad);
  const target = new THREE.RenderTarget(PW, PH, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  renderer.setRenderTarget(target);
  renderer.render(probeScene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, PW, PH)) as unknown as Float32Array;
  // Readback rows are top first, as screenCoordinate rows: out row 0 = probe row 0.
  const gpu = tightRows(raw, PW, PH, 4);

  let maxAbs = 0;
  let maxRel = 0;
  let over = 0;
  let count = 0;
  let firstBad: ChunkProbe["firstBad"];
  for (let row = 0; row < PH; row++) {
    for (let x = 0; x < PW; x++) {
      const a = (x + 0.5) / PW;
      const cosT = a * 2 - 1;
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
      const ray: [number, number, number] = [0, 0, 1];
      const light: [number, number, number] = [sinT, 0, cosT];
      const e = a * 4;
      const pixel: [number, number] = [x * 7 + 3, row * 13 + 5];
      let cpu: number[];
      if (row === 0) {
        cpu = [vdbFrameHenyeyGreenstein(cosT, 0.8), vdbFrameHenyeyGreenstein(cosT, -0.3), vdbFrameHenyeyGreenstein(cosT, 0), vdbFramePhaseWeight(phase, ray, light)];
      } else if (row === 1) {
        cpu = [vdbFrameSliceDepth(range, a), vdbFrameNoise(-1, pixel), vdbFrameNoise(jitterOn, pixel), vdbFrameDistanceFade(50, [0, 0, -a * 60])];
      } else if (row === 2) {
        const absorption = vdbFrameShadowAbsorption(e, albedo);
        const s = vdbFrameShadowStep(e, absorption, ms);
        const depths: [number, number, number] = [s[0] * 3, s[1] * 3, s[2] * 3];
        cpu = [s[1], vdbFrameShadowDepth(depths, ms), vdbFrameShadowTransmittance(depths, ms)[1], absorption];
      } else {
        const shadow: [number, number] = [Math.exp(-e), Math.exp(-e * 0.5)];
        const s0 = vdbFrameShadowStep(e, vdbFrameShadowAbsorption(e, albedo), 0);
        const s: [number, number, number] = [s0[0] * 2, s0[1] * 2, s0[2] * 2];
        cpu = [vdbFrameShadowedLight(phase, ray, light, shadow), vdbFrameShadowDepth(s, 0), vdbFrameShadowTransmittance(s, 0)[0], a > 0.5 ? vdbFramePhaseWeight(phase, ray, light) : 0];
      }
      for (let c = 0; c < 4; c++) {
        const g = gpu[(row * PW + x) * 4 + c]!;
        const d = Math.abs(g - cpu[c]!);
        const rel = d / Math.max(1, Math.abs(cpu[c]!));
        count++;
        maxAbs = Math.max(maxAbs, d);
        maxRel = Math.max(maxRel, rel);
        // f32 vs f64; IGN noise amplifies the fract() argument error (×52.98), so 2e-3 there.
        const tol = row === 1 && c === 2 ? 2e-3 : 2e-4;
        if (rel > tol) {
          over++;
          firstBad ??= { row, x, channel: c, gpu: g, cpu: cpu[c]! };
        }
      }
    }
  }
  target.dispose();
  material.dispose();
  if (params.get("debug") === "1") (result as unknown as Record<string, unknown>).probeGpu = Array.from(gpu);
  return { count, maxAbs, maxRel, over, ...(firstBad ? { firstBad } : {}) };
}

main()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
