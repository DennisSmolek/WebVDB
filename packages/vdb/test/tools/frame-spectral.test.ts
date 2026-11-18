import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  vdbFrameDensityFroxel,
  vdbFrameExtinctionSpectrum,
  vdbFrameFitPhaseGExponent,
  vdbFrameGridCell,
  vdbFrameHenyeyGreenstein,
  vdbFrameInjectIsotropicIrradiance,
  vdbFrameInjectSlice,
  vdbFrameIsotropicDiffusion,
  vdbFrameMediumFromMeasurements,
  vdbFrameMediumMeasurements,
  vdbFrameMediumNames,
  vdbFrameMediumPresets,
  vdbFrameMediumProps,
  vdbFrameMieRgbChannels,
  vdbFrameMieTable,
  vdbFrameOctaveLight,
  vdbFramePackSpectralGroup,
  vdbFrameScatteringOctaves,
  vdbFrameShadowAbsorption,
  vdbFrameShadowMarch,
  vdbFrameShadowStep,
  vdbFrameShadowTotal,
  vdbFrameShadowTransmittance,
  vdbFrameShadowedLightColor,
  vdbFrameSpectralDensityFroxel,
  vdbFrameSpectralEnvironmentGridVoxel,
  vdbFrameSpectralGridAlbedo,
  vdbFrameSpectralGroups,
  vdbFrameSpectralInjectSlice,
  vdbFrameSpectralIntegrate,
  vdbFrameSpectralLightingFroxel,
  vdbFrameSpectralOctaveLight,
  vdbFrameSpectralShadowDepths,
  vdbFrameSpectralShadowMarch,
  vdbFrameSpectralShadowStep,
  vdbFrameSpectralShadowTotal,
  vdbFrameSpectralShadowTransmittance,
  vdbFrameSpectralShadowUnit,
  vdbFrameSpectralShadowedLight,
  vdbFrameSpectralTransmittanceVoxel,
  vdbFrameEnvironmentGridVoxel,
  vdbFrameTransmittanceVoxel,
  vdbFrameUnpackSpectralGroup,
  vdbShIrradianceAt,
  type VDBFrameAtlas,
  type VDBFrameGridLayout,
  type VDBFrameVec3,
  type VDBFrameVec4,
} from "../../src/tools/index.js";
import { getFrameMediumProps, getFrameExtinctionSpectrum, getFrameSpectralGroups } from "../../src/three/frame/index.js";

// K3.5: mjurczyk's spectral media (tools/frame/medium.ts, spectral.ts). Each test pins a property of his formula or
// table that a port error would break; the GPU passes are compared with these twins in e2e/k3-frame-spectral.spec.ts.

type V3 = VDBFrameVec3;
type V4 = VDBFrameVec4;

const close3 = (a: readonly number[], b: readonly number[], digits = 12) => a.forEach((v, i) => expect(v, `channel ${i}`).toBeCloseTo(b[i]!, digits));
const round3 = (x: number) => Math.round(x * 1000) / 1000;

describe("medium presets (his frameMedium.js)", () => {
  it("every preset is the fit of its cited measurements at his precision (3 decimals)", () => {
    // Cloud's g is the 'lorenz-mie-rgb' tables' green asymmetry: the generator (his BHMIE script, K3.4) at 550 nm.
    const green = vdbFrameMieRgbChannels[1]!;
    const greenAsymmetry = vdbFrameMieTable(green.wavelength, green.refractiveIndex).asymmetry;
    expect(greenAsymmetry.toFixed(4)).toBe("0.8533");
    expect(Object.keys(vdbFrameMediumMeasurements).sort()).toEqual([...vdbFrameMediumNames].sort());
    for (const name of vdbFrameMediumNames) {
      const fit = vdbFrameMediumFromMeasurements(vdbFrameMediumMeasurements[name]!, greenAsymmetry);
      const preset = vdbFrameMediumPresets[name]!;
      fit.albedo.forEach((v, c) => expect(Math.abs(v - preset.albedo[c]!), `${name} albedo ${c}`).toBeLessThanOrEqual(5e-4 + 1e-12));
      expect(round3(fit.angstromExponent), `${name} Ångström`).toBe(preset.angstromExponent);
      expect(round3(fit.phaseG), `${name} g`).toBe(preset.phaseG);
      expect(round3(fit.phaseGExponent), `${name} g exponent`).toBe(preset.phaseGExponent);
      expect(fit.phase).toBe(preset.phase);
    }
  }, 120_000);

  it("phaseGExponent reproduces the cited blue and red g through g·(λ/550)^−e (independent of the fit)", () => {
    for (const name of ["white-smoke", "dark-smoke", "haze", "dust"]) {
      const { phaseG, phaseGExponent } = vdbFrameMediumPresets[name]!;
      const { blue, red } = vdbFrameMediumMeasurements[name]!.g;
      // The exponent fits the blue / red ratio (his rounding to 3 decimals moves it by < 1e-3)
      expect(Math.abs(Math.pow(450 / 650, -phaseGExponent) - blue! / red!)).toBeLessThan(1e-3);
      if (name === "dust") continue; // dust keeps its measured green g (0.752 is above both blue and red)
      // His rounding to 3 decimals moves g by < 2e-3
      expect(Math.abs(phaseG * Math.pow(450 / 550, -phaseGExponent) - blue!)).toBeLessThan(2e-3);
      expect(Math.abs(phaseG * Math.pow(650 / 550, -phaseGExponent) - red!)).toBeLessThan(2e-3);
    }
    expect(vdbFrameFitPhaseGExponent(0.69, 0.61)).toBeCloseTo(Math.log(0.69 / 0.61) / Math.log(650 / 450), 15);
  });

  it("getFrameMediumProps: his props (linear albedo, extinction scale 1), null for unknown names", () => {
    expect(vdbFrameMediumProps("custom")).toBeNull();
    expect(vdbFrameMediumProps("toString")).toBeNull();
    expect(getFrameMediumProps("nope")).toBeNull();
    const smoke = getFrameMediumProps("dark-smoke")!;
    expect(smoke.baseColor).toBeInstanceOf(THREE.Color);
    // The exact linear albedo, not an 8-bit sRGB hex
    expect([smoke.baseColor.r, smoke.baseColor.g, smoke.baseColor.b]).toEqual([0.84, 0.86, 0.88]);
    expect(smoke.extinctionScale.toArray()).toEqual([1, 1, 1]);
    expect(smoke).toMatchObject({ angstromExponent: 1.8, phase: "henyey-greenstein", phaseG: 0.577, phaseGExponent: 0.513 });
    expect(getFrameMediumProps("cloud")!.phase).toBe("lorenz-mie-rgb");
    // A copy: changing it does not change the table
    vdbFrameMediumProps("dust")!.albedo[0] = 0;
    expect(vdbFrameMediumPresets.dust!.albedo[0]).toBe(0.98);
  });
});

describe("spectrum and slot groups (his frameSpectrum.js)", () => {
  it("extinction of each channel is scale × (λ / 550)^−α; the three adapter equals the twin", () => {
    for (const alpha of [0, 0.3, 1.8, 4.09, -0.1]) {
      const s = vdbFrameExtinctionSpectrum([2, 1, 0.5], alpha);
      expect(s[0]).toBeCloseTo(2 * (550 / 650) ** alpha, 14);
      expect(s[1]).toBeCloseTo(1, 14);
      expect(s[2]).toBeCloseTo(0.5 * (550 / 450) ** alpha, 14);
      const three = getFrameExtinctionSpectrum({ extinctionScale: { x: 2, y: 1, z: 0.5 }, angstromExponent: alpha }, new THREE.Vector3());
      close3(three.toArray(), s, 14);
    }
    // Rayleigh air (4.09): blue is (650/450)^4.09 = 4.5× red
    const air = getFrameExtinctionSpectrum({ extinctionScale: { x: 1, y: 1, z: 1 }, angstromExponent: "air" }, new THREE.Vector3());
    expect(air.z / air.x).toBeCloseTo((650 / 450) ** 4.09, 12);
    expect(vdbFrameExtinctionSpectrum([-1, 1, 1], 0)[0]).toBe(0);
  });

  it("slot groups: sky + slot 0, then two slots each", () => {
    expect([0, 1, 2, 3, 4, 5, 6].map(vdbFrameSpectralGroups)).toEqual([1, 1, 2, 2, 3, 3, 4]);
    expect([0, 1, 2, 3, 4, 5, 6].map(getFrameSpectralGroups)).toEqual([1, 1, 2, 2, 3, 3, 4]);
  });

  it("group packing: (a.x, a.y.r), (a.y.gb, b.x.rg), (b.x.b, b.y) round-trips", () => {
    const a = { x: [1, 2, 3] as V3, y: [4, 5, 6] as V3 };
    const b = { x: [7, 8, 9] as V3, y: [10, 11, 12] as V3 };
    const maps = vdbFramePackSpectralGroup(a, b);
    expect(maps).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    expect(vdbFrameUnpackSpectralGroup(...maps)).toEqual({ a, b });
  });
});

describe("spectral shader chunks (frameShaderChunks.js L483–622)", () => {
  const octaves = vdbFrameScatteringOctaves({ multipleScattering: "octaves", scatteringOctaves: 4, scatteringOctaveExtinction: 0.5, scatteringOctaveContribution: 0.6, scatteringOctavePhase: 0.5 });

  it("the spectral step and unit are the grey step of each channel (grey albedo: luminance = albedo)", () => {
    for (const ms of [0, 0.42, 1]) {
      const e: V3 = [3, 1, 0.5];
      const albedo = 0.7;
      const step = vdbFrameSpectralShadowStep(e, e.map((x) => x * (1 - albedo)) as V3, ms);
      const unit = vdbFrameSpectralShadowUnit(e, [albedo, albedo, albedo], ms);
      for (let c = 0; c < 3; c++) {
        const grey = vdbFrameShadowStep(e[c]!, vdbFrameShadowAbsorption(e[c]!, [albedo, albedo, albedo]), ms);
        expect([step.x[c], step.y[c], step.z[c]]).toEqual(grey.map((v) => expect.closeTo(v, 12)));
        expect([unit.x[c], unit.y[c], unit.z[c]]).toEqual(grey.map((v) => expect.closeTo(v, 12)));
      }
      // Linear in the optical depth
      const d = vdbFrameSpectralShadowDepths(unit, 2.5);
      close3(d.y, unit.y.map((v) => v * 2.5));
    }
    // Absorption is clamped to the extinction
    expect(vdbFrameSpectralShadowStep([1, 1, 1], [2, 0, 1], 1).z).toEqual([0, 1, 0]);
  });

  it("transmittance and the isotropic total equal the grey ones channel by channel", () => {
    const e: V3 = [2, 1, 0.3];
    const unit = vdbFrameSpectralShadowUnit(e, [0.9, 0.95, 1], 0.42);
    for (const oct of [[0, 0, 0, 0] as V4, octaves]) {
      const t = vdbFrameSpectralShadowTransmittance(unit, 0.42, oct);
      const total = vdbFrameSpectralShadowTotal(t.y, oct, [1, 1, 1]);
      for (let c = 0; c < 3; c++) {
        const g = vdbFrameShadowTransmittance([unit.x[c]!, unit.y[c]!, unit.z[c]!], 0.42, oct);
        expect(t.x[c]).toBeCloseTo(g[0], 14);
        expect(t.y[c]).toBeCloseTo(g[1], 14);
        // White albedo: the spectral octaves are the grey ones
        expect(total[c]).toBeCloseTo(vdbFrameShadowTotal(g[1], oct), 14);
      }
    }
  });

  it("octave n of each channel gets albedoⁿ (one more scattering), an independent sum", () => {
    const tau: V3 = [0.5, 1, 2];
    const albedo: V3 = [0.9, 0.5, 1.2];
    const phase: V4 = [0.8, -0.3, 0.2, 0];
    const cos = 0.3;
    const light = vdbFrameSpectralOctaveLight(octaves, tau, albedo, phase, cos);
    for (let c = 0; c < 3; c++) {
      let sum = 0;
      const a = Math.min(albedo[c]!, 1);
      for (let n = 1; n <= octaves[0]; n++) {
        const g = (x: number) => x * octaves[3] ** n;
        const p = (1 - 0.2) * vdbFrameHenyeyGreenstein(cos, g(0.8)) + 0.2 * vdbFrameHenyeyGreenstein(cos, g(-0.3));
        sum += a ** n * octaves[2] ** n * p * Math.exp(-tau[c]! * octaves[1] ** n);
      }
      expect(light[c]).toBeCloseTo(sum, 13);
    }
    // White albedo: the grey octave light
    const white = vdbFrameSpectralOctaveLight(octaves, tau, [1, 1, 1], phase, cos);
    tau.forEach((t, c) => expect(white[c]).toBeCloseTo(vdbFrameOctaveLight(octaves, t, phase, cos), 14));
  });

  it("the shadowed light equals the grey per-channel light with white albedo", () => {
    const phase: V4 = [0.8, -0.3, 0.2, 0];
    const spectrum: V3 = [0.9, 1.1, 1];
    const ray: V3 = [0, 0, -1];
    const light: V3 = [0.6, 0, -0.8];
    for (const oct of [[0, 0, 0, 0] as V4, octaves]) {
      const shadow = { x: [0.3, 0.3, 0.3] as V3, y: [0.5, 0.5, 0.5] as V3 };
      const spectral = vdbFrameSpectralShadowedLight(phase, spectrum, oct, ray, light, shadow, [1, 1, 1]);
      close3(spectral, vdbFrameShadowedLightColor(phase, spectrum, oct, ray, light, [0.3, 0.5]), 13);
    }
    // Diffusion: per channel phase × direct + (total − direct)
    const s = vdbFrameSpectralShadowedLight([0, 0, 0, 0], [1, 1, 0], [0, 0, 0, 0], ray, light, { x: [0.1, 0.2, 0.3], y: [0.5, 0.4, 0.3] }, [1, 1, 1]);
    close3(s, [0.5, 0.4, 0.3], 14);
  });

  it("grid albedo is 1 − absorption / extinction (1 with no extinction)", () => {
    close3(vdbFrameSpectralGridAlbedo([2, 1, 0.5], [0.2, 0.5, 0.5]), [0.9, 0.5, 0], 14);
    expect(vdbFrameSpectralGridAlbedo([0, 0, 0], [0, 0, 0])).toEqual([1, 1, 1]);
  });
});

// The K3.3 grid layout of frame-grid.test.ts: 2 cascades of 8³ in a 32 × 32 atlas of 4 columns.
const layout: VDBFrameGridLayout = {
  count: 2,
  resolution: 8,
  columns: 4,
  atlasSize: [32, 32],
  gridMin: [
    [-2, -2, -2],
    [-4, -4, -4],
  ],
  gridSize: [
    [4, 4, 4],
    [8, 8, 8],
  ],
  gridVoxel: [
    [0.5, 0.5, 0.5],
    [1, 1, 1],
  ],
};

function atlasOf(channels: number, f: (world: V3) => number[]): VDBFrameAtlas {
  const data = new Float32Array(32 * 32 * channels);
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      const cell = vdbFrameGridCell(8, 4, [x + 0.5, y + 0.5]);
      if (cell.cascade >= 2) continue;
      const m = layout.gridMin[cell.cascade]!;
      const s = layout.gridSize[cell.cascade]!;
      const v = f([m[0] + cell.gridUvw[0] * s[0], m[1] + cell.gridUvw[1] * s[1], m[2] + cell.gridUvw[2] * s[2]]);
      for (let c = 0; c < channels; c++) data[(y * 32 + x) * channels + c] = v[c]!;
    }
  }
  return { data, width: 32, height: 32, channels };
}

describe("spectral grid passes", () => {
  // A ball of extinction 0.8 per unit, albedo 0.9 (grey), in the grid
  const sigma = (w: V3) => (Math.hypot(...w) < 1.5 ? 0.8 : 0);
  const albedo = 0.9;
  const grey = atlasOf(2, (w) => [sigma(w), vdbFrameShadowAbsorption(sigma(w), [albedo, albedo, albedo])]);
  const ext = atlasOf(4, (w) => [sigma(w), sigma(w), sigma(w), 0]);
  const abs = atlasOf(4, (w) => [sigma(w) * (1 - albedo), sigma(w) * (1 - albedo), sigma(w) * (1 - albedo), 0]);
  const lights: V4[] = [
    [0, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 3, 0, 2],
  ];
  const fragCoords: [number, number][] = [
    [4.5, 8 + 4.5],
    [3.5, 8 + 3.5],
    [5.5, 2.5],
  ];

  it("transmittance: with equal channels the slot groups hold the grey pass's values", () => {
    for (const ms of [0, 0.42]) {
      for (const twoSided of [false, true]) {
        for (const fc of fragCoords) {
          const g = vdbFrameTransmittanceVoxel(grey, layout, fc, lights, [0, 1, 0], ms, -1, { twoSided });
          const groups = [0, 1].map((group) => vdbFrameUnpackSpectralGroup(...vdbFrameSpectralTransmittanceVoxel(ext, abs, layout, fc, lights, [0, 1, 0], ms, -1, group, { twoSided })));
          // Group 0: sky up / down totals, slot 0 (total, direct)
          close3(groups[0]!.a.x, [g[0][0], g[0][0], g[0][0]], 6);
          close3(groups[0]!.a.y, [g[0][1], g[0][1], g[0][1]], 6);
          close3(groups[0]!.b.x, [g[0][2], g[0][2], g[0][2]], 6);
          close3(groups[0]!.b.y, [g[2][0], g[2][0], g[2][0]], 6);
          // Group 1: slots 1 and 2
          close3(groups[1]!.a.x, [g[0][3], g[0][3], g[0][3]], 6);
          close3(groups[1]!.a.y, [g[2][1], g[2][1], g[2][1]], 6);
          close3(groups[1]!.b.x, [g[1][0], g[1][0], g[1][0]], 6);
          close3(groups[1]!.b.y, [g[2][2], g[2][2], g[2][2]], 6);
        }
      }
    }
    // A slot past the light count stays 1 (group 3 has no slot 6)
    const g3 = vdbFrameUnpackSpectralGroup(...vdbFrameSpectralTransmittanceVoxel(ext, abs, layout, fragCoords[0]!, lights, [0, 1, 0], 0, -1, 3));
    expect([...g3.a.x, ...g3.a.y, ...g3.b.x, ...g3.b.y]).toEqual(Array(12).fill(1));
  });

  it("transmittance: a blue-heavy spectrum shadows blue more (independent Beer check of the direct light)", () => {
    const spectrum = vdbFrameExtinctionSpectrum([1, 1, 1], 1.8);
    const ext3 = atlasOf(4, (w) => [sigma(w) * spectrum[0], sigma(w) * spectrum[1], sigma(w) * spectrum[2], 0]);
    const abs3 = atlasOf(4, () => [0, 0, 0, 0]);
    const fc = fragCoords[0]!;
    const group = vdbFrameUnpackSpectralGroup(...vdbFrameSpectralTransmittanceVoxel(ext3, abs3, layout, fc, lights, [0, 1, 0], 0, -1, 0));
    const direct = group.b.y;
    expect(direct[2]).toBeLessThan(direct[1]);
    expect(direct[1]).toBeLessThan(direct[0]);
    // Beer: ln T of each channel is proportional to its extinction scale
    expect(Math.log(direct[2]) / Math.log(direct[1])).toBeCloseTo(spectrum[2] / spectrum[1], 6);
    expect(Math.log(direct[0]) / Math.log(direct[1])).toBeCloseTo(spectrum[0] / spectrum[1], 6);
  });

  it("environment grid: equal channels give the grey pass; diffusion pairs keep a white medium's light", () => {
    const envLights = [
      { direction: [0, 1, 0] as V3, color: [0.5, 0.5, 0.5] as V3 },
      { direction: [0, -1, 0] as V3, color: [0.5, 0.5, 0.5] as V3 },
    ];
    for (const ms of [0, 0.42]) {
      for (const fc of [[4.5, 4.5], [3.5, 2.5]] as [number, number][]) {
        // The environment grid atlas is the last cascade (one cascade of 8 tiles: 4 columns)
        const one = { ...layout, count: 1 };
        const g = vdbFrameEnvironmentGridVoxel(grey, one, 4, fc, envLights, [1, 1, 1], ms);
        const s = vdbFrameSpectralEnvironmentGridVoxel(ext, abs, one, 4, fc, envLights, [1, 1, 1], ms);
        close3(s, g, 6);
      }
    }
    const white = atlasOf(4, () => [0, 0, 0, 0]);
    const one = { ...layout, count: 1 };
    const s = vdbFrameSpectralEnvironmentGridVoxel(ext, white, one, 4, [4.5, 4.5], envLights, [1, 1, 1], 0.42);
    // White medium, an opposite pair: the two-sided totals sum to 2 (the white furnace argument)
    close3(s.slice(0, 3), [1, 1, 1], 12);
  });
});

describe("spectral froxel passes", () => {
  // One volume: the unit box at the origin, camera at z = 3 looking down −z, 4 × 4 × 4 froxels.
  const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 10);
  camera.position.set(0, 0, 3);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  const viewToLocal = new THREE.Matrix4().copy(camera.matrixWorld);
  const base = {
    froxelGrid: [4, 4, 4] as V3,
    columns: 2,
    range: [2, 4] as [number, number],
    projectionInverse: camera.projectionMatrixInverse.elements,
    cameraWorld: camera.matrixWorld.elements,
    viewToLocal: viewToLocal.elements,
    maxTexels: 16,
    mipLevels: 0,
    worldScale: 1,
    subSteps: 8,
    jitter: -1,
    fadeRadius: 0,
    opacity: 1,
    baseColor: [0.84, 0.86, 0.88] as V3,
    scatterColor: [0, 0, 0] as V3,
    extinctionAt: (p: V3) => 3 * (1 - Math.hypot(...p)),
  };
  const fc: [number, number] = [1.5, 1.5 + 4];

  it("density: spectrum (1, 1, 1) gives the grey pass's albedo, emission and extinction", () => {
    const grey = vdbFrameDensityFroxel(fc, { ...base, shadows: null, sceneLightCount: 0, lightPositions: [], phase: [0, 0, 0, 0] })!;
    const spectral = vdbFrameSpectralDensityFroxel(fc, { ...base, extinctionSpectrum: [1, 1, 1] })!;
    expect(grey).not.toBeNull();
    close3(spectral[0], grey[0], 13);
    close3(spectral[1], grey[1], 13);
    close3(spectral[2].slice(0, 3), [grey[0][3], grey[0][3], grey[0][3]], 13);
  });

  it("density: the weight is the channel mean; the visible position weighs the mean transmittance", () => {
    const spectrum = vdbFrameExtinctionSpectrum([1, 1, 1], 1.8);
    const s = vdbFrameSpectralDensityFroxel(fc, { ...base, extinctionSpectrum: spectrum })!;
    const grey = vdbFrameSpectralDensityFroxel(fc, { ...base, extinctionSpectrum: [1, 1, 1] })!;
    close3(s[2].slice(0, 3), spectrum.map((k) => k * grey[2][1]!), 12);
    expect(s[0][3]).toBeCloseTo((s[2][0] + s[2][1] + s[2][2]) / 3, 14);
    expect(s[1][3] / s[0][3]).not.toBeCloseTo(grey[1][3] / grey[0][3], 6);
  });

  it("lighting: ambient only gives albedo · ambient / π, and α of each channel from its extinction", () => {
    const density: [V4, V4, V4] = [
      [0.5 * 2, 0.6 * 1, 0.7 * 0.5, (2 + 1 + 0.5) / 3],
      [0.01 * 1.1, 0.02 * 1.1, 0.03 * 1.1, 0.4 * 1.1667],
      [2, 1, 0.5, 0],
    ];
    density[1] = [0.01 * density[0][3], 0.02 * density[0][3], 0.03 * density[0][3], 0.4 * density[0][3]];
    const near: V3 = [0, 0, -2];
    const far: V3 = [0, 0, -2.5];
    const [color, alpha] = vdbFrameSpectralLightingFroxel(density, near, far, new THREE.Matrix4().elements, {
      ambient: [1, 2, 3],
      hemiSky: [0, 0, 0],
      hemiGround: [0, 0, 0],
      lights: [],
      shadowLightCount: 0,
      phase: [0, 0, 0, 0],
      environmentColors: [],
      environmentShadowSlot: 0,
      groupsAt: null,
    });
    const a = [2, 1, 0.5].map((e) => 1 - Math.exp(-e * 0.5));
    const radiance = [0.5 * 1, 0.6 * 2, 0.7 * 3].map((v, c) => v / Math.PI + [0.01, 0.02, 0.03][c]!);
    close3(color.slice(0, 3), radiance.map((r, c) => r * a[c]!), 12);
    expect(color[3]).toBeCloseTo(a[1]!, 14);
    close3(alpha, [a[0]!, a[2]!, 0, a[1]!], 14);
  });

  it("inject: spectrum (1, 1, 1) gives the grey slice and march; the march depths are the unit × optical depth", () => {
    const depths = [0.1, 0.3, 0.05, 0.2];
    const grey = vdbFrameInjectSlice(depths, [0.1, 0.9], 0.5)!;
    const s = vdbFrameSpectralInjectSlice(depths, [1, 1, 1], [0.1, 0.9], 0.5)!;
    expect(s.t).toBeCloseTo(grey.t, 14);
    close3(s.alpha, [grey.alpha, grey.alpha, grey.alpha], 14);
    const g = 0.7;
    const gm = vdbFrameShadowMarch([0.4, 0.5, 0.2], [g, g, g], 0.42);
    const sm = vdbFrameSpectralShadowMarch([0.4, 0.5, 0.2], [1, 1, 1], [g, g, g], 0.42);
    close3(sm.x, [gm[0], gm[0], gm[0]], 13);
    close3(sm.y, [gm[1], gm[1], gm[1]], 13);
    close3(sm.z, [gm[2], gm[2], gm[2]], 13);
    close3(sm.w, [gm[3], gm[3], gm[3]], 13);
  });

  it("integrate: per-channel alpha composites each channel front to back", () => {
    const r = vdbFrameSpectralIntegrate([
      { color: [0.1, 0.2, 0.3], alpha: [0.5, 0.25, 0.1] },
      { color: [0.2, 0.2, 0.2], alpha: [0.5, 0.5, 0.5] },
    ]);
    close3(r.transmittance, [0.25, 0.375, 0.45], 14);
    close3(r.radiance, [0.1 + 0.5 * 0.2, 0.2 + 0.75 * 0.2, 0.3 + 0.9 * 0.2], 14);
  });
});

describe("'volume' lighting isotropic terms (the inject pass's ambient, probe, environment, hemisphere)", () => {
  const identity = new THREE.Matrix4().elements;
  // A slab-like volume: the reduced depth toward a direction depends on its z (asymmetric, so pairs differ)
  const whiteMarch = (spectrum: V3 = [1, 1, 1]) => (d: V3) => {
    const len = Math.hypot(...d);
    const z = d[2] / len;
    const od = 2 + 1.5 * z + 0.5 * (d[0] / len);
    const m = vdbFrameSpectralShadowMarch([od], spectrum, [1, 1, 1], 0.42);
    return m;
  };
  const sphere = Array.from({ length: 6 }, (_, i) => {
    const t = (i / 6) * Math.PI;
    return [Math.cos(t), Math.sin(t) * 0.6, Math.sin(t) * 0.8] as V3;
  });
  const antipodal = sphere.flatMap((d) => [
    { direction: d, color: [0.5, 0.5, 0.5] as V3 },
    { direction: [-d[0], -d[1], -d[2]] as V3, color: [0.5, 0.5, 0.5] as V3 },
  ]);
  const common = { lights: 0b111111, ambient: [0, 0, 0] as V3, worldToLocal: identity, outwardNormal: [0, 0, 1] as V3, multipleScattering: 0.42, albedo: [1, 1, 1] as V3 };

  it("white furnace: diffusion pairs of a white medium give back the environment (any march, both modes)", () => {
    for (const spectrum of [[1, 1, 1], vdbFrameExtinctionSpectrum([1, 1, 1], 1.8)] as V3[]) {
      for (const spectral of [false, true]) {
        const e = vdbFrameInjectIsotropicIrradiance({ ...common, environment: antipodal, spectral, march: whiteMarch(spectrum) });
        close3(e, [6, 6, 6], 12);
        // Hemisphere: sky and ground of equal colour are one pair
        const h = vdbFrameInjectIsotropicIrradiance({ ...common, hemisphere: [{ direction: [0.6, 0, 0.8], sky: [0.4, 0.4, 0.4], ground: [0.4, 0.4, 0.4] }], spectral, march: whiteMarch(spectrum) });
        close3(h, [0.4, 0.4, 0.4], 12);
        // 'outward' IBL with a uniform environment E: 0.5 E (x + y) = E
        const o = vdbFrameInjectIsotropicIrradiance({ ...common, environmentIrradiance: () => [Math.PI * 0.5, Math.PI * 0.5, Math.PI * 0.5], spectral, march: whiteMarch(spectrum) });
        close3(o, [Math.PI * 0.5, Math.PI * 0.5, Math.PI * 0.5], 12);
      }
    }
    // Single scattering loses light (the furnace fails without diffusion)
    const single = vdbFrameInjectIsotropicIrradiance({ ...common, multipleScattering: 0, environment: antipodal, spectral: false, march: (d) => vdbFrameSpectralShadowMarch([2 + Math.abs(d[2])], [1, 1, 1], [1, 1, 1], 0) });
    expect(single[0]).toBeLessThan(6 * 0.5);
  });

  it("the light bitmask gates each term; the probe is three's SH irradiance along the outward normal", () => {
    const sh = Array.from({ length: 9 }, (_, i) => [0.1 * (i + 1), 0.05 * i, 0.2] as V3);
    const march = whiteMarch();
    const all = vdbFrameInjectIsotropicIrradiance({ ...common, ambient: [1, 1, 1], lightProbe: sh, environment: antipodal, spectral: false, march });
    const ambientOnly = vdbFrameInjectIsotropicIrradiance({ ...common, lights: 0b100000, ambient: [1, 1, 1], lightProbe: sh, environment: antipodal, spectral: false, march });
    const probe = vdbShIrradianceAt([0, 0, 1], sh);
    close3(ambientOnly, probe.map((v) => v + 1), 14);
    close3(all, ambientOnly.map((v) => v + 6), 12);
    expect(vdbFrameInjectIsotropicIrradiance({ ...common, lights: 0b001000, ambient: [1, 1, 1], lightProbe: sh, environment: antipodal, spectral: false, march })).toEqual([0, 0, 0]);
  });

  it("vdbShIrradianceAt equals three's SphericalHarmonics3.getIrradianceAt", () => {
    const sh = new THREE.SphericalHarmonics3();
    sh.coefficients.forEach((v, i) => v.set(Math.sin(i + 1), Math.cos(i * 2), 0.1 * i));
    for (const n of [[0, 1, 0], [0.6, 0, 0.8], [-0.48, 0.6, -0.64]] as V3[]) {
      const ours = vdbShIrradianceAt(n, sh.coefficients.map((v) => v.toArray() as V3));
      close3(ours, sh.getIrradianceAt(new THREE.Vector3(...n), new THREE.Vector3()).toArray(), 5);
    }
    // A uniform radiance L projects to c0 = 4π · 0.282095 L; its irradiance is π L at every normal
    const c0 = 4 * Math.PI * 0.282095 * 0.5;
    close3(vdbShIrradianceAt([0.6, 0, 0.8], [[c0, c0, c0]]), [0.5 * Math.PI, 0.5 * Math.PI, 0.5 * Math.PI], 5);
  });

  it("isotropic diffusion pair sums to 2 for zero absorption (frameIsotropicDiffusion)", () => {
    for (const [f, b] of [[0.1, 3], [2, 2], [5, 0]] as [number, number][]) {
      expect(vdbFrameIsotropicDiffusion(f, b, 0) + vdbFrameIsotropicDiffusion(b, f, 0)).toBeCloseTo(2, 14);
    }
  });
});
