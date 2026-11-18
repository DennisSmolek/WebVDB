// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameSpectrum.js (all, 42 lines: wavelengths, extinction spectrum, slot groups);
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L483–622 (frameMeanComponent, frameSpectralShadowUnit,
//   frameSpectralShadowDepths, frameSpectralShadowStep, frameSpectralGridAlbedo, frameSpectralShadowTransmittance,
//   frameSpectralOctaveLight, frameSpectralShadowTotal, frameSpectralShadowedLight, frameUnpackSpectralGroup);
//   src/three-openvdb/frame/webgpu/FrameTransmittanceMaterial.js L278–459 (buildSpectralNodes),
//   FrameEnvironmentGridMaterial.js L104–197 (the spectral branch), FrameDensityMaterial.js L363–537
//   (buildSpectralNodes), FrameLightingMaterial.js L114–275 (createSpectralFragmentNode),
//   FrameInjectMaterial.js L200–532 (the spectral pre-integration and march; the light-probe, environment and
//   hemisphere terms in both modes), as CPU twins;
//   three r187 `getShIrradianceAt` (MIT, three.js authors), the light-probe irradiance the passes call.
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: f64 CPU twins of the TSL (test oracles of the `./three` spectral passes and of the 'volume'-lighting
// environment terms), engine-free (plain arrays). Divergences: DIVERGES.md (K3.5 rows).

import {
  vdbFrameDistanceFade,
  vdbFrameNoise,
  vdbFrameOctavesOff,
  vdbFramePhaseColor,
  vdbFramePhaseWeightScaled,
  vdbFrameShadowDepth,
  vdbFrameShadowOpticalDepth,
  vdbFrameShadowTotal,
  vdbFrameTwoSidedDiffusion,
  vdbFrameMaxScatteringOctaves,
  type Vec2,
  type Vec3,
  type Vec4,
} from "./index.js";
import {
  vdbFrameDensityFroxelSteps,
  vdbFrameFindCascade,
  vdbFrameGridCell,
  vdbFrameGridUvw,
  vdbFrameInsideGrid,
  vdbFrameIsotropicDiffusion,
  vdbFrameSampleGrid,
  vdbFrameSampleShadows,
  vdbFrameSampleTiles,
  type VDBFrameAtlas,
  type VDBFrameDensityInputs,
  type VDBFrameGridLayout,
  type VDBFrameShadowLight,
} from "./grid.js";


/** Wavelengths in nm of the red, green and blue channels. Green (550 nm) is the reference. */
export const vdbFrameSpectrumWavelengths = [650.0, 550.0, 450.0] as const;

/** Spectral slot groups (his `frameSpectralGroupCount`): group 0 is sky up, sky down and slot 0; group g is slots 2g − 1, 2g. */
export const vdbFrameSpectralGroupCount = 4;

/** Groups in use for the number of light slots in use (0 to 6): 1 + ⌈(slots − 1) / 2⌉. */
export const vdbFrameSpectralGroups = (slotCount: number) => 1 + Math.ceil(Math.max(0, slotCount - 1) / 2);

/** The extinction scale of each channel: scale × (λ / 550)^−α (his getFrameExtinctionSpectrum, engine-free; α a number). */
export function vdbFrameExtinctionSpectrum(scale: Vec3, angstromExponent: number): Vec3 {
  const ref = vdbFrameSpectrumWavelengths[1];
  return [0, 1, 2].map((c) => Math.max(0, scale[c]!) * Math.pow(vdbFrameSpectrumWavelengths[c]! / ref, -angstromExponent)) as Vec3;
}

/** Spectral shadow depths { x: extinction, y: effective extinction, z: reduced scattering } of each channel. */
export interface VDBFrameSpectralDepths {
  x: Vec3;
  y: Vec3;
  z: Vec3;
}

/** Spectral shadow values { x: direct, y: total } of each channel. */
export interface VDBFrameSpectralShadow {
  x: Vec3;
  y: Vec3;
}

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const map3 = (f: (c: number) => number): Vec3 => [f(0), f(1), f(2)];

/** frameMeanComponent: the mean of the channels. */
export const vdbFrameMeanComponent = (v: Vec3) => (v[0] + v[1] + v[2]) / 3;

/** frameSpectralShadowUnit: the depths of one unit of extinction for a volume's spectrum and albedo (frameShadowStep per channel). */
export function vdbFrameSpectralShadowUnit(spectrum: Vec3, albedo: Vec3, multipleScattering: number): VDBFrameSpectralDepths {
  const absorbed = map3((c) => spectrum[c]! * (1 - clamp01(albedo[c]!)));
  const reduced = map3((c) => (spectrum[c]! - absorbed[c]!) * multipleScattering);
  return { x: [...spectrum], y: map3((c) => Math.sqrt(absorbed[c]! * 3 * (absorbed[c]! + reduced[c]!))), z: reduced };
}

/** frameSpectralShadowDepths: one volume has one spectrum, so the depths are linear in the optical depth. */
export function vdbFrameSpectralShadowDepths(unit: VDBFrameSpectralDepths, opticalDepth: number): VDBFrameSpectralDepths {
  return { x: map3((c) => unit.x[c]! * opticalDepth), y: map3((c) => unit.y[c]! * opticalDepth), z: map3((c) => unit.z[c]! * opticalDepth) };
}

/** frameSpectralShadowStep: the step depths of each channel from the grid's (extinction, absorption) of each channel. */
export function vdbFrameSpectralShadowStep(extinction: Vec3, absorption: Vec3, multipleScattering: number): VDBFrameSpectralDepths {
  const absorbed = map3((c) => Math.min(absorption[c]!, extinction[c]!));
  const reduced = map3((c) => (extinction[c]! - absorbed[c]!) * multipleScattering);
  return { x: [...extinction], y: map3((c) => Math.sqrt(absorbed[c]! * 3 * (absorbed[c]! + reduced[c]!))), z: reduced };
}

/** frameSpectralGridAlbedo: 1 − absorption / extinction of each channel (clamped), 1 with no extinction. */
export function vdbFrameSpectralGridAlbedo(extinction: Vec3, absorption: Vec3): Vec3 {
  if (!(Math.max(...extinction) > 0)) return [1, 1, 1];
  return map3((c) => clamp01(1 - absorption[c]! / Math.max(extinction[c]!, 1e-8)));
}

/** frameShadowDepth of each channel. */
export function vdbFrameSpectralShadowDepth(depths: VDBFrameSpectralDepths, multipleScattering: number): Vec3 {
  return map3((c) => vdbFrameShadowDepth([depths.x[c]!, depths.y[c]!, depths.z[c]!], multipleScattering));
}

/** frameShadowMarchDepth of each channel (the marches stop when the smallest is past the limit). */
export function vdbFrameSpectralShadowMarchDepth(depths: VDBFrameSpectralDepths, multipleScattering: number, octaves: Vec4 = vdbFrameOctavesOff): Vec3 {
  if (octaves[0] > 0) return map3((c) => depths.x[c]! * Math.pow(octaves[1], octaves[0]));
  return vdbFrameSpectralShadowDepth(depths, multipleScattering);
}

/** frameSpectralShadowTransmittance: (direct, total) of each channel; octaves: total is 1 / (1 + τ). */
export function vdbFrameSpectralShadowTransmittance(depths: VDBFrameSpectralDepths, multipleScattering: number, octaves: Vec4 = vdbFrameOctavesOff): VDBFrameSpectralShadow {
  const d = vdbFrameSpectralShadowDepth(depths, multipleScattering);
  return { x: map3((c) => Math.exp(-depths.x[c]!)), y: map3((c) => (octaves[0] > 0 ? 1 / (depths.x[c]! + 1) : Math.exp(-d[c]!))) };
}

/** frameSpectralOctaveLight: Σₙ albedoⁿ bⁿ phase(g cⁿ) exp(−aⁿ τ) of each channel (each octave is one more scattering). */
export function vdbFrameSpectralOctaveLight(octaves: Vec4, opticalDepth: Vec3, albedo: Vec3, phase: Vec4 | null, cosTheta: number): Vec3 {
  const light: Vec3 = [0, 0, 0];
  const a = map3((c) => clamp01(albedo[c]!));
  const albedoScale: Vec3 = [1, 1, 1];
  let extinctionScale = 1;
  let contribution = 1;
  let gScale = 1;
  for (let i = 0; i < vdbFrameMaxScatteringOctaves; i++) {
    if (i + 1 > octaves[0]) break;
    extinctionScale *= octaves[1];
    contribution *= octaves[2];
    gScale *= octaves[3];
    for (let c = 0; c < 3; c++) albedoScale[c] = albedoScale[c]! * a[c]!;
    const p = phase ? vdbFramePhaseWeightScaled(phase, cosTheta, gScale) : 1;
    for (let c = 0; c < 3; c++) light[c] = light[c]! + albedoScale[c]! * (contribution * p) * Math.exp(-opticalDepth[c]! * extinctionScale);
  }
  return light;
}

/** frameSpectralShadowTotal: total transmittance of isotropic light of each channel from the y shadow values. */
export function vdbFrameSpectralShadowTotal(shadowY: Vec3, octaves: Vec4, albedo: Vec3): Vec3 {
  if (!(octaves[0] > 0)) return [...shadowY];
  const tau = map3((c) => vdbFrameShadowOpticalDepth(shadowY[c]!));
  const octave = vdbFrameSpectralOctaveLight(octaves, tau, albedo, null, 0);
  return map3((c) => Math.exp(-tau[c]!) + octave[c]!);
}

/** frameSpectralShadowedLight: phase × direct + scattered of each channel (octaves: the spectral octave light). */
export function vdbFrameSpectralShadowedLight(phase: Vec4, phaseSpectrum: Vec3, octaves: Vec4, rayDirection: Vec3, lightDirection: Vec3, shadow: VDBFrameSpectralShadow, albedo: Vec3): Vec3 {
  const p = vdbFramePhaseColor(phase, phaseSpectrum, rayDirection, lightDirection);
  const cos = rayDirection[0] * lightDirection[0] + rayDirection[1] * lightDirection[1] + rayDirection[2] * lightDirection[2];
  const scattered =
    octaves[0] > 0
      ? vdbFrameSpectralOctaveLight(
          octaves,
          map3((c) => vdbFrameShadowOpticalDepth(shadow.y[c]!)),
          albedo,
          phase,
          cos,
        )
      : map3((c) => shadow.y[c]! - shadow.x[c]!);
  return map3((c) => shadow.x[c]! * p[c]! + scattered[c]!);
}

/** frameUnpackSpectralGroup: the two entries of a slot group from its 3 RGBA targets. */
export function vdbFrameUnpackSpectralGroup(map0: ArrayLike<number>, map1: ArrayLike<number>, map2: ArrayLike<number>): { a: VDBFrameSpectralShadow; b: VDBFrameSpectralShadow } {
  return {
    a: { x: [map0[0]!, map0[1]!, map0[2]!], y: [map0[3]!, map1[0]!, map1[1]!] },
    b: { x: [map1[2]!, map1[3]!, map2[0]!], y: [map2[1]!, map2[2]!, map2[3]!] },
  };
}

/** The inverse of {@link vdbFrameUnpackSpectralGroup}: the 3 targets of a group. */
export function vdbFramePackSpectralGroup(a: VDBFrameSpectralShadow, b: VDBFrameSpectralShadow): [Vec4, Vec4, Vec4] {
  return [
    [a.x[0], a.x[1], a.x[2], a.y[0]],
    [a.y[1], a.y[2], b.x[0], b.x[1]],
    [b.x[2], b.y[0], b.y[1], b.y[2]],
  ];
}

/** frameIsotropicDiffusion of each channel. */
const isotropic3 = (front: Vec3, back: Vec3, effective: Vec3): Vec3 => map3((c) => vdbFrameIsotropicDiffusion(front[c]!, back[c]!, effective[c]!));

/** frameTwoSidedDiffusion of each channel. */
const twoSided3 = (direct: Vec3, front: Vec3, effective: Vec3, back: Vec3): Vec3 => map3((c) => vdbFrameTwoSidedDiffusion(direct[c]!, front[c]!, effective[c]!, back[c]!));

/** A spectral march result: direct, total, reduced depth and effective depth of each channel. */
export interface VDBFrameSpectralMarch {
  x: Vec3;
  y: Vec3;
  z: Vec3;
  w: Vec3;
}

const unmarched = (): VDBFrameSpectralMarch => ({ x: [1, 1, 1], y: [1, 1, 1], z: [0, 0, 0], w: [0, 0, 0] });

/** The spectral grid sample of the transmittance pass: (extinction, absorption) of each channel, 0 outside the grid. */
export function vdbFrameSpectralGridSample(extinction: VDBFrameAtlas, absorption: VDBFrameAtlas, grid: VDBFrameGridLayout, world: Vec3): { extinction: Vec3; absorption: Vec3 } {
  const found = vdbFrameFindCascade(grid, world, 0);
  if (found < 0) return { extinction: [0, 0, 0], absorption: [0, 0, 0] };
  const uvw = vdbFrameGridUvw(grid, found, world);
  const e = vdbFrameSampleGrid(extinction, grid, found, uvw);
  const a = vdbFrameSampleGrid(absorption, grid, found, uvw);
  return { extinction: [e[0]!, e[1]!, e[2]!], absorption: [a[0]!, a[1]!, a[2]!] };
}

/** The spectral `march` of the transmittance pass: the grey march with the depths of each channel; stops when all channels are past depth 6. */
export function vdbFrameSpectralGridMarch(
  sampleAt: (world: Vec3) => { extinction: Vec3; absorption: Vec3 },
  grid: VDBFrameGridLayout,
  origin: Vec3,
  direction: Vec3,
  maxDistance: number,
  jitter: number,
  multipleScattering: number,
  octaves: Vec4 = vdbFrameOctavesOff,
): VDBFrameSpectralMarch {
  const depths: VDBFrameSpectralDepths = { x: [0, 0, 0], y: [0, 0, 0], z: [0, 0, 0] };
  let traveled = 0;
  const probe: Vec3 = [...origin];
  for (let i = 0; i < 1024; i++) {
    const cascade = vdbFrameFindCascade(grid, probe, 0);
    if (cascade < 0 || traveled >= maxDistance || Math.min(...vdbFrameSpectralShadowMarchDepth(depths, multipleScattering, octaves)) > 6) break;
    const voxel = grid.gridVoxel[cascade]!;
    const perUnit = Math.max(Math.abs(direction[0]) / voxel[0], Math.abs(direction[1]) / voxel[1], Math.abs(direction[2]) / voxel[2]);
    const step = Math.min(1 / perUnit, maxDistance - traveled);
    const s = sampleAt([probe[0] + direction[0] * step * jitter, probe[1] + direction[1] * step * jitter, probe[2] + direction[2] * step * jitter]);
    const d = vdbFrameSpectralShadowStep(s.extinction, s.absorption, multipleScattering);
    for (let c = 0; c < 3; c++) {
      depths.x[c] = depths.x[c]! + d.x[c]! * step;
      depths.y[c] = depths.y[c]! + d.y[c]! * step;
      depths.z[c] = depths.z[c]! + d.z[c]! * step;
      probe[c] = probe[c]! + direction[c]! * step;
    }
    traveled += step;
  }
  const t = vdbFrameSpectralShadowTransmittance(depths, multipleScattering, octaves);
  return { x: t.x, y: t.y, z: depths.z, w: depths.y };
}

const normalize3 = (v: Vec3): Vec3 => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
};

/**
 * The spectral transmittance pass at one grid voxel for one slot group (buildSpectralNodes): the 3 MRT outputs.
 * Group 0: (sky up, sky down) and slot 0; group g: slots 2g − 1 and 2g; a slot entry is (total, direct). Voxels
 * with no extinction within half a voxel keep 1.0. `twoSided`: the slot marches get the slab diffusion.
 */
export function vdbFrameSpectralTransmittanceVoxel(
  extinction: VDBFrameAtlas,
  absorption: VDBFrameAtlas,
  grid: VDBFrameGridLayout,
  fragCoord: Vec2,
  lights: readonly VDBFrameShadowLight[],
  skyDirection: Vec3,
  multipleScattering: number,
  jitterValue: number,
  group: number,
  modes: { octaves?: Vec4; twoSided?: boolean } = {},
): [Vec4, Vec4, Vec4] {
  const octaves = modes.octaves ?? vdbFrameOctavesOff;
  const ones: [Vec4, Vec4, Vec4] = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ];
  const cell = vdbFrameGridCell(grid.resolution, grid.columns, fragCoord);
  if (cell.cascade >= grid.count) return ones;
  const voxel = grid.gridVoxel[cell.cascade]!;
  const m = grid.gridMin[cell.cascade]!;
  const s = grid.gridSize[cell.cascade]!;
  const world: Vec3 = [m[0] + cell.gridUvw[0] * s[0], m[1] + cell.gridUvw[1] * s[1], m[2] + cell.gridUvw[2] * s[2]];
  const sampleAt = (p: Vec3) => vdbFrameSpectralGridSample(extinction, absorption, grid, p);
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    const e = sampleAt([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]]).extinction;
    near += e[0] + e[1] + e[2];
  }
  if (!(near > 0)) return ones;
  const jitter = vdbFrameNoise(jitterValue, fragCoord);
  const sky = normalize3(skyDirection);
  const march = (enabled: boolean, dir: Vec3, maxDistance: number) => (enabled ? vdbFrameSpectralGridMarch(sampleAt, grid, world, dir, maxDistance, jitter, multipleScattering, octaves) : unmarched());
  const marchSlot = (enabled: boolean, slot: number): VDBFrameSpectralMarch => {
    const active = enabled && slot < lights.length;
    const l = lights[Math.min(slot, 5)] ?? ([0, 1, 0, 1] as Vec4);
    const isDirection = l[3] < 1.5;
    const toLight: Vec3 = [l[0] - world[0], l[1] - world[1], l[2] - world[2]];
    const dir = isDirection ? normalize3([l[0], l[1], l[2]]) : normalize3(toLight);
    const front = march(active, dir, isDirection ? 1e20 : Math.hypot(...toLight));
    if (modes.twoSided) {
      const back = march(active, [-dir[0], -dir[1], -dir[2]], 1e20);
      front.y = twoSided3(front.x, front.z, front.w, back.z);
    }
    return front;
  };
  const isSky = group === 0;
  const slotA = group * 2 - 1;
  const first = march(isSky, sky, 1e20);
  const second = march(isSky, [-sky[0], -sky[1], -sky[2]], 1e20);
  const slotFirst = marchSlot(!isSky, Math.max(slotA, 0));
  const slotSecond = marchSlot(true, isSky ? 0 : slotA + 1);
  const skyTotal = (f: VDBFrameSpectralMarch, b: VDBFrameSpectralMarch) => (multipleScattering > 0 ? isotropic3(f.z, b.z, f.w) : f.y);
  const a: VDBFrameSpectralShadow = isSky ? { x: skyTotal(first, second), y: skyTotal(second, first) } : { x: slotFirst.y, y: slotFirst.x };
  const b: VDBFrameSpectralShadow = { x: slotSecond.y, y: slotSecond.x };
  return vdbFramePackSpectralGroup(a, b);
}

/**
 * The spectral environment grid pass at one voxel of the last cascade (FrameEnvironmentGridMaterial, spectral
 * branch): the march of each channel through the grid's extinction and absorption atlases, the voxel's albedo
 * for the octaves; diffusion: the two-sided total of each channel.
 */
export function vdbFrameSpectralEnvironmentGridVoxel(
  extinction: VDBFrameAtlas,
  absorption: VDBFrameAtlas,
  grid: VDBFrameGridLayout,
  envColumns: number,
  fragCoord: Vec2,
  lights: readonly { direction: Vec3; color: Vec3 }[],
  unshadowed: Vec3,
  multipleScattering: number,
  octaves: Vec4 = vdbFrameOctavesOff,
): Vec4 {
  const cascade = grid.count - 1;
  const res = grid.resolution;
  const tile: Vec2 = [Math.floor(fragCoord[0] / res), Math.floor(fragCoord[1] / res)];
  const z = tile[1] * envColumns + tile[0];
  const uvw0: Vec3 = [(fragCoord[0] - tile[0] * res) / res, (fragCoord[1] - tile[1] * res) / res, (z + 0.5) / res];
  const voxel = grid.gridVoxel[cascade]!;
  const m = grid.gridMin[cascade]!;
  const s = grid.gridSize[cascade]!;
  const world: Vec3 = [m[0] + uvw0[0] * s[0], m[1] + uvw0[1] * s[1], m[2] + uvw0[2] * s[2]];
  const uvwOf = (p: Vec3): Vec3 => [(p[0] - m[0]) / s[0], (p[1] - m[1]) / s[1], (p[2] - m[2]) / s[2]];
  const sampleE = (uvw: Vec3) => vdbFrameSampleGrid(extinction, grid, cascade, uvw);
  const sampleA = (uvw: Vec3) => vdbFrameSampleGrid(absorption, grid, cascade, uvw);
  let near = 0;
  for (let i = 0; i < 8; i++) {
    const c: Vec3 = [(i & 1) - 0.5, ((i >> 1) & 1) - 0.5, ((i >> 2) & 1) - 0.5];
    const e = sampleE(uvwOf([world[0] + c[0] * voxel[0], world[1] + c[1] * voxel[1], world[2] + c[2] * voxel[2]]));
    near += e[0]! + e[1]! + e[2]!;
  }
  const outside = vdbFrameSpectralShadowTotal([1, 1, 1], octaves, [1, 1, 1]);
  if (!(z < res && near > 0)) return [unshadowed[0] * outside[0], unshadowed[1] * outside[1], unshadowed[2] * outside[2], 1];
  const e0 = sampleE(uvw0);
  const a0 = sampleA(uvw0);
  const albedo = vdbFrameSpectralGridAlbedo([e0[0]!, e0[1]!, e0[2]!], [a0[0]!, a0[1]!, a0[2]!]);
  const marchDepths = (dir: Vec3): VDBFrameSpectralDepths => {
    const perUnit = Math.max(Math.abs(dir[0]) / voxel[0], Math.abs(dir[1]) / voxel[1], Math.abs(dir[2]) / voxel[2]);
    const step = 1 / perUnit;
    const probe: Vec3 = [world[0] + dir[0] * step * 0.5, world[1] + dir[1] * step * 0.5, world[2] + dir[2] * step * 0.5];
    const depths: VDBFrameSpectralDepths = { x: [0, 0, 0], y: [0, 0, 0], z: [0, 0, 0] };
    for (let i = 0; i < 1024; i++) {
      const uvw = uvwOf(probe);
      if (uvw.some((v) => v < 0 || v > 1) || Math.min(...vdbFrameSpectralShadowMarchDepth(depths, multipleScattering, octaves)) > 6) break;
      const e = sampleE(uvw);
      const a = sampleA(uvw);
      const d = vdbFrameSpectralShadowStep([e[0]!, e[1]!, e[2]!], [a[0]!, a[1]!, a[2]!], multipleScattering);
      for (let c = 0; c < 3; c++) {
        depths.x[c] = depths.x[c]! + d.x[c]! * step;
        depths.y[c] = depths.y[c]! + d.y[c]! * step;
        depths.z[c] = depths.z[c]! + d.z[c]! * step;
        probe[c] = probe[c]! + dir[c]! * step;
      }
    }
    return depths;
  };
  const irradiance: Vec3 = [0, 0, 0];
  for (const light of lights) {
    const depths = marchDepths(light.direction);
    const total =
      multipleScattering > 0
        ? isotropic3(depths.z, marchDepths([-light.direction[0], -light.direction[1], -light.direction[2]]).z, depths.y)
        : vdbFrameSpectralShadowTotal(vdbFrameSpectralShadowTransmittance(depths, multipleScattering, octaves).y, octaves, albedo);
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + light.color[c]! * total[c]!;
  }
  return [Math.max(irradiance[0], 0), Math.max(irradiance[1], 0), Math.max(irradiance[2], 0), 1];
}

/** The inputs of {@link vdbFrameSpectralDensityFroxel}: the grey density inputs, the volume's extinction spectrum, the sky maps. */
export interface VDBFrameSpectralDensityInputs extends Omit<VDBFrameDensityInputs, "shadows" | "sceneLightCount" | "lightPositions" | "phase" | "phaseSpectrum"> {
  /** The volume's extinction scale of each channel (`frameExtinctionSpectrum`). */
  extinctionSpectrum: Vec3;
  /** The grid layout (environment grid and sky maps), or null without shadows. */
  grid?: VDBFrameGridLayout | null;
}

/**
 * The spectral density pass at one froxel of one volume (buildSpectralNodes, environment 'grid' or none): the
 * pre-integration with the transmittance of each channel (the visible position weighs the channel mean), and
 * the 3 MRT outputs (albedo × extinction, weight), (emission × weight, t × weight), (extinction, 0), or null.
 */
export function vdbFrameSpectralDensityFroxel(fragCoord: Vec2, p: VDBFrameSpectralDensityInputs): [Vec4, Vec4, Vec4] | null {
  const f = vdbFrameDensityFroxelSteps(fragCoord, p);
  if (!f) return null;
  const { viewNear, viewFar, localNear, localFar, t0, segment, steps, subStepLength, lod, jitter, mul } = f;
  const lerp3 = (x: Vec3, y: Vec3, t: number): Vec3 => [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, x[2] + (y[2] - x[2]) * t];
  const spectrum = p.extinctionSpectrum;
  let extinctionSum = 0;
  const transmittance: Vec3 = [1, 1, 1];
  let visibleWeight = 0;
  let visibleTSum = 0;
  for (let i = 0; i < steps; i++) {
    const t = t0 + segment * ((i + jitter) / steps);
    const e = p.extinctionAt(lerp3(localNear, localFar, t), lod) * p.opacity * vdbFrameDistanceFade(p.fadeRadius, lerp3(viewNear, viewFar, t));
    const stepT = map3((c) => Math.exp(-spectrum[c]! * e * subStepLength));
    const w = vdbFrameMeanComponent(map3((c) => transmittance[c]! * (1 - stepT[c]!)));
    visibleWeight += w;
    visibleTSum += w * t;
    extinctionSum += e;
    for (let c = 0; c < 3; c++) transmittance[c] = transmittance[c]! * stepT[c]!;
  }
  const extinction = map3((c) => spectrum[c]! * ((extinctionSum / steps) * segment));
  const weight = vdbFrameMeanComponent(extinction);
  if (!(weight > 0)) return null;
  const visibleT = visibleWeight > 0 ? visibleTSum / visibleWeight : 0.5;
  const visibleWorld = mul(p.cameraWorld, lerp3(viewNear, viewFar, visibleT));
  const emission: Vec3 = [...p.scatterColor];
  const albedo = p.baseColor;
  if (p.environmentGrid && p.grid) {
    const g = p.grid;
    const uvw = vdbFrameGridUvw(g, g.count - 1, visibleWorld);
    const outside = vdbFrameSpectralShadowTotal([1, 1, 1], p.octaves ?? vdbFrameOctavesOff, albedo);
    const env = vdbFrameInsideGrid(g.resolution, uvw) ? vdbFrameSampleTiles(p.environmentGrid.atlas, g.resolution, 0, p.environmentGrid.columns, uvw) : map3((c) => p.environmentGrid!.unshadowed[c]! * outside[c]!);
    for (let c = 0; c < 3; c++) emission[c] = emission[c]! + (albedo[c]! * env[c]!) / Math.PI;
  }
  return [
    [albedo[0] * extinction[0], albedo[1] * extinction[1], albedo[2] * extinction[2], weight],
    [emission[0] * weight, emission[1] * weight, emission[2] * weight, visibleT * weight],
    [extinction[0], extinction[1], extinction[2], 0],
  ];
}

/** The lights the spectral lighting pass reads (light probe as 9 SH coefficients, optional). */
export interface VDBFrameSpectralLightingInputs {
  ambient: Vec3;
  hemiSky: Vec3;
  hemiGround: Vec3;
  skyDirection?: Vec3;
  lightProbe?: readonly Vec3[];
  lights: { position: Vec4; color: Vec3; spot?: Vec4; penumbra?: number; falloff?: Vec2 }[];
  shadowLightCount: number;
  phase: Vec4;
  phaseSpectrum?: Vec3;
  octaves?: Vec4;
  environmentColors: Vec3[];
  environmentShadowSlot: number;
  /** The 3 targets of each slot group in use, sampled at a world position (sampleShadows), or null without shadows. */
  groupsAt: ((world: Vec3) => number[][][]) | null;
}

/** three's `getShIrradianceAt` (L2 SH irradiance of a light probe at a unit normal). */
export function vdbShIrradianceAt(normal: Vec3, sh: readonly Vec3[]): Vec3 {
  const [x, y, z] = normal;
  const k = [0.886227, 2 * 0.511664 * y, 2 * 0.511664 * z, 2 * 0.511664 * x, 2 * 0.429043 * x * y, 2 * 0.429043 * y * z, z * z * 0.743125 - 0.247708, 2 * 0.429043 * x * z, 0.429043 * (x * x - y * y)];
  return map3((c) => k.reduce((sum, w, i) => sum + (sh[i]?.[c] ?? 0) * w, 0));
}

const smoothstep = (e0: number, e1: number, x: number) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
};

const distanceAttenuation = (d: number, cutoff: number, decay: number) => {
  const falloff = 1 / Math.max(Math.pow(d, decay), 0.01);
  return cutoff > 0 ? falloff * Math.pow(Math.min(Math.max(1 - Math.pow(d / cutoff, 4), 0), 1), 2) : falloff;
};

/**
 * The spectral lighting pass at one froxel (createSpectralFragmentNode): from the 3 density targets, the slot
 * groups sampled at the visible position. Returns the two atlas outputs (radiance × α of each channel, green α)
 * and (red α, blue α, 0, green α), or zeros when the weight is ≤ 0 (the pass discards; the target is cleared).
 */
export function vdbFrameSpectralLightingFroxel(
  density: readonly [Vec4, Vec4, Vec4],
  viewNear: Vec3,
  viewFar: Vec3,
  cameraWorld: ArrayLike<number>,
  inputs: VDBFrameSpectralLightingInputs,
): [Vec4, Vec4] {
  const [albedoT, emissionT, extinctionT] = density;
  const weight = albedoT[3];
  if (!(weight > 0)) return [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  const m = cameraWorld;
  const extinction: Vec3 = [extinctionT[0], extinctionT[1], extinctionT[2]];
  const visibleT = Math.min(Math.max(emissionT[3] / weight, 0), 1);
  const view: Vec3 = map3((c) => viewNear[c]! + (viewFar[c]! - viewNear[c]!) * visibleT);
  const world: Vec3 = map3((r) => m[r]! * view[0] + m[4 + r]! * view[1] + m[8 + r]! * view[2] + m[12 + r]!);
  const seg: Vec3 = map3((c) => viewFar[c]! - viewNear[c]!);
  const ray = normalize3(map3((r) => m[r]! * seg[0] + m[4 + r]! * seg[1] + m[8 + r]! * seg[2]));
  const albedo = map3((c) => albedoT[c]! / Math.max(extinction[c]!, 1e-8));
  const emission = map3((c) => emissionT[c]! / weight);
  const octaves = inputs.octaves ?? vdbFrameOctavesOff;
  const ones = [1, 1, 1, 1];
  const groupMaps = inputs.groupsAt ? inputs.groupsAt(world) : [];
  const groups = [0, 1, 2, 3].map((g) => {
    const maps = groupMaps[g] ?? [ones, ones, ones];
    return vdbFrameUnpackSpectralGroup(maps[0]!, maps[1]!, maps[2]!);
  });
  const slotEntry = (slot: number) => (slot === 0 ? groups[0]!.b : slot % 2 === 1 ? groups[(slot + 1) / 2]!.a : groups[slot / 2]!.b);
  const skyUp = vdbFrameSpectralShadowTotal(groups[0]!.a.x, octaves, albedo).map((v) => v * 0.5) as Vec3;
  const skyDown = vdbFrameSpectralShadowTotal(groups[0]!.a.y, octaves, albedo).map((v) => v * 0.5) as Vec3;
  const sky = normalize3(inputs.skyDirection ?? [0, 1, 0]);
  const probeUp = inputs.lightProbe ? vdbShIrradianceAt(sky, inputs.lightProbe) : [0, 0, 0];
  const probeDown = inputs.lightProbe ? vdbShIrradianceAt([-sky[0], -sky[1], -sky[2]], inputs.lightProbe) : [0, 0, 0];
  const irradiance: Vec3 = map3((c) => inputs.ambient[c]! + probeUp[c]! * skyUp[c]! + probeDown[c]! * skyDown[c]! + inputs.hemiSky[c]! * skyUp[c]! + inputs.hemiGround[c]! * skyDown[c]!);
  const spectrum = inputs.phaseSpectrum ?? [1, 1, 0];
  inputs.lights.forEach((light, i) => {
    const p = light.position;
    const toLight: Vec3 = [p[0] - world[0], p[1] - world[1], p[2] - world[2]];
    const dir = p[3] < 1.5 ? ([p[0], p[1], p[2]] as Vec3) : normalize3(toLight);
    const nl = normalize3(toLight);
    const spot = p[3] > 2.5 && light.spot ? smoothstep(light.spot[3], light.penumbra ?? 1, nl[0] * light.spot[0] + nl[1] * light.spot[1] + nl[2] * light.spot[2]) : 1;
    const dist = p[3] > 1.5 ? distanceAttenuation(Math.hypot(...toLight), light.falloff?.[0] ?? 0, light.falloff?.[1] ?? 0) : 1;
    const useSlot = i < 6 && i < inputs.shadowLightCount;
    const entry = useSlot ? slotEntry(i) : null;
    const shadow: VDBFrameSpectralShadow = entry ? { x: entry.y, y: entry.x } : { x: [1, 1, 1], y: [1, 1, 1] };
    const l = vdbFrameSpectralShadowedLight(inputs.phase, spectrum, octaves, ray, dir, shadow, albedo);
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + light.color[c]! * l[c]! * spot * dist;
  });
  inputs.environmentColors.forEach((color, i) => {
    const slot = inputs.environmentShadowSlot + i;
    const total = slot < 6 ? slotEntry(slot).x : ([1, 1, 1] as Vec3);
    const t = vdbFrameSpectralShadowTotal(total, octaves, albedo);
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + color[c]! * t[c]!;
  });
  const length = Math.hypot(...seg);
  const alpha = map3((c) => 1 - Math.exp(-extinction[c]! * length));
  const radiance = map3((c) => albedo[c]! * irradiance[c]! * (1 / Math.PI) + emission[c]!);
  return [
    [radiance[0] * alpha[0], radiance[1] * alpha[1], radiance[2] * alpha[2], alpha[1]],
    [alpha[0], alpha[2], 0, alpha[1]],
  ];
}

/** The spectral transmittance group maps at a world position (sampleShadows of each group's 3 targets), for {@link vdbFrameSpectralLightingFroxel}. */
export function vdbFrameSpectralGroupsAt(groupAtlases: readonly (readonly VDBFrameAtlas[])[], grid: VDBFrameGridLayout): (world: Vec3) => number[][][] {
  return (world) => groupAtlases.map((atlases) => vdbFrameSampleShadows(atlases, grid, world));
}

// ── 'volume' lighting: the inject pass's isotropic terms (ambient + probe, environment, hemisphere) ──────

/** A march of the inject pass (`volumeTransmittance` or its spectral twin) from the lit position: (direct, total, reduced, effective). */
export type VDBFrameInjectMarch = (localDirection: Vec3, maxLocalDistance: number) => { x: Vec3; y: Vec3; z: Vec3; w: Vec3 };

/** What the inject pass's light terms read: the light uniforms, the volume's light bitmask and the march. */
export interface VDBFrameInjectLightInputs {
  /** The volume's `lights` bitmask (vdbFrameLights). */
  lights: number;
  ambient: Vec3;
  /** The light probe SH (9 coefficients). */
  lightProbe?: readonly Vec3[];
  /** Environment lights (world directions to the light, isotropic colours). */
  environment?: readonly { direction: Vec3; color: Vec3 }[];
  /** The 'outward' environment: three's IBL irradiance at a world direction (used when there are no environment lights). */
  environmentIrradiance?: ((direction: Vec3) => Vec3) | null;
  /** Hemisphere lights (world up direction, sky and ground colours). */
  hemisphere?: readonly { direction: Vec3; sky: Vec3; ground: Vec3 }[];
  /** World → local direction (the volume's worldToLocal 3 × 3, column-major 4 × 4 elements). */
  worldToLocal: ArrayLike<number>;
  /** The outward normal (world, unit), from the volume centre to the lit position. */
  outwardNormal: Vec3;
  /** multipleScattering value (> 0: diffusion pairs), octaves, the lit albedo (octaves of each channel). */
  multipleScattering: number;
  octaves?: Vec4;
  albedo: Vec3;
  /** Spectral totals (frameSpectralShadowTotal) or grey (frameShadowTotal of the green total, all channels). */
  spectral: boolean;
  march: VDBFrameInjectMarch;
}

const lightFlags = { ambient: 0b100000, hemisphere: 0b000010, environment: 0b000001 };

/**
 * The isotropic light terms of the inject pass ('volume' lighting) at the lit position: ambient + light probe
 * along the outward normal; the environment lights (one march each; with diffusion the two-sided total of the
 * march pair), or the 'outward' IBL irradiance with its march pair; the hemisphere lights (sky up and ground
 * down, a pair). Returns the irradiance before the scene lights. Grey: the march's channels are equal.
 */
export function vdbFrameInjectIsotropicIrradiance(p: VDBFrameInjectLightInputs): Vec3 {
  const octaves = p.octaves ?? vdbFrameOctavesOff;
  const diffusion = p.multipleScattering > 0;
  const m = p.worldToLocal;
  const toLocal = (d: Vec3): Vec3 => map3((r) => m[r]! * d[0] + m[4 + r]! * d[1] + m[8 + r]! * d[2]);
  const total = (y: Vec3): Vec3 => (p.spectral ? vdbFrameSpectralShadowTotal(y, octaves, p.albedo) : map3((c) => vdbFrameShadowTotal(y[c]!, octaves)));
  const pair = (localDirection: Vec3) => {
    const front = p.march(localDirection, 1e6);
    const back = p.march([-localDirection[0], -localDirection[1], -localDirection[2]], 1e6);
    return { x: isotropic3(front.z, back.z, front.w), y: isotropic3(back.z, front.z, back.w) };
  };
  const isotropic = (localDirection: Vec3): Vec3 => (diffusion ? pair(localDirection).x : total(p.march(localDirection, 1e6).y));
  const irradiance: Vec3 = [0, 0, 0];
  const add = (v: Vec3, s: Vec3 | number = 1) => {
    for (let c = 0; c < 3; c++) irradiance[c] = irradiance[c]! + v[c]! * (typeof s === "number" ? s : s[c]!);
  };
  if (p.lights & lightFlags.ambient) {
    add(p.ambient);
    if (p.lightProbe) add(vdbShIrradianceAt(p.outwardNormal, p.lightProbe));
  }
  if (p.lights & lightFlags.environment) {
    if (p.environment && p.environment.length > 0) {
      const env: Vec3 = [0, 0, 0];
      for (const light of p.environment) {
        const t = isotropic(toLocal(light.direction));
        for (let c = 0; c < 3; c++) env[c] = env[c]! + light.color[c]! * t[c]!;
      }
      add(map3((c) => Math.max(env[c]!, 0)));
    } else if (p.environmentIrradiance) {
      const outward = toLocal(p.outwardNormal);
      const n = p.outwardNormal;
      if (diffusion) {
        const pr = pair(outward);
        add(p.environmentIrradiance(n), map3((c) => pr.x[c]! * 0.5));
        add(p.environmentIrradiance([-n[0], -n[1], -n[2]]), map3((c) => pr.y[c]! * 0.5));
      } else {
        add(p.environmentIrradiance(n), total(p.march(outward, 1e6).y));
      }
    }
  }
  if (p.lights & lightFlags.hemisphere) {
    for (const h of p.hemisphere ?? []) {
      const up = toLocal(h.direction);
      let sky: Vec3;
      let ground: Vec3;
      if (diffusion) {
        const pr = pair(up);
        sky = pr.x;
        ground = pr.y;
      } else {
        sky = total(p.march(up, 1e6).y);
        ground = total(p.march([-up[0], -up[1], -up[2]], 1e6).y);
      }
      add(h.sky, map3((c) => 0.5 * sky[c]!));
      add(h.ground, map3((c) => 0.5 * ground[c]!));
    }
  }
  return irradiance;
}

/**
 * The spectral inject shadow march over given extinction samples (already × world step × absorbance): one optical
 * depth for the volume's spectrum; stops when every channel's march depth is past 8. Returns (direct, total,
 * reduced, effective) of each channel. The grey twin is `vdbFrameShadowMarch`.
 */
export function vdbFrameSpectralShadowMarch(scaledExtinctions: readonly number[], spectrum: Vec3, albedo: Vec3, multipleScattering: number, octaves: Vec4 = vdbFrameOctavesOff): VDBFrameSpectralMarch {
  const unit = vdbFrameSpectralShadowUnit(spectrum, albedo, multipleScattering);
  let opticalDepth = 0;
  for (const e of scaledExtinctions) {
    opticalDepth += e;
    if (Math.min(...vdbFrameSpectralShadowMarchDepth(vdbFrameSpectralShadowDepths(unit, opticalDepth), multipleScattering, octaves)) > 8) break;
  }
  const depths = vdbFrameSpectralShadowDepths(unit, opticalDepth);
  const t = vdbFrameSpectralShadowTransmittance(depths, multipleScattering, octaves);
  return { x: t.x, y: t.y, z: depths.z, w: depths.y };
}

/**
 * The spectral view-ray pre-integration of one froxel in the inject pass: the alpha of each channel and the
 * lit position t (the channel-mean visibility weight), or null when discarded (the grey twin is
 * `vdbFrameInjectSlice`).
 */
export function vdbFrameSpectralInjectSlice(opticalDepths: readonly number[], spectrum: Vec3, segment: Vec2, jitter = 0.5): { alpha: Vec3; t: number } | null {
  const n = opticalDepths.length;
  const transmittance: Vec3 = [1, 1, 1];
  let visibleWeight = 0;
  let visibleTSum = 0;
  for (let i = 0; i < n; i++) {
    const t = segment[0] + (segment[1] - segment[0]) * ((i + jitter) / n);
    const stepT = map3((c) => Math.exp(-spectrum[c]! * opticalDepths[i]!));
    const w = vdbFrameMeanComponent(map3((c) => transmittance[c]! * (1 - stepT[c]!)));
    visibleWeight += w;
    visibleTSum += w * t;
    for (let c = 0; c < 3; c++) transmittance[c] = transmittance[c]! * stepT[c]!;
  }
  const alpha = map3((c) => 1 - transmittance[c]!);
  if (visibleWeight <= 0 || Math.max(...alpha) <= 1e-5) return null;
  return { alpha, t: visibleTSum / visibleWeight };
}

/**
 * The spectral integrate composite of one pixel (createSpectralFragmentNode, linear blending, full slices): the
 * froxels' premultiplied colour and per-channel alpha front to back. Returns (radiance, transmittance of each channel).
 */
export function vdbFrameSpectralIntegrate(froxels: readonly { color: Vec3; alpha: Vec3 }[]): { radiance: Vec3; transmittance: Vec3 } {
  const radiance: Vec3 = [0, 0, 0];
  const transmittance: Vec3 = [1, 1, 1];
  for (const f of froxels) {
    if (Math.max(...transmittance) < 0.001) break;
    const a = map3((c) => clamp01(f.alpha[c]!));
    if (!(Math.max(...a) > 0)) continue;
    for (let c = 0; c < 3; c++) {
      radiance[c] = radiance[c]! + transmittance[c]! * f.color[c]! * (a[c]! / Math.max(a[c]!, 1e-6));
      transmittance[c] = transmittance[c]! * (1 - a[c]!);
    }
  }
  return { radiance, transmittance };
}
