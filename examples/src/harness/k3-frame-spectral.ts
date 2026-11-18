/**
 * K3.5 helpers of the k3-frame harness (`?spectral=1`, `?medium=`): GPU-vs-CPU-twin checks of mjurczyk's spectral
 * passes from read-back render targets (the twins: `@pmndrs/vdb/tools` tools/frame/spectral.ts).
 *
 * - `probeSpectral`: the K3.5 TSL chunks (spectral unit / step / transmittance / octave light / shadow total /
 *   shadowed light / grid albedo / group unpack, and the isotropic diffusion pair of the inject pass) vs their twins.
 * - `spectralTransmittanceTwin`: every slot group's 3 targets vs `vdbFrameSpectralTransmittanceVoxel` marched over
 *   the read-back extinction and absorption atlases.
 * - `spectralEnvironmentGridTwin`, `spectralDensityTwin`, `spectralLightingTwin`: the other 'frame' lighting passes.
 * - `spectralIntegrateTwin`: the integrate pass (per-channel alpha) vs `vdbFrameSpectralIntegrate` + the sRGB blend.
 */
import * as THREE from "three";
import { NodeMaterial, type WebGPURenderer } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBFrameRenderer, VDBFrameVolume } from "@pmndrs/vdb/three";
import { vdbFrameChunks } from "@pmndrs/vdb/three";
import {
  sampleWvdbDenseTexture,
  vdbFrameFroxel,
  vdbFrameIsotropicDiffusion,
  vdbFrameSampleShadows,
  vdbFrameSpectralDensityFroxel,
  vdbFrameSpectralEnvironmentGridVoxel,
  vdbFrameSpectralGridAlbedo,
  vdbFrameSpectralIntegrate,
  vdbFrameSpectralLightingFroxel,
  vdbFrameSpectralOctaveLight,
  vdbFrameSpectralShadowStep,
  vdbFrameSpectralShadowTotal,
  vdbFrameSpectralShadowTransmittance,
  vdbFrameSpectralShadowUnit,
  vdbFrameSpectralShadowedLight,
  vdbFrameSpectralTransmittanceVoxel,
  vdbFrameSpectralGroups,
  vdbFrameSrgbBlend,
  vdbFrameUnpackSpectralGroup,
  vdbFrameSpectralShadowDepths,
  vdbFrameMeanComponent,
  type VDBFrameSpectralDensityInputs,
  type WvdbDenseVolume,
} from "@pmndrs/vdb/tools";
import { accumulate, finish, gridLayout, newStats, readAtlas, tightRows, type TwinStats } from "./k3-frame-lighting.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type N = any;
type V3 = [number, number, number];
type V4 = [number, number, number, number];
const { Fn, float, screenCoordinate, vec3, vec4, If } = TSL as unknown as Record<string, N>;

const octavesOf = (v: THREE.Vector4): V4 => [v.x, v.y, v.z, v.w];

/** The K3.5 spectral TSL chunks vs their CPU twins (one RGBA32F value per channel; 8 rows). */
export async function probeSpectral(renderer: WebGPURenderer): Promise<TwinStats> {
  const C = vdbFrameChunks;
  const PW = 64;
  const PH = 8;
  const spectrum: V3 = [0.74, 1.0, 1.43];
  const albedo: V3 = [0.84, 0.86, 0.88];
  const ms = 0.42;
  const octaves: V4 = [4, 0.5, 0.6, 0.5];
  const dual: V4 = [0.8, -0.3, 0.2, 0];
  const phaseSpectrum: V3 = [Math.pow(650 / 550, -0.5), Math.pow(450 / 550, -0.5), 1];
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
    const tau = a.mul(5.0).toVar();
    const spec = vec3(...spectrum);
    const alb = vec3(...albedo);
    const oct = vec4(...octaves);
    const msU = float(ms);
    If(row.lessThan(0.5), () => {
      const unit = C.frameSpectralShadowUnit(spec, alb, msU);
      const d = C.frameSpectralShadowDepths(unit, tau);
      out.assign(vec4(d.y, C.frameMeanComponent(d.z)));
    })
      .ElseIf(row.lessThan(1.5), () => {
        const s = C.frameSpectralShadowStep(spec.mul(tau), spec.mul(tau).mul(vec3(0.1, 0.3, 0.6)), msU);
        out.assign(vec4(s.y, s.z.y));
      })
      .ElseIf(row.lessThan(2.5), () => {
        const d = C.frameSpectralShadowDepths(C.frameSpectralShadowUnit(spec, alb, msU), tau);
        const t = C.frameSpectralShadowTransmittance(d, msU, vec4(0.0));
        out.assign(vec4(t.y, t.x.z));
      })
      .ElseIf(row.lessThan(3.5), () => {
        const d = C.frameSpectralShadowDepths(C.frameSpectralShadowUnit(spec, alb, msU), tau);
        const t = C.frameSpectralShadowTransmittance(d, msU, oct);
        out.assign(vec4(C.frameSpectralShadowTotal(t.y, oct, alb), t.y.x));
      })
      .ElseIf(row.lessThan(4.5), () => {
        out.assign(vec4(C.frameSpectralOctaveLight(oct, spec.mul(tau), alb, vec4(...dual), cosT), C.frameSpectralGridAlbedo(spec.mul(tau), spec.mul(tau).mul(0.2)).y));
      })
      .ElseIf(row.lessThan(5.5), () => {
        const shadow = { x: spec.mul(tau).negate().exp().toVar(), y: vec3(1.0).div(spec.mul(tau).add(1.0)).toVar() };
        out.assign(vec4(C.frameSpectralShadowedLight(vec4(...dual), vec3(...phaseSpectrum), oct, ray, light, shadow, alb), float(0.0)));
      })
      .ElseIf(row.lessThan(6.5), () => {
        const shadow = { x: spec.mul(tau).negate().exp().toVar(), y: spec.mul(tau).mul(-0.5).exp().toVar() };
        out.assign(vec4(C.frameSpectralShadowedLight(vec4(...dual), vec3(1.0, 1.0, 0.0), vec4(0.0), ray, light, shadow, alb), C.frameMeanComponent(spec.mul(a))));
      })
      .Else(() => {
        // The inject pass's isotropic diffusion pair (vec3, spectral) and the group unpack
        const f = spec.mul(tau);
        const b = spec.mul(float(5.0).sub(tau));
        const g = C.frameUnpackSpectralGroup(vec4(a, 1.0, 2.0, 3.0), vec4(4.0, 5.0, 6.0, 7.0), vec4(8.0, 9.0, 10.0, a.mul(3.0)));
        out.assign(vec4(C.frameIsotropicDiffusion(f, b, f.mul(0.1)).x, C.frameIsotropicDiffusion(b, f, vec3(0.0)).z, g.a.x.x.add(g.b.y.z), g.a.y.z.add(g.b.x.x)));
      });
    return out;
  })();
  material.vertexNode = vec4(TSL.positionGeometry.xy, 0.0, 1.0);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  const scene = new THREE.Scene();
  scene.add(quad);
  const target = new THREE.RenderTarget(PW, PH, { type: THREE.FloatType, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  renderer.setRenderTarget(target);
  renderer.render(scene, new THREE.OrthographicCamera());
  renderer.setRenderTarget(null);
  const raw = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, PW, PH)) as unknown as Float32Array;
  const gpu = tightRows(raw, PW, PH, 4);
  target.dispose();
  material.dispose();
  const stats = newStats();
  const mul3 = (v: V3, s: number | V3): V3 => v.map((x, c) => x * (typeof s === "number" ? s : s[c]!)) as V3;
  for (let row = 0; row < PH; row++) {
    for (let x = 0; x < PW; x++) {
      const a = (x + 0.5) / PW;
      const cosT = a * 2 - 1;
      const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
      const ray: V3 = [0, 0, 1];
      const light: V3 = [sinT, 0, cosT];
      const tau = a * 5;
      const unitDepths = vdbFrameSpectralShadowDepths(vdbFrameSpectralShadowUnit(spectrum, albedo, ms), tau);
      let cpu: number[];
      if (row === 0) cpu = [...unitDepths.y, vdbFrameMeanComponent(unitDepths.z)];
      else if (row === 1) {
        const s = vdbFrameSpectralShadowStep(mul3(spectrum, tau), mul3(mul3(spectrum, tau), [0.1, 0.3, 0.6]), ms);
        cpu = [...s.y, s.z[1]];
      } else if (row === 2) {
        const t = vdbFrameSpectralShadowTransmittance(unitDepths, ms, [0, 0, 0, 0]);
        cpu = [...t.y, t.x[2]];
      } else if (row === 3) {
        const t = vdbFrameSpectralShadowTransmittance(unitDepths, ms, octaves);
        cpu = [...vdbFrameSpectralShadowTotal(t.y, octaves, albedo), t.y[0]];
      } else if (row === 4) {
        cpu = [...vdbFrameSpectralOctaveLight(octaves, mul3(spectrum, tau), albedo, dual, cosT), vdbFrameSpectralGridAlbedo(mul3(spectrum, tau), mul3(spectrum, tau * 0.2))[1]];
      } else if (row === 5) {
        const shadow = { x: mul3(spectrum, tau).map((v) => Math.exp(-v)) as V3, y: mul3(spectrum, tau).map((v) => 1 / (v + 1)) as V3 };
        cpu = [...vdbFrameSpectralShadowedLight(dual, phaseSpectrum, octaves, ray, light, shadow, albedo), 0];
      } else if (row === 6) {
        const shadow = { x: mul3(spectrum, tau).map((v) => Math.exp(-v)) as V3, y: mul3(spectrum, tau).map((v) => Math.exp(-0.5 * v)) as V3 };
        cpu = [...vdbFrameSpectralShadowedLight(dual, [1, 1, 0], [0, 0, 0, 0], ray, light, shadow, albedo), vdbFrameMeanComponent(mul3(spectrum, a))];
      } else {
        const f = mul3(spectrum, tau);
        const b = mul3(spectrum, 5 - tau);
        const g = vdbFrameUnpackSpectralGroup([a, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, a * 3]);
        cpu = [vdbFrameIsotropicDiffusion(f[0], b[0], f[0] * 0.1), vdbFrameIsotropicDiffusion(b[2], f[2], 0), g.a.x[0] + g.b.y[2], g.a.y[2] + g.b.x[0]];
      }
      for (let c = 0; c < 4; c++) accumulate(stats, gpu[(row * PW + x) * 4 + c]!, cpu[c]!, 1e-5, 2e-4, { row, x, c });
    }
  }
  return finish(stats);
}

/**
 * Every slot group's transmittance targets vs `vdbFrameSpectralTransmittanceVoxel` on the read-back extinction and
 * absorption atlases (same input, so only the march differs). `step` subsamples the atlas.
 */
export async function spectralTransmittanceTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, step = 3): Promise<TwinStats & { marched: number; groups: number }> {
  const extinction = await readAtlas(renderer, fr.shadowDensityTarget!, 4, 0);
  const absorption = await readAtlas(renderer, fr.shadowDensityTarget!, 4, 1);
  const tm = fr.transmittanceMaterial;
  const grid = gridLayout(fr);
  const count = tm.frameShadowLightCount.value as number;
  const lights = (tm.frameShadowLights.array as THREE.Vector4[]).slice(0, count).map((v) => [v.x, v.y, v.z, v.w] as V4);
  const sky = fr.lightUniforms.frameSkyDirection.value;
  const ms = tm.frameMultipleScattering.value as number;
  const jitter = fr.froxelUniforms.frameJitter.value as number;
  const modes = { octaves: octavesOf(tm.frameScatteringOctaves.value), twoSided: (tm.frameTwoSided.value as number) > 0.5 };
  const groups = vdbFrameSpectralGroups(count);
  const stats = newStats();
  let marched = 0;
  for (let group = 0; group < groups; group++) {
    const target = fr.spectralTransmittanceTargets[group]!;
    const outs = await Promise.all([0, 1, 2].map((i) => readAtlas(renderer, target, 4, i)));
    const w = outs[0]!.width;
    for (let y = 0; y < outs[0]!.height; y += step) {
      for (let x = 0; x < w; x += step) {
        const o = (y * w + x) * 4;
        const gpu = outs.map((t) => [t.data[o]!, t.data[o + 1]!, t.data[o + 2]!, t.data[o + 3]!]);
        const cpu = vdbFrameSpectralTransmittanceVoxel(extinction, absorption, grid, [x + 0.5, y + 0.5], lights, [sky.x, sky.y, sky.z], ms, jitter, group, modes);
        const isMarched = cpu.some((v) => v.some((c) => c !== 1));
        if (isMarched) marched++;
        else if ((x + y) % 7 !== 0) continue;
        for (let t = 0; t < 3; t++) for (let c = 0; c < 4; c++) accumulate(stats, gpu[t]![c]!, cpu[t]![c]!, 2e-3, 4e-3, { group, x, y, t, c });
      }
    }
  }
  return { ...finish(stats), marched, groups };
}

/** The spectral environment grid pass vs `vdbFrameSpectralEnvironmentGridVoxel`. */
export async function spectralEnvironmentGridTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, step = 2): Promise<TwinStats & { marched: number }> {
  const extinction = await readAtlas(renderer, fr.shadowDensityTarget!, 4, 0);
  const absorption = await readAtlas(renderer, fr.shadowDensityTarget!, 4, 1);
  const out = await readAtlas(renderer, fr.environmentGridTarget!, 4);
  const grid = gridLayout(fr);
  const em = fr.environmentGridMaterial;
  const n = em.frameLightCount.value as number;
  const lights = Array.from({ length: n }, (_, i) => ({
    direction: (em.frameLightDirections.array as THREE.Vector3[])[i]!.toArray() as V3,
    color: (em.frameLightColors.array as THREE.Vector3[])[i]!.toArray() as V3,
  }));
  const u = fr.lightUniforms.frameEnvironmentUnshadowed.value;
  const columns = fr.environmentGridUniforms.frameEnvironmentGridColumns.value as number;
  const ms = em.frameMultipleScattering.value as number;
  const octaves = octavesOf(em.frameScatteringOctaves.value);
  const outside = vdbFrameSpectralShadowTotal([1, 1, 1], octaves, [1, 1, 1]);
  const stats = newStats();
  let marched = 0;
  for (let y = 0; y < out.height; y += step) {
    for (let x = 0; x < out.width; x += step) {
      const o = (y * out.width + x) * 4;
      const cpu = vdbFrameSpectralEnvironmentGridVoxel(extinction, absorption, grid, columns, [x + 0.5, y + 0.5], lights, [u.x, u.y, u.z], ms, octaves);
      if (cpu[0] !== u.x * outside[0] || cpu[1] !== u.y * outside[1] || cpu[2] !== u.z * outside[2]) marched++;
      for (let c = 0; c < 4; c++) accumulate(stats, out.data[o + c]!, cpu[c]!, 2e-3, 4e-3, { x, y, c });
    }
  }
  return { ...finish(stats), marched };
}

/** The spectral density pass (3 targets) vs `vdbFrameSpectralDensityFroxel`, one volume, dense source. */
export async function spectralDensityTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, volume: VDBFrameVolume, dense: WvdbDenseVolume, step = 2): Promise<TwinStats & { covered: number }> {
  const targets = await Promise.all([0, 1, 2].map((i) => readAtlas(renderer, fr.densityTarget!, 4, i)));
  const fu = fr.froxelUniforms;
  const material = volume.densityMaterial!;
  const vu = material.volumeUniforms;
  const su = volume.sourceUniforms;
  const ds = fr.densityShadowUniforms;
  const lu = fr.lightUniforms;
  const tex = dense.density;
  const mipLevels = volume.mipmaps && tex.mip ? tex.mip.levels : 0;
  const densityScale = su.densityScale.value as number;
  const extinctionScale = su.extinction.value as number;
  const worldScale = su.worldScale.value as number;
  const cutoff = su.densityCutoff.value as number;
  const spectrum = vu.frameExtinctionSpectrum.value;
  const inputs: VDBFrameSpectralDensityInputs = {
    froxelGrid: [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z],
    columns: fu.froxelColumns.value as number,
    range: [fu.froxelRange.value.x, fu.froxelRange.value.y],
    projectionInverse: fu.froxelProjectionInverse.value.elements,
    cameraWorld: ds.frameCameraMatrixWorld.value.elements,
    viewToLocal: vu.frameViewToLocal.value.elements,
    maxTexels: Math.max(...dense.dimensions),
    mipLevels,
    worldScale,
    subSteps: fu.froxelSubSteps.value as number,
    jitter: fu.frameJitter.value as number,
    fadeRadius: fu.frameFadeRadius.value as number,
    opacity: vu.frameOpacity.value as number,
    baseColor: (su.baseColor.value as THREE.Color).toArray() as V3,
    scatterColor: (vu.frameScatterColor.value as THREE.Color).toArray() as V3,
    extinctionAt: (local, lod) => {
      const v = sampleWvdbDenseTexture(tex, [local[0] + 0.5, local[1] + 0.5, local[2] + 0.5], mipLevels > 0 ? lod : 0);
      return ((v > cutoff ? v : 0) * densityScale * extinctionScale) / worldScale;
    },
    extinctionSpectrum: [spectrum.x, spectrum.y, spectrum.z],
    octaves: octavesOf(ds.frameScatteringOctaves.value),
    grid: (ds.frameUseShadows.value as number) === 1 ? gridLayout(fr) : null,
  };
  if ((ds.frameEnvironmentMode.value as number) === 2) {
    const u = lu.frameEnvironmentUnshadowed.value;
    inputs.environmentGrid = { atlas: await readAtlas(renderer, fr.environmentGridTarget!, 4), columns: fr.environmentGridUniforms.frameEnvironmentGridColumns.value as number, unshadowed: [u.x, u.y, u.z] };
  } else if ((ds.frameEnvironmentMode.value as number) === 0 && fr.densityScene.environment) {
    throw new Error("spectralDensityTwin: the 'outward' environment term (PMREM) has no CPU twin; use 'directional' lights");
  }
  const stats = newStats();
  let covered = 0;
  const w = targets[0]!.width;
  for (let y = 0; y < targets[0]!.height; y += step) {
    for (let x = 0; x < w; x += step) {
      const o = (y * w + x) * 4;
      const froxelSlice = Math.floor(y / inputs.froxelGrid[1]) * inputs.columns + Math.floor(x / inputs.froxelGrid[0]);
      if (froxelSlice >= inputs.froxelGrid[2]) continue;
      const gpu = targets.map((t) => [t.data[o]!, t.data[o + 1]!, t.data[o + 2]!, t.data[o + 3]!]);
      const cpu = vdbFrameSpectralDensityFroxel([x + 0.5, y + 0.5], inputs) ?? [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ];
      if (cpu[0]![3] > 0 || gpu[0]![3]! > 0) covered++;
      else continue;
      for (let t = 0; t < 3; t++) for (let c = 0; c < 4; c++) accumulate(stats, gpu[t]![c]!, cpu[t]![c]!, 2e-3, 4e-3, { x, y, t, c });
    }
  }
  return { ...finish(stats), covered };
}

/** The spectral lighting pass (2 atlas targets) vs `vdbFrameSpectralLightingFroxel` from the read-back density and group targets. */
export async function spectralLightingTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, step = 2): Promise<TwinStats & { lit: number }> {
  const density = await Promise.all([0, 1, 2].map((i) => readAtlas(renderer, fr.densityTarget!, 4, i)));
  const color = await readAtlas(renderer, fr.atlasTarget!, 4, 0);
  const alpha = await readAtlas(renderer, fr.atlasTarget!, 2, 1);
  const fu = fr.froxelUniforms;
  const lu = fr.lightUniforms;
  const lm = fr.lightingMaterial;
  const grid: V3 = [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z];
  const columns = fu.froxelColumns.value as number;
  const range: [number, number] = [fu.froxelRange.value.x, fu.froxelRange.value.y];
  const projectionInverse = fu.froxelProjectionInverse.value.elements;
  const cameraWorld = lm.frameCameraMatrixWorld.value.elements;
  const lightCount = lu.frameLightCount.value as number;
  const lights = (lu.frameLightPositions.array as THREE.Vector4[]).slice(0, lightCount).map((p, i) => {
    const c = (lu.frameLightColors.array as THREE.Vector4[])[i]!;
    return { position: [p.x, p.y, p.z, p.w] as V4, color: [c.x, c.y, c.z] as V3 };
  });
  const envCount = lm.frameEnvironmentLightCount.value as number;
  const envColors = (lu.frameEnvironmentColors.array as THREE.Vector3[]).slice(0, envCount).map((c) => [c.x, c.y, c.z] as V3);
  const groups = lm.frameSpectralGroups.value as number;
  const groupAtlases = await Promise.all(Array.from({ length: groups }, (_, g) => Promise.all([0, 1, 2].map((i) => readAtlas(renderer, fr.spectralTransmittanceTargets[g]!, 4, i)))));
  const layout = gridLayout(fr);
  const groupsAt = groups > 0 ? (world: V3) => groupAtlases.map((atlases) => vdbFrameSampleShadows(atlases, layout, world)) : null;
  const amb = lu.frameAmbientColor.value;
  const hs = lu.frameHemiSkyColor.value;
  const hg = lu.frameHemiGroundColor.value;
  const sky = lu.frameSkyDirection.value;
  const phase = lm.framePhase.value;
  const spectrum = lm.framePhaseSpectrum.value;
  const probe = (lu.frameLightProbe.array as THREE.Vector3[]).map((v) => [v.x, v.y, v.z] as V3);
  const stats = newStats();
  let lit = 0;
  const w = color.width;
  for (let y = 0; y < color.height; y += step) {
    for (let x = 0; x < w; x += step) {
      const o = (y * w + x) * 4;
      const d = density.map((t) => [t.data[o]!, t.data[o + 1]!, t.data[o + 2]!, t.data[o + 3]!] as V4);
      if (!(d[0]![3] > 0)) continue;
      const froxel = vdbFrameFroxel(grid, columns, range, projectionInverse, [x + 0.5, y + 0.5]);
      if (froxel.slice >= grid[2]) continue;
      lit++;
      const [cpuColor, cpuAlpha] = vdbFrameSpectralLightingFroxel(d as [V4, V4, V4], froxel.viewNear, froxel.viewFar, cameraWorld, {
        ambient: [amb.r, amb.g, amb.b],
        hemiSky: [hs.r, hs.g, hs.b],
        hemiGround: [hg.r, hg.g, hg.b],
        skyDirection: [sky.x, sky.y, sky.z],
        lightProbe: probe,
        lights,
        shadowLightCount: lm.frameShadowLightCount.value as number,
        phase: [phase.x, phase.y, phase.z, phase.w],
        phaseSpectrum: [spectrum.x, spectrum.y, spectrum.z],
        octaves: octavesOf(lm.frameScatteringOctaves.value),
        environmentColors: envColors,
        environmentShadowSlot: lm.frameEnvironmentShadowSlot.value as number,
        groupsAt,
      });
      for (let c = 0; c < 4; c++) accumulate(stats, color.data[o + c]!, cpuColor[c]!, 2e-3, 4e-3, { x, y, c });
      const ao = (y * w + x) * 2;
      for (let c = 0; c < 2; c++) accumulate(stats, alpha.data[ao + c]!, cpuAlpha[c]!, 2e-3, 4e-3, { x, y, c: `alpha${c}` });
    }
  }
  return { ...finish(stats), lit };
}

/**
 * The spectral integrate pass vs `vdbFrameSpectralIntegrate` + the per-channel sRGB blend, from the read-back atlas
 * (2 targets). Needs froxels = screen size and no occluder (as the grey integrate twin).
 */
export async function spectralIntegrateTwin(renderer: WebGPURenderer, fr: VDBFrameRenderer, linear: Float32Array, W: number, H: number, background: V3): Promise<{ pixels: number; covered: number; maxAbs: number; over: number; firstBad?: { x: number; y: number; gpu: number[]; cpu: number[] } }> {
  const colorAtlas = await readAtlas(renderer, fr.atlasTarget!, 4, 0);
  const alphaAtlas = await readAtlas(renderer, fr.atlasTarget!, 2, 1);
  const fu = fr.froxelUniforms;
  const grid = [fu.froxelGrid.value.x, fu.froxelGrid.value.y, fu.froxelGrid.value.z];
  if (grid[0] !== W || grid[1] !== H) throw new Error(`?twin=1 needs ?froxels=${W}x${H}xD (got ${grid.join("x")})`);
  const columns = fu.froxelColumns.value as number;
  const sr = fr.integrateMaterial.frameSliceRange.value;
  const aw = colorAtlas.width;
  const tol = 4e-3;
  let maxAbs = 0;
  let over = 0;
  let covered = 0;
  let pixels = 0;
  let firstBad: { x: number; y: number; gpu: number[]; cpu: number[] } | undefined;
  for (let y = 3; y < H; y += 3) {
    for (let x = 3; x < W; x += 3) {
      pixels++;
      const fy = H - 1 - y;
      const froxels: { color: V3; alpha: V3 }[] = [];
      for (let s = sr.x; s < sr.y; s++) {
        const tx = (s % columns) * grid[0]! + x;
        const ty = Math.floor(s / columns) * grid[1]! + fy;
        const o = (ty * aw + tx) * 4;
        const ao = (ty * aw + tx) * 2;
        froxels.push({ color: [colorAtlas.data[o]!, colorAtlas.data[o + 1]!, colorAtlas.data[o + 2]!], alpha: [alphaAtlas.data[ao]!, colorAtlas.data[o + 3]!, alphaAtlas.data[ao + 1]!] });
      }
      const r = vdbFrameSpectralIntegrate(froxels);
      const alpha = r.transmittance.map((t) => 1 - t) as V3;
      const o = (y * W + x) * 4;
      const gpu = [linear[o]!, linear[o + 1]!, linear[o + 2]!];
      const any = Math.max(...alpha) > 0;
      // sRGB blending of each channel with its own alpha (the shader: mix(bg, volume / alpha, alpha) in sRGB)
      const cpu = any ? ([0, 1, 2].map((c) => vdbFrameSrgbBlend(background, r.radiance, Math.max(alpha[c]!, 1e-6))[c]!) as V3) : background;
      if (any) covered++;
      for (let c = 0; c < 3; c++) {
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
