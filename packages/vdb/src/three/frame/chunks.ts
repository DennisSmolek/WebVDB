// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/frameShaderChunks.js L62–203 (frag coord, sample, froxel uniforms
//   and maths, distance fade, LOD and sub-steps, IGN noise, Henyey-Greenstein), L273–315
//   (framePhaseWeight, framePhaseColor), L330–373 (shadow absorption, step, depth, march depth,
//   transmittance), L464–481 (frameShadowedLight), L624–686 (light uniforms), L727–763
//   (createVolumeUniforms, the part that is not the source), L1208–1228 (tile uniforms and vertex);
//   K3.3: L72–80 (frameLoad, frameFullscreenVertex), L403–414 (frameIsotropicDiffusion), L450–462
//   (frameShadowTotal, octaves off), L624–686 (the hemisphere, light-probe, sky and environment light
//   uniforms), L1030–1206 (shadow grid uniforms and nodes).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; the source half of createVolumeUniforms/createVolumeNodes is
// `../volume-source.ts`; Lorenz-Mie, octaves, two-sided/isotropic diffusion, spectral variants,
// hemisphere/probe/environment light uniforms and the grid chunks are not ported yet (K3.3–K3.5);
// `framePhaseWeight` / `framePhaseColor` compile the Mie branch out; `frameShadowTotal` has no
// octave branch (K3.4). The Shadertoy `blackBody()`
// is not ported (R1 §7, DIVERGES V24). Divergences: DIVERGES.md (K3.1 rows), V23–V25.

/**
 * Shared TSL of the froxel passes (K3.1, R1 §6 Step 1): the subset his `'volume'` lighting uses.
 * CPU twins: `@pmndrs/vdb/tools` `vdbFrame*` (tools/frame). Names follow his file so diffs against
 * upstream stay readable.
 *
 * NOTE Shaders use the bottom-left pixel origin of GLSL (gl_FragCoord). three.js nodes use the
 * top-left origin of WebGPU. These helpers convert between them.
 */

import * as THREE from "three";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import type { VDBVolumeNodes, VDBVolumeUniforms } from "../volume-source.js";
import { vdbMaxComponent, vdbMinComponent } from "../volume-source.js";

// NOTE TSL's r187 typings narrow node types; the ported shader code is written against untyped nodes,
// as volume-source.ts is (VDBTslNode).
const { If, ceil, clamp, dot, exp, exp2, float, floor, fract, int, instanceIndex, ivec2, length, log, max, min, mix, mod, positionGeometry, pow, screenCoordinate, select, smoothstep, sqrt, uniform, uniformArray, vec2, vec3, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;


export const minComponent = vdbMinComponent;
export const maxComponent = vdbMaxComponent;

export const frameFragCoord = (targetSize: N): N => vec2(screenCoordinate.x, targetSize.y.sub(screenCoordinate.y));

// NOTE Bottom-left texel of a render target texture
export const frameLoad = (textureNode: N, texel: N, targetSize: N): N => textureNode.load(ivec2(texel.x, int(targetSize.y).sub(texel.y).sub(1)));

// NOTE Fullscreen triangle. Positions are clip coordinates.
export const frameFullscreenVertex: N = vec4(positionGeometry.xy, 0.0, 1.0);

// NOTE Bottom-left uv of a render target texture. Level 0, then sampling is allowed in loops and branches.
export const frameSample = (textureNode: N, uv: N): N => textureNode.sample(vec2(uv.x, float(1.0).sub(uv.y))).level(float(0.0));

// NOTE Froxel grid layout and view reconstruction. Shared by all froxel passes.
export const createFroxelUniforms = () => ({
  froxelGrid: uniform(new THREE.Vector3(1.0, 1.0, 1.0)),
  froxelColumns: uniform(1.0),
  froxelRange: uniform(new THREE.Vector2(0.1, 1.0)),
  froxelProjectionInverse: uniform(new THREE.Matrix4()),
  froxelSubSteps: uniform(6, "int"),
  froxelAtlasSize: uniform(new THREE.Vector2(1.0, 1.0)),
  frameJitter: uniform(0.5),
  frameFadeRadius: uniform(0.0),
});

export type FroxelUniforms = ReturnType<typeof createFroxelUniforms>;

export const frameSliceDepth = (froxel: FroxelUniforms, t: N): N =>
  froxel.froxelRange.x.mul(pow(froxel.froxelRange.y.div(froxel.froxelRange.x), t));

// NOTE Two points on the pixel ray. NDC depth 0.0 and 1.0 are in the frustum in WebGPU and WebGL.
export const frameGetViewPosition = (froxel: FroxelUniforms, uv: N, viewDepth: N): N => {
  const ndc = uv.mul(2.0).sub(1.0);
  const nearPoint = froxel.froxelProjectionInverse.mul(vec4(ndc, 0.0, 1.0));
  const farPoint = froxel.froxelProjectionInverse.mul(vec4(ndc, 1.0, 1.0));
  const near = nearPoint.xyz.div(nearPoint.w);
  const far = farPoint.xyz.div(farPoint.w);
  const t = viewDepth.add(near.z).div(near.z.sub(far.z));

  return mix(near, far, t);
};

export interface FroxelCell {
  tile: N;
  slice: N;
  uv: N;
  sliceNear: N;
  sliceFar: N;
  viewNear: N;
  viewFar: N;
}

// NOTE Slice, uv and view positions of the froxel at an atlas pixel
export const frameGetFroxel = (froxel: FroxelUniforms, fragCoord: N): FroxelCell => {
  const { froxelGrid, froxelColumns } = froxel;
  const tile = floor(fragCoord.div(froxelGrid.xy));
  const slice = tile.y.mul(froxelColumns).add(tile.x);
  const uv = fragCoord.sub(tile.mul(froxelGrid.xy)).div(froxelGrid.xy);
  const sliceNear = frameSliceDepth(froxel, slice.div(froxelGrid.z));
  const sliceFar = frameSliceDepth(froxel, slice.add(1.0).div(froxelGrid.z));

  return {
    tile,
    slice,
    uv,
    sliceNear,
    sliceFar,
    viewNear: frameGetViewPosition(froxel, uv, sliceNear),
    viewFar: frameGetViewPosition(froxel, uv, sliceFar),
  };
};

// NOTE Exponent of the distance fade. 4.0 is FogExp2 with density 2 / frameFadeRadius.
const frameFadeExponent = 4.0;
const frameFadeEnd = Math.exp(-frameFadeExponent);

// NOTE Density scale at a view position. 1.0 at the camera, 0.0 at frameFadeRadius. Radius 0.0: no fade.
export const frameDistanceFade = (froxel: FroxelUniforms, viewPosition: N): N => {
  const x = clamp(length(viewPosition).div(froxel.frameFadeRadius), 0.0, 1.0);
  const fade = exp(x.mul(x).mul(-frameFadeExponent))
    .sub(frameFadeEnd)
    .div(1.0 - frameFadeEnd);

  return select(froxel.frameFadeRadius.greaterThan(0.0), fade, float(1.0));
};

// NOTE Mip level of a froxel from its width or the sub-step length, the larger one
export const getFroxelLod = (froxel: FroxelUniforms, cell: FroxelCell, volume: VDBVolumeNodes, subStepLength: N, worldScale: N): N => {
  const sliceMiddle = cell.sliceNear.add(cell.sliceFar).mul(0.5);
  const neighborUv = cell.uv.add(vec2(float(1.0).div(froxel.froxelGrid.x), 0.0));
  const froxelWidth = frameGetViewPosition(froxel, neighborUv, sliceMiddle).sub(frameGetViewPosition(froxel, cell.uv, sliceMiddle)).length();

  return volume.getLod(max(froxelWidth, subStepLength).div(worldScale)).toVar();
};

// NOTE The mip level the extinction is read from for this froxel width: level 0 without mipmaps, and no level past the
// ones the density grid and the mask grid have. The mask presence is compile-time here (DIVERGES V23).
export const getSampledLod = (source: VDBVolumeUniforms, hasMask: boolean, widthLod: N, mipmaps: boolean): N => {
  if (!mipmaps) {
    return float(0.0);
  }

  const maskLevels = hasMask ? source.mipLevels.mask : widthLod;

  return min(widthLod, min(source.mipLevels.density, maskLevels));
};

// NOTE Sub-steps for the part of a slice inside the volume box: two for each voxel of the mip level that is read, at
// most froxelSubSteps. A thin slice or a far volume needs fewer than froxelSubSteps.
export const getBoxSteps = (volume: VDBVolumeNodes, localNear: N, localFar: N, segment: N, sampledLod: N, froxelSubSteps: N): N => {
  const boxLength = localFar.sub(localNear).length().mul(segment);
  const steps = ceil(exp2(volume.getLod(boxLength).sub(sampledLod)).mul(2.0));

  return int(clamp(steps, 1.0, float(froxelSubSteps))).toVar();
};

// NOTE Per-pixel noise that changes each frame. Interleaved gradient noise (Jimenez 2014,
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare).
// NOTE Returns 0.5 when jitter is off (frameJitter is negative).
export const frameNoise = (jitter: N, pixel: N): N => {
  const noise = fract(float(52.9829189).mul(fract(dot(pixel, vec2(0.06711056, 0.00583715)))));

  return select(jitter.lessThan(0.0), float(0.5), fract(noise.add(jitter)));
};

// NOTE Henyey-Greenstein phase divided by the isotropic phase 1 / (4 PI). g = 0 gives 1.
export const frameHenyeyGreenstein = (cosTheta: N, g: N): N => {
  const g2 = g.mul(g);

  return float(1.0)
    .sub(g2)
    .div(pow(max(float(1.0).add(g2).sub(g.mul(2.0).mul(cosTheta)), 1e-4), 1.5));
};

// NOTE Henyey-Greenstein phase of each channel, g of each channel
const frameHenyeyGreensteinColor = (cosTheta: N, g: N): N => {
  const g2 = g.mul(g);

  return vec3(1.0)
    .sub(g2)
    .div(pow(max(vec3(1.0).add(g2).sub(g.mul(2.0).mul(cosTheta)), vec3(1e-4)), vec3(1.5)));
};

// NOTE framePhase is (forward g, back g, back weight, Lorenz-Mie mode), see phase.ts. Call in a Fn body.
// NOTE rayDirection: from the camera to the sample. lightDirection: from the sample to the light.
// NOTE The Lorenz-Mie branch (framePhase.w > 0) is not compiled in Step 1 (K3.4).
export const framePhaseWeight = (framePhase: N, rayDirection: N, lightDirection: N): N => {
  const cosTheta = dot(rayDirection, lightDirection).toVar();

  return mix(frameHenyeyGreenstein(cosTheta, framePhase.x), frameHenyeyGreenstein(cosTheta, framePhase.y), framePhase.z).toVar();
};

// NOTE Phase of each channel (red, green, blue). Call in a Fn body.
// NOTE phaseSpectrum is (red g scale, blue g scale, flag). Flag 0.0: the same phase for all channels. No Mie RGB branch.
export const framePhaseColor = (framePhase: N, phaseSpectrum: N, rayDirection: N, lightDirection: N): N => {
  const color = vec3(0.0).toVar();

  If(phaseSpectrum.z.lessThan(0.5), () => {
    color.assign(vec3(framePhaseWeight(framePhase, rayDirection, lightDirection)));
  }).Else(() => {
    const cosTheta = dot(rayDirection, lightDirection).toVar();
    const scale = vec3(phaseSpectrum.x, 1.0, phaseSpectrum.y).toVar();

    color.assign(
      mix(
        frameHenyeyGreensteinColor(cosTheta, clamp(scale.mul(framePhase.x), -0.99, 0.99)),
        frameHenyeyGreensteinColor(cosTheta, clamp(scale.mul(framePhase.y), -0.99, 0.99)),
        framePhase.z,
      ),
    );
  });

  return color;
};

// NOTE Absorption of the shadow extinction: the part of the light that the albedo (linear luminance) does not scatter
export const frameShadowAbsorption = (extinction: N, albedo: N): N =>
  extinction.mul(float(1.0).sub(clamp(dot(albedo, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0)));

// NOTE Shadow march values for each unit of length: (extinction, effective extinction, reduced scattering).
// multipleScattering is 1 - g (phase.ts). Diffusion approximation: effective = sqrt(3 * absorption * (absorption + reduced)).
export const frameShadowStep = (extinction: N, absorption: N, multipleScattering: N): N => {
  const absorbed = min(absorption, extinction).toVar();
  const reducedScattering = extinction.sub(absorbed).mul(multipleScattering).toVar();

  return vec3(extinction, sqrt(absorbed.mul(3.0).mul(absorbed.add(reducedScattering))), reducedScattering);
};

// NOTE -log of the total shadow transmittance (direct and scattered light) of the summed march values.
// exp(-effective) / (1 + 0.75 * reduced), never less than the direct light exp(-extinction). 0.0 multipleScattering: direct only.
export const frameShadowDepth = (depths: N, multipleScattering: N): N =>
  select(multipleScattering.greaterThan(0.0), min(depths.x, depths.y.add(log(depths.z.mul(0.75).add(1.0)))), depths.x);

// NOTE Optical depth of the least shadowed light term. Marches stop when it is large. (No octaves in Step 1.)
export const frameShadowMarchDepth = (depths: N, multipleScattering: N): N => frameShadowDepth(depths, multipleScattering);

// NOTE Shadow values (direct, total) transmittance. Direct light gets the phase function, scattered light (total - direct) is isotropic.
export const frameShadowTransmittance = (depths: N, multipleScattering: N): N =>
  vec2(exp(depths.x.negate()), exp(frameShadowDepth(depths, multipleScattering).negate()));

// NOTE Light of a shadowed light of each channel, divided by the light color: phase * direct + scattered. Call in a Fn body.
// Diffusion or single scattering: the scattered light (total - direct) is isotropic. (No octaves in Step 1.)
export const frameShadowedLight = (framePhase: N, phaseSpectrum: N, rayDirection: N, lightDirection: N, shadow: N): N => {
  const light = framePhaseColor(framePhase, phaseSpectrum, rayDirection, lightDirection).mul(shadow.x).toVar();

  light.addAssign(vec3(shadow.y.sub(shadow.x)));

  return light;
};

// NOTE Total transmittance of isotropic light (environment, sky) with diffusion multiple scattering.
// Two-sided slab diffusion: 2 * (back + z0) / (front + back + 2 z0) of the reduced depths, z0 = 2/3, times exp(-effective front).
// Light that is scattered and not absorbed is not lost: opposite directions sum to 2. In a white volume, a uniform
// environment then gives the same light inside and outside (white furnace test). Front is toward the light.
const frameDiffusionExtrapolation = 2.0 / 3.0;

export const frameIsotropicDiffusion = (frontReduced: N, backReduced: N, frontEffective: N): N =>
  backReduced
    .add(frameDiffusionExtrapolation)
    .mul(2.0)
    .div(frontReduced.add(backReduced).add(2.0 * frameDiffusionExtrapolation))
    .mul(exp(frontEffective.negate()));

// NOTE Total transmittance of isotropic light (no phase) from the y shadow value. Octaves off (K3.4): the value itself.
export const frameShadowTotal = (shadowY: N): N => float(shadowY);

export const frameMaxLights = 16;

export const frameMaxHemisphereLights = 4;
export const frameMaxEnvironmentLights = 12;

// NOTE Scene lights in world space. Lights are in shadow slot order: directional, point, spot.
// NOTE Position w: 1.0 directional (xyz is the direction to the light), 2.0 point, 3.0 spot.
// NOTE Color rgb is color * intensity, color w is the spot penumbra cosine.
// NOTE Spot xyz is the direction from the target to the light, spot w is the cone cosine.
// NOTE Falloff x is the light distance, falloff y is the decay. Point and spot lights only.
// NOTE 'volume' lighting (inject) reads the ambient and the scene light slots; the hemisphere, probe and
// environment uniforms are read by 'frame' lighting (K3.3).
export const createLightUniforms = () => ({
  frameAmbientColor: uniform(new THREE.Color(0x000000)),
  frameLightProbe: uniformArray(
    Array.from({ length: 9 }, () => new THREE.Vector3()),
    "vec3",
  ),
  // NOTE Sums of all hemisphere lights, for combined lighting. Sky direction is from the first one.
  frameHemiSkyColor: uniform(new THREE.Color(0x000000)),
  frameHemiGroundColor: uniform(new THREE.Color(0x000000)),
  frameSkyDirection: uniform(new THREE.Vector3(0.0, 1.0, 0.0)),
  // NOTE Each hemisphere light, for per-volume lighting. Directions point up (sky).
  frameHemiCount: uniform(0, "int"),
  frameHemiSkyColors: uniformArray(
    Array.from({ length: frameMaxHemisphereLights }, () => new THREE.Vector3()),
    "vec3",
  ),
  frameHemiGroundColors: uniformArray(
    Array.from({ length: frameMaxHemisphereLights }, () => new THREE.Vector3()),
    "vec3",
  ),
  frameHemiDirections: uniformArray(
    Array.from({ length: frameMaxHemisphereLights }, () => new THREE.Vector3()),
    "vec3",
  ),
  frameLightCount: uniform(0, "int"),
  frameLightPositions: uniformArray(
    Array.from({ length: frameMaxLights }, () => new THREE.Vector4()),
    "vec4",
  ),
  frameLightColors: uniformArray(
    Array.from({ length: frameMaxLights }, () => new THREE.Vector4()),
    "vec4",
  ),
  frameLightSpots: uniformArray(
    Array.from({ length: frameMaxLights }, () => new THREE.Vector4()),
    "vec4",
  ),
  frameLightFalloffs: uniformArray(
    Array.from({ length: frameMaxLights }, () => new THREE.Vector2()),
    "vec2",
  ),
  // NOTE Environment lights (settings.environmentLighting). World directions to the light.
  // Colors are isotropic irradiance, no phase. Count is 0 with the 'outward' environment.
  frameEnvironmentCount: uniform(0, "int"),
  frameEnvironmentDirections: uniformArray(
    Array.from({ length: frameMaxEnvironmentLights }, () => new THREE.Vector3(0.0, 1.0, 0.0)),
    "vec3",
  ),
  frameEnvironmentColors: uniformArray(
    Array.from({ length: frameMaxEnvironmentLights }, () => new THREE.Vector3()),
    "vec3",
  ),
  // NOTE Environment grid lights with no shadow, clamped to 0 or more. Used outside the environment grid.
  frameEnvironmentUnshadowed: uniform(new THREE.Vector3()),
});

export type LightUniforms = ReturnType<typeof createLightUniforms>;

// NOTE Values of one frame volume that are not the source (the source half is vdbVolumeUniforms). The renderer sets
// them before each draw (setVolumeUniforms in renderer.ts).
export const createFrameVolumeUniforms = () => ({
  frameViewToLocal: uniform(new THREE.Matrix4()),
  frameLocalToView: uniform(new THREE.Matrix4()),
  frameWorldToLocal: uniform(new THREE.Matrix4()),
  frameScatterColor: uniform(new THREE.Color(0x000000)),
  // NOTE Extinction scale of each channel (spectrum.ts). Only spectral shaders read it.
  frameExtinctionSpectrum: uniform(new THREE.Vector3(1.0, 1.0, 1.0)),
  frameAbsorbance: uniform(1.0),
  frameOpacity: uniform(1.0),
  // NOTE Phase of this volume for 'volume' lighting: (forward g, back g, back weight, Lorenz-Mie weight)
  framePhase: uniform(new THREE.Vector4()),
  // NOTE Phase of each channel (phase.ts getFramePhaseSpectrum)
  framePhaseSpectrum: uniform(new THREE.Vector3()),
});

export type FrameVolumeUniforms = ReturnType<typeof createFrameVolumeUniforms>;

export const frameMaxShadowCascades = 4;

// NOTE World-space shadow grid cascades. Cascade 0 is the finest, near the camera.
// All cascades are z slices (tiles) in one 2D atlas.
export const createGridUniforms = (frameJitter: N) => ({
  frameGridMin: uniformArray(
    Array.from({ length: frameMaxShadowCascades }, () => new THREE.Vector3()),
    "vec3",
  ),
  frameGridSize: uniformArray(
    Array.from({ length: frameMaxShadowCascades }, () => new THREE.Vector3()),
    "vec3",
  ),
  frameGridVoxel: uniformArray(
    Array.from({ length: frameMaxShadowCascades }, () => new THREE.Vector3()),
    "vec3",
  ),
  frameGridCount: uniform(0, "int"),
  frameGridResolution: uniform(1.0),
  frameGridColumns: uniform(1.0),
  frameGridAtlasSize: uniform(new THREE.Vector2(1.0, 1.0)),
  frameJitter,
});

export type GridUniforms = ReturnType<typeof createGridUniforms>;

// NOTE Environment grid: environment light of each voxel of the last shadow grid cascade, in its own atlas
export const createEnvironmentGridUniforms = () => ({
  frameEnvironmentGridCascade: uniform(0, "int"),
  frameEnvironmentGridColumns: uniform(1.0),
  frameEnvironmentGridAtlasSize: uniform(new THREE.Vector2(1.0, 1.0)),
});

export type EnvironmentGridUniforms = ReturnType<typeof createEnvironmentGridUniforms>;

export const createGridNodes = (grid: GridUniforms) => {
  const { frameGridMin, frameGridSize, frameGridCount, frameGridResolution, frameGridColumns } = grid;

  const gridUvw = (cascade: N, worldPosition: N): N => worldPosition.sub(frameGridMin.element(cascade)).div(frameGridSize.element(cascade));

  // NOTE Half a voxel inside the cascade. Then the linear filter does not read the clamped edge.
  const insideGrid = (uvw: N): N => {
    const inset = float(0.5).div(frameGridResolution);

    return minComponent(uvw)
      .greaterThanEqual(inset)
      .and(maxComponent(uvw).lessThanEqual(float(1.0).sub(inset)));
  };

  // NOTE Finest cascade from the first cascade that contains the position, or -1
  const findCascade = (worldPosition: N, firstCascade: N): N => {
    let cascade: N = int(-1);

    for (let i = frameMaxShadowCascades - 1; i >= 0; i--) {
      cascade = select(
        int(i)
          .lessThan(frameGridCount)
          .and(int(i).greaterThanEqual(firstCascade))
          .and(insideGrid(gridUvw(int(i), worldPosition))),
        int(i),
        cascade,
      );
    }

    return cascade;
  };

  // NOTE Hardware bilinear filter in the slice, manual linear filter between slices.
  // NOTE The grid is the z slices (tiles) from tileOffset in an atlas with columns tile columns.
  // resolution: voxels on each side of the grid (each tile is resolution x resolution).
  const sampleTiles = (atlas: N, resolution: N, tileOffset: N, columns: N, atlasSize: N, uvw: N): N => {
    const z = uvw.z.mul(resolution).sub(0.5);
    const z0 = clamp(floor(z), 0.0, resolution.sub(1.0));
    const z1 = min(z0.add(1.0), resolution.sub(1.0));
    const blend = clamp(z.sub(z0), 0.0, 1.0);
    const xy = clamp(uvw.xy.mul(resolution), vec2(0.5), vec2(resolution.sub(0.5)));

    const tileUv = (index: N): N => {
      const tile = vec2(mod(index, columns), floor(index.div(columns)));

      return tile.mul(resolution).add(xy).div(atlasSize);
    };

    return mix(frameSample(atlas, tileUv(tileOffset.add(z0))), frameSample(atlas, tileUv(tileOffset.add(z1))), blend);
  };

  const sampleGrid = (atlas: N, cascade: N, uvw: N): N =>
    sampleTiles(atlas, frameGridResolution, float(cascade).mul(frameGridResolution), frameGridColumns, grid.frameGridAtlasSize, uvw);

  // NOTE Environment grid: the last cascade only, in its own atlas. outside is the value outside it.
  // NOTE Call in a Fn body. Returns a vec4 variable.
  const sampleEnvironmentGrid = (map: N, environmentGrid: EnvironmentGridUniforms, worldPosition: N, outside: N): N => {
    const value = vec4(outside).toVar();
    const uvw = gridUvw(environmentGrid.frameEnvironmentGridCascade, worldPosition).toVar();

    If(insideGrid(uvw), () => {
      value.assign(sampleTiles(map, frameGridResolution, float(0.0), environmentGrid.frameEnvironmentGridColumns, environmentGrid.frameEnvironmentGridAtlasSize, uvw));
    });

    return value;
  };

  // NOTE Transmittance of the shadow slots at a world position. 1.0 outside the grid.
  // Call in a Fn body. Returns one vec4 variable for each map.
  const sampleShadows = (maps: N[], worldPosition: N): N[] => {
    const shadows = maps.map(() => vec4(1.0).toVar());
    const cascade = findCascade(worldPosition, int(0)).toVar();

    If(cascade.greaterThanEqual(0), () => {
      const uvw = gridUvw(cascade, worldPosition).toVar();

      maps.forEach((map, index) => {
        shadows[index]!.assign(sampleGrid(map, cascade, uvw));
      });

      // NOTE Blend to the next cascade near the sides of this cascade. Then cascade borders are not visible.
      const sideDistance = min(uvw, vec3(1.0).sub(uvw));
      const blend = float(1.0).sub(smoothstep(0.0, 0.1, minComponent(sideDistance))).toVar();

      If(blend.greaterThan(0.0), () => {
        const nextCascade = findCascade(worldPosition, cascade.add(1)).toVar();

        If(nextCascade.greaterThanEqual(0), () => {
          const nextUvw = gridUvw(nextCascade, worldPosition).toVar();

          maps.forEach((map, index) => {
            shadows[index]!.assign(mix(shadows[index]!, sampleGrid(map, nextCascade, nextUvw), blend));
          });
        });
      });
    });

    return shadows;
  };

  // NOTE Cascade and grid position of an atlas pixel
  const getGridCell = (fragCoord: N): { cascade: N; gridUvw: N } => {
    const resolution = frameGridResolution;
    const tile = floor(fragCoord.div(resolution));
    const tileIndex = tile.y.mul(frameGridColumns).add(tile.x);
    const cascade = floor(tileIndex.div(resolution));
    const z = tileIndex.sub(cascade.mul(resolution));

    return {
      cascade: int(cascade),
      gridUvw: vec3(fragCoord.sub(tile.mul(resolution)), z.add(0.5)).div(resolution),
    };
  };

  return { gridUvw, insideGrid, findCascade, sampleTiles, sampleGrid, sampleEnvironmentGrid, sampleShadows, getGridCell };
};

export type GridNodes = ReturnType<typeof createGridNodes>;

// NOTE Instanced quad for froxel passes. One instance for each tile (slice), the quad covers the volume rectangle.
export const createTileUniforms = () => ({
  frameTileSize: uniform(new THREE.Vector2(1.0, 1.0)),
  frameTileColumns: uniform(1.0),
  frameTargetSize: uniform(new THREE.Vector2(1.0, 1.0)),
});

export type TileUniforms = ReturnType<typeof createTileUniforms>;

// NOTE Froxel area of all volumes (view.ts getFroxelArea). Passes over the whole froxel grid draw only it.
export const createAreaUniforms = () => ({
  frameAreaRect: uniform(new THREE.Vector4()),
  frameAreaSliceStart: uniform(0.0),
});

export const frameTileVertex = (tileUniforms: TileUniforms, rect: N, sliceStart: N): N => {
  const { frameTileSize, frameTileColumns, frameTargetSize } = tileUniforms;
  const slice = sliceStart.add(float(instanceIndex));
  const tile = vec2(mod(slice, frameTileColumns), floor(slice.div(frameTileColumns)));
  const cell = mix(rect.xy, rect.zw, positionGeometry.xy.mul(0.5).add(0.5));

  return vec4(tile.mul(frameTileSize).add(cell).div(frameTargetSize).mul(2.0).sub(1.0), 0.0, 1.0);
};
