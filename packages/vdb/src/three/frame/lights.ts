// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameLights.js (all, 9 lines).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript. Divergences: none.

/** Light types of the `VDBFrameVolume` `lights` bitmask (his `frameLights`). */
export const vdbFrameLights = {
  useAmbientLights: 0b100000,
  usePointLights: 0b010000,
  useDirectionalLights: 0b001000,
  useSpotLights: 0b000100,
  useHemisphereLights: 0b000010,
  useEnvironment: 0b000001,
} as const;

/** Every light type (his FrameVolume default `lights`). */
export const vdbFrameAllLights = Object.values(vdbFrameLights).reduce((mask, flag) => mask | flag, 0);
