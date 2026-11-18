// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/frameLayout.js (all, 55 lines).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: TypeScript; the renderer is typed structurally. Divergences: none.

import { Vector2 } from "three";

const bufferSize = new Vector2();
const canvasSize = new Vector2();

/** Tiles of tileWidth x tileHeight in an atlas that is approximately square. */
export const getAtlasLayout = (tileWidth: number, tileHeight: number, tileCount: number) => {
  const columns = Math.ceil(Math.sqrt((tileCount * tileHeight) / tileWidth));
  const rows = Math.ceil(tileCount / columns);

  return { columns, width: tileWidth * columns, height: tileHeight * rows };
};

// NOTE A size that changes by less than this share is kept. The address bar of a mobile browser shows and hides while
// the page scrolls, and a new grid size drops the temporal history and makes new render targets.
const steadyChange = 0.15;

const steady = (size: number, held: number) => (held > 0 && Math.abs(size - held) <= held * steadyChange ? held : size);

export interface FroxelSizeSettings {
  width: number;
  height: number;
  pixelsPerFroxel: number;
}

export interface FroxelHeldSize {
  canvasX: number;
  canvasY: number;
  bufferX: number;
  bufferY: number;
}

export interface FroxelSizeRenderer {
  getDrawingBufferSize(target: Vector2): Vector2;
  getSize(target: Vector2): Vector2;
}

/**
 * Froxel grid size: the settings, but not more than the drawing buffer pixels (and, with
 * `pixelsPerFroxel` N > 0, not more than one froxel per N x N canvas pixels). `state` holds the
 * sizes through small changes (see his frameLayout.js notes).
 */
export const getFroxelSize = (
  settings: FroxelSizeSettings,
  renderer: FroxelSizeRenderer,
  target: { width: number; height: number } = { width: 1, height: 1 },
  state: FroxelHeldSize | null = null,
) => {
  renderer.getDrawingBufferSize(bufferSize);

  let capX = bufferSize.x;
  let capY = bufferSize.y;

  if (settings.pixelsPerFroxel > 0) {
    renderer.getSize(canvasSize);

    if (state) {
      state.canvasX = steady(canvasSize.x, state.canvasX);
      state.canvasY = steady(canvasSize.y, state.canvasY);
      state.bufferX = steady(bufferSize.x, state.bufferX);
      state.bufferY = steady(bufferSize.y, state.bufferY);
      canvasSize.set(state.canvasX, state.canvasY);
      bufferSize.set(state.bufferX, state.bufferY);
    }

    capX = Math.min(bufferSize.x, Math.ceil(canvasSize.x / settings.pixelsPerFroxel));
    capY = Math.min(bufferSize.y, Math.ceil(canvasSize.y / settings.pixelsPerFroxel));
  } else if (state) {
    // NOTE Turned on again later, it starts from the sizes of that frame
    state.canvasX = state.canvasY = state.bufferX = state.bufferY = 0;
  }

  target.width = Math.max(1, Math.min(Math.round(settings.width), capX));
  target.height = Math.max(1, Math.min(Math.round(settings.height), capY));

  return target;
};
