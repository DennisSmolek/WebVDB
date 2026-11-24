/**
 * The raw WebGPU handles behind a three.js `WebGPURenderer` (task K2.1).
 *
 * Same helpers and semantics as `@pmndrs/upscaler` and `@pmndrs/denoiser`
 * (`getDevice`, `getGPUTexture`), so every pmndrs tool in a frame shares the
 * renderer's one `GPUDevice`: create the pool with
 * `VDBGridPool.create(getDevice(renderer))`.
 *
 * Typed structurally (`VDBThreeRendererLike`, the downsampler's pattern), so this
 * file needs nothing from three at runtime. It reads two backend internals,
 * checked against three r187 (the peer range):
 * - `renderer.backend.device`: the `GPUDevice` (null until `renderer.init()`
 *   resolves; absent on the WebGL fallback backend);
 * - `renderer.backend.get(texture).texture`: the `GPUTexture` behind a three
 *   texture once it is allocated.
 *
 * A renderer built without a `device` parameter owns its device and destroys
 * it in `renderer.dispose()`, and with it every buffer on it (the pool's
 * too). To keep a pool alive past the renderer, create the device yourself and
 * pass it: `new WebGPURenderer({ device })`.
 */

import { VDBError } from "../core/errors.js";

/** The parts of three's WebGPU backend the adapter reads (structural, three r187). */
export interface VDBThreeBackendLike {
  /** True on three's WebGPU backend (false on the WebGL fallback). */
  readonly isWebGPUBackend?: boolean;
  /** The backend's `GPUDevice`, set once the renderer is initialized. */
  readonly device?: GPUDevice | null;
  /** three's per-object backend record (texture or buffer data). */
  get(object: object): { texture?: GPUTexture; buffer?: GPUBuffer } & Record<string, unknown>;
}

/**
 * The part of a three.js `WebGPURenderer` the adapter uses (structural, so
 * `three` stays an optional peer). `backend` is typed loosely because
 * `@types/three` does not declare the internals read here (`get`); they are
 * checked at runtime instead.
 */
export interface VDBThreeRendererLike {
  /** The renderer's backend object (checked at runtime). */
  readonly backend: object;
}

/** Throws unless `renderer` is an initialized `WebGPURenderer` on its WebGPU backend; returns that backend. */
export function webgpuBackendOf(renderer: unknown, what = "getDevice"): VDBThreeBackendLike & { readonly device: GPUDevice } {
  const backend = (renderer as { backend?: VDBThreeBackendLike } | null | undefined)?.backend;
  if (backend?.isWebGPUBackend !== true || !backend.device || typeof backend.get !== "function") {
    throw new VDBError(
      "VDB_UNSUPPORTED_DEVICE",
      `three adapter: ${what} requires an initialized WebGPURenderer (await renderer.init()) on its WebGPU backend; ` +
        "the WebGL fallback backend has no GPUDevice.",
    );
  }
  return backend as VDBThreeBackendLike & { readonly device: GPUDevice };
}

/**
 * The `GPUDevice` an initialized `WebGPURenderer` renders with. Create the
 * pool (and the other pmndrs tools) on it so they share one device.
 *
 * @throws VDBError `VDB_UNSUPPORTED_DEVICE` before `await renderer.init()`, or on the WebGL fallback backend.
 */
export function getDevice(renderer: VDBThreeRendererLike): GPUDevice {
  return webgpuBackendOf(renderer, "getDevice").device;
}

/**
 * The raw `GPUTexture` behind a three texture (render-target attachment,
 * `DepthTexture`, `StorageTexture`...). It must already be allocated: render to
 * it once or call `renderer.initTexture(texture)`.
 *
 * @param renderer The initialized `WebGPURenderer` whose device the pool lives on.
 * @param texture The three texture (or render-target attachment) to look up.
 *
 * @throws VDBError `VDB_UNSUPPORTED_DEVICE` for an uninitialized or non-WebGPU
 *   renderer; `VDB_BAD_OPTION` if the texture has no GPU allocation yet.
 */
export function getGPUTexture(renderer: VDBThreeRendererLike, texture: object): GPUTexture {
  const backend = webgpuBackendOf(renderer, "getGPUTexture");
  const gpuTexture = backend.get(texture)?.texture;
  if (!gpuTexture) {
    const named = texture as { name?: string; uuid?: string };
    throw new VDBError(
      "VDB_BAD_OPTION",
      `three adapter: no GPUTexture behind "${named.name || named.uuid || "texture"}" yet. ` +
        "Render to it once or call renderer.initTexture(texture) first.",
    );
  }
  return gpuTexture;
}
