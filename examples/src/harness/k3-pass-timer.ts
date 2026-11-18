/**
 * K3.2c: per-pass GPU times of the froxel renderer from three's own render-pass timestamps
 * (`WebGPURenderer({ trackTimestamp: true })`, the `timestamp-query` feature).
 *
 * three r187 writes a begin/end timestamp per render call and, after `resolveTimestampsAsync('render')`,
 * keeps each call's duration in `backend.timestampQueryPool.render.timestamps` by uid. This maps uids
 * to the frame renderer's pass names: `VDBFrameRenderer.passHook` brackets each pass, and the
 * renderer's inspector hook (`inspector.beginRender(uid)`) sees every render call inside the bracket.
 * Two three internals (harness only, checked at install): the uid is made unique per call (three's
 * `'r:<frameCalls>:<context id>:f<frame>'` repeats when a pass renders the same scene, camera and
 * target twice in one frame), and the pool's `timestamps` map is read after the resolve.
 */
import type { WebGPURenderer } from "three/webgpu";
import type { VDBFrameRenderer } from "@pmndrs/vdb/three";

type Backend = {
  updateTimeStampUID?: (ctx: object) => void;
  get(ctx: object): { timestampUID?: string };
  timestampQueryPool?: Record<string, { timestamps?: Map<string, number> } | null>;
};
type Inspector = { beginRender?: (uid: string, ...rest: unknown[]) => void };

export interface PassTimer {
  /** Durations (ms) by pass name of the render calls since the last `collect()`; `other` = outside any pass. */
  collect(): Promise<Record<string, number>>;
  /** Removes the hooks. */
  dispose(): void;
}

/** Installs the hooks on `renderer` and `frameRenderer` (see the file comment). */
export function installPassTimer(renderer: WebGPURenderer, frameRenderer: VDBFrameRenderer): PassTimer {
  const backend = renderer.backend as unknown as Backend;
  const inspector = (renderer as unknown as { inspector: Inspector }).inspector;
  if (typeof backend.updateTimeStampUID !== "function" || typeof backend.get !== "function" || typeof inspector?.beginRender !== "function") {
    throw new Error("installPassTimer: three r187 internals not found (backend.updateTimeStampUID / inspector.beginRender)");
  }
  const stack: string[] = [];
  const passOf = new Map<string, string>();
  let serial = 0;

  const updateUid = backend.updateTimeStampUID.bind(backend);
  backend.updateTimeStampUID = (ctx: object) => {
    updateUid(ctx);
    const data = backend.get(ctx);
    // Keep three's ':f<frame>' suffix (its resolve parses it); make the rest unique per call.
    if (typeof data.timestampUID === "string") data.timestampUID = data.timestampUID.replace(/:f(\d+)$/, `#${++serial}:f$1`);
  };
  const beginRender = inspector.beginRender.bind(inspector);
  inspector.beginRender = (uid: string, ...rest: unknown[]) => {
    passOf.set(uid, stack[stack.length - 1] ?? "other");
    beginRender(uid, ...rest);
  };
  const previousHook = frameRenderer.passHook;
  frameRenderer.passHook = {
    begin: (pass, detail) => {
      stack.push(pass);
      previousHook?.begin?.(pass, detail);
    },
    end: (pass, detail) => {
      // Pop back to (and including) this pass, tolerant of a missing end.
      const i = stack.lastIndexOf(pass);
      if (i >= 0) stack.length = i;
      previousHook?.end?.(pass, detail);
    },
  };

  return {
    async collect() {
      await renderer.resolveTimestampsAsync("render");
      const pool = backend.timestampQueryPool?.render;
      const out: Record<string, number> = {};
      for (const [uid, ms] of pool?.timestamps ?? []) {
        const pass = passOf.get(uid) ?? "other";
        out[pass] = (out[pass] ?? 0) + ms;
      }
      pool?.timestamps?.clear();
      passOf.clear();
      return out;
    },
    dispose() {
      backend.updateTimeStampUID = updateUid;
      inspector.beginRender = beginRender;
      frameRenderer.passHook = previousHook;
    },
  };
}
