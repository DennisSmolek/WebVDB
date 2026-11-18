/**
 * K3.6: the three r187 internals the mesh shadows and the emission light rely on (his `FrameVolumeShadows` /
 * `FrameEmissionLight` use them; R1 §3 B bucket, §8), each behind a loud check in the style of `VDBGridNode`'s
 * self-test: a check runs before the first use that could leave three in a bad state, and throws
 * `VDB_ENGINE_MISMATCH` (cached, rethrown on every later use) when three r187's behaviour is not there.
 *
 * 1. `new LightsNode().setLights([light]).setupLightsNode(builder)` returns the light node every material shares
 *    for that light (three keeps one per light in a module WeakMap), an `AnalyticLightNode` with a `shadowNode`
 *    field and `disposeShadow()`, which drops the shadow node and bumps the node's `version` (the lights cache key
 *    includes it, so receivers rebuild). His `resetLightShadow`: a new `shadow.filterNode` only takes effect when
 *    the light's shadow node is built again.
 * 2. `ShadowNode` / `PointShadowNode` call `shadow.filterNode(inputs)` at build with `{ depthTexture, shadowCoord,
 *    shadow, depthLayer }` (point: `{ depthTexture, bd3D, dp, distance, shadow }`); the default filters are the
 *    `three/tsl` exports `BasicShadowFilter`, `PCFShadowFilter`, `VSMShadowFilter` (by `shadowMap.type`; the
 *    PCF-soft slot is empty in r187) and `BasicPointShadowFilter` / `PointShadowFilter`. Checked at the filter call.
 * 3. `renderer._nodes.nodeFrame.renderId` is a number that changes once per `render()` (his frame identity; the
 *    collector already reads it since K3.1). `NodeUpdateType.RENDER` `updateBefore` runs once per render with
 *    `frame.renderer / scene / camera / renderId`.
 * 4. `Light.registerNode(nodeClass)` sets the node class lights of a type are shaded with (`prototype._lightNode`).
 * 5. `renderer._precompilationState` exists (null outside `compileAsync`); light passes skip while it is set, as
 *    three's own `ShadowNode.updateBefore`.
 * 6. `RendererUtils.resetRendererState` / `restoreRendererState` save and restore the renderer around nested passes.
 */

import * as THREE from "three";
import { AnalyticLightNode, LightsNode, NodeUpdateType, REVISION, RendererUtils } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBError } from "../../core/errors.js";

const T = TSL as unknown as Record<string, unknown>;

/** The three revision these checks are written against (the package's three peer range is >=0.187.0 <0.188.0). */
export const VDB_FRAME_THREE_REVISION = "187";

const mismatch = (what: string): VDBError =>
  new VDBError(
    "VDB_ENGINE_MISMATCH",
    `three adapter (frame mesh shadows / emission light): three r${REVISION} ${what}. These features use three r187 node-system ` +
      "internals (written against three r187, peer range >=0.187.0 <0.188.0); this three version is unsupported for them.",
  );

let lightResetResult: true | VDBError | null = null;

type LightNodeLike = { light?: unknown; isAnalyticLightNode?: boolean; shadowNode?: unknown; disposeShadow?: () => void; version?: number };

/** The shared light node of a light, through LightsNode.setupLightsNode (internal 1). */
export function vdbFrameSharedLightNode(light: THREE.Light): LightNodeLike | null {
  const nodeData = {};
  const lightsNode = new LightsNode() as unknown as { setLights(l: THREE.Light[]): { setupLightsNode(b: unknown): LightNodeLike[] } };
  const [node] = lightsNode.setLights([light]).setupLightsNode({ getDataFromNode: () => nodeData, context: { materialLightings: [] } });
  return node ?? null;
}

/**
 * Internal 1, self-tested once per page on a throwaway `DirectionalLight` that casts shadows: two `LightsNode`s
 * give the same `AnalyticLightNode` for it; `disposeShadow()` drops a shadow node and bumps `version`.
 * @throws VDBError `VDB_ENGINE_MISMATCH`
 */
export function vdbFrameCheckLightReset(): void {
  if (lightResetResult === true) return;
  if (lightResetResult) throw lightResetResult;
  let error: VDBError | null = null;
  const light = new THREE.DirectionalLight();
  light.castShadow = true;
  try {
    if (typeof (LightsNode.prototype as unknown as { setupLightsNode?: unknown }).setupLightsNode !== "function") {
      error = mismatch("has no LightsNode.setupLightsNode");
    } else {
      const a = vdbFrameSharedLightNode(light);
      const b = vdbFrameSharedLightNode(light);
      if (!a || !a.isAnalyticLightNode || !(a instanceof AnalyticLightNode) || a.light !== light) {
        error = mismatch("LightsNode.setupLightsNode did not return the light's AnalyticLightNode");
      } else if (a !== b) {
        error = mismatch("gave each LightsNode its own node for one light (the shadow reset needs the shared node)");
      } else if (!("shadowNode" in a) || a.shadowNode !== null || typeof a.disposeShadow !== "function") {
        error = mismatch("has no AnalyticLightNode.shadowNode / disposeShadow()");
      } else {
        let disposed = false;
        const version = a.version ?? 0;
        a.shadowNode = { dispose: () => (disposed = true) };
        a.disposeShadow();
        if (!disposed || a.shadowNode !== null || !((a.version ?? 0) > version)) {
          error = mismatch("AnalyticLightNode.disposeShadow() did not drop the shadow node and bump its version (receivers would keep the old shadow)");
        }
      }
    }
  } catch (e) {
    error = e instanceof VDBError ? e : mismatch(`failed the light-reset self-test (${e instanceof Error ? e.message : String(e)})`);
  } finally {
    light.dispose();
  }
  lightResetResult = error ?? true;
  if (error) throw error;
}

/** The shadow filters of internal 2 (three/tsl exports), checked present. */
export function vdbFrameShadowFilters(): { maps: (unknown | null)[]; basicPoint: (inputs: unknown) => unknown; point: (inputs: unknown) => unknown } {
  const names = ["BasicShadowFilter", "PCFShadowFilter", "VSMShadowFilter", "BasicPointShadowFilter", "PointShadowFilter"];
  for (const name of names) if (typeof T[name] !== "function") throw mismatch(`has no three/tsl ${name}`);
  return {
    // NOTE In shadow map type order; PCFSoftShadowMap has no filter (three r187 ShadowNode `_shadowFilterLib`)
    maps: [T.BasicShadowFilter, T.PCFShadowFilter, null, T.VSMShadowFilter],
    basicPoint: T.BasicPointShadowFilter as (inputs: unknown) => unknown,
    point: T.PointShadowFilter as (inputs: unknown) => unknown,
  };
}

/** Internal 2 at the call: the inputs three passes to `shadow.filterNode`. */
export function vdbFrameCheckFilterInputs(inputs: Record<string, unknown>, point: boolean): void {
  if (!inputs || !inputs.depthTexture || (point ? !inputs.bd3D : !inputs.shadowCoord)) {
    throw mismatch(`called shadow.filterNode without ${point ? "{ depthTexture, bd3D }" : "{ depthTexture, shadowCoord }"}`);
  }
}

/** Internal 3: the render id of the renderer's node frame. @throws VDBError `VDB_ENGINE_MISMATCH` */
export function vdbFrameRenderId(renderer: object): number {
  const id = (renderer as { _nodes?: { nodeFrame?: { renderId?: unknown } } })._nodes?.nodeFrame?.renderId;
  if (typeof id !== "number") throw mismatch("has no renderer._nodes.nodeFrame.renderId");
  return id;
}

/** Internal 3: `NodeUpdateType.RENDER`. */
export function vdbFrameRenderUpdateType(): string {
  const type = (NodeUpdateType as unknown as Record<string, string>).RENDER;
  if (type !== "render") throw mismatch("has no NodeUpdateType.RENDER");
  return type;
}

/** Internal 4: registers a light node class. @throws VDBError `VDB_ENGINE_MISMATCH` */
export function vdbFrameRegisterLightNode(lightClass: typeof THREE.Light, nodeClass: unknown): void {
  const register = (lightClass as unknown as { registerNode?: (c: unknown) => void }).registerNode;
  if (typeof register !== "function") throw mismatch("has no Light.registerNode");
  register.call(lightClass, nodeClass);
  if ((lightClass.prototype as unknown as { _lightNode?: unknown })._lightNode !== nodeClass) throw mismatch("Light.registerNode did not set prototype._lightNode");
}

/** Internal 5: whether three is precompiling (`compileAsync`). @throws VDBError `VDB_ENGINE_MISMATCH` */
export function vdbFramePrecompiling(renderer: object): boolean {
  if (!("_precompilationState" in renderer)) throw mismatch("has no renderer._precompilationState");
  return (renderer as { _precompilationState: unknown })._precompilationState != null;
}

/** Internal 6. @throws VDBError `VDB_ENGINE_MISMATCH` */
export function vdbFrameRendererUtils(): { resetRendererState(renderer: unknown, state: object): object; restoreRendererState(renderer: unknown, state: object): void } {
  const utils = RendererUtils as unknown as Record<string, unknown>;
  if (typeof utils.resetRendererState !== "function" || typeof utils.restoreRendererState !== "function") throw mismatch("has no RendererUtils.resetRendererState / restoreRendererState");
  return utils as never;
}

/** @internal Tests: forget the cached light-reset result. */
export function vdbFrameResetInternalsCache(): void {
  lightResetResult = null;
}
