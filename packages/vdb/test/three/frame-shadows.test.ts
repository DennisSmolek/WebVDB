/**
 * K3.6: the host side of the volume shadows and the emission light in Node (no GPU): his settings, the three r187
 * internals self-tests (and that they throw VDB_ENGINE_MISMATCH when three behaves otherwise), the shadow filter
 * install / restore on lights, and the emission light's emitters, keys and probe cascades over a VDBFrameVolume with
 * an emission channel. The passes run in e2e/k3-frame-shadows.spec.ts.
 */

import * as THREE from "three";
import { LightsNode } from "three/webgpu";
import { afterEach, describe, expect, it } from "vitest";
import { readWvdb, wvdbDenseVolume, writeWvdb } from "../../src/tools/index.js";
import { vdbCie1931Fit, vdbFrameBlackbodyTable, type VDBColorMatchingFunctions, type VDBFrameVec3 } from "../../src/tools/internal.js";
import { getFrameBlackbodyMap } from "../../src/three/frame/blackbody.js";
import { VDBFrameEmissionLight, VDBFrameEmissionLightNode, VDBFrameRenderer, VDBFrameVolume, vdbDenseTextures, vdbFrameBlackbodyEmission, vdbFrameCheckLightReset } from "../../src/three/internal.js";
import { vdbFrameRegisterLightNode, vdbFrameRenderId, vdbFrameResetInternalsCache, vdbFrameCheckFilterInputs, vdbFramePrecompiling } from "../../src/three/frame/three-internals.js";

// A renderer as far as the host code reaches without a draw (initTarget clears a new map)
const fakeRenderer = () =>
  ({
    isWebGPURenderer: true,
    coordinateSystem: THREE.WebGPUCoordinateSystem,
    shadowMap: { type: THREE.PCFShadowMap, enabled: true },
    getRenderTarget: () => null,
    getActiveCubeFace: () => 0,
    getActiveMipmapLevel: () => 0,
    getClearAlpha: () => 1,
    getClearColor: (c: THREE.Color) => c,
    setClearColor: () => undefined,
    setRenderTarget: () => undefined,
    clear: () => undefined,
  }) as never;

/** A 16³ volume with density everywhere and emission in its middle (writeWvdb → the dense source). */
const emissiveVolume = () => {
  const n = 16;
  const density = new Float32Array(n * n * n).fill(0.5);
  const flames = new Float32Array(n * n * n).map((_, i) => {
    const x = i % n;
    const y = Math.floor(i / n) % n;
    const z = Math.floor(i / (n * n));
    return Math.max(0, 1 - Math.hypot(x - 7.5, y - 7.5, z - 7.5) / 6);
  });
  const bbox: [[number, number, number], [number, number, number]] = [
    [-5, -5, -5],
    [5, 5, 5],
  ];
  const { bytes } = writeWvdb(
    [
      { name: "density", class: "fog volume", background: 0, bbox, finest: { dimensions: [n, n, n], values: density } },
      { name: "flames", class: "fog volume", background: 0, bbox, finest: { dimensions: [n, n, n], values: flames } },
    ],
    { lods: 1 },
  );
  return vdbDenseTextures(wvdbDenseVolume(readWvdb(bytes), { emission: "flames" }));
};

afterEach(() => vdbFrameResetInternalsCache());

describe("K3.6 settings and three internals", () => {
  it("his castShadow* settings and defaults", () => {
    const fr = new VDBFrameRenderer();
    expect(fr.settings).toMatchObject({ castShadows: false, castShadowMapSize: 512, castShadowCascades: 1, castShadowDistance: 4000, castShadowSteps: 32, castShadowUpdatesPerFrame: 1, castShadowCoarse: false });
    expect(fr.usesShadowGrid()).toBe(true);
    fr.settings.lighting = "volume";
    expect(fr.usesShadowGrid()).toBe(false);
    fr.settings.castShadows = true;
    expect(fr.usesShadowGrid()).toBe(true);
  });

  it("the light-shadow reset self-test passes on three r187", () => {
    expect(() => vdbFrameCheckLightReset()).not.toThrow();
  });

  it("fail-before: a LightsNode that does not share one node per light throws VDB_ENGINE_MISMATCH", () => {
    const proto = LightsNode.prototype as unknown as { setupLightsNode: (b: unknown) => unknown[] };
    const real = proto.setupLightsNode;
    proto.setupLightsNode = function (this: { _lights: THREE.Light[] }) {
      return this._lights.map((light) => ({ isAnalyticLightNode: true, light, shadowNode: null, disposeShadow() {} }));
    };
    try {
      expect(() => vdbFrameCheckLightReset()).toThrow(/VDB_ENGINE_MISMATCH.*AnalyticLightNode/);
      // cached: rethrown without running again
      proto.setupLightsNode = real;
      expect(() => vdbFrameCheckLightReset()).toThrow(/VDB_ENGINE_MISMATCH/);
    } finally {
      proto.setupLightsNode = real;
    }
  });

  it("fail-before: disposeShadow that does not bump the node version (receivers keep the old shadow) throws", async () => {
    const { AnalyticLightNode } = await import("three/webgpu");
    const proto = AnalyticLightNode.prototype as unknown as { disposeShadow: () => void };
    const real = proto.disposeShadow;
    proto.disposeShadow = function (this: { shadowNode: { dispose(): void } | null }) {
      this.shadowNode?.dispose();
      this.shadowNode = null;
    };
    try {
      expect(() => vdbFrameCheckLightReset()).toThrow(/VDB_ENGINE_MISMATCH.*version/);
    } finally {
      proto.disposeShadow = real;
    }
  });

  it("the other internals: render id, precompilation state, filter inputs and light node registration", () => {
    expect(vdbFrameRenderId({ _nodes: { nodeFrame: { renderId: 3 } } })).toBe(3);
    expect(() => vdbFrameRenderId({})).toThrow(/VDB_ENGINE_MISMATCH.*renderId/);
    expect(vdbFramePrecompiling({ _precompilationState: null })).toBe(false);
    expect(() => vdbFramePrecompiling({})).toThrow(/VDB_ENGINE_MISMATCH/);
    expect(() => vdbFrameCheckFilterInputs({ depthTexture: {}, shadowCoord: {} }, false)).not.toThrow();
    expect(() => vdbFrameCheckFilterInputs({ depthTexture: {} }, false)).toThrow(/VDB_ENGINE_MISMATCH.*filterNode/);
    expect(() => vdbFrameCheckFilterInputs({ depthTexture: {}, shadowCoord: {} }, true)).toThrow(/bd3D/);
    class NoRegistry {}
    expect(() => vdbFrameRegisterLightNode(NoRegistry as never, class {})).toThrow(/VDB_ENGINE_MISMATCH.*registerNode/);
  });
});

describe("K3.6 volume shadows: filters on lights", () => {
  it("castShadows puts the volume filter on each shadow light (keeping a user filter as its base); off restores it", () => {
    const fr = new VDBFrameRenderer();
    const scene = new THREE.Scene();
    const sun = new THREE.DirectionalLight();
    const spot = new THREE.SpotLight();
    const quiet = new THREE.PointLight();
    const userFilter = () => null;
    sun.castShadow = true;
    spot.castShadow = true;
    (spot.shadow as unknown as { filterNode: unknown }).filterNode = userFilter;
    scene.add(sun, spot, quiet);
    const renderer = fakeRenderer();

    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(0);

    fr.settings.castShadows = true;
    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(2);
    const filterOf = (l: THREE.Light) => (l.shadow as unknown as { filterNode: { isVDBFrameVolumeShadowFilter?: boolean } | null }).filterNode;
    expect(filterOf(sun)?.isVDBFrameVolumeShadowFilter).toBe(true);
    expect(filterOf(spot)?.isVDBFrameVolumeShadowFilter).toBe(true);
    expect(filterOf(quiet)).toBeFalsy();
    expect(fr.volumeShadows.entries.get(spot)!.userFilter).toBe(userFilter);
    // directional lights get 4 cascade tiles in one atlas; spot lights one map
    expect(fr.volumeShadows.entries.get(sun)!.target.width).toBe(512 * 4);
    expect(fr.volumeShadows.entries.get(spot)!.target.width).toBe(512);
    // the filter checks what three passes it
    expect(() => (filterOf(sun) as unknown as (i: unknown) => unknown)({})).toThrow(/VDB_ENGINE_MISMATCH/);

    // a new map size makes new maps; a removed light frees its entry
    fr.settings.castShadowCoarse = true;
    scene.remove(spot);
    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(1);
    expect(fr.volumeShadows.entries.get(sun)!.size).toBe(64);
    expect(filterOf(spot)).toBe(userFilter);

    fr.settings.castShadows = false;
    fr.volumeShadows.update(renderer, scene);
    expect(fr.volumeShadows.entries.size).toBe(0);
    expect(filterOf(sun)).toBeNull();
    fr.dispose();
  });
});

describe("K7.1 volume shadow update budget (his c13f2d5)", () => {
  // updateMaps without a GPU: the shadow grid is "ready" and the map draws are recorded, not drawn
  const budgetRig = () => {
    const fr = new VDBFrameRenderer();
    fr.settings.castShadows = true;
    fr.settings.castShadowUpdatesPerFrame = 1;
    const shadows = fr.volumeShadows;
    const drawnLights: string[] = [];
    Object.assign(fr, { prepareShadowGrid: () => true, shadowDensityTarget: { textures: [null], dispose() {} } });
    shadows.drawPass = (_r, entry) => void drawnLights.push(entry.light.name);
    const spot = (name: string, x: number) => {
      const light = new THREE.SpotLight();
      light.name = name;
      light.castShadow = true;
      light.position.set(x, 10, 0);
      return light;
    };
    return { fr, shadows, drawnLights, spot, renderer: fakeRenderer(), camera: new THREE.PerspectiveCamera() };
  };

  it("fail-before: a shadow light of another scene takes no part of the budget (nor a light removed after update)", () => {
    const { fr, shadows, drawnLights, spot, renderer, camera } = budgetRig();
    const sceneA = new THREE.Scene();
    const sceneB = new THREE.Scene();
    const lightA = spot("A", 0);
    const lightB = spot("B", 5);
    sceneA.add(lightA);
    sceneB.add(lightB);
    sceneA.updateMatrixWorld(true);
    sceneB.updateMatrixWorld(true);
    // One frame renderer in two scenes: B's light gets its entry first (the oldest-first order then puts it first)
    shadows.update(renderer, sceneB);
    shadows.update(renderer, sceneA);
    expect([...shadows.entries.keys()].map((l) => l.name)).toEqual(["B", "A"]);

    // Rendering scene A: only A's light is in the rendered scene, so it is drawn and nothing waits
    shadows.updateMaps(renderer, sceneA, camera, 1);
    expect(drawnLights).toEqual(["A"]);
    expect(shadows.lastUpdate).toEqual({ drawn: 1, waiting: 0 });
    expect(shadows.entries.get(lightA)!.enabled.value).toBe(1);
    expect(shadows.entries.get(lightB)!.enabled.value).toBe(0);

    // Rendering scene B: B's light is drawn
    drawnLights.length = 0;
    shadows.updateMaps(renderer, sceneB, camera, 2);
    expect(drawnLights).toEqual(["B"]);
    expect(shadows.lastUpdate).toEqual({ drawn: 1, waiting: 0 });

    // A light removed between update() (which would free it) and the maps: no draw, no budget
    const lightC = spot("C", -5);
    sceneA.add(lightC);
    sceneA.updateMatrixWorld(true);
    shadows.update(renderer, sceneA);
    lightA.position.x = 1;
    sceneA.updateMatrixWorld(true);
    sceneA.remove(lightC);
    drawnLights.length = 0;
    shadows.updateMaps(renderer, sceneA, camera, 3);
    expect(drawnLights).toEqual(["A"]);
    expect(shadows.lastUpdate).toEqual({ drawn: 1, waiting: 0 });
    fr.dispose();
  });

  it("lights of the rendered scene still share the budget (oldest first; the rest wait)", () => {
    const { fr, shadows, drawnLights, spot, renderer, camera } = budgetRig();
    const scene = new THREE.Scene();
    const a = spot("A", 0);
    const b = spot("B", 5);
    scene.add(a, b);
    scene.updateMatrixWorld(true);
    shadows.update(renderer, scene);
    shadows.updateMaps(renderer, scene, camera, 1);
    expect(drawnLights).toEqual(["A"]);
    expect(shadows.lastUpdate).toEqual({ drawn: 1, waiting: 1 });
    shadows.updateMaps(renderer, scene, camera, 2);
    expect(drawnLights).toEqual(["A", "B"]);
    expect(shadows.lastUpdate).toEqual({ drawn: 1, waiting: 0 });
    // a hidden parent still takes the light out
    const group = new THREE.Group();
    scene.add(group);
    group.add(b);
    group.visible = false;
    b.position.x = 6;
    scene.updateMatrixWorld(true);
    shadows.updateMaps(renderer, scene, camera, 3);
    expect(shadows.entries.get(b)!.enabled.value).toBe(0);
    expect(shadows.lastUpdate).toEqual({ drawn: 0, waiting: 0 });
    fr.dispose();
  });
});

describe("K3.6 emission light host side", () => {
  it("registers its light node with three (Light.registerNode) and keeps his props", () => {
    const light = new VDBFrameEmissionLight({ intensity: 25, resolution: 32 });
    expect((VDBFrameEmissionLight.prototype as unknown as { _lightNode: unknown })._lightNode).toBe(VDBFrameEmissionLightNode);
    expect(light).toMatchObject({ intensity: 25, resolution: 32, range: 3, cascades: 1, distance: 512, steps: 24, isLight: true });
    expect(light.probeTarget.width).toBe(384);
    expect(light.probeTarget.textures.length).toBe(3);
    light.dispose();
  });

  it("the blackbody hook (D33): the analytic-fit table by default, the caller's colour matching functions when given", () => {
    const planck = vdbFrameBlackbodyEmission("planck");
    expect(planck.table).toBe(getFrameBlackbodyMap());
    expect(planck.table.image.data).toBe(vdbFrameBlackbodyTable(vdbCie1931Fit));
    expect(vdbFrameBlackbodyEmission("soot", 2).table).toBe(planck.table);
    // e.g. the exact CIE table a parity harness injects (here: the fit with a red-shifted x̄)
    const cmf: VDBColorMatchingFunctions = (w) => {
      const [, y, z] = vdbCie1931Fit(w);
      const shifted: VDBFrameVec3 = [vdbCie1931Fit(w - 5)[0], y, z];
      return shifted;
    };
    const custom = vdbFrameBlackbodyEmission("soot", 1.39, { cmf });
    expect(custom.table).not.toBe(planck.table);
    expect(custom.table.image.data).toBe(vdbFrameBlackbodyTable(cmf));
    expect(custom.table.image.data).not.toEqual(planck.table.image.data);
    expect(vdbFrameBlackbodyEmission("planck", 0, { cmf }).table).toBe(custom.table);
    expect(custom.spectrum).toBe("soot");
    expect(custom.sootExponent.value).toBeCloseTo(1.39, 6);
  });

  it("emitters: visible volumes with an emission channel; keys change with the volume; cascades cover them", () => {
    const source = emissiveVolume();
    const fr = new VDBFrameRenderer();
    const hook = vdbFrameBlackbodyEmission("soot", 1.39);
    const volume = new VDBFrameVolume(source, { frameRenderer: fr, emissionColor: hook, temperatureRange: [0, 3000] });
    const plain = new VDBFrameVolume(vdbDenseTextures(wvdbDenseVolume(readWvdb(writeWvdb([{ name: "density", background: 0, bbox: [[0, 0, 0], [1, 1, 1]], finest: { dimensions: [2, 2, 2], values: new Float32Array(8).fill(1) } }], { lods: 1 }).bytes))), { frameRenderer: fr });
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();
    const light = new VDBFrameEmissionLight();
    volume.position.set(20, 0, 0);
    scene.add(volume, plain, light);
    scene.updateMatrixWorld(true);

    const first = light.prepareProbes(scene, camera)!;
    expect(light.emitterCount).toBe(1);
    expect(first.map((c) => c.slot)).toEqual([0]);
    expect(light.slotCount).toBe(1);
    expect(light.directionTarget.width).toBe(16);
    expect(light.emitters[0]!.center.toArray()).toEqual([20, 0, 0]);
    expect(light.emitters[0]!.radius).toBeCloseTo(5 * Math.sqrt(3), 9);
    // the last cascade: the emitter box ([15, 25] × [−5, 5]²) grown by (3 − 1)/2 × 10
    expect(light.cascadeCount).toBe(1);
    expect(light.cascadeMin[0]!.toArray()).toEqual([5, -15, -15]);
    expect(light.cascadeSize[0]!.toArray()).toEqual([30, 30, 30]);
    // nothing changed: no passes
    expect(light.prepareProbes(scene, camera)).toBeNull();
    // the colour hook's soot exponent, a prop, or the transform change the key
    hook.sootExponent.value = 2;
    expect(light.prepareProbes(scene, camera)).toHaveLength(1);
    volume.emissiveIntensity = 3;
    expect(light.prepareProbes(scene, camera)).toHaveLength(1);
    volume.position.x = 21;
    scene.updateMatrixWorld(true);
    expect(light.prepareProbes(scene, camera)).toHaveLength(1);
    // a hidden volume is not an emitter
    volume.visible = false;
    expect(light.prepareProbes(scene, camera)).toBeNull();
    expect(light.emitterCount).toBe(0);
    // the direction material is the volume's (its source compiled in), made once
    const material = light.getDirectionMaterial(volume);
    expect(light.getDirectionMaterial(volume)).toBe(material);
    expect(material.source).toBe(source);
    light.dispose();
    volume.dispose();
    expect(volume.emissionDirectionMaterial).toBeNull();
  });
});
