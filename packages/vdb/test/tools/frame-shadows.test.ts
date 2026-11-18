/**
 * K3.6: CPU twins of the volume shadows on meshes and the emission light (`tools/frame/shadows.ts`,
 * `tools/frame/emission.ts`, `tools/frame/blackbody.ts`), each against an independent derivation: Beer's law over
 * analytic slabs and chords, the CSM practical split, the L1 projection of a point light, the Planckian locus.
 * The GPU passes are checked against these twins in e2e/k3-frame-shadows.spec.ts.
 */

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  vdbFrameBlackbodyColor,
  vdbFrameBlackbodyRgb,
  vdbFrameBlackbodyTable,
  vdbFrameBlackbodyTableSize,
  vdbFrameBoxSegment,
  vdbFrameEmissionDirection,
  vdbFrameEmissionDirectionTexel,
  vdbFrameEmissionDirections,
  vdbFrameEmissionFalloff,
  vdbFrameEmissionIrradiance,
  vdbFrameEmissionProbe,
  vdbFrameEmissionProbeGrid,
  vdbFrameEmissionReduce,
  vdbFrameEmissionSpectrum,
  vdbFrameEmitterBounds,
  vdbFrameShadowCameraDepth,
  vdbFrameShadowCascadeBreaks,
  vdbFrameShadowCascadeFar,
  vdbFrameShadowCascadeIndex,
  vdbFrameShadowCascadeUv,
  vdbFrameShadowMapTexel,
  vdbFrameShadowPassRay,
  vdbFrameShadowReceiver,
  type VDBFrameEmitter,
  type VDBFrameVec3 as Vec3,
} from "../../src/tools/internal.js";

const close = (a: number, b: number, tol: number) => expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol);

describe("volume shadows: the Beer shadow map pass", () => {
  it("frameShadowDepth: distance from the light (perspective), along the forward axis (orthographic)", () => {
    const ray = { origin: [1, 2, 3] as Vec3, forward: [0, 0, -1] as Vec3 };
    expect(vdbFrameShadowCameraDepth({ ...ray, perspective: 1 }, [4, 6, 3])).toBe(5);
    expect(vdbFrameShadowCameraDepth({ ...ray, perspective: 0 }, [4, 6, -7])).toBe(10);
  });

  it("the pass ray: orthographic rays run along the view axis from the camera plane; perspective rays from the camera", () => {
    const ortho = new THREE.OrthographicCamera(-10, 10, 10, -10, 0, 100);
    ortho.position.set(3, 50, -2);
    ortho.lookAt(8, 0, 4);
    ortho.updateMatrixWorld();
    ortho.updateProjectionMatrix();
    const r = vdbFrameShadowPassRay(ortho.projectionMatrix.elements, ortho.projectionMatrixInverse.elements, ortho.matrixWorld.elements, [0.5, -0.25]);
    const forward = ortho.getWorldDirection(new THREE.Vector3());
    for (let k = 0; k < 3; k++) close(r.direction[k]!, forward.getComponent(k), 1e-9);
    // On the camera plane, at the pixel's offset (x 0.5 × 10, y −0.25 × 10 in camera space)
    const p = new THREE.Vector3(5, -2.5, 0).applyMatrix4(ortho.matrixWorld);
    for (let k = 0; k < 3; k++) close(r.origin[k]!, p.getComponent(k), 1e-9);

    const persp = new THREE.PerspectiveCamera(60, 1, 0.5, 200);
    persp.position.set(-4, 7, 9);
    persp.lookAt(0, 0, 0);
    persp.updateMatrixWorld();
    const q = vdbFrameShadowPassRay(persp.projectionMatrix.elements, persp.projectionMatrixInverse.elements, persp.matrixWorld.elements, [0, 0]);
    expect(q.origin).toEqual([-4, 7, 9]);
    const toTarget = new THREE.Vector3(4, -7, -9).normalize();
    close(q.direction[0], toTarget.x, 1e-9);
    close(q.direction[1], toTarget.y, 1e-9);
    close(q.direction[2], toTarget.z, 1e-9);
  });

  // A ray along −y through a slab of extinction σ between y = 2 and y = 6, in a box y ∈ [0, 10]
  const slab = (sigma: number) => (p: Vec3) => (p[1] > 2 && p[1] < 6 ? sigma : 0);
  const box = { min: [-1, 0, -1] as Vec3, max: [1, 10, 1] as Vec3 };
  const ray = { origin: [0, 20, 0] as Vec3, forward: [0, -1, 0] as Vec3, perspective: 0, far: 40 };

  it("a slab: τ = σ × thickness, first / last depths at its faces (÷ far), on the step lattice", () => {
    const { map } = vdbFrameShadowMapTexel(slab(0.5), [0, 20, 0], [0, -1, 0], box, 100, ray);
    // 100 steps of 0.1 from y = 10 down; the steps whose centres are inside (2, 6) are 40
    close(map[3], 0.5 * 4, 1e-9);
    // first non-empty step starts at y = 6 (depth 14), the last ends at y = 2 (depth 18)
    close(map[0] / map[3], 14 / 40, 1e-9);
    close(map[1] / map[3], 18 / 40, 1e-9);
    expect(map[2]).toBe(0);
  });

  it("empty rays and rays that miss the box give 0 (the linear filter stays right)", () => {
    expect(vdbFrameShadowMapTexel(() => 0, [0, 20, 0], [0, -1, 0], box, 32, ray).map).toEqual([0, 0, 0, 0]);
    expect(vdbFrameShadowMapTexel(slab(1), [5, 20, 0], [0, -1, 0], box, 32, ray).map).toEqual([0, 0, 0, 0]);
    expect(vdbFrameShadowMapTexel(slab(1), [0, 20, 0], [0, -1, 0], null, 32, ray).map).toEqual([0, 0, 0, 0]);
  });

  it("the march stops past optical depth 8 (the step that crosses it is the last)", () => {
    const { map } = vdbFrameShadowMapTexel(() => 10, [0, 20, 0], [0, -1, 0], box, 100, ray);
    // each step adds 10 × 0.1 = 1: stops after the 9th step (τ 9 > 8 checked before the 10th)
    close(map[3], 9, 1e-9);
    close(map[1] / map[3], (10 + 0.9) / 40, 1e-9);
  });

  it("spectral: output 0 holds the channel mean, output 1 τ per channel; the march stops when every channel is past 8", () => {
    const rgb = (p: Vec3): Vec3 => (p[1] > 2 && p[1] < 6 ? [0.5, 1, 2] : [0, 0, 0]);
    const { map, spectral } = vdbFrameShadowMapTexel(rgb, [0, 20, 0], [0, -1, 0], box, 100, ray, true);
    close(spectral![0], 2, 1e-9);
    close(spectral![1], 4, 1e-9);
    close(spectral![2], 8, 1e-9);
    close(map[3], 14 / 3, 1e-9);
    close(map[0] / map[3], 14 / 40, 1e-9);
  });

  it("the receiver: no shadow in front of the first density, Beer's law at the far side, linear in between", () => {
    const tau = 2;
    const texel: [number, number, number, number] = [tau * (14 / 40), tau * (18 / 40), 0, tau];
    expect(vdbFrameShadowReceiver(texel, 10, 40)).toBe(1);
    close(vdbFrameShadowReceiver(texel, 16, 40) as number, Math.exp(-1), 1e-12);
    close(vdbFrameShadowReceiver(texel, 30, 40) as number, Math.exp(-2), 1e-12);
    expect(vdbFrameShadowReceiver([0, 0, 0, 5e-5], 30, 40)).toBe(1);
    const s = vdbFrameShadowReceiver(texel, 30, 40, [1, 2, 3]) as Vec3;
    close(s[2], Math.exp(-3), 1e-12);
    close(vdbFrameShadowReceiver(texel, 30, 40, [1, 2, 3], true) as number, (Math.exp(-1) + Math.exp(-2) + Math.exp(-3)) / 3, 1e-12);
  });
});

describe("volume shadows: directional cascades", () => {
  it("the practical split of three's CSMShadowNode (λ = ½), the last break at far", () => {
    expect(vdbFrameShadowCascadeBreaks(1, 0.1, 1000)).toEqual([1000]);
    const b = vdbFrameShadowCascadeBreaks(4, 1, 1000);
    for (let i = 1; i < 4; i++) close(b[i - 1]!, 0.5 * (1 + (999 * i) / 4) + 0.5 * Math.pow(1000, i / 4), 1e-9);
    expect(b[3]).toBe(1000);
    expect([...b].sort((x, y) => x - y)).toEqual(b);
  });

  it("the covered distance: box far + diagonal rounded up to half octaves, within camera far and castShadowDistance", () => {
    close(vdbFrameShadowCascadeFar(0.1, 1e4, 4000, 200, 100), Math.pow(2, 8.5), 1e-9);
    expect(vdbFrameShadowCascadeFar(0.1, 1e4, 300, 200, 100)).toBe(300);
    expect(vdbFrameShadowCascadeFar(0.1, 250, 4000, 200, 100)).toBe(250);
    // a box behind the camera: at least 2 × near
    close(vdbFrameShadowCascadeFar(1, 1e4, 4000, -50, 10), 2, 1e-9);
  });

  it("the cascade of a receiver and its atlas uv (half a texel in from the tile sides)", () => {
    const breaks = [10, 30, 90, 200];
    expect(vdbFrameShadowCascadeIndex(5, breaks, 4)).toBe(0);
    expect(vdbFrameShadowCascadeIndex(50, breaks, 4)).toBe(2);
    expect(vdbFrameShadowCascadeIndex(150, breaks, 4)).toBe(3);
    expect(vdbFrameShadowCascadeIndex(150, breaks, 2)).toBe(1);
    const a = vdbFrameShadowCascadeUv([0, 0], 2, 512);
    expect(a.uv).toEqual([(0.5 + 2) / 4, 0.5]);
    expect(a.insideTile).toBe(true);
    const edge = vdbFrameShadowCascadeUv([-1.2, 1], 1, 512);
    expect(edge.insideTile).toBe(false);
    close(edge.uv[0], (0.5 / 512 + 1) / 4, 1e-12);
    close(edge.uv[1], 0.5 / 512, 1e-12);
  });
});

describe("emission light: direction, reduce and probe passes", () => {
  it("16 Fibonacci directions, unit length, y spread evenly; orthonormal tangent frames", () => {
    for (let i = 0; i < vdbFrameEmissionDirections; i++) {
      const { direction: d, tangent: t, bitangent: b } = vdbFrameEmissionDirection(i);
      close(Math.hypot(...d), 1, 1e-12);
      close(d[1], 1 - (2 * i + 1) / 16, 1e-12);
      close(Math.hypot(...t), 1, 1e-12);
      close(Math.hypot(...b), 1, 1e-12);
      close(d[0] * t[0] + d[1] * t[1] + d[2] * t[2], 0, 1e-12);
      close(t[0] * b[0] + t[1] * b[1] + t[2] * b[2], 0, 1e-12);
    }
  });

  it("box segment and emitter bounds", () => {
    expect(vdbFrameBoxSegment([-1, 0, 0], [1, 0, 0])).toEqual([0.25, 0.75]);
    expect(vdbFrameBoxSegment([2, 0, 0], [3, 0, 0])[0]).toBeGreaterThan(vdbFrameBoxSegment([2, 0, 0], [3, 0, 0])[1]);
    const m = new THREE.Matrix4().compose(new THREE.Vector3(5, 1, -2), new THREE.Quaternion(), new THREE.Vector3(4, 4, 4));
    const b = vdbFrameEmitterBounds(m.elements);
    expect(b.center).toEqual([5, 1, -2]);
    close(b.radius, 2 * Math.sqrt(3), 1e-12);
    expect(b.min).toEqual([3, -1, -4]);
  });

  // A cube of side L at c with constant extinction σ and emission e (local samplers), absorbance 1
  const cube = (sigma: number, e: Vec3, L = 10, c: Vec3 = [1, 2, 3], steps = 64): VDBFrameEmitter => {
    const m = new THREE.Matrix4().compose(new THREE.Vector3(...c), new THREE.Quaternion(), new THREE.Vector3(L, L, L));
    const bounds = vdbFrameEmitterBounds(m.elements);
    return {
      sampleExtinction: () => sigma,
      sampleEmission: () => e,
      getLod: () => 0,
      worldToLocal: m.clone().invert().elements,
      center: bounds.center,
      radius: bounds.radius,
      worldScale: L,
      absorbance: 1,
      steps,
    };
  };

  it("a direction texel is e × Σ ray area × (1 − exp(−σ × chord)) (Beer, chords from an independent slab test)", () => {
    const sigma = 0.05;
    const e: Vec3 = [2, 1, 0.5];
    const emitter = cube(sigma, e);
    for (const index of [0, 5, 11]) {
      const texel = vdbFrameEmissionDirectionTexel(emitter, index);
      const { direction, tangent, bitangent } = vdbFrameEmissionDirection(index);
      const R = emitter.radius;
      const area = ((2 * R) / 8) ** 2;
      let expected = 0;
      for (let r = 0; r < 64; r++) {
        const u = (((r % 8) + 0.5) / 8) * 2 - 1;
        const v = ((Math.floor(r / 8) + 0.5) / 8) * 2 - 1;
        const start = [0, 1, 2].map((k) => emitter.center[k]! + direction[k]! * R + tangent[k]! * u * R + bitangent[k]! * v * R);
        // slab test against the world cube [c − 5, c + 5]
        let t0 = -Infinity;
        let t1 = Infinity;
        for (let k = 0; k < 3; k++) {
          const a = (emitter.center[k]! - 5 - start[k]!) / -direction[k]!;
          const b = (emitter.center[k]! + 5 - start[k]!) / -direction[k]!;
          t0 = Math.max(t0, Math.min(a, b));
          t1 = Math.min(t1, Math.max(a, b));
        }
        t0 = Math.max(0, t0);
        t1 = Math.min(2 * R, t1);
        if (t1 > t0) expected += area * (1 - Math.exp(-sigma * (t1 - t0)));
      }
      for (let k = 0; k < 3; k++) close(texel.intensity[k]!, e[k]! * expected, 1e-9 * expected);
      expect(texel.intensity[3]).toBe(R);
    }
  });

  it("reduce: the mean of the 16 directions; an optically thin cube's emission centre is its centre", () => {
    const emitter = cube(0.002, [1, 1, 1]);
    const texels = Array.from({ length: 16 }, (_, i) => vdbFrameEmissionDirectionTexel(emitter, i));
    const reduced = vdbFrameEmissionReduce(texels);
    close(reduced.intensity[0], texels.reduce((s, t) => s + t.intensity[0], 0) / 16, 1e-12);
    expect(reduced.center[3]).toBe(1);
    for (let k = 0; k < 3; k++) close(reduced.center[k]!, emitter.center[k]!, 0.05);
    // a dark volume shows nothing
    expect(vdbFrameEmissionReduce(Array.from({ length: 16 }, (_, i) => vdbFrameEmissionDirectionTexel(cube(0.01, [0, 0, 0]), i))).center).toEqual([0, 0, 0, 0]);
  });

  it("probe: the L1 projection of a point light (E/4 + E/2 · n·l), softened by r/4, inverse square far away", () => {
    const light = { intensity: [8, 4, 2, 1] as [number, number, number, number], center: [0, 0, 0, 1] as [number, number, number, number] };
    const p = vdbFrameEmissionProbe([0, 0, 10], [light]);
    const sd2 = 100 + 0.0625;
    close(p[0][0], 8 / sd2 / 4, 1e-12);
    // facing the light: E/4 + E/2 × d²/(d² + s²)
    const facing = vdbFrameEmissionIrradiance(p, [0, 0, -1]);
    close(facing[0], (8 / sd2) * (0.25 + (0.5 * 100) / sd2), 1e-12);
    // facing away: 1/4 − 1/2 < 0 → 0
    expect(vdbFrameEmissionIrradiance(p, [0, 0, 1])[0]).toBe(0);
    const far = vdbFrameEmissionIrradiance(vdbFrameEmissionProbe([0, 0, 20], [light]), [0, 0, -1]);
    close(far[0] / facing[0], 0.25, 0.001);
    // an invisible emitter (centre w 0) adds nothing; two emitters add
    expect(vdbFrameEmissionProbe([0, 0, 10], [{ ...light, center: [0, 0, 0, 0] }])[0]).toEqual([0, 0, 0, 0]);
    close(vdbFrameEmissionProbe([0, 0, 10], [light, light])[1][0], 2 * p[1][0], 1e-12);
  });

  it("probe cascades: the last is the emitters' box grown by (range − 1)/2 × its longest side; finer ones snap around the camera", () => {
    const boxes = [{ min: [0, 0, 0] as Vec3, max: [100, 50, 100] as Vec3 }];
    const one = vdbFrameEmissionProbeGrid({ resolution: 16, range: 3, cascades: 1, distance: 512 }, boxes, [0, 0, 0]);
    expect(one.count).toBe(1);
    expect(one.min[0]).toEqual([-100, -100, -100]);
    expect(one.size[0]).toEqual([300, 250, 300]);
    expect(one.center).toEqual([50, 25, 50]);
    const two = vdbFrameEmissionProbeGrid({ resolution: 16, range: 3, cascades: 2, distance: 512 }, boxes, [50, 25, 50]);
    expect(two.count).toBe(2);
    // fine half size min(512, 300 / 2) / 2 = 75; voxel 2^ceil(log2(150 / 15)) = 16; size 256; min snapped to 16
    expect(two.size[0]).toEqual([256, 256, 256]);
    expect(two.min[0]).toEqual([-32, -64, -32]);
    expect(two.min[1]).toEqual([-100, -100, -100]);
    // a fine cascade away from every emitter is skipped
    expect(vdbFrameEmissionProbeGrid({ resolution: 16, range: 1, cascades: 2, distance: 10 }, boxes, [5000, 0, 0]).count).toBe(1);
    expect(vdbFrameEmissionProbeGrid({ resolution: 99, range: 1, cascades: 9, distance: 10 }, boxes, [0, 0, 0]).resolution).toBe(32);
    expect(one.key).not.toBe(two.key);
  });

  it("falloff past the last cascade: the squared distance ratio, ≤ 1", () => {
    close(vdbFrameEmissionFalloff([10, 0, 0], [20, 0, 0], [0, 0, 0]), 0.25, 1e-12);
    expect(vdbFrameEmissionFalloff([10, 0, 0], [5, 0, 0], [0, 0, 0])).toBe(1);
  });
});

describe("emission colour: the CIE 1931 'planck' / 'soot' table", () => {
  // XYZ from linear sRGB (IEC 61966-2-1 inverse); the table's colours are scaled, so only the chromaticity is compared
  const chromaticity = ([r, g, b]: Vec3) => {
    const X = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const Z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
    return [X / (X + Y + Z), Y / (X + Y + Z)];
  };

  it("'planck' lies on the Planckian locus (CIE 1931 x, y at 3000 K and 6500 K)", () => {
    const [x3, y3] = chromaticity(vdbFrameBlackbodyRgb(3000, 0));
    close(x3, 0.4369, 0.002);
    close(y3, 0.4041, 0.002);
    const [x6, y6] = chromaticity(vdbFrameBlackbodyRgb(6500, 0));
    close(x6, 0.3135, 0.002);
    close(y6, 0.3236, 0.002);
  });

  it("hot is blue, cool is red; soot (λ^−α) is bluer than planck at the same temperature; largest channel 1", () => {
    const cool = vdbFrameBlackbodyRgb(1000, 0);
    const hot = vdbFrameBlackbodyRgb(30000, 0);
    expect(cool[0]).toBe(1);
    expect(cool[2]).toBeLessThan(0.01);
    expect(hot[2]).toBe(1);
    const soot = vdbFrameBlackbodyRgb(2000, 1.39);
    const planck = vdbFrameBlackbodyRgb(2000, 0);
    expect(soot[1] / soot[0]).toBeGreaterThan(planck[1] / planck[0]);
  });

  it("the table read: texel values at texel temperatures, linear between; spectra map to (1, exponent)", () => {
    const t = vdbFrameBlackbodyTable();
    const T = (column: number) => 500 * Math.pow(80, column / (vdbFrameBlackbodyTableSize - 1));
    const c = vdbFrameBlackbodyColor(T(100), 0);
    for (let k = 0; k < 3; k++) close(c[k]!, t[100 * 4 + k]!, 1e-6);
    const mid = vdbFrameBlackbodyColor(Math.sqrt(T(100) * T(101)), 0);
    for (let k = 0; k < 3; k++) close(mid[k]!, (t[100 * 4 + k]! + t[101 * 4 + k]!) / 2, 1e-6);
    expect(vdbFrameEmissionSpectrum("planck", 2)).toEqual([1, 0]);
    expect(vdbFrameEmissionSpectrum("soot", 9)).toEqual([1, 3]);
    expect(vdbFrameEmissionSpectrum("soot", Number.NaN)).toEqual([1, 1.39]);
  });
});
