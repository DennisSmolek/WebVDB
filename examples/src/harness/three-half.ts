/**
 * K7.H harness: a NanoVDB Half grid in `@pmndrs/vdb/three`. No fixtures: the page builds a fog ramp
 * (`vdbBuildFromDense`) and its Half copy (`vdbQuantize(…, 'half')`), puts both in one pool and renders each
 * with one `VDBVolumeMaterial` (`gridId` swapped, jitter off) through a `VDBGridNode` made with
 * `types: ['float', 'half']`; then a second pool holding only the Half grid renders through a node whose
 * types default to the pool's (`['half']`). The frames go to `window.__VDB_THREE_HALF__` with the per-pixel
 * difference; e2e/three-half.spec.ts gates them.
 */
import * as THREE from "three";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, VDBVolumeMaterial, vdbProxyGeometry } from "@pmndrs/vdb/three";
import { vdbBuildFromDense, vdbQuantize } from "@pmndrs/vdb/tools";
import { createDeviceRenderer } from "../demos/device-renderer";

interface Result {
  done: boolean;
  error?: string;
  adapter?: string;
  width?: number;
  height?: number;
  /** Pixels whose RGB differs from the background by more than 6/255, per frame. */
  nonBackground?: { float: number; half: number; halfOnly: number };
  /** Max and mean absolute RGB difference (0..255) between the float and half frames, and the half-only frame. */
  diff?: { max: number; mean: number; maxHalfOnly: number };
  nodeTypes?: { mixed: string[]; halfOnly: string[] };
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_THREE_HALF__?: Result;
  }
}

const W = 160;
const H = 160;
const BACKGROUND = 0x223049;
const root = document.getElementById("demo-root")!;
const result: Result = { done: false, uncaptured: [] };

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

/** A 48^3 fog ball with a ramp and some structure: values that a half can't hold exactly. */
function fogBall(): Uint32Array {
  const n = 48;
  const values = new Float32Array(n * n * n);
  const c = (n - 1) / 2;
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++)
      for (let z = 0; z < n; z++) {
        const d = Math.hypot(x - c, y - c, z - c) / (n * 0.45);
        values[(x * n + y) * n + z] = d < 1 ? (1 - d) * (0.6 + 0.4 * Math.sin(x * 0.7) * Math.cos(z * 0.5)) + 1e-3 / 3 : 0;
      }
  return vdbBuildFromDense(values, [n, n, n], { origin: [-24, -24, -24], voxelSize: 0.05, background: 0 });
}

async function run(): Promise<void> {
  const { renderer, device, adapterName } = await createDeviceRenderer();
  result.adapter = adapterName;
  result.width = W;
  result.height = H;
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const floatImage = fogBall();
  const halfImage = vdbQuantize(floatImage, "half");

  const rt = new THREE.RenderTarget(W, H);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  const bg = new THREE.Color(BACKGROUND);
  const c = bg.clone().convertLinearToSRGB();
  const bgBytes = [Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255)];

  const camera = new THREE.PerspectiveCamera(45, W / H, 0.01, 100);
  camera.position.set(2.2, 1.3, 2.6);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  async function frame(scene: THREE.Scene): Promise<{ rgb: Uint8Array; nonBackground: number }> {
    renderer.setRenderTarget(rt);
    renderer.render(scene, camera);
    const pixels = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as Uint8Array;
    renderer.setRenderTarget(null);
    const packed = W * 4;
    const stride = (pixels.byteLength - packed) / (H - 1);
    const rgb = new Uint8Array(W * H * 3);
    let nonBackground = 0;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const i = y * stride + x * 4;
        const o = (y * W + x) * 3;
        for (let k = 0; k < 3; k++) rgb[o + k] = pixels[i + k]!;
        if (Math.abs(rgb[o]! - bgBytes[0]!) + Math.abs(rgb[o + 1]! - bgBytes[1]!) + Math.abs(rgb[o + 2]! - bgBytes[2]!) > 6) nonBackground++;
      }
    return { rgb, nonBackground };
  }

  // Float and Half in one pool, one node compiled for both.
  const pool = await VDBGridPool.create(device);
  const floatId = pool.add(floatImage);
  const halfId = pool.add(halfImage);
  pool.flush();
  await pool.settled();
  const grids = new VDBGridNode(renderer, pool, { types: ["float", "half"], helpers: ["rayClipSupport"] });
  const material = new VDBVolumeMaterial(renderer, grids, floatId, { jitter: false, densityScale: 40 });
  const mesh = new THREE.Mesh(vdbProxyGeometry(pool.info(floatId)), material);
  const scene = new THREE.Scene();
  scene.background = bg;
  scene.add(mesh);
  grids.update();
  const f = await frame(scene);
  material.gridId.value = halfId;
  const h = await frame(scene);

  // A pool with only the Half grid: the node's default types are the pool's.
  const halfPool = await VDBGridPool.create(device);
  const onlyId = halfPool.add(halfImage);
  halfPool.flush();
  await halfPool.settled();
  const halfGrids = new VDBGridNode(renderer, halfPool, { helpers: ["rayClipSupport"] });
  const halfMaterial = new VDBVolumeMaterial(renderer, halfGrids, onlyId, { jitter: false, densityScale: 40 });
  const halfScene = new THREE.Scene();
  halfScene.background = bg;
  halfScene.add(new THREE.Mesh(vdbProxyGeometry(halfPool.info(onlyId)), halfMaterial));
  halfGrids.update();
  const o = await frame(halfScene);

  let max = 0;
  let sum = 0;
  let maxHalfOnly = 0;
  for (let i = 0; i < f.rgb.length; i++) {
    const d = Math.abs(f.rgb[i]! - h.rgb[i]!);
    max = Math.max(max, d);
    sum += d;
    maxHalfOnly = Math.max(maxHalfOnly, Math.abs(h.rgb[i]! - o.rgb[i]!));
  }
  result.nonBackground = { float: f.nonBackground, half: h.nonBackground, halfOnly: o.nonBackground };
  result.diff = { max, mean: sum / f.rgb.length, maxHalfOnly };
  result.nodeTypes = { mixed: [...grids.options.types], halfOnly: [...halfGrids.options.types] };
  log(`float ${f.nonBackground} px, half ${h.nonBackground} px, half-only pool ${o.nonBackground} px; max diff ${max}/255, mean ${result.diff.mean.toFixed(3)}`, "ok");
  pool.dispose();
  halfPool.dispose();
}

root.textContent = "";
run()
  .catch((e: unknown) => {
    result.error = e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e);
    log(`error: ${result.error}`, "fail");
  })
  .finally(() => {
    result.done = true;
    window.__VDB_THREE_HALF__ = result;
  });
