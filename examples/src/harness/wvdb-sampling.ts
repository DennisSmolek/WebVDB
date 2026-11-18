/**
 * K4.21 harness: `.wvdb` bake sampling A/B (`sampling: 'nearest'` vs `'box'`), image metric only (D25).
 *
 * For each volume the page bakes the sparse NanoVDB grid to `.wvdb` in memory (`nvdbToWvdb`) with both
 * samplings at each resolution, reads each file back and renders it through the K3.0 dense source
 * (`wvdbDenseVolume` -> `vdbDenseTextures` -> `vdbVolumeNodes`, no mip atlas, `clamp01` texels) with the K3 harness's
 * emission-absorption march. Two references from the same two camera poses:
 *
 * - `ea`: the SAME march and box, density read from the full-resolution sparse grid in a `VDBGridPool`
 *   (`vdbSample`, trilinear, clamped to [0, 1] as the dense texels are). Only the bake differs: this is the
 *   like-for-like reference and the primary metric.
 * - `vvm`: the sparse grid drawn by `VDBVolumeMaterial` (its own lighting and step model), as the task asked.
 *   The two renderers differ, so its absolute PSNR is low for every variant; only the ranking carries.
 *
 * Frames go to `window.__VDB_WVDB_SAMPLING__` (base64 RGBA8, rows top first); e2e/wvdb-sampling.spec.ts computes the
 * metrics with scripts/k3/metrics.mjs. Query: `?w=&h=` (224), `?res=128,64`, `?volumes=smoke,bunny_cloud,wdas`,
 * `?steps=` (192), `?extinction=` (20).
 */
import * as THREE from "three";
import { NodeMaterial, type WebGPURenderer } from "three/webgpu";
import { Fn, Loop, cameraPosition, clamp, exp, float, length, modelWorldMatrix, modelWorldMatrixInverse, positionLocal, uniform, vec3, vec4 } from "three/tsl";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, VDBVolumeMaterial, vdbDenseTextures, vdbProxyGeometry, vdbSample, vdbVolumeNodes, vdbVolumeUniforms } from "@pmndrs/vdb/three";
import { NvdbFile, buildFromVdb, nvdbToWvdb, parseVdb, readWvdb, wvdbDenseVolume, type WvdbSampling } from "@pmndrs/vdb/tools";
import { createDeviceRenderer } from "../demos/device-renderer";

// The K3 frame files take TSL through untyped nodes (three r187's typings reject most shader code).
/* eslint-disable @typescript-eslint/no-explicit-any */
type Node = any;

interface VariantResult {
  key: string;
  sampling: WvdbSampling;
  resolution: number;
  dims: number[];
  fileBytes: number;
  meanDensity: number;
  nonZeroVoxels: number;
  /** Mean of the 8-bit density texels over the whole box (0..1): total mass in texel units. */
  texelMean: number;
  frames: string[];
  nonBackground: number[];
}
interface VolumeResult {
  name: string;
  source: string;
  box: number[][];
  ea: string[];
  eaNonBackground: number[];
  vvm: string[];
  vvmNonBackground: number[];
  variants: VariantResult[];
}
interface Result {
  done: boolean;
  error?: string;
  missing?: string;
  adapter?: string;
  width?: number;
  height?: number;
  steps?: number;
  extinction?: number;
  volumes: VolumeResult[];
  uncaptured: string[];
}
declare global {
  interface Window {
    __VDB_WVDB_SAMPLING__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const W = Number(params.get("w") ?? 224);
const H = Number(params.get("h") ?? 224);
const STEPS = Number(params.get("steps") ?? 192);
const EXTINCTION = Number(params.get("extinction") ?? 20);
const RESOLUTIONS = (params.get("res") ?? "128,64").split(",").map(Number);
const WANTED = (params.get("volumes") ?? "smoke,bunny_cloud,wdas").split(",");
const SAMPLINGS: WvdbSampling[] = ["nearest", "box"];
const BACKGROUND = new THREE.Color(0.12, 0.16, 0.22);
const POSES: [number, number, number][] = [
  [Math.PI * 0.25, Math.PI * 0.16, 1.7],
  [Math.PI * 0.85, Math.PI * 0.34, 2.0],
];

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, volumes: [], uncaptured: [] };
window.__VDB_WVDB_SAMPLING__ = result;
const log = (text: string, cls = ""): void => {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
};

interface Spec {
  name: string;
  url: string;
  densityScale: number;
}
const SPECS: Spec[] = [
  { name: "smoke", url: "/fixtures/vdb-samples/smoke.vdb", densityScale: 4 },
  { name: "bunny_cloud", url: "/fixtures/vdb-samples/bunny_cloud.vdb", densityScale: 4 },
  { name: "wdas", url: "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb", densityScale: 40 },
].filter((s) => WANTED.includes(s.name));

async function loadSource(spec: Spec): Promise<Uint32Array | null> {
  const resp = await fetch(spec.url);
  if (!resp.ok) {
    result.missing = `${spec.url} -> ${resp.status}`;
    return null;
  }
  const buf = await resp.arrayBuffer();
  if (spec.url.endsWith(".vdb")) {
    const grid = parseVdb(buf).grids.find((g) => g.valueType === "float");
    if (!grid) throw new Error(`${spec.url}: no float grid`);
    return buildFromVdb(grid);
  }
  return NvdbFile.fromArrayBuffer(buf).gridImage(0);
}

interface Frame {
  rgba: string;
  nonBackground: number;
}

async function run(): Promise<void> {
  const { renderer, device, adapterName } = await createDeviceRenderer();
  result.adapter = adapterName;
  result.width = W;
  result.height = H;
  result.steps = STEPS;
  result.extinction = EXTINCTION;
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const rt = new THREE.RenderTarget(W, H);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  const bgSrgb = BACKGROUND.clone().convertLinearToSRGB();
  const bgBytes = [bgSrgb.r, bgSrgb.g, bgSrgb.b].map((c) => Math.round(c * 255));

  async function frame(scene: THREE.Scene, camera: THREE.Camera, r: WebGPURenderer): Promise<Frame> {
    r.setRenderTarget(rt);
    r.render(scene, camera);
    const pixels = (await r.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as Uint8Array;
    r.setRenderTarget(null);
    const tight = W * 4;
    const stride = H > 1 ? (pixels.byteLength - tight) / (H - 1) : tight;
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(pixels.subarray((H - 1 - y) * stride, (H - 1 - y) * stride + tight), y * tight);
    let nonBackground = 0;
    for (let i = 0; i < out.length; i += 4) {
      if (Math.abs(out[i]! - bgBytes[0]!) + Math.abs(out[i + 1]! - bgBytes[1]!) + Math.abs(out[i + 2]! - bgBytes[2]!) > 6) nonBackground++;
    }
    let bin = "";
    for (let i = 0; i < out.length; i += 0x8000) bin += String.fromCharCode(...out.subarray(i, i + 0x8000));
    return { rgba: btoa(bin), nonBackground };
  }

  for (const spec of SPECS) {
    const source = await loadSource(spec);
    if (!source) {
      log(`fixture missing: ${result.missing}`, "fail");
      return;
    }
    log(`${spec.name}: source loaded (${source.byteLength} B image)`);

    // The bakes, read back as dense volumes (no mip atlas: isolate the sampling).
    const baked: { key: string; sampling: WvdbSampling; resolution: number; fileBytes: number; header: { meanDensity?: number; nonZeroVoxels?: number }; volume: ReturnType<typeof wvdbDenseVolume> }[] = [];
    for (const resolution of RESOLUTIONS) {
      for (const sampling of SAMPLINGS) {
        const out = nvdbToWvdb(source, { resolution, sampling });
        const file = readWvdb(out.bytes);
        const volume = wvdbDenseVolume(file, { mipmaps: false });
        const h = out.header as unknown as { grids?: { meanDensity?: number; nonZeroVoxels?: number }[] };
        baked.push({ key: `${sampling}/${resolution}`, sampling, resolution, fileBytes: out.bytes.length, header: h.grids?.[0] ?? {}, volume });
        log(`${spec.name} ${sampling} ${resolution}: ${volume.dimensions.join("x")} texels, ${out.bytes.length} B`);
      }
    }
    const box = baked[0]!.volume.bbox ?? [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]];
    for (const b of baked) {
      const bb = b.volume.bbox;
      if (!bb || bb.some((v, i) => v.some((c, a) => Math.abs(c - box[i]![a]!) > 1e-9))) throw new Error(`${spec.name}: bakes have different boxes`);
    }
    const center = new THREE.Vector3(...box[0]!).add(new THREE.Vector3(...box[1]!)).multiplyScalar(0.5);
    const size = new THREE.Vector3(...box[1]!).sub(new THREE.Vector3(...box[0]!));
    const worldScale = Math.max(size.x, size.y, size.z);
    const diag = size.length();

    // The sparse reference grid.
    const pool = await VDBGridPool.create(device);
    const fullId = pool.add(source);
    pool.flush();
    await pool.settled();
    const grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
    grids.update();

    const camera = new THREE.PerspectiveCamera(45, W / H, diag * 0.01, diag * 100);
    const place = ([az, el, rf]: [number, number, number]): void => {
      const r = diag * rf;
      camera.position.set(center.x + r * Math.cos(el) * Math.cos(az), center.y + r * Math.sin(el), center.z + r * Math.cos(el) * Math.sin(az));
      camera.lookAt(center);
      camera.updateMatrixWorld();
    };

    // One emission-absorption march (the K3.0 harness's loop): `density(localPos)` is the extinction per world unit.
    const march = (density: (p: Node) => Node, boxSegment: (a: Node, b: Node) => Node): NodeMaterial => {
      const material = new NodeMaterial();
      material.side = THREE.BackSide;
      material.depthWrite = false;
      const background = uniform(BACKGROUND.clone());
      material.fragmentNode = (Fn as any)(() => {
        const start = modelWorldMatrixInverse.mul(vec4(cameraPosition, 1.0)).xyz.toVar();
        const delta = positionLocal.sub(start).toVar();
        const seg = boxSegment(start, positionLocal).toVar();
        const dt = seg.y.sub(seg.x).div(float(STEPS)).toVar();
        const stepWorld = length(delta).mul(dt).mul(worldScale).toVar();
        const transmittance = float(1.0).toVar();
        const radiance = vec3(0.0).toVar();
        (Loop as any)(STEPS, ({ i }: { i: Node }) => {
          const t = seg.x.add(float(i).add(0.5).mul(dt));
          const p = start.add(delta.mul(t)).toVar();
          const tau = density(p).mul(stepWorld);
          radiance.addAssign(transmittance.mul(float(1.0).sub(exp(tau.negate()))).mul(vec3(1.0, 0.97, 0.92)));
          transmittance.mulAssign(exp(tau.negate()));
        });
        return vec4(radiance.add(background.mul(transmittance)), 1.0);
      })();
      return material;
    };

    const boxMesh = (material: THREE.Material): THREE.Mesh => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
      mesh.position.copy(center);
      mesh.scale.copy(size);
      return mesh;
    };
    const render = async (object: THREE.Object3D): Promise<Frame[]> => {
      const scene = new THREE.Scene();
      scene.background = BACKGROUND.clone();
      scene.add(object);
      const out: Frame[] = [];
      for (const pose of POSES) {
        place(pose);
        out.push(await frame(scene, camera, renderer));
      }
      return out;
    };

    const vol: VolumeResult = { name: spec.name, source: spec.url, box, ea: [], eaNonBackground: [], vvm: [], vvmNonBackground: [], variants: [] };

    // Dense variants first: their nodes also give the reference the same box intersection.
    let boxSegment: ((a: Node, b: Node) => Node) | null = null;
    for (const b of baked) {
      const dense = vdbDenseTextures(b.volume);
      const u = vdbVolumeUniforms(dense, { extinction: EXTINCTION, normalizeDensity: false, worldScale });
      const nodes = vdbVolumeNodes(dense, u, { useMipmaps: false });
      boxSegment ??= nodes.boxSegment as (a: Node, b: Node) => Node;
      // sampleExtinction is per world unit (divided by worldScale); the march multiplies by stepWorld.
      const material = march((p) => nodes.sampleExtinction(p, float(0.0)), nodes.boxSegment as (a: Node, b: Node) => Node);
      const frames = await render(boxMesh(material));
      let sum = 0;
      const texels = b.volume.density.data as Uint8Array;
      for (const t of texels) sum += t;
      vol.variants.push({
        key: b.key,
        sampling: b.sampling,
        resolution: b.resolution,
        dims: [...b.volume.dimensions],
        fileBytes: b.fileBytes,
        meanDensity: b.header.meanDensity ?? NaN,
        nonZeroVoxels: b.header.nonZeroVoxels ?? NaN,
        texelMean: sum / (255 * texels.length),
        frames: frames.map((f) => f.rgba),
        nonBackground: frames.map((f) => f.nonBackground),
      });
      material.dispose();
      dense.dispose();
      log(`${spec.name} ${b.key}: ${frames.map((f) => f.nonBackground).join(", ")} px`);
    }

    // Like-for-like reference: the same march over the sparse grid, clamped to [0, 1] like the 8-bit texels.
    const eaMaterial = march(
      (p) => (clamp as any)((vdbSample as any)(grids, fullId, modelWorldMatrix.mul(vec4(p, 1.0)).xyz), 0.0, 1.0).mul(EXTINCTION).div(worldScale),
      boxSegment!,
    );
    const ea = await render(boxMesh(eaMaterial));
    vol.ea = ea.map((f) => f.rgba);
    vol.eaNonBackground = ea.map((f) => f.nonBackground);
    log(`${spec.name} ea reference: ${vol.eaNonBackground.join(", ")} px`);

    // The task's reference: VDBVolumeMaterial on the pool.
    const vvm = new VDBVolumeMaterial(renderer, grids, fullId, { jitter: false, densityScale: spec.densityScale });
    const vvmFrames = await render(new THREE.Mesh(vdbProxyGeometry(pool.info(fullId)), vvm));
    vol.vvm = vvmFrames.map((f) => f.rgba);
    vol.vvmNonBackground = vvmFrames.map((f) => f.nonBackground);
    log(`${spec.name} VDBVolumeMaterial reference: ${vol.vvmNonBackground.join(", ")} px`);

    result.volumes.push(vol);
    pool.dispose();
  }
}

run()
  .catch((err: unknown) => {
    result.error = err instanceof Error ? `${err.message}\n${err.stack ?? ""}` : String(err);
    log(result.error, "fail");
  })
  .finally(() => {
    result.done = true;
  });
