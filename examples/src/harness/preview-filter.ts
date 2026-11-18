/**
 * K4.19 harness: coarse-first preview filter A/B (image metric only; timings wait for a quiet machine, D25).
 *
 * For each volume (smoke, and the WDAS quarter cloud) the page puts the full grid and every preview variant
 * in one pool, then renders each with one `VDBVolumeMaterial` (`gridId` swapped, jitter off):
 *
 * - previews: `resampleGrid(source, { resolution: 48, filter, quantize: 'fpn' })` for filter in
 *   box / point / trilinear, on the source's active cell box ("aligned") and on the same lattice shifted by
 *   half a preview voxel on every axis ("shifted", `bbox` + `dimensions`), the aliasing check;
 * - each preview is drawn with `densityScale x voxelRatio` (K4.16: the material integrates per index voxel);
 * - the reference is the full-resolution grid at `densityScale`, from the same two camera poses.
 *
 * The frames go to `window.__VDB_PREVIEW_FILTER__` (base64 RGBA8) and e2e/preview-filter.spec.ts computes
 * PSNR and the fraction of pixels > 8/255 with scripts/k3/metrics.mjs.
 *
 * Query: `?w=&h=` (default 224), `?resolution=` (default 48), `?volumes=smoke,wdas`.
 */
import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { VDBGridPool } from "@pmndrs/vdb";
import { VDBGridNode, VDBVolumeMaterial, vdbProxyGeometry } from "@pmndrs/vdb/three";
import { NanoVDBFile, buildFromVdb, inspect, parseVdb, resampleGrid, type ResampleFilter } from "@pmndrs/vdb/tools";
import { createDeviceRenderer } from "../demos/device-renderer";

interface VariantInfo {
  key: string;
  filter: ResampleFilter;
  shift: "aligned" | "shifted";
  dims: number[];
  voxelCount: number;
  imageBytes: number;
  voxelRatio: number;
  densityScale: number;
  worldBBox: number[][];
}

interface VolumeResult {
  name: string;
  source: string;
  fullDensityScale: number;
  fullVoxelCount: number;
  fullWorldBBox: number[][];
  previewVoxel: number;
  /** pose index -> base64 RGBA8 (rows top-down). */
  full: string[];
  variants: (VariantInfo & { frames: string[]; nonBackground: number[] })[];
  fullNonBackground: number[];
}

interface Result {
  done: boolean;
  error?: string;
  missing?: string;
  adapter?: string;
  width?: number;
  height?: number;
  resolution?: number;
  poses?: number[][];
  volumes: VolumeResult[];
  uncaptured: string[];
}

declare global {
  interface Window {
    __VDB_PREVIEW_FILTER__?: Result;
  }
}

const params = new URLSearchParams(location.search);
const W = Number(params.get("w") ?? 224);
const H = Number(params.get("h") ?? 224);
const RESOLUTION = Number(params.get("resolution") ?? 48);
const WANTED = (params.get("volumes") ?? "smoke,wdas").split(",");
const BACKGROUND = 0x223049;
const FILTERS: ResampleFilter[] = ["box", "point", "trilinear"];
// Demo 02's two test poses: [azimuth, elevation, radius in diagonals].
const POSES: [number, number, number][] = [
  [Math.PI * 0.25, Math.PI * 0.16, 1.7],
  [Math.PI * 0.85, Math.PI * 0.34, 2.0],
];

const root = document.getElementById("demo-root")!;
const result: Result = { done: false, volumes: [], uncaptured: [] };
window.__VDB_PREVIEW_FILTER__ = result;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

interface VolumeSpec {
  name: string;
  url: string;
  densityScale: number;
}
const SPECS: VolumeSpec[] = [
  { name: "smoke", url: "/fixtures/vdb-samples/smoke.vdb", densityScale: 4 },
  // The Fp8 bake of the quarter cloud is both the full grid and the preview source (29 MB instead of a 68 MB .vdb).
  { name: "wdas", url: "/fixtures/wdas/wdas_cloud_quarter_fp8.nvdb", densityScale: 40 },
  // High-frequency content for the aliasing check (off by default: `?volumes=smoke,wdas,explosion,fire`).
  { name: "explosion", url: "/fixtures/vdb-samples/explosion.vdb", densityScale: 4 },
  { name: "fire", url: "/fixtures/vdb-samples/fire.vdb", densityScale: 4 },
].filter((s) => WANTED.includes(s.name));

/** The source: a NanoVDB image the previews are resampled from and the full grid the pool holds. */
async function loadSource(spec: VolumeSpec): Promise<Uint32Array | null> {
  const resp = await fetch(spec.url);
  if (!resp.ok) {
    result.missing = `${spec.url} -> ${resp.status}`;
    return null;
  }
  const buf = await resp.arrayBuffer();
  if (spec.url.endsWith(".vdb")) {
    const vdb = parseVdb(buf);
    const grid = vdb.grids.find((g) => g.valueType === "float");
    if (!grid) throw new Error(`${spec.url}: no float grid`);
    return buildFromVdb(grid);
  }
  return NanoVDBFile.fromArrayBuffer(buf).gridImage(0);
}

async function run(): Promise<void> {
  const { renderer, device, adapterName } = await createDeviceRenderer();
  result.adapter = adapterName;
  result.width = W;
  result.height = H;
  result.resolution = RESOLUTION;
  result.poses = POSES;
  device.addEventListener("uncapturederror", (e) => result.uncaptured.push((e as GPUUncapturedErrorEvent).error.message));

  const rt = new THREE.RenderTarget(W, H);
  rt.texture.colorSpace = THREE.SRGBColorSpace;
  const bg = new THREE.Color(BACKGROUND);
  const bgBytes = (() => {
    const c = bg.clone().convertLinearToSRGB();
    return [Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255)];
  })();

  async function frame(scene: THREE.Scene, camera: THREE.Camera, r: WebGPURenderer): Promise<{ rgba: string; nonBackground: number }> {
    r.setRenderTarget(rt);
    r.render(scene, camera);
    const pixels = (await r.readRenderTargetPixelsAsync(rt, 0, 0, W, H)) as Uint8Array;
    r.setRenderTarget(null);
    const packed = W * 4;
    const stride = H > 1 ? (pixels.byteLength - packed) / (H - 1) : packed;
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(pixels.subarray((H - 1 - y) * stride, (H - 1 - y) * stride + packed), y * packed);
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

    // Previews: three filters, aligned and shifted by half a preview voxel.
    const pool = await VDBGridPool.create(device);
    const fullId = pool.add(source);
    const variants: (VariantInfo & { id: number; frames: string[]; nonBackground: number[] })[] = [];
    for (const filter of FILTERS) {
      const base = resampleGrid(source, { resolution: RESOLUTION, filter, quantize: "fpn" });
      const h = base.voxelSize / 2;
      const shiftedBox: [[number, number, number], [number, number, number]] = [
        [base.bbox[0][0] + h, base.bbox[0][1] + h, base.bbox[0][2] + h],
        [base.bbox[1][0] + h, base.bbox[1][1] + h, base.bbox[1][2] + h],
      ];
      const shifted = resampleGrid(source, { dimensions: base.dimensions, bbox: shiftedBox, filter, quantize: "fpn" });
      for (const [shift, g] of [["aligned", base], ["shifted", shifted]] as const) {
        const id = pool.add(g.image);
        const info = pool.info(id);
        variants.push({
          key: `${filter}/${shift}`,
          filter,
          shift,
          id,
          dims: [...g.dimensions],
          voxelCount: g.voxelCount,
          imageBytes: g.image.byteLength,
          voxelRatio: g.voxelSize / g.sourceVoxelSize[0],
          densityScale: spec.densityScale * (g.voxelSize / g.sourceVoxelSize[0]),
          worldBBox: [[...info.worldBBox.min], [...info.worldBBox.max]],
          frames: [],
          nonBackground: [],
        });
      }
    }
    pool.flush();
    await pool.settled();

    const grids = new VDBGridNode(renderer, pool, { helpers: ["rayClipSupport"] });
    const material = new VDBVolumeMaterial(renderer, grids, fullId, { jitter: false, densityScale: spec.densityScale });
    const mesh = new THREE.Mesh(vdbProxyGeometry(pool.info(fullId)), material);
    const scene = new THREE.Scene();
    scene.background = bg;
    scene.add(mesh);
    grids.update();

    const fullInfo = pool.info(fullId);
    const box = new THREE.Box3(new THREE.Vector3(...fullInfo.worldBBox.min), new THREE.Vector3(...fullInfo.worldBBox.max));
    const center = box.getCenter(new THREE.Vector3());
    const diag = box.getSize(new THREE.Vector3()).length();
    const camera = new THREE.PerspectiveCamera(45, W / H, diag * 0.01, diag * 100);
    const place = ([az, el, rf]: [number, number, number]): void => {
      const r = diag * rf;
      camera.position.set(center.x + r * Math.cos(el) * Math.cos(az), center.y + r * Math.sin(el), center.z + r * Math.cos(el) * Math.sin(az));
      camera.lookAt(center);
      camera.updateMatrixWorld();
    };

    const show = (id: number, densityScale: number): void => {
      material.gridId.value = id;
      material.densityScale.value = densityScale;
      mesh.geometry.dispose();
      mesh.geometry = vdbProxyGeometry(pool.info(id));
    };

    const vol: VolumeResult = {
      name: spec.name,
      source: spec.url,
      fullDensityScale: spec.densityScale,
      fullVoxelCount: inspect(source).voxelCount,
      fullWorldBBox: [[...fullInfo.worldBBox.min], [...fullInfo.worldBBox.max]],
      previewVoxel: variants[0] ? variants[0].voxelRatio : 0,
      full: [],
      fullNonBackground: [],
      variants: [],
    };
    show(fullId, spec.densityScale);
    for (const pose of POSES) {
      place(pose);
      const f = await frame(scene, camera, renderer);
      vol.full.push(f.rgba);
      vol.fullNonBackground.push(f.nonBackground);
    }
    log(`${spec.name}: full rendered (${vol.fullNonBackground.join(", ")} px)`);
    for (const v of variants) {
      show(v.id, v.densityScale);
      for (const pose of POSES) {
        place(pose);
        const f = await frame(scene, camera, renderer);
        v.frames.push(f.rgba);
        v.nonBackground.push(f.nonBackground);
      }
      const { id: _id, ...rest } = v;
      vol.variants.push(rest);
      log(`${spec.name} ${v.key}: voxel ratio ${v.voxelRatio.toFixed(2)}, ${v.nonBackground.join(", ")} px`);
    }
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
