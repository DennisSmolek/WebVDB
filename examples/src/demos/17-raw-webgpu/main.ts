/**
 * Demo 17 · Rendering without three (K6.9, the K6 exit-gate showcase): the core frame renderer on raw WebGPU, with no
 * three.js and no engine. The page owns the device, the encoder and every pass that is not a volume; the core records
 * its compute kernels into that encoder and hands back textures the page's own shaders read.
 *
 * - The WDAS cloud (sparse, Fp8, `vdbSparseSourceGpu`; the committed CC0 JangaFX cloud when the git-ignored fixture is
 *   absent) and a CC0 EmberGen explosion frame (sparse density + temperature, emissive), in one pool and one source.
 * - 'frame' lighting (or 'volume') with a procedural HDR sky as the environment (an equirect GPU texture the core
 *   reduces on the GPU) and a sun that requests a volume shadow map.
 * - Shadows both ways: the ground, the obelisk, the plinth and the level set read the core's Beer map
 *   (`vdbVolumeShadowWgsl`: volume → mesh); the page's own sun depth map, with the obelisk and the level set in it, goes
 *   back in as an occluder (`VDBFrameInput.occluders`: mesh → volume).
 * - The explosion lights the ground through the emission probes (`vdbEmissionProbeWgsl`).
 * - A level set (the OpenVDB icosahedron, or a procedural one) through the core's raw level-set pass
 *   (`vdbCreateLevelSetPassGpu`, G-buffer output) shaded by the page with the same lighting.
 * - The volumes composite either through the core's pass (`frame.render`) or in the page's own present shader
 *   (`vdbFrameIntegrateWgsl`, the host-side seam).
 *
 * The core APIs used here are internal until 0.2.0 (K6.8 makes them public), so the page imports
 * `packages/vdb/src/core/internal.ts` by path. The docs page "Rendering without three" walks through this file.
 *
 * Query (with the shell's `presentation`, `still`, `panel`): `?quality=low|medium|high`, `?lighting=frame|volume`,
 * `?composite=core|host`, `?cloud=wdas|cc0`, `?levelset=icosahedron|procedural`, `?test=1` (still camera, no loop,
 * runs the statistical checks of e2e/demo-17.spec.ts and puts them in `__DEMO17__.info.checks`).
 */
import "../webgpu-gate.js";
import { fixtureUrl } from "../assets.js";
import { VDBGridPool } from "@pmndrs/vdb";
import { NvdbFile, buildFromVdb, parseVdb, readWvdb, vdbBuildFromDense, wvdbDenseToNvdb, wvdbDenseVolume } from "@pmndrs/vdb/tools";
import {
  vdbCreateFrameRenderer,
  vdbCreateLevelSetPassGpu,
  vdbSparseSourceGpu,
  type VDBFrameInput,
  type VDBFrameLight,
  type VDBFrameOccluder,
  type VDBFrameRendererGpu,
  type VDBFrameVolumeGpu,
} from "../../../../packages/vdb/src/core/internal.js";
import { MiB, PRESENTATION, QUALITY, STILL, fetchWithProgress, initialQuality, mountShowcase, params, probe, snapshotOf, type Quality, type Snapshot } from "../showcase-ui.js";
import { RawPresenter, RawTargets, boxMatrix, lookAt, multiply, normalize, orthographic, perspective, rawCamera, type RawCamera, type Vec3 } from "./raw-frame.js";
import { SKY_HEIGHT, SKY_WIDTH, skyAmbient, skyTexels } from "./sky.js";
import { createPlaceholders, createScenePipelines, meshVertices, packScene, SCENE_FLOATS, VERTEX_FLOATS, type Box } from "./scene-passes.js";
import cloudWvdbUrl from "../../../../packages/vdb/test/tools/fixtures/wvdb/cloud_04_variant_0000.wvdb?url";
import explosionWvdbUrl from "../../../../packages/vdb/test/tools/fixtures/wvdb/ground_explosion_0010.wvdb?url";

const ui = mountShowcase({
  number: "17",
  title: "Rendering without three",
  blurb:
    "The froxel renderer on raw WebGPU, no three.js: our own passes draw the sky, ground and a level set, read the volumes' shadow map and fire light, and feed our depth map back so meshes shadow the cloud.",
  stateKey: "__DEMO17__",
});
const state = ui.state;

const TEST = params.get("test") === "1";
const WDAS_URL = fixtureUrl("wdas/wdas_cloud_quarter_fp8.nvdb");
const ICOSAHEDRON_URL = fixtureUrl("vdb-samples/icosahedron.vdb");

type Lighting = "frame" | "volume";
type Composite = "core" | "host";

/** The adjustable state (controls write it, the frame reads it). */
const look = {
  quality: initialQuality(),
  lighting: (params.get("lighting") === "volume" ? "volume" : "frame") as Lighting,
  composite: (params.get("composite") === "host" ? "host" : "core") as Composite,
  sunElevation: 42,
  sunAzimuth: 28,
  sunIntensity: 2.4,
  fireLight: 60,
  sky: 1,
  exposure: 0.85,
  volumeShadows: true,
  occluders: true,
  obeliskX: 92,
  flames: 150,
};

/** One frame's switches beyond `look` (the test checks turn single features off). */
interface FrameOptions {
  volumeShadows?: boolean;
  occluders?: boolean;
  emission?: boolean;
  volumes?: boolean;
  levelSet?: boolean;
  composite?: Composite;
}

// ------------------------------------------------------------------------------------------------ scene layout
// World units ~ metres, y up. The cloud floats over the origin; the sun sits low in +x, so the obelisk stands between
// it and the cloud (its shadow crosses the cloud) and the cloud's shadow falls toward −x, where the explosion burns.
const CLOUD_CENTER: Vec3 = [0, 140, 0];
const CLOUD_SIZE = 230;
const EXPLOSION_AT: Vec3 = [-125, 0, -20];
const EXPLOSION_HEIGHT = 110;
const PLINTH: Box = { center: [40, 6, 150], size: [44, 12, 44], albedo: [0.42, 0.4, 0.37] };
const LEVEL_SET_SIZE = 38;
const LEVEL_SET_AT: Vec3 = [40, 12 + LEVEL_SET_SIZE / 2, 150];
const GROUND_HALF = 1400;
const SHADOW_HALF = 480;
const SHADOW_CENTER: Vec3 = [-20, 60, 20];
const obelisk = (): Box => ({ center: [look.obeliskX, 120, 52], size: [16, 240, 16], albedo: [0.32, 0.3, 0.28] });

async function main(): Promise<void> {
  //* 1. The host's device (the pool's required limits), its canvas and targets.
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter.");
  state.adapter = [adapter.info.vendor, adapter.info.architecture, adapter.info.description].filter(Boolean).join(" ");
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    state.uncaptured.push(message);
    ui.fail(`WebGPU: ${message}`);
  });
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:100%;height:100%;touch-action:none;cursor:grab";
  ui.app.append(canvas);
  const presenter = new RawPresenter(device, canvas, PRESENTATION);
  const targets = new RawTargets(device, "demo-17");

  //* 2. The volumes: one pool, one sparse source, two volumes (cloud, explosion).
  ui.status("loading the cloud…", null);
  const pool = await VDBGridPool.create(device, { label: "demo-17 volumes" });
  const wantWdas = params.get("cloud") !== "cc0";
  const wdas = wantWdas && (await probe(WDAS_URL));
  const source = vdbSparseSourceGpu(pool, { types: ["fp8", "float"], lod: "coarse-pyramid", label: "demo-17 sparse" });
  let cloudSlot;
  let cloudName: string;
  if (wdas) {
    const bytes = await fetchWithProgress(WDAS_URL, (got, total) => ui.status(`loading the WDAS cloud… ${MiB(got)}`, total ? got / total : null));
    const image = NvdbFile.fromArrayBuffer(bytes).gridImage(0);
    cloudSlot = source.volume({ density: pool.add(image, { label: "wdas" }) }, { image });
    cloudName = "WDAS cloud (Fp8, sparse)";
  } else {
    if (wantWdas) ui.note("WDAS cloud not found (fixtures/wdas/, local only): the committed CC0 JangaFX cloud instead.");
    const dense = wvdbDenseVolume(readWvdb(new Uint8Array(await (await fetch(cloudWvdbUrl)).arrayBuffer())));
    const grid = wvdbDenseToNvdb(dense);
    cloudSlot = source.volume({ density: pool.add(grid.image, { label: "cc0 cloud" }) }, { indexBox: grid.textureIndexBox, stats: grid.sparseSourceOptions.stats });
    cloudName = "JangaFX cloud (CC0, sparse)";
  }
  // The explosion: density and temperature of one EmberGen frame (CC0), as two sparse grids of one volume
  // his explosion setup: the "flames" grid is the emission (temperature) channel, scaled by 6 (frame.js L25–47)
  const explosionDense = wvdbDenseVolume(readWvdb(new Uint8Array(await (await fetch(explosionWvdbUrl)).arrayBuffer())), { emission: "flames", scales: { flames: 6 } });
  const exDensity = wvdbDenseToNvdb(explosionDense);
  const exEmission = wvdbDenseToNvdb(explosionDense, { channel: "emission" });
  const explosionSlot = source.volume(
    { density: pool.add(exDensity.image, { label: "explosion density" }), emission: pool.add(exEmission.image, { label: "explosion temperature" }) },
    { indexBox: exDensity.textureIndexBox, auxIndexBoxes: { emission: exEmission.textureIndexBox }, stats: exDensity.sparseSourceOptions.stats },
  );

  //* 3. The level set, in its own pool (its pass compiles the pool's types: float only).
  ui.status("loading the level set…", null);
  const lsPool = await VDBGridPool.create(device, { label: "demo-17 level set" });
  const levelSet = await loadLevelSet(lsPool);
  const lsMain = vdbCreateLevelSetPassGpu(device, lsPool, { colorFormat: RawTargets.COLOR_FORMAT, depthFormat: RawTargets.DEPTH_FORMAT, output: "gbuffer", types: ["float"], label: "demo-17 level set" });
  // the same pipeline (one compile) drawn from the sun into its depth map
  const lsShadow = vdbCreateLevelSetPassGpu(device, lsPool, { colorFormat: RawTargets.COLOR_FORMAT, depthFormat: RawTargets.DEPTH_FORMAT, output: "gbuffer", types: ["float"], label: "demo-17 level set (sun)" });

  //* 4. The core renderer and its two volumes.
  const frame = vdbCreateFrameRenderer(device, { composite: { colorFormat: RawTargets.COLOR_FORMAT, depthFormat: RawTargets.DEPTH_FORMAT }, label: "demo-17 frame" });
  const cloudIndex = cloudSlot.indexBox;
  const cloudDims = [0, 1, 2].map((k) => cloudIndex.max[k]! - cloudIndex.min[k]! + 1);
  const cloudScale = CLOUD_SIZE / Math.max(...cloudDims);
  const cloudHalf = cloudDims.map((d) => (d * cloudScale) / 2);
  const cloud = frame.addVolume(cloudSlot, {
    matrixWorld: boxMatrix(
      CLOUD_CENTER.map((c, k) => c - cloudHalf[k]!),
      CLOUD_CENTER.map((c, k) => c + cloudHalf[k]!),
    ),
    extinction: 40,
    baseColor: [1, 1, 1],
    hdda: true,
  });
  // the box of his bake (the 128³ texels are not cubic voxels): its z side is the height once stood up
  const [exLo, exHi] = explosionDense.bbox ?? [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]];
  const exScale = EXPLOSION_HEIGHT / (exHi[2]! - exLo[2]!);
  const [sx, sy, sz] = [0, 1, 2].map((k) => (exHi[k]! - exLo[k]!) * exScale) as Vec3;
  // his explosion is Z-up: −90° about x, standing on the ground (frame.js L384–425)
  const explosion = frame.addVolume(explosionSlot, {
    matrixWorld: [sx, 0, 0, 0, 0, 0, -sy, 0, 0, sz, 0, 0, EXPLOSION_AT[0], EXPLOSION_AT[1] + sz / 2, EXPLOSION_AT[2], 1],
    // his values are 0–3000 K at intensity 15 (frame 10 is mostly smoke with embers); hotter and brighter here, so the
    // one committed frame reads as fire from across the scene
    baseColor: [0.2, 0.19, 0.18],
    extinction: 25,
    temperatureRange: [0, 4500],
    emissiveIntensity: look.flames,
    emissionColor: { spectrum: "planck" },
  });
  const volumes: VDBFrameVolumeGpu[] = [cloud, explosion];

  //* 5. The environment: a procedural HDR sky as an equirect rgba16float texture (also the background).
  const sky = device.createTexture({ label: "demo-17 sky", size: [SKY_WIDTH, SKY_HEIGHT], format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  let skyRevision = 0;
  let skyKey = "";
  let ambient = skyAmbient([0, 1, 0], [1, 1, 1]);
  const sunColor: Vec3 = [1.0, 0.94, 0.86];

  //* 6. The page's own passes and their resources.
  ui.status("compiling shaders…", null);
  const pipelines = await createScenePipelines(device, RawTargets.COLOR_FORMAT, RawTargets.DEPTH_FORMAT, RawTargets.COLOR_FORMAT);
  const placeholders = createPlaceholders(device);
  const sceneBuffer = device.createBuffer({ label: "demo-17 scene", size: SCENE_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const shadowSceneBuffer = device.createBuffer({ label: "demo-17 scene (sun)", size: SCENE_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const group0 = device.createBindGroup({ layout: pipelines.group0, entries: [{ binding: 0, resource: { buffer: sceneBuffer } }] });
  const shadowGroup0 = device.createBindGroup({ layout: pipelines.group0, entries: [{ binding: 0, resource: { buffer: shadowSceneBuffer } }] });
  const linear = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
  const skySampler = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge" });
  const compare = device.createSampler({ compare: "less", magFilter: "linear", minFilter: "linear" });
  let meshBuffer: GPUBuffer | null = null;
  let meshVertexCount = 0;
  let meshKey = "";
  const ensureMeshes = (): boolean => {
    const key = String(look.obeliskX);
    if (key === meshKey) return false;
    const vertices = meshVertices(GROUND_HALF, [0.25, 0.235, 0.2], [obelisk(), PLINTH]);
    if (!meshBuffer || meshBuffer.size < vertices.byteLength) {
      meshBuffer?.destroy();
      meshBuffer = device.createBuffer({ label: "demo-17 meshes", size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    device.queue.writeBuffer(meshBuffer, 0, vertices);
    meshVertexCount = vertices.length / VERTEX_FLOATS;
    meshKey = key;
    return true;
  };
  // The sun's depth map (meshes + the level set) and the colour target the level-set pass needs beside it
  let shadowSize = 0;
  let shadowDepth: GPUTexture | null = null;
  /** The occluder input: one object per drawn map (a new view or matrix each frame would read as a new map and redraw the transmittance). */
  let occluder: VDBFrameOccluder | null = null;
  let shadowColor: GPUTexture | null = null;
  let shadowRevision = 0;
  let shadowKey = "";
  let sunViewProjection: number[] = [];
  let levelSetGbuffer: GPUTexture | null = null;

  //* 7. Camera: an orbit around the scene.
  const orbit = { yaw: 0.2, pitch: 0.13, distance: 430, target: [-15, 85, 10] as Vec3 };
  let autoOrbit = !STILL && !TEST && PRESENTATION === "direct";

  let frameId = 0;
  let inFlight = false;
  const frameTimes: number[] = [];
  let lighting: { inputs: readonly object[]; group: GPUBindGroup } | null = null;
  let lastUpdate: ReturnType<VDBFrameRendererGpu["update"]> | null = null;

  function sunVectors(): { toSun: Vec3; direction: Vec3 } {
    const e = (look.sunElevation * Math.PI) / 180;
    const a = (look.sunAzimuth * Math.PI) / 180;
    const toSun: Vec3 = [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];
    return { toSun, direction: [-toSun[0], -toSun[1], -toSun[2]] };
  }

  function applySettings(q: Quality): void {
    const p = QUALITY[q];
    Object.assign(frame.settings, {
      width: p.width,
      height: p.height,
      depth: p.depth,
      subSteps: p.subSteps,
      shadowResolution: p.shadowResolution,
      shadowDistance: 2000,
      lighting: look.lighting,
      multipleScattering: true,
      phase: "dual-lobe",
      phaseG: 0.8,
      phaseBackG: -0.3,
      phaseBackWeight: 0.2,
      environmentLighting: "directional",
      castShadows: true,
      castShadowMapSize: q === "high" ? 1024 : 512,
      castShadowCascades: 1,
      castShadowSteps: p.shadowSteps,
      // temporal reprojection converges the jittered froxels on screen; the readback e2e wants deterministic frames
      jitter: PRESENTATION === "direct" && !TEST,
      temporal: PRESENTATION === "direct" && !TEST,
    });
    for (const v of volumes) v.params.shadowSteps = p.shadowSteps;
  }
  applySettings(look.quality);

  /** Records and submits one frame; resolves when it is on screen (readback: read back and shown). */
  async function renderFrame(o: FrameOptions = {}): Promise<void> {
    inFlight = true;
    try {
      const q = QUALITY[look.quality];
      const { width, height } = presenter.resize(q.pixelRatio);
      targets.resize(width, height);
      if (!levelSetGbuffer || levelSetGbuffer.width !== width || levelSetGbuffer.height !== height) {
        levelSetGbuffer?.destroy();
        levelSetGbuffer = device.createTexture({ label: "demo-17 level-set gbuffer", size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      }

      // Camera
      const { yaw, pitch, distance, target } = orbit;
      const eye: Vec3 = [target[0] + distance * Math.cos(pitch) * Math.sin(yaw), target[1] + distance * Math.sin(pitch), target[2] + distance * Math.cos(pitch) * Math.cos(yaw)];
      eye[1] = Math.max(eye[1], 3);
      const near = 1;
      const far = 8000;
      const camera = rawCamera(lookAt(eye, target), perspective((50 * Math.PI) / 180, width / height, near, far), near, far);

      // Sun and sky (the sky follows the sun: a new revision re-runs the core's environment kernels)
      const { toSun, direction } = sunVectors();
      const key = `${look.sunElevation},${look.sunAzimuth}`;
      if (key !== skyKey) {
        device.queue.writeTexture({ texture: sky }, skyTexels(toSun, sunColor), { bytesPerRow: SKY_WIDTH * 8 }, [SKY_WIDTH, SKY_HEIGHT]);
        ambient = skyAmbient(toSun, sunColor);
        skyKey = key;
        skyRevision++;
      }

      // The sun's orthographic camera over the scene, and its depth map (redrawn when the sun or a mesh moved)
      const meshesMoved = ensureMeshes();
      const size = look.quality === "high" ? 2048 : 1024;
      if (size !== shadowSize) {
        shadowDepth?.destroy();
        shadowColor?.destroy();
        shadowDepth = device.createTexture({ label: "demo-17 sun depth", size: [size, size], format: RawTargets.DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
        shadowColor = device.createTexture({ label: "demo-17 sun colour (unused)", size: [size, size], format: RawTargets.COLOR_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
        shadowSize = size;
        shadowKey = "";
      }
      const sunEye: Vec3 = [SHADOW_CENTER[0] + toSun[0] * 1500, SHADOW_CENTER[1] + toSun[1] * 1500, SHADOW_CENTER[2] + toSun[2] * 1500];
      const sunCamera = rawCamera(lookAt(sunEye, SHADOW_CENTER), orthographic(-SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, SHADOW_HALF, 1, 3000), 1, 3000);
      sunViewProjection = sunCamera.viewProjection;

      const sunLight: VDBFrameLight = {
        id: "sun",
        kind: "directional",
        direction,
        color: [sunColor[0] * look.sunIntensity, sunColor[1] * look.sunIntensity, sunColor[2] * look.sunIntensity],
        shadow: { size: frame.settings.castShadowMapSize, view: "fit" },
      };
      const showVolumes = o.volumes ?? true;
      for (const v of volumes) {
        v.visible = showVolumes;
        v.castsShadow = showVolumes;
      }
      const useOccluders = o.occluders ?? look.occluders;

      const scene = sceneUniforms(camera, direction, width, height, o);
      device.queue.writeBuffer(sceneBuffer, 0, scene);
      const encoder = device.createCommandEncoder({ label: "demo-17 frame" });
      //* The frame loop: the host's pools first, then the core's source, in the host's encoder
      pool.update(encoder);
      lsPool.update(encoder);
      source.update(encoder);

      const sKey = `${key},${look.obeliskX},${size}`;
      if (sKey !== shadowKey || meshesMoved) {
        device.queue.writeBuffer(shadowSceneBuffer, 0, sceneUniforms(sunCamera, direction, size, size, o));
        const pass = encoder.beginRenderPass({
          label: "demo-17 sun depth map",
          colorAttachments: [{ view: shadowColor!.createView(), loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "discard" }],
          depthStencilAttachment: { view: shadowDepth!.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "store" },
        });
        pass.setPipeline(pipelines.shadow);
        pass.setBindGroup(0, shadowGroup0);
        pass.setVertexBuffer(0, meshBuffer!);
        pass.draw(meshVertexCount);
        lsShadow.record(pass, { view: sunCamera.view, projection: sunCamera.projection, viewport: [0, 0, size, size] }, [{ grid: levelSet.id }]);
        pass.end();
        shadowKey = sKey;
        shadowRevision++;
        // mesh → volume: the core reads our map with the matrix it was drawn with; a new revision redraws its transmittance
        occluder = { light: "sun", depth: shadowDepth!.createView({ dimension: "2d-array" }), viewProjection: [sunViewProjection], revision: shadowRevision };
      }

      //* The core: one update per encoder (grids, transmittance, environment, shadow maps, emission, froxels)
      const input: VDBFrameInput = {
        frameId: frameId++,
        camera: { view: camera.view, projection: camera.projection, near, far, id: "main" },
        viewport: { drawingBuffer: [width, height], canvas: [canvas.clientWidth || width, canvas.clientHeight || height] },
        lights: [sunLight],
        environment: { id: "procedural sky", revision: skyRevision, texture: sky, projection: "equirect", intensity: look.sky },
        // mesh → volume: the page's depth map of the sun, read with the matrix it was drawn with
        occluders: useOccluders && occluder ? [occluder] : [],
        emissionLight: { resolution: q.emission, range: 3, cascades: 1, steps: 24 },
      };
      const u = frame.update(encoder, input);
      lastUpdate = u;

      //* The page's opaque passes, reading the core's outputs (volume shadow map, emission probes)
      const group1 = lightingGroup(u);
      const main = encoder.beginRenderPass({
        label: "demo-17 opaque",
        colorAttachments: [{ view: targets.hdr.createView(), loadOp: "clear", clearValue: [0, 0, 0, 1], storeOp: "store" }],
        depthStencilAttachment: { view: targets.depth.createView(), depthLoadOp: "clear", depthClearValue: 1, depthStoreOp: "store" },
      });
      main.setBindGroup(0, group0);
      main.setBindGroup(1, group1);
      main.setPipeline(pipelines.sky);
      main.draw(3);
      main.setPipeline(pipelines.mesh);
      main.setVertexBuffer(0, meshBuffer!);
      main.draw(meshVertexCount);
      main.end();
      if (o.levelSet ?? true) {
        // The core's level-set pass: a G-buffer (normal, distance) and frag_depth, depth-tested against our meshes
        const ls = encoder.beginRenderPass({
          label: "demo-17 level set",
          colorAttachments: [{ view: levelSetGbuffer.createView(), loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }],
          depthStencilAttachment: { view: targets.depth.createView(), depthLoadOp: "load", depthStoreOp: "store" },
        });
        lsMain.record(ls, { view: camera.view, projection: camera.projection, viewport: [0, 0, width, height] }, [{ grid: levelSet.id }]);
        ls.end();
        const resolve = encoder.beginRenderPass({ label: "demo-17 level-set shade", colorAttachments: [{ view: targets.hdr.createView(), loadOp: "load", storeOp: "store" }] });
        resolve.setPipeline(pipelines.resolve);
        resolve.setBindGroup(0, group0);
        resolve.setBindGroup(1, group1);
        resolve.setBindGroup(2, device.createBindGroup({ layout: pipelines.group2, entries: [{ binding: 0, resource: levelSetGbuffer.createView() }] }));
        resolve.draw(3);
        resolve.end();
      }

      //* The volumes over the opaque frame: the core's composite pass, or our present shader (vdbFrameIntegrate)
      const composite = o.composite ?? look.composite;
      if (composite === "core") {
        targets.copySceneColor(encoder);
        frame.render(encoder, targets.composite(false));
      }
      const present = await presenter.pipeline(composite === "host");
      presenter.present(encoder, present, targets, {
        exposure: look.exposure,
        toneMap: true,
        integrate: composite === "host" ? { outputs: u.outputs, near, far, srgbBlending: frame.settings.srgbBlending } : null,
      });
      const t0 = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      frameTimes.push(performance.now() - t0);
      if (frameTimes.length > 30) frameTimes.shift();
      await presenter.show();
      state.frames++;
      state.frameMs = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
      describe(u, width, height);
    } finally {
      inFlight = false;
    }
  }

  function sceneUniforms(camera: RawCamera, direction: Vec3, width: number, height: number, o: FrameOptions): ArrayBuffer {
    return packScene({
      viewProjection: camera.viewProjection,
      projectionInverse: camera.projectionInverse,
      cameraWorld: camera.world,
      sunViewProjection,
      cameraPosition: camera.position,
      cameraForward: camera.forward,
      sunDirection: direction,
      shadowTexel: 1 / Math.max(1, shadowSize),
      sunColor: [sunColor[0] * look.sunIntensity, sunColor[1] * look.sunIntensity, sunColor[2] * look.sunIntensity],
      emissionIntensity: (o.emission ?? true) ? look.fireLight : 0,
      skyAmbient: ambient.sky.map((v) => v * look.sky) as Vec3,
      environmentIntensity: look.sky,
      groundAmbient: ambient.ground.map((v) => v * look.sky) as Vec3,
      shadowBias: 0.0008,
      fogColor: [0.4 * look.sky, 0.47 * look.sky, 0.58 * look.sky],
      fogDensity: 1 / 4500,
      sunDiscColor: sunColor.map((v) => v * look.sunIntensity * 40) as Vec3,
      sunDiscCos: Math.cos((0.8 * Math.PI) / 180),
      levelSetColor: [0.78, 0.6, 0.42],
      normalOffset: 2.0,
      volumeShadows: o.volumeShadows ?? look.volumeShadows,
      meshShadows: true,
      emission: (o.emission ?? true) && look.fireLight > 0,
      viewport: [width, height],
    });
  }

  /** Group 1 of the page's passes: our depth map, the core's volume shadow map + view, its emission probes + grid. */
  function lightingGroup(u: ReturnType<VDBFrameRendererGpu["update"]>): GPUBindGroup {
    const map = u.outputs.volumeShadows.get("sun");
    const probes = u.outputs.emissionProbes;
    // Rebuild when any input is a new object (the core bumps outputs.version when it reallocates a texture)
    const inputs = [shadowDepth!, map?.texture ?? placeholders.volumeShadowMap, map?.view ?? placeholders.volumeShadowView, probes?.texture ?? placeholders.emissionProbes, probes?.grid ?? placeholders.emissionGrid] as const;
    if (lighting && lighting.inputs.every((x, i) => x === inputs[i])) return lighting.group;
    const group = device.createBindGroup({
      label: "demo-17 lighting",
      layout: pipelines.group1,
      entries: [
        { binding: 0, resource: inputs[0].createView() },
        { binding: 1, resource: compare },
        { binding: 2, resource: inputs[1].createView({ dimension: "2d-array" }) },
        { binding: 3, resource: { buffer: inputs[2] } },
        { binding: 4, resource: linear },
        { binding: 5, resource: inputs[3].createView({ dimension: "3d" }) },
        { binding: 6, resource: { buffer: inputs[4] } },
        { binding: 7, resource: sky.createView() },
        { binding: 8, resource: skySampler },
      ],
    });
    lighting = { inputs, group };
    return group;
  }

  function describe(u: ReturnType<VDBFrameRendererGpu["update"]>, width: number, height: number): void {
    const map = u.outputs.volumeShadows.get("sun");
    const fx = u.outputs.froxels?.layout;
    state.info = {
      ...state.info,
      cloud: wdas ? "wdas" : "cc0",
      levelSet: levelSet.kind,
      lighting: frame.settings.lighting,
      composite: look.composite,
      quality: look.quality,
      size: [width, height],
      froxels: fx ? [fx.width, fx.height, fx.depth] : null,
      volumes: u.volumes,
      recorded: u.recorded,
      shadowMap: map ? { layers: map.layers, size: map.size, enabled: map.enabled, revision: map.revision } : null,
      emissionProbes: u.outputs.emissionProbes ? { resolution: u.outputs.emissionProbes.state.resolution, emitters: u.outputs.emissionProbes.state.emitters } : null,
      occluders: look.occluders ? 1 : 0,
      occluderRevision: shadowRevision,
      redraw: JSON.parse(JSON.stringify(frame.redrawStats)),
      poolBytes: pool.byteLength,
      levelSetBytes: lsPool.byteLength,
    };
    ui.stats(
      `${state.adapter || "WebGPU"}\n` +
        `${cloudName} + EmberGen explosion (CC0) · ${MiB(pool.byteLength)} pool\n` +
        `level set: ${levelSet.label} · ${MiB(lsPool.byteLength)}\n` +
        `froxels ${fx ? `${fx.width}×${fx.height}×${fx.depth}` : "—"} · '${frame.settings.lighting}' lighting · composite: ${look.composite === "core" ? "core pass" : "our shader"}\n` +
        `volume shadow map ${map ? `${map.size}² × ${map.layers}` : "—"} · sun depth map ${shadowSize}² · emission probes ${u.outputs.emissionProbes ? `${u.outputs.emissionProbes.state.resolution}³` : "—"}\n` +
        `frame ${state.frameMs!.toFixed(1)} ms (submit → done, this page; not a benchmark)`,
    );
  }

  //* Controls (the gallery strip)
  if (!TEST) {
    ui.select<Lighting>("lighting", "lighting", [{ value: "frame", label: "'frame' (grids)" }, { value: "volume", label: "'volume' (march)" }], look.lighting, (v) => {
      look.lighting = v;
      applySettings(look.quality);
    });
    ui.select<Composite>("composite", "composite", [{ value: "core", label: "core pass (render)" }, { value: "host", label: "our shader (vdbFrameIntegrate)" }], look.composite, (v) => (look.composite = v));
    ui.select<Quality>("quality", "quality", (["low", "medium", "high"] as const).map((v) => ({ value: v, label: QUALITY[v].label })), look.quality, (v) => {
      look.quality = v;
      applySettings(v);
    });
    ui.range("elevation", "sun elevation", 5, 75, 1, look.sunElevation, (v) => (look.sunElevation = v), (v) => `${v}°`);
    ui.range("azimuth", "sun azimuth", -180, 180, 1, look.sunAzimuth, (v) => (look.sunAzimuth = v), (v) => `${v}°`);
    ui.range("obelisk", "obelisk x", -20, 160, 1, look.obeliskX, (v) => (look.obeliskX = v));
    ui.range("flames", "flames", 0, 200, 1, look.flames, (v) => (explosion.params.emissiveIntensity = look.flames = v));
    ui.range("fire", "fire light", 0, 150, 1, look.fireLight, (v) => (look.fireLight = v));
    ui.range("sky", "sky", 0, 2, 0.05, look.sky, (v) => (look.sky = v));
    ui.range("exposure", "exposure", 0.3, 2.5, 0.05, look.exposure, (v) => (look.exposure = v));
    ui.checkbox("volume-shadows", "vol → mesh", look.volumeShadows, (v) => (look.volumeShadows = v));
    ui.checkbox("occluders", "mesh → vol", look.occluders, (v) => (look.occluders = v));
  }

  //* Orbit: drag to turn, wheel to zoom; the first touch stops the slow auto orbit.
  if (!TEST) {
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
      autoOrbit = false;
    });
    canvas.addEventListener("pointerup", (e) => {
      canvas.releasePointerCapture(e.pointerId);
      canvas.style.cursor = "grab";
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!canvas.hasPointerCapture(e.pointerId)) return;
      orbit.yaw -= e.movementX * 0.006;
      orbit.pitch = Math.max(0.02, Math.min(1.3, orbit.pitch + e.movementY * 0.006));
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        autoOrbit = false;
        orbit.distance = Math.max(120, Math.min(2000, orbit.distance * Math.exp(e.deltaY * 0.001)));
      },
      { passive: false },
    );
  }

  //* First frames: compile everything, then report.
  ui.status("first frame (compiling the core's kernels)…", null);
  await frame.prepare({ lighting: ["frame", "volume"] });
  await lsMain.prepare();
  await renderFrame();
  ui.status("");
  state.loading = false;

  if (TEST) {
    await runChecks();
  } else if (PRESENTATION === "direct") {
    let last = performance.now();
    const loop = (now: number): void => {
      if (ui.error) return;
      requestAnimationFrame(loop);
      if (inFlight) return;
      if (autoOrbit) orbit.yaw += ((now - last) / 1000) * 0.04;
      last = now;
      renderFrame().catch((e) => ui.fail(e));
    };
    requestAnimationFrame(loop);
  } else {
    // readback without ?test: redraw on control changes only
    const redraw = (): void => {
      if (!inFlight) renderFrame().catch((e) => ui.fail(e));
    };
    ui.strip.addEventListener("input", redraw);
    ui.strip.addEventListener("change", redraw);
  }
  state.ready = true;

  /**
   * `?test=1` (e2e/demo-17.spec.ts): statistical image checks. Each turns one feature off and counts the pixels that
   * change against the full frame (sum of |Δ| over RGB > 6, sRGB bytes) and their mean signed luma change.
   */
  async function runChecks(): Promise<void> {
    const shot = async (o: FrameOptions = {}): Promise<Uint8ClampedArray> => {
      await renderFrame(o);
      return presenter.last!.slice();
    };
    const diff = (a: Uint8ClampedArray, b: Uint8ClampedArray, threshold = 6) => {
      let changed = 0;
      let luma = 0;
      let maxChannel = 0;
      let over2 = 0;
      for (let i = 0; i < a.length; i += 4) {
        const d = Math.abs(a[i]! - b[i]!) + Math.abs(a[i + 1]! - b[i + 1]!) + Math.abs(a[i + 2]! - b[i + 2]!);
        const m = Math.max(Math.abs(a[i]! - b[i]!), Math.abs(a[i + 1]! - b[i + 1]!), Math.abs(a[i + 2]! - b[i + 2]!));
        maxChannel = Math.max(maxChannel, m);
        if (m > 2) over2++;
        if (d > threshold) {
          changed++;
          luma += (b[i]! + b[i + 1]! + b[i + 2]! - a[i]! - a[i + 1]! - a[i + 2]!) / 3;
        }
      }
      return { changed, meanLuma: changed ? luma / changed : 0, maxChannel, over2, pixels: a.length / 4 };
    };
    const base = await shot();
    const w = presenter.last ? targets.width : 0;
    const h = targets.height;
    // Determinism and the zero-build second frame
    const again = await countBuilds(device, () => shot());
    const repeat = diff(base, again.result, 0);
    const checks = {
      repeat: { identical: repeat.changed === 0 && repeat.maxChannel === 0, builds: again.builds },
      noVolumes: diff(base, await shot({ volumes: false })),
      noVolumeShadows: diff(base, await shot({ volumeShadows: false })),
      noOccluders: diff(base, await shot({ occluders: false })),
      noEmission: diff(base, await shot({ emission: false })),
      noLevelSet: diff(base, await shot({ levelSet: false })),
      hostComposite: diff(base, await shot({ composite: "host" })),
    };
    // Back to the full frame for the snapshot (and the screenshot)
    const final = await shot();
    const snapshot: Snapshot = snapshotOf(final, w, h);
    state.info = { ...state.info, checks, snapshot, finalEqualsBase: diff(base, final, 0).changed === 0 };
    (state as unknown as { done: boolean }).done = true;
  }
}

/** Counts the pipelines and shader modules the device creates while `fn` runs. */
async function countBuilds<T>(device: GPUDevice, fn: () => Promise<T>): Promise<{ builds: number; result: T }> {
  const d = device as unknown as Record<string, (...a: unknown[]) => unknown>;
  const names = ["createShaderModule", "createComputePipeline", "createComputePipelineAsync", "createRenderPipeline", "createRenderPipelineAsync"];
  const saved = names.map((n) => d[n]!);
  let builds = 0;
  names.forEach((n, i) => {
    d[n] = (...a: unknown[]) => {
      builds++;
      return saved[i]!.apply(device, a);
    };
  });
  try {
    const result = await fn();
    return { builds, result };
  } finally {
    names.forEach((n, i) => (d[n] = saved[i]!));
  }
}

// ------------------------------------------------------------------------------------------------ level set

interface LoadedLevelSet {
  id: number;
  kind: "icosahedron" | "procedural";
  label: string;
}

/**
 * The OpenVDB icosahedron level set (parsed and built in the page), else a procedural one (a ring around an orb, a
 * narrow-band SDF built with `vdbBuildFromDense`), placed on the plinth by an instance transform.
 */
async function loadLevelSet(pool: VDBGridPool): Promise<LoadedLevelSet> {
  const want = params.get("levelset") ?? "icosahedron";
  let image: Uint32Array | null = null;
  let kind: LoadedLevelSet["kind"] = "procedural";
  if (want !== "procedural" && (await probe(ICOSAHEDRON_URL))) {
    const vdb = parseVdb(await (await fetch(ICOSAHEDRON_URL)).arrayBuffer());
    const grid = vdb.grids.find((g) => g.valueType === "float");
    if (grid) {
      image = buildFromVdb(grid);
      kind = "icosahedron";
    }
  } else if (want !== "procedural") {
    ui.note("icosahedron.vdb not found (fixtures/vdb-samples/, pnpm fixtures:vdb-samples): a procedural level set instead.");
  }
  if (!image) image = proceduralLevelSet();
  const id = pool.add(image, { label: kind });
  const box = pool.info(id).worldBBox;
  const center = [0, 1, 2].map((k) => (box.min[k]! + box.max[k]!) / 2);
  const s = LEVEL_SET_SIZE / Math.max(...[0, 1, 2].map((k) => box.max[k]! - box.min[k]!));
  pool.setTransform(id, [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, LEVEL_SET_AT[0] - s * center[0]!, LEVEL_SET_AT[1] - s * center[1]!, LEVEL_SET_AT[2] - s * center[2]!, 1]);
  return { id, kind, label: kind === "icosahedron" ? "OpenVDB icosahedron (CC BY-SA 4.0)" : "procedural ring and orb" };
}

/** A ring around an orb as a narrow-band level set (voxel units, band ±3). */
function proceduralLevelSet(): Uint32Array {
  const n = 72;
  const band = 3;
  const values = new Float32Array(n * n * n);
  const c = (n - 1) / 2;
  const tilt = 1.15;
  const [ct, st] = [Math.cos(tilt), Math.sin(tilt)];
  for (let x = 0; x < n; x++)
    for (let y = 0; y < n; y++)
      for (let z = 0; z < n; z++) {
        const px = x - c;
        const py = y - c;
        const pz = z - c;
        const orb = Math.hypot(px, py, pz) - 14;
        // a torus tilted about x
        const ty = py * ct - pz * st;
        const tz = py * st + pz * ct;
        const ring = Math.hypot(Math.hypot(px, tz) - 25, ty) - 5;
        const d = Math.min(orb, ring);
        values[(x * n + y) * n + z] = Math.max(-band, Math.min(band, d));
      }
  return vdbBuildFromDense(values, [n, n, n], { background: band, activeThreshold: 1e-6, gridClass: "LevelSet", gridName: "ring and orb" });
}

main().catch((e: unknown) => ui.fail(e));
