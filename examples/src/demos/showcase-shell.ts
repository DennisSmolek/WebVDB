/**
 * The one page shell of gallery demos 11-16 (the shared design spec): a full-bleed canvas, a caption card top-left
 * (number · title, one sentence, "← all demos", a slot for demo widgets, a status line with a progress bar, a dim
 * stats line, a note for fallbacks, errors), a small control strip bottom-right (selects, ranges, checkboxes,
 * buttons), a flash and toast for state changes, gentle auto camera motion that stops on interaction, and the e2e
 * plumbing: the readback presentation (an offscreen HalfFloat target read back to a 2D canvas and tone mapped on
 * the CPU, as three's own output transform is skipped there), a frame summary, a coverage probe.
 *
 * Query (every gallery page): `?presentation=readback` (headless SwiftShader e2e, as demo 10), `?still=1` (no
 * camera motion), `?panel=0` (no caption or controls), `?gallery=0` (don't load the git-ignored `fixtures/gallery/`
 * assets: the procedural fallbacks), `?debug=1` (the demo may expose its objects on `window`).
 *
 * Demos 11, 12, 15 and 16 build their own renderer and use `mountShowcase` + `Presenter`; demos 13 and 14 take the
 * whole `createViewer` (renderer, camera, orbit, loop). Frame times are submit-to-done wall times of one frame on
 * this page: provisional (D25), not a benchmark.
 */
import * as THREE from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createDeviceRenderer } from "./device-renderer.js";
import { aces, PANEL, PRESENTATION, snapshotOf, srgbByte, STILL, type ShowcaseApi, type ShowcaseUi, type Snapshot } from "./showcase-ui.js";

// The engine-free half (flags, caption card, controls, fetch helpers, snapshot) lives in showcase-ui.ts.
export * from "./showcase-ui.js";

/**
 * The page's canvas and viewport in either presentation. `direct`: the renderer's own canvas. `readback`: three
 * renders into an offscreen linear HalfFloat target with depth (the froxel integrate pass reads the scene depth;
 * `present()` makes one, `target()` more), `readTarget()` turns one into sRGB bytes (rows top first, padding
 * dropped; ACES tone mapped with the renderer's exposure when `renderer.toneMapping` is ACESFilmic, since three
 * skips its output transform when rendering to a target) and `show()` draws bytes on the 2D canvas.
 */
export class Presenter {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx2d: CanvasRenderingContext2D | null;
  width = 1;
  height = 1;
  pixelRatio = 1;
  private last: Uint8ClampedArray<ArrayBuffer> | null = null;
  private readonly targets: THREE.RenderTarget[] = [];
  private main: THREE.RenderTarget | null = null;

  constructor(
    readonly app: HTMLElement,
    readonly renderer: THREE.WebGPURenderer,
    directCanvas?: HTMLCanvasElement,
    /** Needed by `present()` in the direct presentation (waits for the submit). */
    private readonly device?: GPUDevice,
  ) {
    this.canvas = directCanvas ?? document.createElement("canvas");
    this.canvas.id = "view";
    app.append(this.canvas);
    this.ctx2d = PRESENTATION === "readback" ? this.canvas.getContext("2d") : null;
  }

  /** A HalfFloat target with depth, sized with the viewport. */
  target(): THREE.RenderTarget {
    const rt = new THREE.RenderTarget(2, 2, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
    rt.setSize(this.pixelWidth, this.pixelHeight);
    this.targets.push(rt);
    return rt;
  }

  get pixelWidth(): number {
    return PRESENTATION === "direct" ? Math.round(this.width * this.pixelRatio) : this.width;
  }

  get pixelHeight(): number {
    return PRESENTATION === "direct" ? Math.round(this.height * this.pixelRatio) : this.height;
  }

  /** Fits the viewport to the page and updates the camera aspect. */
  resize(pixelRatio: number, camera: THREE.PerspectiveCamera): void {
    this.width = Math.max(1, this.app.clientWidth || window.innerWidth);
    this.height = Math.max(1, this.app.clientHeight || window.innerHeight);
    this.pixelRatio = pixelRatio;
    if (PRESENTATION === "direct") {
      this.renderer.setPixelRatio(pixelRatio);
      this.renderer.setSize(this.width, this.height, false);
    } else {
      this.canvas.width = this.width;
      this.canvas.height = this.height;
    }
    for (const rt of this.targets) rt.setSize(this.pixelWidth, this.pixelHeight);
    camera.aspect = this.width / this.height;
    camera.updateProjectionMatrix();
  }

  /** Reads a HalfFloat target back as sRGB RGBA bytes. */
  async readTarget(rt: THREE.RenderTarget): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const w = rt.width;
    const h = rt.height;
    const raw = (await this.renderer.readRenderTargetPixelsAsync(rt, 0, 0, w, h)) as unknown as Uint16Array;
    const tight = w * 4;
    const stride = h > 1 ? (raw.length - tight) / (h - 1) : tight;
    if (!Number.isInteger(stride) || stride < tight) throw new Error(`readback layout: ${raw.length} halves for ${w}x${h}`);
    const mapped = this.renderer.toneMapping === THREE.ACESFilmicToneMapping;
    const exposure = this.renderer.toneMappingExposure;
    const tone = [0, 0, 0];
    const out = new Uint8ClampedArray(w * h * 4);
    const half = THREE.DataUtils.fromHalfFloat;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = y * stride + x * 4;
        const d = (y * w + x) * 4;
        tone[0] = half(raw[s]!);
        tone[1] = half(raw[s + 1]!);
        tone[2] = half(raw[s + 2]!);
        if (mapped) aces(tone[0], tone[1], tone[2], exposure, tone);
        out[d] = srgbByte(tone[0]!);
        out[d + 1] = srgbByte(tone[1]!);
        out[d + 2] = srgbByte(tone[2]!);
        out[d + 3] = 255;
      }
    }
    return out;
  }

  /** Draws read-back bytes (the viewport's size) on the 2D canvas and keeps them for `capture()`. */
  show(bytes: Uint8ClampedArray<ArrayBuffer>): void {
    this.ctx2d!.putImageData(new ImageData(bytes, this.width, this.height), 0, 0);
    this.last = bytes;
  }

  /**
   * Renders the scene: direct, into the canvas (resolves when the GPU is done); readback, into the page's own
   * target, read back, shown (unless `show: false`) and returned.
   */
  async present(scene: THREE.Scene, camera: THREE.Camera, options: { show?: boolean } = {}): Promise<Uint8ClampedArray<ArrayBuffer> | null> {
    const { renderer } = this;
    if (PRESENTATION === "direct") {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
      await this.device?.queue.onSubmittedWorkDone();
      return null;
    }
    this.main ??= this.target();
    renderer.setRenderTarget(this.main);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const bytes = await this.readTarget(this.main);
    if (options.show !== false) this.show(bytes);
    return bytes;
  }

  /** A summary of the last shown read-back frame, or null. `background` (sRGB bytes) counts the pixels that differ. */
  capture(background?: readonly [number, number, number]): Snapshot | null {
    return this.last ? snapshotOf(this.last, this.width, this.height, background) : null;
  }
}

/** Slow auto-orbit that stops for good on the first user interaction (never starts with `?still=1` or in readback). Call `update()` every frame. */
export function autoOrbit(controls: { autoRotate: boolean; autoRotateSpeed: number; addEventListener(type: "start", cb: () => void): void }, speed = 0.35): void {
  controls.autoRotate = !STILL && PRESENTATION === "direct";
  controls.autoRotateSpeed = speed;
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
  });
}

export interface Viewer {
  renderer: THREE.WebGPURenderer;
  device: GPUDevice;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  orbit: OrbitControls;
  presenter: Presenter;
  /** The demo's own objects: hidden by the coverage probe. */
  content: THREE.Group;
  /** Auto camera motion (sway or orbit around `orbit.target`); off with `?still=1` and after any interaction. */
  motion: { enabled: boolean; rebase(): void };
  /** Starts the render loop; `tick` runs before each frame. */
  start(tick?: (now: number) => void, statsLine?: () => string): void;
  api: ShowcaseApi;
}

/**
 * The device-first renderer (`device-renderer.ts`), the presenter, an `OrbitControls` camera with gentle auto
 * motion, and the loop that keeps `ui.state` current (demos 13 and 14).
 */
export async function createViewer(
  ui: ShowcaseUi,
  options: { fov?: number; near?: number; far?: number; exposure?: number; motion?: "sway" | "orbit"; swayDegrees?: number; speed?: number; pixelRatio?: number },
): Promise<Viewer> {
  const directCanvas = PRESENTATION === "direct" ? document.createElement("canvas") : undefined;
  const created = await createDeviceRenderer(directCanvas ? { canvas: directCanvas } : {});
  const { renderer, device } = created;
  device.addEventListener("uncapturederror", (e) => {
    const message = (e as GPUUncapturedErrorEvent).error.message;
    ui.state.uncaptured.push(message);
    console.error(`WebGPU: ${message}`);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") ui.fail(`WebGPU device lost: ${info.message}`);
  });
  ui.state.adapter = created.adapterName || "unreported adapter";
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = options.exposure ?? 1;

  const presenter = new Presenter(ui.app, renderer, directCanvas, device);
  const scene = new THREE.Scene();
  const content = new THREE.Group();
  scene.add(content);
  const camera = new THREE.PerspectiveCamera(options.fov ?? 40, 1, options.near ?? 0.05, options.far ?? 500);
  const orbit = new OrbitControls(camera, presenter.canvas);
  orbit.enableDamping = PRESENTATION === "direct";
  orbit.dampingFactor = 0.08;

  // Auto motion: sway the azimuth around where the camera was put (or orbit), until the user touches the view.
  const motionMode = options.motion ?? "sway";
  const sway = THREE.MathUtils.degToRad(options.swayDegrees ?? 18);
  const speed = options.speed ?? 0.12;
  const base = new THREE.Spherical();
  let motionStart = performance.now();
  const motion = {
    enabled: !STILL,
    rebase(): void {
      base.setFromVector3(camera.position.clone().sub(orbit.target));
      motionStart = performance.now();
    },
  };
  orbit.addEventListener("start", () => {
    motion.enabled = false;
  });
  motion.rebase();
  const offset = new THREE.Vector3();
  const applyMotion = (now: number): void => {
    if (!motion.enabled) return;
    const t = (now - motionStart) / 1000;
    const s = base.clone();
    // ease in from the rest pose so a rebase never jumps
    s.theta = motionMode === "orbit" ? base.theta + t * speed : base.theta + sway * Math.sin(t * speed * 2 * Math.PI * 0.25);
    offset.setFromSpherical(s);
    camera.position.copy(orbit.target).add(offset);
    camera.lookAt(orbit.target);
  };

  const pixelRatio = options.pixelRatio ?? Math.min(2, window.devicePixelRatio || 1);
  const setViewport = (): void => presenter.resize(pixelRatio, camera);
  setViewport();
  window.addEventListener("resize", setViewport);

  let busy = false;
  async function coverage(): Promise<number> {
    if (PRESENTATION !== "readback") throw new Error("coverage() needs ?presentation=readback");
    while (busy) await new Promise((r) => setTimeout(r, 5));
    busy = true;
    try {
      const withContent = await presenter.present(scene, camera, { show: false });
      content.visible = false;
      const without = await presenter.present(scene, camera, { show: false });
      content.visible = true;
      let n = 0;
      for (let i = 0; i < withContent!.length; i += 4) {
        const d = Math.abs(withContent![i]! - without![i]!) + Math.abs(withContent![i + 1]! - without![i + 1]!) + Math.abs(withContent![i + 2]! - without![i + 2]!);
        if (d > 12) n++;
      }
      return n;
    } finally {
      content.visible = true;
      busy = false;
    }
  }

  const api: ShowcaseApi = { capture: () => presenter.capture(), coverage };
  let frameMs = 0;
  let lastStats = 0;

  function start(tick?: (now: number) => void, statsLine?: () => string): void {
    const loop = (now: number): void => {
      requestAnimationFrame(loop);
      if (busy || ui.state.error) return;
      busy = true;
      applyMotion(now);
      orbit.update();
      try {
        tick?.(now);
      } catch (err) {
        busy = false;
        ui.fail(err);
        return;
      }
      const t0 = performance.now();
      presenter
        .present(scene, camera)
        .then(() => {
          const ms = performance.now() - t0;
          frameMs = frameMs === 0 ? ms : frameMs * 0.9 + ms * 0.1;
          ui.state.frames++;
          ui.state.frameMs = frameMs;
          if (now - lastStats > 400 || ui.state.frames <= 3) {
            lastStats = now;
            ui.stats(`${ui.state.adapter} · ${presenter.pixelWidth}×${presenter.pixelHeight} · ${frameMs.toFixed(1)} ms/frame (provisional)` + (statsLine ? `\n${statsLine()}` : ""));
          }
        })
        .catch((err: unknown) => ui.fail(err))
        .finally(() => {
          busy = false;
        });
    };
    requestAnimationFrame(loop);
  }

  return { renderer, device, scene, camera, orbit, presenter, content, motion, start, api };
}
