// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/webgpu/FrameRenderer.js L291–296 and L642–699 (isNeural, the grid mip targets and their
//   redraw, renderGridMips); commit 4e2d46f (the 'multi-scale' mips and levels).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
// Changes: the network is data for a generic MLP evaluator (shape uniform + weight storage buffer) and its feature
// levels are mips of one 3D texture; 'multi-scale' uses K7.3's core mips. Divergences: DIVERGES.md (K6.3 rows).

/**
 * K6.3 (design §3.6): the resources of the transmittance kernel's pluggable scattered-light term.
 *
 * - `'mlp'` (EXPERIMENTAL, Q5; his `'neural'` maps to it): the network as data (`VDBFrameMlp`: a shape uniform and a
 *   weight storage buffer, the source-free kernel's one storage buffer, D10) and the box mips of its feature levels
 *   (`gridMips`: one rgba16float 3D texture of side R/2 with ≤ 5 mips, redrawn with the shadow grid).
 * - `'multi-scale'` (K7.3): `vdbCreateMultiScaleMips` on the last shadow grid cascade (rg16float, edge `'zero'` or his
 *   `'clamp'`, `multiScaleLevels`), recorded in its own compute pass after the grid, and the `VdbFrameMultiScale`
 *   uniform.
 *
 * Grey 'frame' lighting only; elsewhere these modes run as 1D diffusion (his `isNeural`), as the TSL path.
 */

import { vdbCreateMultiScaleMips, vdbFrameMultiScaleParams, VDB_FRAME_MULTI_SCALE_PARAMS_BYTES, type VDBMultiScaleMips } from "../compute/multi-scale.js";
import { vdbFrameMlpCheck, vdbFrameMlpMipSizes, vdbFrameMlpShape, VDB_FRAME_MLP_MAX_LAYERS, type VDBFrameMlp } from "../../tools/frame/mlp.js";
import type { VDBFrameMultiScaleEdge } from "../../tools/frame/multi-scale.js";
import { GRID_MIP_BINDINGS, TRANSMITTANCE_MLP_BINDINGS, TRANSMITTANCE_MULTI_SCALE_BINDINGS, type VDBFrameScatteringKernel } from "./wgsl/lighting.js";
import { vdbFrameGridMipSpec } from "./kernels.js";
import { computePipeline } from "./pipelines.js";
import { BUFFER_USAGE, TEXTURE_USAGE, VDBRetireQueue } from "./resources.js";
import type { VDBFrameSettings } from "./settings.js";

/** The last cascade of the shadow grid, as the scattered-light terms read it. */
export interface VDBFrameScatteringGrid {
  resolution: number;
  /** World box and voxel of the last cascade. */
  min: readonly number[];
  size: readonly number[];
  voxel: readonly number[];
  /** Its texel origin in the shadow grid stack. */
  origin: readonly number[];
}

/** The scattered-light term of an update: the kernel family and the uniform values. */
export interface VDBFrameScatteringPlan {
  kind: VDBFrameScatteringKernel;
  /** The transmittance redraw signature's part (network identity, levels, edge, mip versions). */
  signature: unknown[];
}

const MLP_SHAPE_BYTES = 16 + 16 * VDB_FRAME_MLP_MAX_LAYERS;

/** The `'mlp'` and `'multi-scale'` resources of one renderer (see the file comment). */
export class VDBFrameScattering {
  readonly #device: GPUDevice;
  readonly #label: string;
  readonly #retire: VDBRetireQueue;
  readonly #warn: (key: string, message: string) => void;
  // 'mlp'
  #mlp: VDBFrameMlp | null = null;
  #mlpValid = false;
  #mlpVersion = 0;
  #weights: GPUBuffer | null = null;
  #shape: GPUBuffer | null = null;
  #mips: GPUTexture | null = null;
  #mipsResolution = 0;
  /** The grid version the feature mips hold (−1: none). */
  #mipsDrawn = -1;
  // 'multi-scale'
  #ms: VDBMultiScaleMips | null = null;
  #msKey = "";
  #msParams: GPUBuffer | null = null;
  #msDrawn = -1;
  #msVersion = 0;

  constructor(device: GPUDevice, label: string, retire: VDBRetireQueue, warn: (key: string, message: string) => void) {
    this.#device = device;
    this.#label = label;
    this.#retire = retire;
    this.#warn = warn;
  }

  /**
   * The kernel family of this update ('frame' lighting, grey): `'neural'` / `'mlp'` with a valid network → `'mlp'`,
   * `'multi-scale'` → `'multi-scale'`, else `'none'` (a bad or missing network warns once and runs as 1D diffusion).
   */
  kind(settings: VDBFrameSettings, mlp: VDBFrameMlp | null): VDBFrameScatteringKernel {
    const mode = settings.multipleScattering;
    if (settings.spectral) return "none";
    if (mode === "multi-scale") return "multi-scale";
    if (mode !== "neural" && mode !== "mlp") return "none";
    if (mlp !== this.#mlp) {
      this.#mlp = mlp;
      const problem = mlp ? vdbFrameMlpCheck(mlp) : null;
      this.#mlpValid = !!mlp && !problem;
      this.#mlpVersion++;
      if (this.#weights) this.#retire.retire(this.#weights);
      this.#weights = null;
      if (!mlp) this.#warn("mlp-none", `multipleScattering '${mode}' has no network (mlp is null); the light slots use 1D diffusion (true).`);
      else if (problem) this.#warn(`mlp-shape-${this.#mlpVersion}`, `the mlp ${mlp.name ?? ""} does not fit the evaluator (${problem}); using 1D diffusion.`);
    }
    return this.#mlpValid ? "mlp" : "none";
  }

  /**
   * Prepares the resources of `kind` for the last cascade `grid` (drawn at `gridVersion`): uploads a new network,
   * (re)allocates the mips, writes the multi-scale uniform. Returns the plan; `recordPass` / `recordEncoder` draw the
   * mips when the grid moved.
   */
  plan(kind: VDBFrameScatteringKernel, settings: VDBFrameSettings, grid: VDBFrameScatteringGrid, multipleScattering: number): VDBFrameScatteringPlan {
    if (kind === "mlp") {
      const mlp = this.#mlp!;
      if (!this.#weights) {
        this.#weights = this.#device.createBuffer({ label: `${this.#label} mlp weights`, size: Math.max(16, mlp.weights.byteLength), usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST });
        this.#device.queue.writeBuffer(this.#weights, 0, mlp.weights.buffer as ArrayBuffer, mlp.weights.byteOffset, mlp.weights.byteLength);
        this.#shape ??= this.#device.createBuffer({ label: `${this.#label} mlp shape`, size: MLP_SHAPE_BYTES, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
        this.#device.queue.writeBuffer(this.#shape, 0, vdbFrameMlpShape(mlp).buffer as ArrayBuffer);
      }
      if (!this.#mips || this.#mipsResolution !== grid.resolution) {
        this.#retire.retire(this.#mips);
        const sizes = vdbFrameMlpMipSizes(grid.resolution);
        this.#mips = this.#device.createTexture({
          label: `${this.#label} mlp feature mips`,
          size: [sizes[0]!, sizes[0]!, sizes[0]!],
          dimension: "3d",
          format: "rgba16float",
          mipLevelCount: sizes.length,
          usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.TEXTURE_BINDING | TEXTURE_USAGE.COPY_SRC,
        });
        this.#mipsResolution = grid.resolution;
        this.#mipsDrawn = -1;
      }
      return { kind, signature: ["mlp", this.#mlpVersion, this.#mips] };
    }
    if (kind === "multi-scale") {
      const levels = Math.min(8, Math.max(2, Math.round(settings.multiScaleLevels)));
      const edge: VDBFrameMultiScaleEdge = settings.multiScaleEdge === "clamp" ? "clamp" : "zero";
      const key = `${grid.resolution}:${levels}:${edge}`;
      if (!this.#ms || this.#msKey !== key) {
        if (this.#ms) this.#retire.retire(this.#ms);
        this.#ms = vdbCreateMultiScaleMips(this.#device, { size: [grid.resolution, grid.resolution, grid.resolution], levels, edge, channels: [0, 1], label: `${this.#label} multi-scale` });
        this.#msKey = key;
        this.#msDrawn = -1;
        this.#msVersion++;
      }
      this.#msParams ??= this.#device.createBuffer({ label: `${this.#label} multi-scale params`, size: VDB_FRAME_MULTI_SCALE_PARAMS_BYTES, usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST });
      const r = grid.resolution;
      this.#device.queue.writeBuffer(
        this.#msParams,
        0,
        vdbFrameMultiScaleParams({
          boxMin: [grid.min[0]!, grid.min[1]!, grid.min[2]!],
          boxSize: [grid.size[0]!, grid.size[1]!, grid.size[2]!],
          unit: Math.max(grid.voxel[0]!, grid.voxel[1]!, grid.voxel[2]!),
          levels,
          multipleScattering,
          baseOrigin: [grid.origin[0]!, grid.origin[1]!, grid.origin[2]!],
          baseSize: [r, r, r],
        }),
      );
      return { kind, signature: ["multi-scale", key, this.#msVersion] };
    }
    return { kind, signature: ["none"] };
  }

  /** `'mlp'`: whether the feature mips must be drawn for the grid at `gridVersion`. */
  mipsStale(kind: VDBFrameScatteringKernel, gridVersion: number): boolean {
    if (kind === "mlp") return this.#mipsDrawn !== gridVersion;
    if (kind === "multi-scale") return this.#msDrawn !== gridVersion;
    return false;
  }

  /** `'mlp'`: records the feature mip chain (one dispatch per mip) into `pass` from the shadow grid `stack`. */
  recordMips(pass: GPUComputePassEncoder, group0: GPUBindGroup, stack: GPUTexture, gridVersion: number): void {
    const mips = this.#mips!;
    const b = GRID_MIP_BINDINGS;
    for (let level = 0; level < mips.mipLevelCount; level++) {
      const built = computePipeline(this.#device, vdbFrameGridMipSpec({ first: level === 0 }));
      const source = level === 0 ? stack.createView({ dimension: "3d" }) : mips.createView({ dimension: "3d", baseMipLevel: level - 1, mipLevelCount: 1 });
      const out = mips.createView({ dimension: "3d", baseMipLevel: level, mipLevelCount: 1 });
      pass.setPipeline(built.pipeline);
      pass.setBindGroup(0, group0);
      pass.setBindGroup(
        1,
        this.#device.createBindGroup({
          label: `${this.#label} grid mip ${level + 1} group 1`,
          layout: built.layouts[1]!,
          entries: [
            { binding: b.source, resource: source },
            { binding: b.out, resource: out },
          ],
        }),
      );
      const side = Math.max(1, mips.width >> level);
      pass.dispatchWorkgroups(Math.ceil(side / 4), Math.ceil(side / 4), Math.ceil(side / 4));
    }
    this.#mipsDrawn = gridVersion;
  }

  /** `'multi-scale'`: records K7.3's mip build (its own compute pass) into `encoder` from the last cascade. */
  recordMultiScale(encoder: GPUCommandEncoder, stack: GPUTexture, origin: readonly number[], gridVersion: number): void {
    this.#ms!.record(encoder, stack, { origin: [origin[0]!, origin[1]!, origin[2]!] });
    this.#msDrawn = gridVersion;
  }

  /** The transmittance kernel's group 2 for `kind` (null for `'none'`). */
  group2(kind: VDBFrameScatteringKernel, layout: GPUBindGroupLayout | undefined): GPUBindGroup | null {
    if (!layout) return null;
    if (kind === "mlp") {
      const m = TRANSMITTANCE_MLP_BINDINGS;
      return this.#device.createBindGroup({
        label: `${this.#label} transmittance mlp group 2`,
        layout,
        entries: [
          { binding: m.weights, resource: { buffer: this.#weights! } },
          { binding: m.shape, resource: { buffer: this.#shape! } },
          { binding: m.mips, resource: this.#mips!.createView({ dimension: "3d" }) },
        ],
      });
    }
    if (kind === "multi-scale") {
      const m = TRANSMITTANCE_MULTI_SCALE_BINDINGS;
      return this.#device.createBindGroup({
        label: `${this.#label} transmittance multi-scale group 2`,
        layout,
        entries: [
          { binding: m.params, resource: { buffer: this.#msParams! } },
          { binding: m.mips, resource: this.#ms!.view },
        ],
      });
    }
    return null;
  }

  /** @internal (harness probes) the feature mips texture and the multi-scale chain. */
  get debug(): { mips: GPUTexture | null; multiScale: VDBMultiScaleMips | null; mlp: VDBFrameMlp | null } {
    return { mips: this.#mips, multiScale: this.#ms, mlp: this.#mlpValid ? this.#mlp : null };
  }

  dispose(): void {
    this.#weights?.destroy();
    this.#shape?.destroy();
    this.#mips?.destroy();
    this.#ms?.destroy();
    this.#msParams?.destroy();
    this.#weights = this.#shape = this.#msParams = null;
    this.#mips = null;
    this.#ms = null;
  }
}
