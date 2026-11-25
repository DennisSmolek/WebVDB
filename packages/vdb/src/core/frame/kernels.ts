/**
 * The kernels of the core frame renderer as buildable specs (K6.1, design §3.1, §5.1): the composed WGSL
 * of each pipeline and its bind-group layouts (groups 0–2, no gaps). The renderer records them; the
 * `k6-core` harness runs each alone against its CPU twin; tests check their limits and bytes.
 *
 * Group 0 (every kernel): `VdbFrameUniforms`, `VdbFrameLights`, a linear clamp sampler, the environment buffer
 * (K6.2b; the environment kernels have their own group 0: `VdbFrameEnvParams`).
 * Group 1: the kernel's volumes / group records and textures. Group 2: the source (gather only).
 */

import { VDBError } from "../errors.js";
import type { VDBSlot } from "../types.js";
import type { VDBVolumeSource } from "../source/types.js";
import { frameChunksWgsl } from "./wgsl/chunks.js";
import { GATHER_BINDINGS, vdbFrameGatherWgsl } from "./wgsl/gather.js";
import {
  FRAME_GATHER_BINDINGS,
  GRID_DENSITY_BINDINGS,
  GRID_MIP_BINDINGS,
  TRANSMITTANCE_BINDINGS,
  TRANSMITTANCE_MLP_BINDINGS,
  TRANSMITTANCE_MULTI_SCALE_BINDINGS,
  vdbFrameGridDensityWgsl,
  vdbFrameGridMipWgsl,
  vdbFrameLightingGatherWgsl,
  vdbFrameTransmittanceWgsl,
  type VDBFrameScatteringKernel,
} from "./wgsl/lighting.js";
import { RESOLVE_BINDINGS, vdbFrameResolveWgsl } from "./wgsl/resolve.js";
import {
  ENV_BLUR_BINDINGS,
  ENV_GRID_BINDINGS,
  ENV_PREFILTER_BINDINGS,
  ENV_REDUCE_BINDINGS,
  vdbFrameEnvAreaWgsl,
  vdbFrameEnvBlurWgsl,
  vdbFrameEnvCubeWgsl,
  vdbFrameEnvironmentGridWgsl,
  vdbFrameEnvReduceWgsl,
} from "./wgsl/environment.js";
import { VDB_FRAME_COMPOSITE_COLOR_WGSL, VDB_FRAME_INTEGRATE_TAPS_WGSL, VDB_FRAME_SLICE_COORD_WGSL, vdbFrameCompositeWgsl } from "./wgsl/integrate.js";
import { VDB_FRAME_MAX_VOLUMES, vdbFrameUniformsWgsl } from "./uniforms.js";
import { vdbHashText, type VDBComputeSpec, type VDBRenderSpec } from "./pipelines.js";
import { VOLUME_SHADOW_BINDINGS, vdbFrameVolumeShadowWgsl } from "./wgsl/shadows.js";
import {
  EMISSION_DIRECTION_BINDINGS,
  EMISSION_PROBE_BINDINGS,
  EMISSION_REDUCE_BINDINGS,
  vdbFrameEmissionDirectionWgsl,
  vdbFrameEmissionProbeWgsl,
  vdbFrameEmissionReduceWgsl,
} from "./wgsl/emission.js";
import { vdbFrameSourceFunctionsWgsl } from "./wgsl/gather.js";
import { STAGE } from "./resources.js";
import { vdbFrameOccluderLayoutEntries } from "./wgsl/occluders.js";

/** Where a source's group-2 bindings start. */
export const SOURCE_SLOT: VDBSlot = { group: 2, firstBinding: 0 };

const COMPUTE_FRAGMENT = STAGE.COMPUTE | STAGE.FRAGMENT;

/** Group 0 of every frame kernel. */
export const GROUP0_LAYOUT: GPUBindGroupLayoutDescriptor = {
  entries: [
    { binding: 0, visibility: COMPUTE_FRAGMENT, buffer: { type: "uniform" } },
    { binding: 1, visibility: COMPUTE_FRAGMENT, buffer: { type: "uniform" } },
    { binding: 2, visibility: COMPUTE_FRAGMENT, sampler: { type: "filtering" } },
    // K6.2b: the environment buffer (written by envReduce, read as a uniform by the kernels after it)
    { binding: 3, visibility: COMPUTE_FRAGMENT, buffer: { type: "uniform" } },
    // K6.3: the Lorenz-Mie phase tables (V47): 2048 texels as binary16 in a 16 KiB uniform (no sampled texture)
    { binding: 4, visibility: COMPUTE_FRAGMENT, buffer: { type: "uniform" } },
  ],
};

/** Group 0 of the environment kernels: their parameters (`VdbFrameEnvParams`). */
export const ENV_GROUP0_LAYOUT: GPUBindGroupLayoutDescriptor = {
  entries: [{ binding: 0, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } }],
};

/** Pipeline-key options of the gather kernel. */
export interface VDBFrameGatherKey {
  readonly mix: "mixture" | "over";
  readonly multi: boolean;
  readonly maxVolumes?: number;
  /** K6.2: `'frame'` lighting fuses his density and lighting passes (the mix does not apply: his sums are additive). */
  readonly lighting?: "volume" | "frame";
  /** K6.5: a validated user emission colour hook compiled into the group's kernels (sources with emission). */
  readonly hook?: string | null;
}

/** The source's VDB module (sparse, K6.7) and its `vdbSrc*` WGSL at group 2, then the frame uniforms. */
function sourcePrelude(source: VDBVolumeSource, maxVolumes: number | undefined, environment = false): string {
  // One VDB module per shader (design §5.1): a sparse source (K6.7) gives it; a dense source has none.
  const withModule = source as VDBVolumeSource & { moduleWgsl?: (slot: VDBSlot) => string };
  if (source.kind !== "dense" && typeof withModule.moduleWgsl !== "function") {
    throw new VDBError("VDB_BAD_OPTION", "vdbCreateFrameRenderer: a sparse source must come from vdbSparseSourceGpu (K6.7).");
  }
  return [
    ...(typeof withModule.moduleWgsl === "function" ? [withModule.moduleWgsl(SOURCE_SLOT)] : []),
    source.wgsl({ slot: SOURCE_SLOT, profile: "march" }),
    vdbFrameUniformsWgsl(maxVolumes ?? VDB_FRAME_MAX_VOLUMES, { environment }),
  ].join("\n");
}

/** The gather kernel's WGSL for `source` (its `vdbSrc*` at group 2). */
export function vdbFrameGatherKernelWgsl(source: VDBVolumeSource, key: VDBFrameGatherKey): string {
  const hook = source.channels.emission ? (key.hook ?? null) : null;
  const body =
    key.lighting === "frame"
      ? vdbFrameLightingGatherWgsl({ multi: key.multi, channels: source.channels, hook })
      : vdbFrameGatherWgsl({ mix: key.mix, multi: key.multi, channels: source.channels, hook });
  return `${sourcePrelude(source, key.maxVolumes, true)}\n${body}`;
}

/** The source layout part of a label: its kind, channels and LOD levels. */
const sourceChannels = (s: VDBVolumeSource) =>
  `${s.kind} ${["density", ...(["mask", "albedo", "emission"] as const).filter((c) => s.channels[c])].join("+")}${s.lodLevels > 0 ? "+mips" : ""}`;

/** The gather kernel's spec. */
export function vdbFrameGatherSpec(source: VDBVolumeSource, key: VDBFrameGatherKey): VDBComputeSpec {
  const frame = key.lighting === "frame";
  const b = frame ? FRAME_GATHER_BINDINGS : GATHER_BINDINGS;
  const group1: GPUBindGroupLayoutEntry[] = [
    { binding: b.volumes, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
    { binding: b.group, visibility: STAGE.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    { binding: b.scatter, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
  ];
  if (key.multi) group1.push({ binding: b.partials, visibility: STAGE.COMPUTE, storageTexture: { access: "read-write", format: "r32uint", viewDimension: "3d" } });
  if (frame) {
    group1.push({ binding: FRAME_GATHER_BINDINGS.transmittance, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } });
    group1.push({ binding: FRAME_GATHER_BINDINGS.environmentGrid, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } });
  }
  if (source.channels.emission) group1.push({ binding: b.emissionLut, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } });
  // K6.5b: the host occluder maps (4 fixed slots)
  group1.push(...vdbFrameOccluderLayoutEntries(STAGE.COMPUTE));
  const hooked = source.channels.emission && key.hook ? `, hook ${vdbHashText(key.hook)}` : "";
  return {
    label: `vdb frame gather (${frame ? "frame lighting" : key.mix}${key.multi ? ", multi-group" : ""}; ${sourceChannels(source)}${hooked})`,
    code: vdbFrameGatherKernelWgsl(source, key),
    entryPoint: "vdbFrameGatherCs",
    groups: [GROUP0_LAYOUT, { entries: group1 }, { entries: source.layoutEntries(SOURCE_SLOT, STAGE.COMPUTE) }],
  };
}

/** Pipeline-key options of the shadow grid density kernel. */
export interface VDBFrameGridDensityKey {
  /** More than one caster source group: accumulate in the packed r32uint partials. */
  readonly multi: boolean;
  readonly maxVolumes?: number;
}

/** The shadow grid density kernel's spec (K6.2; one dispatch per caster source group). */
export function vdbFrameGridDensitySpec(source: VDBVolumeSource, key: VDBFrameGridDensityKey): VDBComputeSpec {
  const b = GRID_DENSITY_BINDINGS;
  const group1: GPUBindGroupLayoutEntry[] = [
    { binding: b.volumes, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
    { binding: b.group, visibility: STAGE.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
    { binding: b.grid, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
  ];
  if (key.multi) group1.push({ binding: b.partials, visibility: STAGE.COMPUTE, storageTexture: { access: "read-write", format: "r32uint", viewDimension: "3d" } });
  return {
    label: `vdb frame grid density${key.multi ? " (multi-group)" : ""}; ${sourceChannels(source)}`,
    code: `${sourcePrelude(source, key.maxVolumes)}\n${vdbFrameGridDensityWgsl({ multi: key.multi, channels: source.channels })}`,
    entryPoint: "vdbFrameGridDensityCs",
    groups: [GROUP0_LAYOUT, { entries: group1 }, { entries: source.layoutEntries(SOURCE_SLOT, STAGE.COMPUTE) }],
  };
}

/**
 * The transmittance kernel's spec (K6.2; source-free). K6.3: `scattering` (a pipeline key) adds group 2: `'mlp'` the
 * weights (the kernel's one storage buffer, D10), the shape uniform and the feature mips; `'multi-scale'` K7.3's
 * parameters and mips.
 */
export function vdbFrameTransmittanceSpec(o: { scattering?: VDBFrameScatteringKernel } = {}): VDBComputeSpec {
  const b = TRANSMITTANCE_BINDINGS;
  const scattering = o.scattering ?? "none";
  const groups: GPUBindGroupLayoutDescriptor[] = [
    GROUP0_LAYOUT,
    {
      entries: [
        { binding: b.grid, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: b.transmittance, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
        // K6.5b: the host occluder maps (4 fixed slots)
        ...vdbFrameOccluderLayoutEntries(STAGE.COMPUTE),
      ],
    },
  ];
  if (scattering === "mlp") {
    const m = TRANSMITTANCE_MLP_BINDINGS;
    groups.push({
      entries: [
        { binding: m.weights, visibility: STAGE.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: m.shape, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
        { binding: m.mips, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
      ],
    });
  } else if (scattering === "multi-scale") {
    const m = TRANSMITTANCE_MULTI_SCALE_BINDINGS;
    groups.push({
      entries: [
        { binding: m.params, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
        { binding: m.mips, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
      ],
    });
  }
  return {
    label: `vdb frame transmittance${scattering === "none" ? "" : ` (${scattering})`}`,
    code: `${vdbFrameUniformsWgsl(VDB_FRAME_MAX_VOLUMES, { environment: true })}\n${vdbFrameTransmittanceWgsl({ scattering })}`,
    entryPoint: "vdbFrameTransmittanceCs",
    groups,
  };
}

/**
 * K6.3 `gridMips`: the box mips of the `'mlp'` feature levels (source-free). `first` reads the last shadow grid
 * cascade from the stack, else a one-mip view of the level before; writes one mip (rgba16float storage).
 */
export function vdbFrameGridMipSpec(o: { first: boolean }): VDBComputeSpec {
  const b = GRID_MIP_BINDINGS;
  return {
    label: `vdb frame grid mip${o.first ? " (from the shadow grid)" : ""}`,
    code: `${vdbFrameUniformsWgsl()}\n${vdbFrameGridMipWgsl()}`,
    entryPoint: o.first ? "vdbFrameGridMipFirstCs" : "vdbFrameGridMipCs",
    groups: [
      GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.source, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
          { binding: b.out, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
        ],
      },
    ],
  };
}

/** `envPrefilter` (K6.2b): the host equirect (2D) or cube (a 6-layer 2D array view) texture → the work map. */
export function vdbFrameEnvPrefilterSpec(projection: "equirect" | "cube"): VDBComputeSpec {
  const b = ENV_PREFILTER_BINDINGS;
  const cube = projection === "cube";
  return {
    label: `vdb frame environment prefilter (${projection})`,
    code: cube ? vdbFrameEnvCubeWgsl() : vdbFrameEnvAreaWgsl(),
    entryPoint: cube ? "vdbFrameEnvCubeCs" : "vdbFrameEnvAreaCs",
    groups: [
      ENV_GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.source, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: cube ? "2d-array" : "2d" } },
          { binding: b.work, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
        ],
      },
    ],
  };
}

/** `envPrefilter`'s GGX (roughness 0.4) convolution of the work map into his 64 × 32 map (K6.2b). */
export function vdbFrameEnvBlurSpec(): VDBComputeSpec {
  const b = ENV_BLUR_BINDINGS;
  return {
    label: "vdb frame environment blur",
    code: vdbFrameEnvBlurWgsl(),
    entryPoint: "vdbFrameEnvBlurCs",
    groups: [
      ENV_GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.work, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: b.map, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
        ],
      },
    ],
  };
}

/** `envReduce` (K6.2b): the environment buffer (lights, SH) from the maps; one workgroup. */
export function vdbFrameEnvReduceSpec(): VDBComputeSpec {
  const b = ENV_REDUCE_BINDINGS;
  return {
    label: "vdb frame environment reduce",
    code: vdbFrameEnvReduceWgsl(),
    entryPoint: "vdbFrameEnvReduceCs",
    groups: [
      ENV_GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.work, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: b.map, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: b.environment, visibility: STAGE.COMPUTE, buffer: { type: "storage" } },
        ],
      },
    ],
  };
}

/** The environment grid kernel's spec (K6.2b; source-free, the last cascade). */
export function vdbFrameEnvironmentGridSpec(): VDBComputeSpec {
  const b = ENV_GRID_BINDINGS;
  return {
    label: "vdb frame environment grid",
    code: `${vdbFrameUniformsWgsl(VDB_FRAME_MAX_VOLUMES, { environment: true })}\n${vdbFrameEnvironmentGridWgsl()}`,
    entryPoint: "vdbFrameEnvironmentGridCs",
    groups: [
      GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.grid, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
          { binding: b.out, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
        ],
      },
    ],
  };
}

/** K6.5: the volume shadow map kernel's spec (source-free: the shadow grid stack; one dispatch per map piece). */
export function vdbFrameVolumeShadowSpec(): VDBComputeSpec {
  const b = VOLUME_SHADOW_BINDINGS;
  return {
    label: "vdb frame volume shadow map",
    code: `${vdbFrameUniformsWgsl()}\n${frameChunksWgsl(["grid"])}\n${vdbFrameVolumeShadowWgsl()}`,
    entryPoint: "vdbFrameVolumeShadowCs",
    groups: [
      GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.piece, visibility: STAGE.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
          { binding: b.grid, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
          { binding: b.map, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d-array" } },
        ],
      },
    ],
  };
}

/** K6.5: the emission direction kernel's spec (per source group with emitters; the source must have emission). */
export function vdbFrameEmissionDirectionSpec(source: VDBVolumeSource, key: { maxVolumes?: number; hook?: string | null }): VDBComputeSpec {
  if (!source.channels.emission) throw new VDBError("VDB_BAD_OPTION", "vdbFrameEmissionDirectionSpec: the source has no emission channel.");
  const b = EMISSION_DIRECTION_BINDINGS;
  const hook = key.hook ?? null;
  const code = `${sourcePrelude(source, key.maxVolumes)}
${frameChunksWgsl(["constants", "froxel", "box", "lod"])}
@group(1) @binding(${b.volumes}) var<uniform> vdbFrameVolumes: VdbFrameVolumes;
${vdbFrameSourceFunctionsWgsl(source.channels, { lutBinding: b.lut, hook })}
${vdbFrameEmissionDirectionWgsl()}`;
  return {
    label: `vdb frame emission direction; ${sourceChannels(source)}${hook ? `, hook ${vdbHashText(hook)}` : ""}`,
    code,
    entryPoint: "vdbFrameEmissionDirectionCs",
    groups: [
      GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.volumes, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
          { binding: b.emitters, visibility: STAGE.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
          { binding: b.lut, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: b.directions, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
        ],
      },
      { entries: source.layoutEntries(SOURCE_SLOT, STAGE.COMPUTE) },
    ],
  };
}

/** K6.5: the emission reduce kernel's spec (source-free). */
export function vdbFrameEmissionReduceSpec(): VDBComputeSpec {
  const b = EMISSION_REDUCE_BINDINGS;
  return {
    label: "vdb frame emission reduce",
    code: `${vdbFrameUniformsWgsl()}\n${vdbFrameEmissionReduceWgsl()}`,
    entryPoint: "vdbFrameEmissionReduceCs",
    groups: [
      GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.emitters, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
          { binding: b.directions, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: b.out, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
        ],
      },
    ],
  };
}

/** K6.5: the emission probe kernel's spec (source-free). */
export function vdbFrameEmissionProbeSpec(): VDBComputeSpec {
  const b = EMISSION_PROBE_BINDINGS;
  return {
    label: "vdb frame emission probes",
    code: `${vdbFrameUniformsWgsl()}\n${vdbFrameEmissionProbeWgsl()}`,
    entryPoint: "vdbFrameEmissionProbeCs",
    groups: [
      GROUP0_LAYOUT,
      {
        entries: [
          { binding: b.grid, visibility: STAGE.COMPUTE, buffer: { type: "uniform" } },
          { binding: b.emitters, visibility: STAGE.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: b.probes, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
        ],
      },
    ],
  };
}

/** The resolve kernel's WGSL. */
export function vdbFrameResolveKernelWgsl(o: { temporal: boolean }): string {
  return `${vdbFrameUniformsWgsl()}\n${vdbFrameResolveWgsl(o)}`;
}

/** The resolve kernel's spec. */
export function vdbFrameResolveSpec(o: { temporal: boolean }): VDBComputeSpec {
  const b = RESOLVE_BINDINGS;
  const group1: GPUBindGroupLayoutEntry[] = [
    { binding: b.scatter, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
    { binding: b.integrated, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
  ];
  if (o.temporal) {
    group1.push(
      { binding: b.historyIn, visibility: STAGE.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
      { binding: b.historyOut, visibility: STAGE.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
    );
  }
  return {
    label: `vdb frame resolve${o.temporal ? " (temporal)" : ""}`,
    code: vdbFrameResolveKernelWgsl(o),
    entryPoint: "vdbFrameResolveCs",
    groups: [GROUP0_LAYOUT, { entries: group1 }],
  };
}

/** Pipeline-key options of the composite pass. */
export interface VDBFrameCompositeKey {
  readonly colorFormat: GPUTextureFormat;
  /** `'depth'`: a depth texture (`texture_depth_2d`); `'r32float'`: a colour texture of view-independent depth. */
  readonly depth: "depth" | "r32float";
  readonly srgb: boolean;
}

/** The composite pass's WGSL. */
export function vdbFrameCompositeKernelWgsl(key: Pick<VDBFrameCompositeKey, "depth" | "srgb">): string {
  return `${vdbFrameUniformsWgsl()}\n${vdbFrameCompositeWgsl(key)}`;
}

/** The composite pass's spec (premultiplied blending without sRGB; the shader blends itself with it). */
export function vdbFrameCompositeSpec(key: VDBFrameCompositeKey): VDBRenderSpec {
  const group1: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: STAGE.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
    { binding: 1, visibility: STAGE.FRAGMENT, sampler: { type: "filtering" } },
    { binding: 2, visibility: STAGE.FRAGMENT, texture: { sampleType: key.depth === "depth" ? "depth" : "unfilterable-float", viewDimension: "2d" } },
  ];
  if (key.srgb) group1.push({ binding: 3, visibility: STAGE.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } });
  const premultiplied: GPUBlendState = {
    color: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
  };
  return {
    label: `vdb frame composite (${key.srgb ? "sRGB blend" : "premultiplied"}, ${key.depth})`,
    code: vdbFrameCompositeKernelWgsl(key),
    vertexEntryPoint: "vdbFrameCompositeVs",
    fragmentEntryPoint: "vdbFrameCompositeFs",
    groups: [GROUP0_LAYOUT, { entries: group1 }],
    targets: [{ format: key.colorFormat, ...(key.srgb ? {} : { blend: premultiplied }) }],
  };
}

/** Values per row of the chunk probe: inputs, then outputs. */
export const CHUNK_PROBE_INPUTS = 12;
export const CHUNK_PROBE_OUTPUTS = 29;
const ROW = CHUNK_PROBE_INPUTS + CHUNK_PROBE_OUTPUTS;

/**
 * The `?probe=view` kernel (harness only): row i of a read-write storage buffer holds 12 inputs and gets 24
 * outputs of the source-free chunks (slice depth, view position, noise, fade, box segment, lod, HG, phase
 * colour, shadow step / depth / transmittance, isotropic diffusion, slice coordinate, two-tap integrate,
 * sRGB composite), each against its tools/frame twin. Group 1 binding 0: the buffer (its one storage buffer).
 */
export function vdbFrameChunkProbeSpec(): VDBComputeSpec {
  const code = `${vdbFrameUniformsWgsl()}
${frameChunksWgsl(["constants", "froxel", "noise", "fade", "box", "lod", "phase", "shadow", "octaves"])}
${VDB_FRAME_SLICE_COORD_WGSL}
${VDB_FRAME_INTEGRATE_TAPS_WGSL}
${VDB_FRAME_COMPOSITE_COLOR_WGSL}
@group(1) @binding(0) var<storage, read_write> vdbFrameProbe: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn vdbFrameChunkProbeCs(@builtin(global_invocation_id) gid: vec3u) {
  let rows = arrayLength(&vdbFrameProbe) / ${ROW}u;
  if (gid.x >= rows) {
    return;
  }
  let at = gid.x * ${ROW}u;
  var x: array<f32, ${CHUNK_PROBE_INPUTS}>;
  for (var i = 0u; i < ${CHUNK_PROBE_INPUTS}u; i++) {
    x[i] = vdbFrameProbe[at + i];
  }
  let o = at + ${CHUNK_PROBE_INPUTS}u;
  // x0 t, x1 u, x2 v, x3 view depth, x4 jitter, x5 cos, x6 g, x7 extinction, x8 absorption, x9 ms, x10 c, x11 alpha
  vdbFrameProbe[o + 0u] = vdbFrameSliceDepth(x[0]);
  let p = vdbFrameViewPosition(vec2f(x[1], x[2]), x[3]);
  vdbFrameProbe[o + 1u] = p.x;
  vdbFrameProbe[o + 2u] = p.y;
  vdbFrameProbe[o + 3u] = p.z;
  vdbFrameProbe[o + 4u] = vdbFrameNoise(x[4], vec2f(x[1], x[2]) * 977.0);
  vdbFrameProbe[o + 5u] = vdbFrameDistanceFade(p);
  let seg = vdbFrameBoxSegment(vec3f(x[1], x[2], x[5]) * 2.0 - 1.0, vec3f(x[6], x[7] * 0.1, x[0]) * 2.0 - 0.5);
  vdbFrameProbe[o + 6u] = seg.x;
  vdbFrameProbe[o + 7u] = seg.y;
  vdbFrameProbe[o + 8u] = vdbFrameLod(x[3] * 0.01, 128.0);
  vdbFrameProbe[o + 9u] = vdbFrameHenyeyGreenstein(x[5], x[6]);
  let pc = vdbFramePhaseColor(vec4f(x[6], -0.3, 0.2, 0.0), vec3f(1.2, 0.8, 1.0), x[5]);
  vdbFrameProbe[o + 10u] = pc.x;
  vdbFrameProbe[o + 11u] = pc.z;
  let st = vdbFrameShadowStep(x[7], x[8], x[9]);
  vdbFrameProbe[o + 12u] = st.y;
  vdbFrameProbe[o + 13u] = st.z;
  let depths = st * 3.0;
  vdbFrameProbe[o + 14u] = vdbFrameShadowDepth(depths, x[9]);
  let tr = vdbFrameShadowTransmittance(depths, x[9], vec4f(0.0));
  vdbFrameProbe[o + 15u] = tr.x;
  vdbFrameProbe[o + 16u] = tr.y;
  vdbFrameProbe[o + 17u] = vdbFrameIsotropicDiffusion(st.z, x[8], st.y);
  let kc = vdbFrameSliceCoord(x[3], vdbFrame.range.xy, vdbFrame.grid.z);
  vdbFrameProbe[o + 18u] = kc.x;
  vdbFrameProbe[o + 19u] = kc.y;
  let taps = vdbFrameIntegrateTaps(vec4f(x[7], x[8], x[9], x[11] * 0.5), vec4f(x[7] + x[1], x[8] + x[2], x[9], x[11]), x[10]);
  vdbFrameProbe[o + 20u] = taps.x;
  vdbFrameProbe[o + 21u] = taps.w;
  let composite = vdbFrameCompositeColor(vec4f(x[1] * 0.5, x[2] * 0.5, x[6] * 0.5, 1.0 - x[11]), vec4f(x[0], x[5] * 0.5 + 0.5, x[10], 1.0), 1u, 1u);
  vdbFrameProbe[o + 22u] = composite.x;
  vdbFrameProbe[o + 23u] = composite.w;
  // K6.3: the Lorenz-Mie phases (the LUT), two-sided diffusion, the octave total and phased octave light (count 3, a 0.6, b 0.45, c 0.7)
  vdbFrameProbe[o + 24u] = vdbFramePhaseWeight(vec4f(x[6], -0.3, 0.2, 0.5), x[5]);
  vdbFrameProbe[o + 25u] = vdbFramePhaseColor(vec4f(0.0, 0.0, 0.0, 2.0), vec3f(1.0), x[5]).x;
  vdbFrameProbe[o + 26u] = vdbFrameTwoSidedDiffusion(tr.x, depths.z, depths.y, x[8]);
  let octaves = vec4f(3.0, 0.6, 0.45, 0.7);
  vdbFrameProbe[o + 27u] = vdbFrameShadowTotal(x[11], octaves);
  vdbFrameProbe[o + 28u] = vdbFrameOctaveLight(octaves, x[7], vec4f(x[6], -0.3, 0.2, 0.0), x[5]);
}
`;
  return {
    label: "vdb frame chunk probe",
    code,
    entryPoint: "vdbFrameChunkProbeCs",
    groups: [GROUP0_LAYOUT, { entries: [{ binding: 0, visibility: STAGE.COMPUTE, buffer: { type: "storage" } }] }],
  };
}
