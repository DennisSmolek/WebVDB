// Portions derived from mjurczyk/openvdb (MIT), commit 75e7345 (v0.26.0):
//   src/three-openvdb/frame/core/FrameEmissionLight.js L1–287 (FrameEmissionLightBase: emitters, keys, probe
//   cascades, prepareProbes); src/three-openvdb/frame/webgpu/FrameEmissionLight.js L59–658 (probe grid uniforms,
//   the direction / reduce / probe pass materials, FrameEmissionLightNode, FrameEmissionLight: targets, slots,
//   updateProbes, dispose, the node registration).
// Copyright (c) mjurczyk/openvdb contributors. Licensed under the MIT License (see THIRD_PARTY.md).
// Changes: TypeScript; one class (his CPU base and WebGPU subclass merged); the direction pass reads each emitter
// through its own volume source (one direction material per source, as the other passes, DIVERGES V30) instead of
// one material that swaps 3D textures; no CPU mip step (the sources hold their LOD); the pass materials belong to
// the light (his are shared by all lights and never freed); the probe cascade maths is engine-free
// (`@pmndrs/vdb/tools` `vdbFrameEmissionProbeGrid`, with twins of every pass); every three r187 internal goes
// through `three-internals.ts` (VDB_ENGINE_MISMATCH on another three). Divergences: DIVERGES.md (K3.6 rows).

/**
 * K3.6: light from the emissive grids of every visible `VDBFrameVolume` (his `FrameEmissionLight`). Add one to the
 * scene; three's lit node materials (`MeshStandardMaterial`, `MeshLambertMaterial`, …) receive it as irradiance, like a
 * light probe. Per changed volume: its emitted light along 16 directions × 8² rays (direction pass), the mean and the
 * luminance-weighted centre (reduce pass); then L1 irradiance probes of all volumes as point lights, in up to 4
 * cascades of ≤ 32³ (probe pass). A lit pixel reads 6 texels (12 near a cascade side), for any number of volumes.
 * The passes run before the first lit mesh of each render and only when a volume or the cascades changed.
 *
 *   const light = new VDBFrameEmissionLight({ intensity: 25, resolution: 32 });
 *   scene.add(light);   // volumes need an emission channel; colour: VDBFrameVolume `emissionColor`
 */

import * as THREE from "three";
import { AnalyticLightNode, NodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import type { VDBTslNode } from "../grid-node.js";
import { vdbVolumeNodes, type VDBThreeVolumeSource, type VDBVolumeNodesOptions, type VDBVolumeUniforms } from "../volume-source.js";
import { createFrameVolumeUniforms, createGridNodes, frameFragCoord, frameFullscreenVertex, frameLoad, type FrameVolumeUniforms, type GridUniforms } from "./chunks.js";
import {
  frameEmissionAtlasColumns as atlasColumns,
  frameEmissionAtlasSize as atlasSize,
  frameEmissionDirections as directionCount,
  frameEmissionGoldenAngle as goldenAngle,
  frameEmissionMaxCascades as maxCascades,
  frameEmissionRays as rayCount,
  vdbFrameEmissionProbeGrid,
  vdbFrameEmitterBounds,
} from "../../tools/frame/emission.js";
import { setVolumeUniforms } from "./renderer.js";
import { vdbFramePrecompiling, vdbFrameRegisterLightNode, vdbFrameRenderUpdateType, vdbFrameRendererUtils } from "./three-internals.js";
import type { VDBFrameVolume } from "./volume.js";
import type { FrameVolumeRecord } from "./view.js";
import type { Vec3 } from "../../tools/frame/index.js";

const { Discard, Fn, If, Loop, Break, abs, clamp, cos, cross, dot, exp, float, floor, int, inverseSqrt, ivec2, luminance, max, min, mix, normalWorld, normalize, outputStruct, positionGeometry, positionWorld, property, renderGroup, select, sin, smoothstep, sqrt, texture, uniform, uniformArray, vec3, vec4 } =
  TSL as unknown as Record<string, VDBTslNode>;

type N = VDBTslNode;
type RendererLike = { render(scene: THREE.Object3D, camera: THREE.Camera): void; setRenderTarget(t: THREE.RenderTarget | null): void; autoClear: boolean };

// NOTE Probe cascades in the uniform names of createGridNodes. Cascade 0 is the finest.
const createProbeGridUniforms = (group: N | null = null) => {
  const nodes = {
    frameGridMin: uniformArray(
      Array.from({ length: maxCascades }, () => new THREE.Vector3()),
      "vec3",
    ),
    frameGridSize: uniformArray(
      Array.from({ length: maxCascades }, () => new THREE.Vector3(1.0, 1.0, 1.0)),
      "vec3",
    ),
    frameGridCount: uniform(0, "int"),
    frameGridResolution: uniform(1.0),
    frameGridColumns: uniform(atlasColumns),
    frameGridAtlasSize: uniform(new THREE.Vector2(atlasSize, atlasSize)),
  };

  if (group) {
    Object.values(nodes).forEach((node) => node.setGroup(group));
  }

  return nodes as unknown as GridUniforms;
};

const setProbeGridUniforms = (uniforms: GridUniforms, light: VDBFrameEmissionLight) => {
  const min = uniforms.frameGridMin.array as THREE.Vector3[];
  const size = uniforms.frameGridSize.array as THREE.Vector3[];

  for (let i = 0; i < maxCascades; i++) {
    min[i]!.copy(light.cascadeMin[i]!);
    size[i]!.copy(light.cascadeSize[i]!);
  }

  uniforms.frameGridCount.value = light.cascadeCount;
  uniforms.frameGridResolution.value = light.probeResolution;
};

const fullscreenPositions = [-1, -1, 0, 3, -1, 0, -1, 3, 0];
const quadPositions = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0];

const createGeometry = (positions: number[]) => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));

  return geometry;
};

const createTarget = (count: number, type: THREE.TextureDataType, filter: THREE.MinificationTextureFilter & THREE.MagnificationTextureFilter, width = 1, height = 1) =>
  new THREE.RenderTarget(width, height, {
    count,
    type,
    format: THREE.RGBAFormat,
    minFilter: filter,
    magFilter: filter,
    generateMipmaps: false,
    depthBuffer: false,
  });

const setPassMaterial = (material: THREE.Material) => {
  material.side = THREE.DoubleSide;
  material.depthTest = false;
  material.depthWrite = false;
  material.blending = THREE.NoBlending;
};

/**
 * Light that one volume sends in each of 16 directions: one texel per direction, one row per volume (slot). Parallel
 * rays cover the bounding sphere; each adds the emission with the volume's transmittance. Output 0: intensity
 * (emission × ray area) and the radius. Output 1: the luminance-weighted world position. One material per volume
 * source (the source is compiled in). Twin: `vdbFrameEmissionDirectionTexel`.
 */
export class VDBFrameEmissionDirectionMaterial extends NodeMaterial {
  readonly volumeUniforms: FrameVolumeUniforms = createFrameVolumeUniforms();
  readonly sourceUniforms: VDBVolumeUniforms;
  readonly source: VDBThreeVolumeSource;
  readonly emissionColor: VDBVolumeNodesOptions["emissionColor"];
  readonly frameSlot = uniform(0.0);
  readonly frameTargetSize = uniform(new THREE.Vector2(directionCount, 1.0));
  readonly frameEmitterCenter = uniform(new THREE.Vector3());
  readonly frameEmitterRadius = uniform(1.0);
  readonly frameSteps = uniform(24, "int");
  mipmaps: boolean;

  constructor(o: { source: VDBThreeVolumeSource; sourceUniforms: VDBVolumeUniforms; mipmaps: boolean; emissionColor?: VDBVolumeNodesOptions["emissionColor"] }) {
    super();

    this.name = "VDBFrameEmissionDirectionMaterial";
    this.source = o.source;
    this.sourceUniforms = o.sourceUniforms;
    this.mipmaps = o.mipmaps;
    this.emissionColor = o.emissionColor;
    this.buildNodes();

    setPassMaterial(this);
  }

  setMipmaps(mipmaps: boolean): void {
    if (mipmaps !== this.mipmaps) {
      this.mipmaps = mipmaps;
      this.buildNodes();
      this.needsUpdate = true;
    }
  }

  buildNodes(): void {
    const volume = vdbVolumeNodes(this.source, this.sourceUniforms, { useMipmaps: this.mipmaps, ...(this.emissionColor ? { emissionColor: this.emissionColor } : {}) });
    const { frameWorldToLocal, frameAbsorbance } = this.volumeUniforms;
    const frameWorldScale = this.sourceUniforms.worldScale;
    const { frameSlot, frameTargetSize, frameEmitterCenter, frameEmitterRadius, frameSteps } = this;

    const outputIntensity = property("vec4");
    const outputCenter = property("vec4");

    // NOTE The quad covers the row of the slot
    this.vertexNode = vec4(positionGeometry.x, frameSlot.add(positionGeometry.y.mul(0.5).add(0.5)).div(frameTargetSize.y).mul(2.0).sub(1.0), 0.0, 1.0);

    this.colorNode = Fn(() => {
      const index = floor(frameFragCoord(frameTargetSize).x).toVar();

      // NOTE Fibonacci sphere direction, from the volume to the viewer
      const y = float(1.0).sub(index.mul(2.0).add(1.0).div(directionCount)).toVar();
      const ring = sqrt(max(0.0, float(1.0).sub(y.mul(y))));
      const phi = index.mul(goldenAngle);
      const direction = vec3(cos(phi).mul(ring), y, sin(phi).mul(ring)).toVar();
      const helper = select(abs(direction.y).lessThan(0.999), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
      const tangent = normalize(cross(helper, direction)).toVar();
      const bitangent = cross(direction, tangent).toVar();

      const diameter = frameEmitterRadius.mul(2.0).toVar();
      const spacing = diameter.div(rayCount).toVar();
      const stepCount = float(frameSteps).toVar();
      const intensity = vec3(0.0).toVar();
      const center = vec4(0.0).toVar();

      Loop({ start: int(0), end: int(rayCount * rayCount), type: "int", condition: "<", name: "r" }, ({ r }: { r: N }) => {
        const u = float(r.mod(int(rayCount))).add(0.5).div(rayCount).mul(2.0).sub(1.0);
        const v = float(r.div(int(rayCount))).add(0.5).div(rayCount).mul(2.0).sub(1.0);
        const worldStart = frameEmitterCenter.add(direction.mul(frameEmitterRadius)).add(tangent.mul(u.mul(frameEmitterRadius))).add(bitangent.mul(v.mul(frameEmitterRadius))).toVar();
        const worldEnd = worldStart.sub(direction.mul(diameter)).toVar();
        const localStart = frameWorldToLocal.mul(vec4(worldStart, 1.0)).xyz.toVar();
        const localEnd = frameWorldToLocal.mul(vec4(worldEnd, 1.0)).xyz.toVar();
        const segment = volume.boxSegment(localStart, localEnd).toVar();

        If(segment.x.lessThan(segment.y), () => {
          const segmentLength = segment.y.sub(segment.x).toVar();
          const stepLength = segmentLength.mul(diameter).div(stepCount).toVar();
          // NOTE Smoke uses the mip level of the step. Emission uses level 0: power grows with temperature^4, mipmaps lose it.
          const lod = volume.getLod(stepLength.div(frameWorldScale)).toVar();
          const transmittance = float(1.0).toVar();

          Loop({ start: int(0), end: frameSteps, type: "int", condition: "<", name: "s" }, ({ s }: { s: N }) => {
            const t = segment.x.add(segmentLength.mul(float(s).add(0.5).div(stepCount))).toVar();
            const localPosition = mix(localStart, localEnd, t).toVar();
            const extinction = volume.sampleExtinction(localPosition, lod).toVar();

            If(extinction.greaterThan(0.0), () => {
              // NOTE Absorbance scales the shadow of the volume, the same as for scene lights
              const alpha = float(1.0).sub(exp(extinction.mul(stepLength).negate()));
              const radiance = volume.sampleEmission(localPosition, float(0.0)).mul(transmittance.mul(alpha)).toVar();

              intensity.addAssign(radiance);
              center.addAssign(vec4(mix(worldStart, worldEnd, t), 1.0).mul(luminance(radiance)));
              transmittance.mulAssign(exp(extinction.mul(frameAbsorbance).mul(stepLength).negate()));

              If(transmittance.lessThan(0.01), () => {
                Break();
              });
            });
          });
        });
      });

      const area = spacing.mul(spacing);

      outputIntensity.assign(vec4(intensity.mul(area), frameEmitterRadius));
      outputCenter.assign(center.mul(area));

      return vec4(0.0);
    })();

    this.outputNode = outputStruct(outputIntensity, outputCenter);
  }
}

/** One texel per volume: the mean intensity of all directions, and the centre of the visible emission. Twin: `vdbFrameEmissionReduce`. */
export class VDBFrameEmissionReduceMaterial extends NodeMaterial {
  readonly frameTargetSize = uniform(new THREE.Vector2(1.0, 1.0));
  readonly frameDirectionSize = uniform(new THREE.Vector2(directionCount, 1.0));
  readonly frameDirectionIntensity = texture(new THREE.Texture());
  readonly frameDirectionCenter = texture(new THREE.Texture());

  constructor() {
    super();

    this.name = "VDBFrameEmissionReduceMaterial";

    const outputIntensity = property("vec4");
    const outputCenter = property("vec4");

    this.vertexNode = frameFullscreenVertex;

    this.colorNode = Fn(() => {
      const row = int(floor(frameFragCoord(this.frameTargetSize).y)).toVar();
      const intensity = vec4(0.0).toVar();
      const center = vec4(0.0).toVar();

      Loop(directionCount, ({ i }: { i: N }) => {
        const texel = ivec2(i, row);

        intensity.addAssign(frameLoad(this.frameDirectionIntensity, texel, this.frameDirectionSize));
        center.addAssign(frameLoad(this.frameDirectionCenter, texel, this.frameDirectionSize));
      });

      const visible = center.w.greaterThan(0.0);

      outputIntensity.assign(intensity.div(directionCount));
      outputCenter.assign(select(visible, vec4(center.xyz.div(max(center.w, 1e-12)), 1.0), vec4(0.0)));

      return vec4(0.0);
    })();

    this.outputNode = outputStruct(outputIntensity, outputCenter);

    setPassMaterial(this);
  }
}

/**
 * L1 irradiance of all volumes at each probe, E(n) = x + dot(yzw, n) per colour channel: each volume is a point light
 * at its emission centre, its radius softening the distance near it. Twin: `vdbFrameEmissionProbe`.
 */
export class VDBFrameEmissionProbeMaterial extends NodeMaterial {
  readonly gridUniforms = createProbeGridUniforms();
  readonly frameEmitterCount = uniform(0, "int");
  readonly frameEmitterSize = uniform(new THREE.Vector2(1.0, 1.0));
  readonly frameEmitterIntensity = texture(new THREE.Texture());
  readonly frameEmitterCenter = texture(new THREE.Texture());

  constructor() {
    super();

    this.name = "VDBFrameEmissionProbeMaterial";

    const outputRed = property("vec4");
    const outputGreen = property("vec4");
    const outputBlue = property("vec4");

    const grid = createGridNodes(this.gridUniforms);
    const { frameGridMin, frameGridSize, frameGridCount, frameGridResolution, frameGridAtlasSize } = this.gridUniforms;

    this.vertexNode = frameFullscreenVertex;

    this.colorNode = Fn(() => {
      const fragCoord = frameFragCoord(frameGridAtlasSize).toVar();

      // NOTE Tiles right of the atlas columns are not used. Smaller grids leave them empty.
      If(fragCoord.x.greaterThanEqual(frameGridResolution.mul(atlasColumns)), () => {
        Discard();
      });

      const cell = grid.getGridCell(fragCoord);

      If(cell.cascade.greaterThanEqual(frameGridCount), () => {
        Discard();
      });

      const probe = frameGridMin.element(cell.cascade).add(cell.gridUvw.mul(frameGridSize.element(cell.cascade))).toVar();

      const constant = vec3(0.0).toVar();
      const directionRed = vec3(0.0).toVar();
      const directionGreen = vec3(0.0).toVar();
      const directionBlue = vec3(0.0).toVar();

      Loop(this.frameEmitterCount, ({ i }: { i: N }) => {
        const texel = ivec2(int(0), i);
        const center = frameLoad(this.frameEmitterCenter, texel, this.frameEmitterSize).toVar();

        If(center.w.greaterThan(0.0), () => {
          const emitter = frameLoad(this.frameEmitterIntensity, texel, this.frameEmitterSize).toVar();
          const toProbe = probe.sub(center.xyz).toVar();
          const distanceSquared = dot(toProbe, toProbe).toVar();
          const soft = emitter.w.mul(0.25);
          const softDistanceSquared = distanceSquared.add(soft.mul(soft)).toVar();
          const irradiance = emitter.rgb.div(softDistanceSquared).toVar();

          // NOTE L1 projection of a point light: E(n) = E / 4 + E / 2 * dot(l, n). The direction fades near the centre.
          const toLight = toProbe.mul(inverseSqrt(max(distanceSquared, 1e-12))).mul(distanceSquared.div(softDistanceSquared).mul(-0.5)).toVar();

          constant.addAssign(irradiance.mul(0.25));
          directionRed.addAssign(toLight.mul(irradiance.r));
          directionGreen.addAssign(toLight.mul(irradiance.g));
          directionBlue.addAssign(toLight.mul(irradiance.b));
        });
      });

      outputRed.assign(vec4(constant.r, directionRed));
      outputGreen.assign(vec4(constant.g, directionGreen));
      outputBlue.assign(vec4(constant.b, directionBlue));

      return vec4(0.0);
    })();

    this.outputNode = outputStruct(outputRed, outputGreen, outputBlue);

    setPassMaterial(this);
  }
}

/**
 * The irradiance at a lit point from the finest probe cascade that contains it; near a cascade side the next one
 * blends in (as the shadow grid); past the last cascade its clamped edge probes fall off with distance squared.
 */
export class VDBFrameEmissionLightNode extends AnalyticLightNode<THREE.Light> {
  static get type(): string {
    return "VDBFrameEmissionLightNode";
  }

  readonly probeMaps: N[];
  readonly gridUniforms: GridUniforms;
  readonly probeCenter: N;
  readonly probeActive: N;

  constructor(light: VDBFrameEmissionLight | null = null) {
    super(light as never);

    // NOTE The passes run before the first lit mesh of each render, as shadow maps
    const render = vdbFrameRenderUpdateType();
    (this as unknown as { updateType: string; updateBeforeType: string }).updateType = render;
    (this as unknown as { updateType: string; updateBeforeType: string }).updateBeforeType = render;

    const maps = light ? light.probeTarget.textures : [new THREE.Texture(), new THREE.Texture(), new THREE.Texture()];

    this.probeMaps = maps.map((map) => texture(map));

    // NOTE Render group, as three's lights: lit materials without node changes update only shared uniforms
    this.gridUniforms = createProbeGridUniforms(renderGroup);
    this.probeCenter = uniform(new THREE.Vector3()).setGroup(renderGroup);
    this.probeActive = uniform(0.0).setGroup(renderGroup);
  }

  private get emissionLight(): VDBFrameEmissionLight {
    return (this as unknown as { light: VDBFrameEmissionLight }).light;
  }

  override updateBefore(frame: never): boolean | undefined {
    const { renderer, scene, camera } = frame as { renderer: object; scene: THREE.Scene; camera: THREE.Camera };

    // NOTE No passes while three precompiles materials (compileAsync), as shadow maps
    if (vdbFramePrecompiling(renderer)) {
      return undefined;
    }

    this.emissionLight.updateProbes(renderer as unknown as RendererLike, scene, camera);

    return undefined;
  }

  override update(frame: never): boolean | undefined {
    super.update(frame);

    const light = this.emissionLight;

    setProbeGridUniforms(this.gridUniforms, light);
    this.probeCenter.value.copy(light.probeCenter);
    this.probeActive.value = light.emitterCount > 0 ? 1.0 : 0.0;

    return undefined;
  }

  override setup(builder: never): null {
    const grid = createGridNodes(this.gridUniforms);
    const { frameGridMin, frameGridSize, frameGridCount, frameGridResolution } = this.gridUniforms;
    const { probeCenter, probeActive } = this;
    const [redMap, greenMap, blueMap] = this.probeMaps;

    const irradiance = Fn(() => {
      const position = positionWorld.toVar();
      const red = vec4(0.0).toVar();
      const green = vec4(0.0).toVar();
      const blue = vec4(0.0).toVar();
      const falloff = float(1.0).toVar();

      const sampleProbes = (cascade: N, uvw: N, weight: N) => {
        red.assign(mix(red, grid.sampleGrid(redMap, cascade, uvw), weight));
        green.assign(mix(green, grid.sampleGrid(greenMap, cascade, uvw), weight));
        blue.assign(mix(blue, grid.sampleGrid(blueMap, cascade, uvw), weight));
      };

      If(frameGridCount.greaterThan(0), () => {
        const cascade = grid.findCascade(position, int(0)).toVar();

        If(cascade.greaterThanEqual(0), () => {
          const uvw = grid.gridUvw(cascade, position).toVar();

          sampleProbes(cascade, uvw, float(1.0));

          const sideDistance = min(uvw, vec3(1.0).sub(uvw));
          const blend = float(1.0).sub(smoothstep(0.0, 0.1, min(sideDistance.x, min(sideDistance.y, sideDistance.z)))).toVar();

          If(blend.greaterThan(0.0), () => {
            const nextCascade = grid.findCascade(position, cascade.add(1)).toVar();

            If(nextCascade.greaterThanEqual(0), () => {
              sampleProbes(nextCascade, grid.gridUvw(nextCascade, position), blend);
            });
          });
        }).Else(() => {
          const lastCascade = frameGridCount.sub(1).toVar();
          const inset = float(0.5).div(frameGridResolution);
          const uvw = clamp(grid.gridUvw(lastCascade, position), vec3(inset), vec3(float(1.0).sub(inset))).toVar();
          const toEdge = frameGridMin.element(lastCascade).add(uvw.mul(frameGridSize.element(lastCascade))).sub(probeCenter);
          const toPoint = position.sub(probeCenter);

          sampleProbes(lastCascade, uvw, float(1.0));
          falloff.assign(min(1.0, dot(toEdge, toEdge).div(max(dot(toPoint, toPoint), 1e-6))));
        });
      });

      const normal = normalWorld;

      return max(vec3(red.x.add(dot(red.yzw, normal)), green.x.add(dot(green.yzw, normal)), blue.x.add(dot(blue.yzw, normal))), vec3(0.0)).mul(falloff.mul(probeActive));
    })();

    const b = builder as unknown as { context: { irradiance: N } };
    const colorNode = (this as unknown as { colorNode: N }).colorNode;

    b.context.irradiance.addAssign(colorNode.mul(irradiance));

    return null;
  }
}

/** Props of {@link VDBFrameEmissionLight} (his `FrameEmissionLight` props; same names and defaults). */
export interface VDBFrameEmissionLightProps {
  color?: THREE.ColorRepresentation;
  intensity?: number;
  /** Probes on each axis of each cascade, 2–32 (default 16). */
  resolution?: number;
  /** Size of the last cascade as a multiple of the box of all emitting volumes; past it light falls off with distance² (default 3). */
  range?: number;
  /** 1: one grid for all volumes. Up to 4: finer grids around the camera (default 1). */
  cascades?: number;
  /** The finer cascades cover this distance from the camera (default 512). */
  distance?: number;
  /** Steps of each ray in a volume (default 24). */
  steps?: number;
}

/** One emitting volume of the current render (his `createEmitter`; the record `setVolumeUniforms` reads). */
interface Emitter extends FrameVolumeRecord {
  frameVolume: VDBFrameVolume | null;
  key: string;
  center: THREE.Vector3;
  radius: number;
}

const unitBox = new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5));
const cameraPosition = new THREE.Vector3();

const createEmitter = (): Emitter => ({
  frameVolume: null,
  key: "",
  matrix: new THREE.Matrix4(),
  matrixWorld: new THREE.Matrix4(),
  worldToLocal: new THREE.Matrix4(),
  localToView: new THREE.Matrix4(),
  viewToLocal: new THREE.Matrix4(),
  worldBox: new THREE.Box3(),
  viewNear: 0,
  viewFar: 0,
  ready: true,
  center: new THREE.Vector3(),
  radius: 0.0,
});

// NOTE An id for each source object (the emitter key: a new source or a replaced grid draws again)
const sourceIds = new WeakMap<object, number>();
let nextSourceId = 1;

const sourceKey = (frameVolume: VDBFrameVolume): string => {
  const source = frameVolume.source as unknown as { version?: number; lod?: string; lodLevels: number; textures?: Record<string, { map: THREE.Texture; mip: THREE.Texture | null } | null> };
  let id = sourceIds.get(source);

  if (id === undefined) {
    id = nextSourceId++;
    sourceIds.set(source, id);
  }

  const values: (number | string)[] = [id, source.version ?? -1, source.lodLevels, source.lod ?? "-"];

  if (source.textures) {
    for (const name of ["density", "mask", "emission"]) {
      const t = source.textures[name];

      values.push(t ? `${t.map.id}.${t.map.version}` : "-", t?.mip ? `${t.mip.id}.${t.mip.version}` : "-");
    }
  }

  // NOTE The colour hook's runtime value (vdbFrameBlackbodyEmission: the soot exponent)
  const hook = frameVolume.emissionColor as unknown as { sootExponent?: { value: number } } | undefined;

  values.push(hook?.sootExponent ? hook.sootExponent.value : "-");

  return values.join(":");
};

/** One registration of the light node class (internal 4), checked at the first construction. */
let registered = false;

export class VDBFrameEmissionLight extends THREE.Light {
  readonly isVDBFrameEmissionLight = true as const;

  resolution: number;
  range: number;
  cascades: number;
  distance: number;
  steps: number;

  readonly cascadeMin = Array.from({ length: maxCascades }, () => new THREE.Vector3());
  readonly cascadeSize = Array.from({ length: maxCascades }, () => new THREE.Vector3(1.0, 1.0, 1.0));
  cascadeCount = 0;
  probeResolution = 2;
  readonly probeCenter = new THREE.Vector3();

  emitters: Emitter[] = [];
  emitterCount = 0;
  slotKeys: string[] = [];
  slotCount = 0;
  probeKey = "";

  // NOTE Float targets for the volumes: intensities in world units do not overflow
  directionTarget = createTarget(2, THREE.FloatType, THREE.NearestFilter);
  emitterTarget = createTarget(2, THREE.FloatType, THREE.NearestFilter);
  // NOTE The probe atlas has the size for the largest grid and all cascades: lit materials keep its binding
  readonly probeTarget = createTarget(3, THREE.HalfFloatType, THREE.LinearFilter, atlasSize, atlasSize);

  // NOTE One draw for each changed volume, in one render
  readonly directionScene = new THREE.Scene();
  directionPool: THREE.Mesh[] = [];
  readonly reduceMaterial = new VDBFrameEmissionReduceMaterial();
  readonly probeMaterial = new VDBFrameEmissionProbeMaterial();
  readonly reduceScene: THREE.Scene;
  readonly probeScene: THREE.Scene;
  // NOTE WebGPURenderer calls updateProjectionMatrix() on the camera
  readonly passCamera = new THREE.OrthographicCamera();
  // NOTE RendererUtils fills this object
  rendererState: object = {};
  /** Volumes drawn in the last probe update and whether the probes were drawn (not his; tests and timing). */
  readonly lastUpdate = { directions: 0, probes: false };

  constructor(props: VDBFrameEmissionLightProps = {}) {
    const { color = 0xffffff, intensity = 1.0, resolution = 16, range = 3.0, cascades = 1, distance = 512, steps = 24 } = props;

    super(color, intensity);

    if (!registered) {
      vdbFrameRegisterLightNode(VDBFrameEmissionLight as unknown as typeof THREE.Light, VDBFrameEmissionLightNode);
      registered = true;
    }

    (this as { type: string }).type = "VDBFrameEmissionLight";
    this.resolution = resolution;
    this.range = range;
    this.cascades = cascades;
    this.distance = distance;
    this.steps = steps;

    this.directionScene.matrixWorldAutoUpdate = false;

    const createPassScene = (material: THREE.Material) => {
      const scene = new THREE.Scene();
      const mesh = new THREE.Mesh(createGeometry(fullscreenPositions), material);

      mesh.frustumCulled = false;
      scene.matrixWorldAutoUpdate = false;
      scene.add(mesh);

      return scene;
    };

    this.reduceScene = createPassScene(this.reduceMaterial);
    this.probeScene = createPassScene(this.probeMaterial);
  }

  getSteps(): number {
    return Math.max(1, Math.round(this.steps));
  }

  /** The direction-pass material of a volume (its source compiled in), made on first use. */
  getDirectionMaterial(frameVolume: VDBFrameVolume): VDBFrameEmissionDirectionMaterial {
    const mipmaps = frameVolume.mipmaps && frameVolume.source.lodLevels > 0;
    let material = frameVolume.emissionDirectionMaterial;

    if (!material) {
      material = new VDBFrameEmissionDirectionMaterial({ source: frameVolume.source, sourceUniforms: frameVolume.sourceUniforms, mipmaps, ...(frameVolume.emissionColor ? { emissionColor: frameVolume.emissionColor } : {}) });
      frameVolume.emissionDirectionMaterial = material;
    }

    material.setMipmaps(mipmaps);

    return material;
  }

  // NOTE Emissive volumes of the scene (visible, with an emission channel)
  collectEmitters(scene: THREE.Object3D): void {
    let count = 0;
    const steps = this.getSteps();

    scene.traverseVisible((object) => {
      const frameVolume = object as VDBFrameVolume;

      if (!frameVolume.isVDBFrameVolume || !frameVolume.source.channels.emission) {
        return;
      }

      for (const volume of frameVolume.volumes) {
        if (!volume.ready) {
          continue;
        }

        if (!this.emitters[count]) {
          this.emitters[count] = createEmitter();
        }

        const emitter = this.emitters[count++]!;

        emitter.frameVolume = frameVolume;
        emitter.matrixWorld.multiplyMatrices(frameVolume.matrixWorld, volume.matrix);
        emitter.worldToLocal.copy(emitter.matrixWorld).invert();
        emitter.worldBox.copy(unitBox).applyMatrix4(emitter.matrixWorld);

        const bounds = vdbFrameEmitterBounds(emitter.matrixWorld.elements);

        emitter.center.fromArray(bounds.center);
        emitter.radius = bounds.radius;
        emitter.key = [
          sourceKey(frameVolume),
          frameVolume.extinction,
          frameVolume.getDensityScale(),
          frameVolume.densityCutoff,
          frameVolume.absorbance,
          frameVolume.emissiveIntensity,
          frameVolume.temperatureRange.x,
          frameVolume.temperatureRange.y,
          frameVolume.offset3D.x,
          frameVolume.offset3D.y,
          frameVolume.offset3D.z,
          frameVolume.wrap3D,
          frameVolume.mipmaps,
          steps,
          ...emitter.matrixWorld.elements,
        ].join(",");
      }
    });

    this.emitterCount = count;
  }

  // NOTE The probe cascades (tools/frame/emission.ts). Returns a key that changes with them.
  updateProbeGrid(camera: THREE.Camera): string {
    cameraPosition.setFromMatrixPosition(camera.matrixWorld);

    const boxes = this.emitters.slice(0, this.emitterCount).map((e) => ({ min: e.worldBox.min.toArray() as Vec3, max: e.worldBox.max.toArray() as Vec3 }));
    const grid = vdbFrameEmissionProbeGrid({ resolution: this.resolution, range: this.range, cascades: this.cascades, distance: this.distance }, boxes, cameraPosition.toArray() as Vec3);

    this.probeResolution = grid.resolution;
    this.probeCenter.fromArray(grid.center);

    for (let i = 0; i < grid.count; i++) {
      this.cascadeMin[i]!.fromArray(grid.min[i]!);
      this.cascadeSize[i]!.fromArray(grid.size[i]!);
    }

    this.cascadeCount = grid.count;

    return grid.key;
  }

  // NOTE The changed volumes as { emitter, slot }, or null when the probes stay the same.
  // Slots grow to the next power of two. New targets are empty, so all volumes draw again.
  prepareProbes(scene: THREE.Object3D, camera: THREE.Camera): { emitter: Emitter; slot: number }[] | null {
    this.collectEmitters(scene);

    if (this.emitterCount > this.slotCount) {
      this.slotCount = Math.pow(2, Math.ceil(Math.log2(this.emitterCount)));
      this.slotKeys = [];
      this.resizeSlots();
    }

    const changed: { emitter: Emitter; slot: number }[] = [];

    for (let i = 0; i < this.emitterCount; i++) {
      const emitter = this.emitters[i]!;

      if (this.slotKeys[i] !== emitter.key) {
        this.slotKeys[i] = emitter.key;
        changed.push({ emitter, slot: i });
      }
    }

    if (this.emitterCount === 0) {
      this.probeKey = "";

      return null;
    }

    const probeKey = `${this.emitterCount},${this.updateProbeGrid(camera)}`;

    if (changed.length === 0 && probeKey === this.probeKey) {
      return null;
    }

    this.probeKey = probeKey;

    return changed;
  }

  // NOTE New targets, because setSize() destroys the GPU textures that materials still bind
  resizeSlots(): void {
    this.directionTarget.dispose();
    this.emitterTarget.dispose();
    this.directionTarget = createTarget(2, THREE.FloatType, THREE.NearestFilter, directionCount, this.slotCount);
    this.emitterTarget = createTarget(2, THREE.FloatType, THREE.NearestFilter, 1, this.slotCount);
  }

  getDirectionMesh(index: number): THREE.Mesh {
    if (!this.directionPool[index]) {
      const mesh = new THREE.Mesh(createGeometry(quadPositions), new THREE.MeshBasicMaterial());

      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.onBeforeRender = () => {
        const { emitter, slot } = mesh.userData as { emitter: Emitter; slot: number };
        const material = mesh.material as unknown as VDBFrameEmissionDirectionMaterial;

        setVolumeUniforms(material, emitter.frameVolume!, emitter);
        material.frameSlot.value = slot;
        material.frameTargetSize.value.set(directionCount, this.slotCount);
        material.frameEmitterCenter.value.copy(emitter.center);
        material.frameEmitterRadius.value = emitter.radius;
        material.frameSteps.value = this.getSteps();
      };

      this.directionScene.add(mesh);
      this.directionPool[index] = mesh;
    }

    return this.directionPool[index]!;
  }

  /** Runs the passes of changed volumes. The light node calls it before the lit meshes of each render. */
  updateProbes(renderer: RendererLike, scene: THREE.Object3D, viewCamera: THREE.Camera): void {
    this.lastUpdate.directions = 0;
    this.lastUpdate.probes = false;

    const changed = this.prepareProbes(scene, viewCamera);

    if (!changed) {
      return;
    }

    changed.forEach(({ emitter, slot }, index) => {
      const mesh = this.getDirectionMesh(index);

      mesh.material = this.getDirectionMaterial(emitter.frameVolume!);
      mesh.userData.emitter = emitter;
      mesh.userData.slot = slot;
    });

    const { reduceMaterial, probeMaterial, reduceScene, probeScene, passCamera: camera } = this;
    const utils = vdbFrameRendererUtils();

    this.rendererState = utils.resetRendererState(renderer, this.rendererState);

    // NOTE Each pass writes all its texels. Rows of volumes that did not change stay.
    renderer.autoClear = false;

    try {
      if (changed.length > 0) {
        this.directionPool.forEach((mesh, index) => {
          mesh.visible = index < changed.length;
        });

        renderer.setRenderTarget(this.directionTarget);
        renderer.render(this.directionScene, camera);

        reduceMaterial.frameTargetSize.value.set(1, this.slotCount);
        reduceMaterial.frameDirectionSize.value.set(directionCount, this.slotCount);
        reduceMaterial.frameDirectionIntensity.value = this.directionTarget.textures[0];
        reduceMaterial.frameDirectionCenter.value = this.directionTarget.textures[1];

        renderer.setRenderTarget(this.emitterTarget);
        renderer.render(reduceScene, camera);
      }

      setProbeGridUniforms(probeMaterial.gridUniforms, this);
      probeMaterial.frameEmitterCount.value = this.emitterCount;
      probeMaterial.frameEmitterSize.value.set(1, this.slotCount);
      probeMaterial.frameEmitterIntensity.value = this.emitterTarget.textures[0];
      probeMaterial.frameEmitterCenter.value = this.emitterTarget.textures[1];

      renderer.setRenderTarget(this.probeTarget);
      renderer.render(probeScene, camera);
      this.lastUpdate.directions = changed.length;
      this.lastUpdate.probes = true;
    } finally {
      utils.restoreRendererState(renderer, this.rendererState);
    }
  }

  override dispose(): void {
    super.dispose();

    this.slotKeys = [];
    this.slotCount = 0;
    this.probeKey = "";
    this.directionTarget.dispose();
    this.emitterTarget.dispose();
    this.probeTarget.dispose();
    this.directionPool.forEach((mesh) => mesh.geometry.dispose());
    this.directionPool = [];
    this.reduceMaterial.dispose();
    this.probeMaterial.dispose();

    for (const scene of [this.reduceScene, this.probeScene]) {
      (scene.children[0] as THREE.Mesh | undefined)?.geometry.dispose();
    }
  }
}
