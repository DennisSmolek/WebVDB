/**
 * `VDBPoints`: the points of a parsed point data grid as round, instanced
 * sprites (K3.7, R1 §6 Step 7), three r187 `PointsNodeMaterial`.
 *
 *   const grid = parseVdb(bytes).grids.find((g) => g.valueType === 'ptdataidx32');
 *   const points = new VDBPoints(vdbPointRenderData(grid), { color: 0xffffff });
 *   scene.add(points);
 *
 * Portions derived from mjurczyk/openvdb (MIT), commit 75e7345
 * (`src/three-openvdb/points/webgpu/VDBPoints.js`, `points/pointsCore.js`):
 * a `Sprite` with one instance per point (WebGPU point primitives are 1 pixel),
 * the round alpha-tested disc, the attenuated size in pixels with a 1-pixel
 * minimum, `Cd` colours times the material colour, the default diameter of half
 * the mean spacing. Copyright (c) mjurczyk/openvdb contributors, MIT (see
 * THIRD_PARTY_NOTICES.md). Changes: takes the engine-free `vdbPointRenderData` (tools)
 * instead of decoding the grid itself; the pixel scale reads the render
 * target's height when one is set (DIVERGES.md).
 *
 * Positions are world space of the grid (the point list's own transform);
 * place them with the object's transform as any three object. The sprite sits
 * at the origin with the points in local space, so frustum culling is off.
 */

import * as THREE from "three";
import { InstancedBufferAttribute, PointsNodeMaterial, Sprite } from "three/webgpu";
import * as TSL from "three/tsl";
import { VDBError } from "../core/errors.js";
import type { VDBTslNode } from "./grid-node.js";

const { instancedBufferAttribute, materialColor, max, modelViewMatrix, uniform, vec4 } = TSL as unknown as Record<string, VDBTslNode>;

/** What `VDBPoints` draws (tools `vdbPointRenderData` returns one). */
export interface VDBPointsData {
  /** World positions of the grid, xyz per point. */
  positions: Float32Array;
  /** Linear RGB per point (multiplied by the material colour), or null. */
  colors?: Float32Array | null;
  /** Points to draw (default `positions.length / 3`). */
  count?: number;
  /** Mean point spacing in world units: the default diameter is half of it. Default 1. */
  spacing?: number;
}

/** Options of {@link VDBPoints}: `PointsNodeMaterial` parameters plus the point diameter. */
export interface VDBPointsOptions extends THREE.SpriteMaterialParameters {
  /** World diameter of a point. Default `spacing * 0.5` (a gap between neighbours). */
  size?: number | null;
}

type ScalarUniform = ReturnType<typeof TSL.uniform> & { value: number };

let discTexture: THREE.DataTexture | null = null;

/** The 64x64 round alpha map every `VDBPoints` shares (alpha-tested at 0.5). */
export function vdbPointDiscTexture(): THREE.DataTexture {
  if (!discTexture) {
    const size = 64;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = (x + 0.5) / size - 0.5;
        const dy = (y + 0.5) / size - 0.5;
        const v = Math.sqrt(dx * dx + dy * dy) <= 0.5 ? 255 : 0;
        data.set([v, v, v, 255], (y * size + x) * 4);
      }
    }
    discTexture = new THREE.DataTexture(data, size, size);
    discTexture.needsUpdate = true;
  }
  return discTexture;
}

const worldPosition = new THREE.Vector3();
const worldQuaternion = new THREE.Quaternion();
const worldScale = new THREE.Vector3();
const canvasSize = new THREE.Vector2();

/**
 * Sprite size (pixels at view depth 1, before the half-height factor) for a world diameter: his
 * `getAttenuatedPointSize`, `diameter * maxScale * zoom / tan(fov / 2)` (fov 50 for non-perspective cameras).
 */
export function vdbAttenuatedPointSize(object: THREE.Object3D, camera: THREE.Camera, diameter: number): number {
  const perspective = camera as THREE.PerspectiveCamera;
  const fov = perspective.isPerspectiveCamera ? perspective.fov : 50;
  const zoom = (camera as { zoom?: number }).zoom ?? 1;
  object.matrixWorld.decompose(worldPosition, worldQuaternion, worldScale);
  return (diameter * Math.max(worldScale.x, worldScale.y, worldScale.z) * zoom) / Math.tan(THREE.MathUtils.degToRad(fov) * 0.5);
}

interface RendererSizeLike {
  getSize(target: THREE.Vector2): THREE.Vector2;
  getRenderTarget?(): { height: number } | null;
}

/**
 * Round points of a point list as one instanced sprite (see the file comment).
 * `count` instances; the material is a `PointsNodeMaterial` (`color`, `opacity`
 * and the other sprite parameters apply; `Cd` multiplies the colour).
 */
export class VDBPoints extends Sprite {
  /** Type flag: true. */
  readonly isVDBPoints = true;
  declare material: PointsNodeMaterial;
  /** Points drawn. */
  pointCount: number;
  /** Mean spacing of the points (world units). */
  spacing: number;
  /** World diameter of a point, or null for `spacing * 0.5`. */
  size: number | null;
  /** Point diameter in pixels at view depth 1 (set before each render). */
  readonly pixelScale: ScalarUniform;
  /** The instanced position attribute (world xyz per point). */
  readonly positionAttribute: InstancedBufferAttribute;
  /** The instanced colour attribute (linear RGB per point), or null without colours. */
  readonly colorAttribute: InstancedBufferAttribute | null;

  /**
   * Builds the sprite from render data (tools `vdbPointRenderData`).
   *
   * @param data World positions (xyz triples), optional colours, count and spacing.
   * @param options Sprite material parameters plus `size`, the world diameter of a point.
   */
  constructor(data: VDBPointsData, options: VDBPointsOptions = {}) {
    if (!data || !(data.positions instanceof Float32Array) || data.positions.length % 3 !== 0) {
      throw new VDBError("VDB_BAD_OPTION", "VDBPoints: data.positions must be a Float32Array of xyz triples (tools vdbPointRenderData).");
    }
    const count = data.count ?? data.positions.length / 3;
    if (!Number.isInteger(count) || count < 0 || count * 3 > data.positions.length) {
      throw new VDBError("VDB_BAD_OPTION", `VDBPoints: count ${count} does not fit ${data.positions.length / 3} positions.`);
    }
    if (data.colors && data.colors.length < count * 3) {
      throw new VDBError("VDB_BAD_OPTION", "VDBPoints: colors must hold count * 3 values.");
    }
    const { size = null, ...materialProps } = options;
    super(
      new PointsNodeMaterial({
        alphaMap: vdbPointDiscTexture(),
        alphaTest: 0.5,
        ...materialProps,
        // sizeNode attenuates (with a 1-pixel minimum): sprites under a pixel are not drawn.
        sizeAttenuation: false,
      } as never) as never,
    );
    this.pointCount = count;
    this.spacing = data.spacing ?? 1;
    this.size = size;
    this.pixelScale = uniform(1) as ScalarUniform;
    this.positionAttribute = new InstancedBufferAttribute(data.positions, 3);
    this.colorAttribute = data.colors ? new InstancedBufferAttribute(data.colors, 3) : null;
    const position = instancedBufferAttribute(this.positionAttribute);
    const viewDepth = modelViewMatrix.mul(vec4(position, 1.0)).z.negate();
    const m = this.material as PointsNodeMaterial & Record<string, VDBTslNode>;
    m.positionNode = position;
    m.sizeNode = max((this.pixelScale as VDBTslNode).div(viewDepth), 1.0);
    m.colorNode = this.colorAttribute ? vec4(instancedBufferAttribute(this.colorAttribute), 1.0).mul(materialColor) : null;
    m.needsUpdate = true;
    (this as unknown as { count: number }).count = count;
    this.frustumCulled = false;
    this.onBeforeRender = (renderer, _scene, camera) => this.updatePixelScale(renderer as unknown as RendererSizeLike, camera);
  }

  /** The world diameter in use: `size`, or half the spacing. */
  get diameter(): number {
    return this.size ?? this.spacing * 0.5;
  }

  /**
   * Sets {@link pixelScale} for a camera and the renderer's current target (called before each render).
   *
   * @param renderer The renderer whose current target size gives the pixel height.
   * @param camera The camera whose field of view (or zoom) sets the scale.
   */
  updatePixelScale(renderer: RendererSizeLike, camera: THREE.Camera): void {
    const target = renderer.getRenderTarget?.() ?? null;
    const height = target ? target.height : renderer.getSize(canvasSize).y;
    this.pixelScale.value = vdbAttenuatedPointSize(this, camera, this.diameter) * height * 0.5;
  }

  /** Frees the material (the shared disc texture stays). */
  dispose(): void {
    this.material.dispose();
  }
}
