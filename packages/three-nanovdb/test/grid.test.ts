import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { NanoVDBFile } from "nanovdb-wgsl";
import { NanoVDBGrid } from "../src/grid.js";

/**
 * `NanoVDBGrid` unit tests against a real baked fixture. `.nvdb` files are
 * git-ignored (regenerate with `pnpm fixtures:bake`), so this suite is
 * skipped wholesale when the fixtures directory is absent — same pattern as
 * `packages/nanovdb-wgsl/test/nvdb-file.test.ts`.
 */

const fixturesDir = new URL("../../../fixtures/primitives/", import.meta.url);
const fixturePath = new URL("box_fog_float.nvdb", fixturesDir);
const fixturesPresent = existsSync(fixturesDir) && existsSync(fixturePath);

async function readArrayBuffer(url: URL): Promise<ArrayBuffer> {
  const buf = await readFile(url);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

async function loadGrid(): Promise<NanoVDBGrid> {
  const buffer = await readArrayBuffer(fixturePath);
  const file = NanoVDBFile.fromArrayBuffer(buffer);
  return NanoVDBGrid.fromFile(file, 0);
}

describe.skipIf(!fixturesPresent)("NanoVDBGrid (box_fog_float fixture)", () => {
  it("passes through metadata from the parsed file", async () => {
    const grid = await loadGrid();
    expect(grid.metadata.name).toBe("box_fog");
    expect(grid.metadata.gridType).toBe("Float");
    expect(grid.metadata.gridClass).toBe("FogVolume");
    expect(grid.byteLength).toBe(grid.image.byteLength);
    expect(grid.byteLength).toBe(grid.metadata.gridByteSize);
  });

  it("maps gridType to the PNanoVDB GridType id", async () => {
    const grid = await loadGrid();
    expect(grid.gridTypeId).toBe(1); // Float
  });

  it("throws for an unsupported grid type", () => {
    const bogusMetadata = {
      name: "bogus",
      gridType: "Double",
      gridClass: "Unknown",
      worldBBox: { min: [0, 0, 0], max: [1, 1, 1] },
      indexBBox: { min: [0, 0, 0], max: [1, 1, 1] },
      voxelSize: [1, 1, 1],
      voxelCount: 0,
      gridByteSize: 0,
    } as const;
    expect(() => new NanoVDBGrid({ image: new Uint32Array(0), metadata: bogusMetadata })).toThrow(/unsupported/i);
  });

  it("worldBBox()/indexBBox() match the sidecar ground truth", async () => {
    const grid = await loadGrid();
    // From fixtures/primitives/box_fog_float.sidecar.json
    const world = grid.worldBBox();
    expect([world.min.x, world.min.y, world.min.z]).toEqual([-40, -20, -30]);
    expect([world.max.x, world.max.y, world.max.z]).toEqual([41, 21, 31]);

    const index = grid.indexBBox();
    expect([index.min.x, index.min.y, index.min.z]).toEqual([-40, -20, -30]);
    expect([index.max.x, index.max.y, index.max.z]).toEqual([40, 20, 30]);
  });

  it("indexToWorld() composed with worldToIndex() is (approximately) the identity", async () => {
    const grid = await loadGrid();
    const roundTrip = grid.worldToIndex().clone().multiply(grid.indexToWorld());
    const identity = new THREE.Matrix4();
    for (let i = 0; i < 16; i++) {
      expect(roundTrip.elements[i]).toBeCloseTo(identity.elements[i]!, 10);
    }
  });

  it("indexToWorld() maps indexBBox.min onto worldBBox.min exactly", async () => {
    const grid = await loadGrid();
    const m = grid.indexToWorld();
    const idxBox = grid.indexBBox();
    const worldBox = grid.worldBBox();

    const gotMin = idxBox.min.clone().applyMatrix4(m);
    expect(gotMin.toArray()).toEqual(worldBox.min.toArray());
  });

  it("indexToWorld() maps (indexBBox.max + voxelSize) onto worldBBox.max — NanoVDB's " +
    "worldBBox spans the far face of the last voxel, one voxel-width beyond indexBBox.max", async () => {
    const grid = await loadGrid();
    const m = grid.indexToWorld();
    const idxBox = grid.indexBBox();
    const worldBox = grid.worldBBox();
    const [sx, sy, sz] = grid.metadata.voxelSize;

    const farCorner = idxBox.max.clone().add(new THREE.Vector3(sx, sy, sz));
    const gotMax = farCorner.applyMatrix4(m);
    expect(gotMax.toArray()).toEqual(worldBox.max.toArray());
  });

  it("proxyGeometry() dimensions match the world bbox size", async () => {
    const grid = await loadGrid();
    const worldBox = grid.worldBBox();
    const size = worldBox.getSize(new THREE.Vector3());
    const geometry = grid.proxyGeometry();

    geometry.computeBoundingBox();
    const geomBox = geometry.boundingBox!;
    const geomSize = geomBox.getSize(new THREE.Vector3());

    expect(geomSize.x).toBeCloseTo(size.x, 6);
    expect(geomSize.y).toBeCloseTo(size.y, 6);
    expect(geomSize.z).toBeCloseTo(size.z, 6);

    // Geometry is translated so it sits exactly on the world bbox (not just
    // centered at the origin) — SPEC §3.1's `scene.add(new Mesh(grid.proxyGeometry(), ...))`
    // usage has no separate position transform.
    expect(geomBox.min.toArray()).toEqual(worldBox.min.toArray());
    expect(geomBox.max.toArray()).toEqual(worldBox.max.toArray());
  });

  it("storageAttribute is created lazily and cached (same object on repeat access)", async () => {
    const grid = await loadGrid();
    const a = grid.storageAttribute;
    const b = grid.storageAttribute;
    expect(a).toBe(b);
    expect(a.itemSize).toBe(1);
    expect(a.count).toBe(grid.image.length);
    expect(a.array).toBe(grid.image);
  });
});
