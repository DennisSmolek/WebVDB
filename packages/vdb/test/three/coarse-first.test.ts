/**
 * K4.16: `vdbStreamCoarseFirst` streams a coarse-first `.nvdb` into one pool id:
 * `add(preview, { capacity: full size })`, then `replace(id, full)`. A stub pool
 * records the calls; the GPU path is the e2e (e2e/coarse-first.spec.ts).
 */
import { describe, expect, it } from "vitest";
import { BoxGeometry } from "three";
import { vdbStreamCoarseFirst, type VDBCoarseFirstEvent } from "../../src/three/index.js";
import { buildFromDense, NvdbFormatError, writeNvdb } from "../../src/tools/index.js";

function ball(dims: number, name: string, voxelSize: number): Uint32Array {
  const v = new Float32Array(dims ** 3);
  const c = (dims - 1) / 2;
  for (let x = 0; x < dims; x++)
    for (let y = 0; y < dims; y++)
      for (let z = 0; z < dims; z++) v[(x * dims + y) * dims + z] = Math.max(0, 1 - Math.hypot(x - c, y - c, z - c) / (dims / 2));
  return buildFromDense(v, [dims, dims, dims], { gridName: name, voxelSize });
}

const LOD8 = ball(8, "density.lod8", 4);
const LOD16 = ball(16, "density.lod16", 2);
const FULL = ball(32, "density", 1);
const OTHER = ball(8, "temperature", 4);

type Call = { op: "add"; image: Uint32Array; capacity?: number; label?: string } | { op: "replace"; id: number; image: Uint32Array };

function stubPool() {
  const calls: Call[] = [];
  return {
    calls,
    add(image: Uint32Array, options: { capacity?: number; label?: string } = {}) {
      calls.push({ op: "add", image, ...options });
      return 7;
    },
    replace(id: number, image: Uint32Array) {
      calls.push({ op: "replace", id, image });
    },
  };
}

/** A pull-driven stream of `step`-byte chunks that records how far it was read and whether it was cancelled. */
function chunked(bytes: Uint8Array, step: number) {
  const state = { pulled: 0, cancelled: false };
  const stream = new ReadableStream<Uint8Array>(
    {
      pull(c) {
        if (state.pulled >= bytes.length) return c.close();
        c.enqueue(bytes.slice(state.pulled, state.pulled + step));
        state.pulled += step;
      },
      cancel() {
        state.cancelled = true;
      },
    },
    { highWaterMark: 0 },
  );
  return { stream, state };
}

describe("vdbStreamCoarseFirst (K4.16)", () => {
  it("adds the coarsest grid with the family's largest size reserved, then replaces finer ones in place", async () => {
    const file = new Uint8Array(writeNvdb([LOD8, LOD16, FULL], { codec: "zip" }));
    const pool = stubPool();
    const events: VDBCoarseFirstEvent[] = [];
    const { stream } = chunked(file, 257);
    const result = await vdbStreamCoarseFirst(pool, stream, { onGrid: (e) => events.push({ ...e }) });

    expect(result.id).toBe(7);
    expect(pool.calls.map((c) => c.op)).toEqual(["add", "replace", "replace"]);
    const add = pool.calls[0] as Extract<Call, { op: "add" }>;
    expect(add.image).toEqual(LOD8);
    expect(add.capacity).toBe(FULL.byteLength);
    expect(add.label).toBe("density");
    expect((pool.calls[2] as Extract<Call, { op: "replace" }>).image).toEqual(FULL);
    expect(pool.calls.slice(1).every((c) => c.op === "replace" && c.id === 7)).toBe(true);

    expect(events.map((e) => [e.action, e.grid.lod, e.final, e.voxelRatio])).toEqual([
      ["add", 8, false, 4],
      ["replace", 16, false, 2],
      ["replace", null, true, 1],
    ]);
    expect(result.firstBytes).toBe(events[0]!.bytesReceived);
    expect(result.fullBytes).toBe(file.length);
    expect(result.firstBytes).toBeLessThan(result.fullBytes);
  });

  it("selects a family by name, skips other grids, ignores a coarser grid after a finer one, and stops reading at the full grid", async () => {
    const file = new Uint8Array(writeNvdb([OTHER, LOD16, FULL, LOD8, OTHER], { codec: "none" }));
    const pool = stubPool();
    const { stream, state } = chunked(file, 4096);
    const res = await vdbStreamCoarseFirst(pool, stream, { gridName: "density", label: "smoke" });
    expect(pool.calls.map((c) => [c.op, c.image.byteLength])).toEqual([
      ["add", LOD16.byteLength],
      ["replace", FULL.byteLength],
    ]);
    expect((pool.calls[0] as Extract<Call, { op: "add" }>).label).toBe("smoke");
    expect(res.shown.length).toBe(2);
    expect(state.cancelled).toBe(true);
    expect(state.pulled).toBeLessThan(file.length);
  });

  it("accepts a Response and an async iterable", async () => {
    const file = new Uint8Array(writeNvdb([LOD8, FULL], { codec: "zip" }));
    const a = stubPool();
    await vdbStreamCoarseFirst(a, new Response(file));
    expect(a.calls.map((c) => c.op)).toEqual(["add", "replace"]);
    const b = stubPool();
    async function* gen() {
      for (let i = 0; i < file.length; i += 1000) yield file.subarray(i, i + 1000);
    }
    await vdbStreamCoarseFirst(b, gen());
    expect(b.calls.map((c) => c.op)).toEqual(["add", "replace"]);
    await expect(vdbStreamCoarseFirst(stubPool(), new Response("nope", { status: 404 }))).rejects.toThrow(/404/);
  });

  it("rejects a truncated stream (after the preview was shown) and a stream without the full grid", async () => {
    const file = new Uint8Array(writeNvdb([LOD8, FULL], { codec: "zip" }));
    const pool = stubPool();
    await expect(vdbStreamCoarseFirst(pool, new Response(file.slice(0, file.length - 10)))).rejects.toThrow(NvdbFormatError);
    expect(pool.calls.map((c) => c.op)).toEqual(["add"]);

    const previewOnly = new Uint8Array(writeNvdb([LOD8]));
    await expect(vdbStreamCoarseFirst(stubPool(), new Response(previewOnly))).rejects.toThrow(/without the full grid "density"/);
  });

  it("honours an AbortSignal", async () => {
    const file = new Uint8Array(writeNvdb([LOD8, FULL]));
    const controller = new AbortController();
    controller.abort(new Error("stop"));
    await expect(vdbStreamCoarseFirst(stubPool(), new Response(file), { signal: controller.signal })).rejects.toThrow(/stop/);
  });

  describe("bind (K5.4c)", () => {
    function bindPool() {
      const log: string[] = [];
      const pool = {
        ...stubPool(),
        flush: () => log.push("flush"),
        info: (id: number) => {
          log.push(`info ${id}`);
          return { indexBBox: { min: [0, 0, 0], max: [7, 7, 7] }, indexToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] } as never;
        },
        setMajorant: (id: number, m: unknown) => log.push(`majorant ${id} ${m === null ? "null" : "set"}`),
      };
      return { pool, log };
    }

    it("scales the material by base x voxelRatio, rebuilds the proxy, updates the node and re-applies the majorant", async () => {
      const file = new Uint8Array(writeNvdb([LOD8, LOD16, FULL], { codec: "zip" }));
      const { pool, log } = bindPool();
      const material = { densityScale: { value: 5 } };
      const first = new BoxGeometry(1, 1, 1);
      const mesh = { geometry: first };
      const scales: number[] = [];
      const geoms: unknown[] = [];
      let disposed = 0;
      first.dispose = () => void disposed++;
      const majorants: boolean[] = [];
      await vdbStreamCoarseFirst(pool, new Response(file), {
        bind: {
          material,
          mesh,
          grids: { update: () => log.push("update") },
          majorant: (e) => {
            majorants.push(e.final);
            return e.final ? ({ brickLog2: 3 } as never) : null;
          },
        },
        onGrid: () => {
          scales.push(material.densityScale.value);
          geoms.push(mesh.geometry);
        },
      });
      expect(scales).toEqual([20, 10, 5]); // base 5 captured at the first swap, x 4, x 2, x 1
      expect(majorants).toEqual([false, false, true]);
      expect(log.filter((l) => l.startsWith("majorant"))).toEqual(["majorant 7 set"]);
      expect(log.filter((l) => l === "update").length).toBe(3);
      // bind runs before onGrid: flush, then update, in order, per swap.
      expect(log.indexOf("flush")).toBeLessThan(log.indexOf("update"));
      expect(new Set(geoms).size).toBe(3);
      expect(geoms[0]).not.toBe(first);
      expect(disposed).toBe(1); // the geometry the caller supplied is disposed on the first swap
    });

    it("an explicit densityScale wins, members may be filled in later, and a pool without the calls is refused", async () => {
      const file = new Uint8Array(writeNvdb([LOD8, FULL], { codec: "none" }));
      const { pool } = bindPool();
      const material = { densityScale: { value: 1 } };
      const bind: { material?: typeof material; densityScale?: number } = { densityScale: 3 };
      const seen: number[] = [];
      await vdbStreamCoarseFirst(pool, new Response(file), {
        bind,
        onGrid: (e) => {
          bind.material ??= material; // assigned on the first event; applied from the next swap
          seen.push(material.densityScale.value);
          void e;
        },
      });
      expect(seen).toEqual([1, 3]);
      await expect(vdbStreamCoarseFirst(stubPool(), new Response(file), { bind: { grids: { update() {} } } })).rejects.toThrow(/flush\(\)/);
    });
  });
});
