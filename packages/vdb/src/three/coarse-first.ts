/**
 * `vdbStreamCoarseFirst` (K4.16): stream a coarse-first `.nvdb`
 * (`vdbBakeCoarseFirst`, docs/dev/design/K4-5-coarse-first.md option a) into one
 * pool id. The preview is added as soon as its bytes are in, with the full
 * grid's size reserved, and each finer grid replaces it in place:
 *
 *     const res = await fetch(url);
 *     const { id } = await vdbStreamCoarseFirst(pool, res, {
 *       onGrid: (e) => { if (e.final) material.densityScale.value = base; },
 *     });
 *
 *   1. `id = pool.add(preview, { capacity })`, where `capacity` is the largest
 *      grid of the family announced so far (a segment lists every grid's
 *      FileMetaData before any data, so the full grid's size is known);
 *   2. `pool.replace(id, finer)` for each finer grid, the full grid last.
 *
 * The id, its bind-group slot and any instance transform stay. Each image's
 * own Map places it, and a preview covers the full grid's world box, so the
 * world placement is unchanged. Things the caller re-derives after a swap
 * (the `onGrid` callback is the place, or the opt-in `bind` option, which does these
 * three steps for you after every swap):
 * - `replace` drops a brick majorant: call `pool.setMajorant` again;
 * - `VDBVolumeMaterial` marches in index voxels, so a preview with k-times
 *   larger voxels has 1/k the optical depth at the same `densityScale`;
 *   scale it by `voxelRatio` while a preview shows, and back to 1 at `final`;
 * - the proxy mesh: a new `vdbProxyGeometry(pool.info(id))` (the preview's
 *   box is the full box grown to whole preview voxels).
 *
 * Into the froxel renderer (G16a): at the first event make the `VDBGridNode`
 * with `types: event.types` (every grid type of the family) and the sparse
 * source with `followImage: true`; the source re-reads its boxes after each
 * swap and the `VDBFrameVolume` follows, with no new node, source, volume or
 * shader. Froxels integrate in world space: no `voxelRatio` scale there.
 *
 * Without `bind` it touches only `pool.add` / `pool.replace`, so it works with
 * any `VDBGridPool` host. It lives in `./three` because `tools` may not import
 * the core (D20).
 */

import type * as THREE from "three";

import { VDBError } from "../core/errors.js";
import type { VDBBrickMajorant } from "../core/index.js";
import { VDB_GRID_TYPE_NAMES, type VDBGridTypeName } from "../core/types.js";
import { vdbProxyGeometry, type VDBProxyGridInfo } from "./material.js";
import {
  NvdbStreamReader,
  nvdbResponseLength,
  parseNvdbLodName,
  type NvdbGridMetadata,
  type NvdbProgress,
  type NvdbStreamGrid,
  type NvdbStreamReaderOptions,
} from "../tools/index.js";

/** The part of a `VDBGridPool` the loader uses. */
export interface VDBCoarseFirstPoolLike {
  add(image: Uint32Array, options?: { capacity?: number; label?: string }): number;
  replace(id: number, image: Uint32Array): void;
}

/** The extra pool calls `bind` makes (a `VDBGridPool` has them all). */
export interface VDBCoarseFirstBindPoolLike {
  flush(): void;
  info(id: number): VDBProxyGridInfo;
  setMajorant(id: number, majorant: VDBBrickMajorant | null): void;
}

/**
 * What to keep in step with the pool id after each swap (K5.4c). Every member is optional; the fields are read at
 * each swap, so they may be assigned later (for example in `onGrid`, once the first grid exists).
 */
export interface VDBCoarseFirstBind {
  /**
   * A material to scale: `material.densityScale.value = base * event.voxelRatio`, so a preview with k-times larger
   * voxels keeps the full grid's optical depth and the full grid gets `base` back. `base` is `densityScale` below,
   * else the material's value the first time `bind` is applied to it.
   */
  material?: { densityScale: { value: number } };
  /** The base density scale for `material`. Default: the material's current value at the first swap. */
  densityScale?: number;
  /** A proxy mesh: its geometry is replaced (and the old one disposed) with `vdbProxyGeometry(pool.info(id))`. */
  mesh?: { geometry: THREE.BufferGeometry };
  /** `vdbProxyGeometry` padding for `mesh` (default 0). */
  proxyPad?: number;
  /** A `VDBGridNode`: `pool.flush()` then `grids.update()`, so its shaders see the swapped grid. */
  grids?: { update(): void };
  /**
   * Brick majorant for the grid just shown (`replace` drops the old one). Return `null` / `undefined` to leave the grid
   * without one, e.g. for a preview.
   */
  majorant?: (event: VDBCoarseFirstEvent) => VDBBrickMajorant | null | undefined;
}

/** What a coarse-first stream delivers. */
export type VDBCoarseFirstSource = Response | ReadableStream<Uint8Array> | AsyncIterable<Uint8Array>;

/** One grid shown in the pool. */
export interface VDBCoarseFirstEvent {
  /** The pool id (the same for every grid of the family). */
  id: number;
  /** The grid as the stream reader emitted it (`lod` is null for the full grid). */
  grid: NvdbStreamGrid;
  /** True for the full grid: the last swap. */
  final: boolean;
  /** `'add'` for the first grid shown, `'replace'` after. */
  action: "add" | "replace";
  /** File bytes received when the grid was shown (decoded bytes; see the e2e for compressed counts). */
  bytesReceived: number;
  /** Expected file size when the grid was shown, or null if unknown (K4.20; see `NvdbProgress.bytesTotal`). */
  bytesTotal: number | null;
  /** This grid's voxel size over the full grid's (x axis): 1 for the full grid, > 1 for a preview. */
  voxelRatio: number;
  /**
   * G16a: the grid types of the family's grids (the file lists every grid's metadata before any data, so this holds
   * the full grid's type from the first event on), sorted. A `VDBGridNode` made with `types: event.types` decodes the
   * preview and the full grid with one module (a baked FpN preview and an Fp8 full grid differ), so a source made
   * with `followImage` swaps without a recompile.
   */
  types: VDBGridTypeName[];
}

export interface VDBCoarseFirstOptions {
  /** Which family to load, by the full grid's name. Default: the first grid's family. */
  gridName?: string;
  /** Pool label for the id. Default: the full grid's name. */
  label?: string;
  /** Limits for the stream reader. */
  reader?: NvdbStreamReaderOptions;
  /**
   * Keeps a material, proxy mesh, grid node and majorant in step with each swap, before `onGrid` runs (see
   * {@link VDBCoarseFirstBind} and {@link vdbCoarseFirstApply}). Needs a pool with `flush` / `info` / `setMajorant`
   * (a `VDBGridPool`).
   */
  bind?: VDBCoarseFirstBind;
  /** Called after each `pool.add` / `pool.replace` (preview, then finer grids, then the full grid), after `bind`. */
  onGrid?: (event: VDBCoarseFirstEvent) => void;
  /** Called after every received chunk with the byte progress (K4.20). `bytesTotal` is the Response's Content-Length, else derived from the file's metadata. */
  onProgress?: (progress: NvdbProgress) => void;
  /** Aborts the read (rejects with the signal's reason; the id, if added, stays in the pool). */
  signal?: AbortSignal;
}

export interface VDBCoarseFirstResult {
  id: number;
  /** Every grid shown, in order (the preview(s), then the full grid). */
  shown: VDBCoarseFirstEvent[];
  /** File bytes received when the first grid was added. */
  firstBytes: number;
  /** File bytes received when the full grid replaced the preview. */
  fullBytes: number;
  /** Expected file size when the full grid arrived, or null if unknown. */
  bytesTotal: number | null;
}

/** The base density scale of each bound material (the value seen at its first swap, unless `bind.densityScale` says). */
const baseScale = new WeakMap<object, number>();

/**
 * The work `bind` does after a swap, callable on its own from an async `onGrid` (or any code that shows a
 * {@link VDBCoarseFirstEvent}): majorant, proxy geometry, `pool.flush()` + `grids.update()`, density scale. It never
 * touches what `bind` leaves out.
 */
export function vdbCoarseFirstApply(
  pool: VDBCoarseFirstPoolLike & Partial<VDBCoarseFirstBindPoolLike>,
  event: VDBCoarseFirstEvent,
  bind: VDBCoarseFirstBind,
): void {
  const need = <K extends keyof VDBCoarseFirstBindPoolLike>(name: K): NonNullable<VDBCoarseFirstBindPoolLike[K]> => {
    const fn = pool[name];
    if (typeof fn !== "function") {
      throw new VDBError("VDB_BAD_OPTION", `vdbStreamCoarseFirst: \`bind\` needs a pool with ${name}() (a VDBGridPool).`);
    }
    return fn.bind(pool) as NonNullable<VDBCoarseFirstBindPoolLike[K]>;
  };
  if (bind.majorant) {
    const majorant = bind.majorant(event);
    if (majorant) need("setMajorant")(event.id, majorant);
  }
  if (bind.mesh) {
    const next = vdbProxyGeometry(need("info")(event.id), bind.proxyPad === undefined ? {} : { pad: bind.proxyPad });
    bind.mesh.geometry.dispose();
    bind.mesh.geometry = next;
  }
  if (bind.grids) {
    need("flush")();
    bind.grids.update();
  }
  const material = bind.material;
  if (material) {
    const base = bind.densityScale ?? baseScale.get(material) ?? material.densityScale.value;
    baseScale.set(material, base);
    material.densityScale.value = base * event.voxelRatio;
  }
}

const familyOf = (name: string): string => parseNvdbLodName(name).baseName;

/** Finest first: the full grid (lod null) beats any preview; a larger resolution beats a smaller. */
function finer(a: NvdbStreamGrid, b: NvdbStreamGrid | null): boolean {
  if (b === null) return true;
  if (b.lod === null) return false;
  if (a.lod === null) return true;
  return a.lod > b.lod;
}

function chunks(source: VDBCoarseFirstSource): { iterate: AsyncIterable<Uint8Array>; cancel: () => Promise<void> } {
  const stream = source instanceof Response ? source.body : source;
  if (source instanceof Response && !source.ok) {
    throw new VDBError("VDB_BAD_OPTION", `vdbStreamCoarseFirst: the response is ${source.status} ${source.statusText} (${source.url}).`);
  }
  if (stream === null) throw new VDBError("VDB_BAD_OPTION", "vdbStreamCoarseFirst: the response has no body.");
  if (typeof (stream as ReadableStream<Uint8Array>).getReader === "function") {
    const reader = (stream as ReadableStream<Uint8Array>).getReader();
    return {
      iterate: {
        async *[Symbol.asyncIterator]() {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) return;
            yield value;
          }
        },
      },
      cancel: () => reader.cancel().catch(() => undefined),
    };
  }
  const iterable = stream as AsyncIterable<Uint8Array>;
  return { iterate: iterable, cancel: async () => undefined };
}

/**
 * Streams a coarse-first `.nvdb` into one pool id: `add` the first grid of the
 * family with the family's largest size reserved, then `replace` with each
 * finer grid. Resolves when the full grid is in the pool (the rest of the
 * stream is cancelled). Rejects with the reader's `NvdbFormatError` on bad
 * or truncated data.
 */
export async function vdbStreamCoarseFirst(
  pool: VDBCoarseFirstPoolLike & Partial<VDBCoarseFirstBindPoolLike>,
  source: VDBCoarseFirstSource,
  options: VDBCoarseFirstOptions = {},
): Promise<VDBCoarseFirstResult> {
  const signal = options.signal;
  signal?.throwIfAborted();
  const wire = source instanceof Response ? nvdbResponseLength(source) : null;
  const reader = new NvdbStreamReader({
    ...options.reader,
    ...(wire !== null && options.reader?.bytesTotal === undefined ? { bytesTotal: wire } : {}),
    ...(options.onProgress ? { onProgress: options.onProgress } : {}),
  });
  const { iterate, cancel } = chunks(source);

  let family = options.gridName ?? null;
  let id = -1;
  let current: NvdbStreamGrid | null = null;
  const shown: VDBCoarseFirstEvent[] = [];

  const familyGrids = (): NvdbGridMetadata[] => reader.grids.filter((g) => familyOf(g.name) === family);
  const fullVoxel = (grid: NvdbStreamGrid): number => {
    const full = familyGrids().find((g) => g.name === family);
    return full ? grid.metadata.voxelSize[0] / full.voxelSize[0] : 1;
  };
  // The reader accepts only the pool's scalar types (Float, Fp4, Fp8, Fp16, FpN: the names in lower case).
  const familyTypes = (): VDBGridTypeName[] => {
    const names = new Set(familyGrids().map((g) => g.gridType.toLowerCase()));
    return VDB_GRID_TYPE_NAMES.filter((t) => names.has(t));
  };

  try {
    for await (const chunk of iterate) {
      signal?.throwIfAborted();
      for (const grid of reader.push(chunk)) {
        family ??= grid.baseName;
        if (grid.baseName !== family || !finer(grid, current)) continue;
        const final = grid.lod === null;
        let action: "add" | "replace";
        if (id < 0) {
          const capacity = Math.max(...familyGrids().map((g) => g.gridByteSize));
          id = pool.add(grid.image, { capacity, label: options.label ?? family });
          action = "add";
        } else {
          pool.replace(id, grid.image);
          action = "replace";
        }
        current = grid;
        const event: VDBCoarseFirstEvent = { id, grid, final, action, bytesReceived: grid.bytesReceived, bytesTotal: grid.bytesTotal, voxelRatio: fullVoxel(grid), types: familyTypes() };
        shown.push(event);
        if (options.bind) vdbCoarseFirstApply(pool, event, options.bind);
        options.onGrid?.(event);
        if (final) {
          await cancel();
          return { id, shown, firstBytes: shown[0]!.bytesReceived, fullBytes: grid.bytesReceived, bytesTotal: grid.bytesTotal };
        }
      }
    }
    reader.end();
  } catch (err) {
    await cancel();
    throw err;
  }
  throw new VDBError(
    "VDB_BAD_IMAGE",
    family === null
      ? "vdbStreamCoarseFirst: the stream held no grid."
      : `vdbStreamCoarseFirst: the stream ended without the full grid "${family}" (grids: ${reader.grids.map((g) => g.name).join(", ")}).`,
  );
}
