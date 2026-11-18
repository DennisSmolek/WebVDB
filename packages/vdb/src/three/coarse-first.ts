/**
 * `vdbStreamCoarseFirst` (K4.16): stream a coarse-first `.nvdb`
 * (`bakeCoarseFirst`, docs/design/K4-5-coarse-first.md option a) into one
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
 * (the `onGrid` callback is the place):
 * - `replace` drops a brick majorant: call `pool.setMajorant` again;
 * - `VDBVolumeMaterial` marches in index voxels, so a preview with k-times
 *   larger voxels has 1/k the optical depth at the same `densityScale`;
 *   scale it by `voxelRatio` while a preview shows, and back to 1 at `final`;
 * - the proxy mesh: a new `vdbProxyGeometry(pool.info(id))` (the preview's
 *   box is the full box grown to whole preview voxels).
 *
 * Engine-free: it touches only `pool.add` / `pool.replace`, so it works with
 * any `VDBGridPool` host. It lives in `./three` because `tools` may not import
 * the core (D20).
 */

import { VDBError } from "../core/errors.js";
import { NvdbStreamReader, parseNvdbLodName, type GridMetadata, type NvdbStreamGrid, type NvdbStreamReaderOptions } from "../tools/index.js";

/** The part of a `VDBGridPool` the loader uses. */
export interface VDBCoarseFirstPoolLike {
  add(image: Uint32Array, options?: { capacity?: number; label?: string }): number;
  replace(id: number, image: Uint32Array): void;
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
  /** This grid's voxel size over the full grid's (x axis): 1 for the full grid, > 1 for a preview. */
  voxelRatio: number;
}

export interface VDBCoarseFirstOptions {
  /** Which family to load, by the full grid's name. Default: the first grid's family. */
  gridName?: string;
  /** Pool label for the id. Default: the full grid's name. */
  label?: string;
  /** Limits for the stream reader. */
  reader?: NvdbStreamReaderOptions;
  /** Called after each `pool.add` / `pool.replace` (preview, then finer grids, then the full grid). */
  onGrid?: (event: VDBCoarseFirstEvent) => void;
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
  pool: VDBCoarseFirstPoolLike,
  source: VDBCoarseFirstSource,
  options: VDBCoarseFirstOptions = {},
): Promise<VDBCoarseFirstResult> {
  const signal = options.signal;
  signal?.throwIfAborted();
  const reader = new NvdbStreamReader(options.reader);
  const { iterate, cancel } = chunks(source);

  let family = options.gridName ?? null;
  let id = -1;
  let current: NvdbStreamGrid | null = null;
  const shown: VDBCoarseFirstEvent[] = [];

  const familyGrids = (): GridMetadata[] => reader.grids.filter((g) => familyOf(g.name) === family);
  const fullVoxel = (grid: NvdbStreamGrid): number => {
    const full = familyGrids().find((g) => g.name === family);
    return full ? grid.metadata.voxelSize[0] / full.voxelSize[0] : 1;
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
        const event: VDBCoarseFirstEvent = { id, grid, final, action, bytesReceived: grid.bytesReceived, voxelRatio: fullVoxel(grid) };
        shown.push(event);
        options.onGrid?.(event);
        if (final) {
          await cancel();
          return { id, shown, firstBytes: shown[0]!.bytesReceived, fullBytes: grid.bytesReceived };
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
