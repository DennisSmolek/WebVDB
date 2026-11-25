/**
 * Resources of the core frame renderer (K6.1, design §4.1–§4.2): the retire queue and the froxel texture
 * allocation. The renderer owns every texture and buffer it makes; a replaced one is retired and destroyed
 * two updates later (the pool's contract: work recorded with it in this and the previous update has been
 * submitted by then). `dispose()` drains the queue.
 */

/** Something with a `destroy()` (a GPUTexture, GPUBuffer or GPUQuerySet). */
export interface VDBDestroyable {
  destroy(): void;
}

/** Destroys retired objects two updates after they were retired. */
export class VDBRetireQueue {
  #items: { tick: number; object: VDBDestroyable }[] = [];
  #tick = 0;

  /** Objects waiting to be destroyed. */
  get pending(): number {
    return this.#items.length;
  }

  /** Keeps `object` until two more {@link advance} calls, then destroys it. */
  retire(object: VDBDestroyable | null | undefined): void {
    if (object) this.#items.push({ tick: this.#tick, object });
  }

  /** One update: destroys what was retired two or more updates ago. */
  advance(): void {
    this.#tick++;
    const keep: { tick: number; object: VDBDestroyable }[] = [];
    for (const item of this.#items) {
      if (this.#tick - item.tick >= 2) item.object.destroy();
      else keep.push(item);
    }
    this.#items = keep;
  }

  /** Destroys everything now (`dispose()`). */
  drain(): void {
    for (const item of this.#items) item.object.destroy();
    this.#items = [];
  }
}

/** `GPUTextureUsage` values, restated (the globals do not exist in Node). */
export const TEXTURE_USAGE = { COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10 } as const;
/** `GPUBufferUsage` values. */
export const BUFFER_USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
/** `GPUShaderStage` values. */
export const STAGE = { VERTEX: 0x1, FRAGMENT: 0x2, COMPUTE: 0x4 } as const;

/** A froxel-sized 3D storage texture (write by compute, read by sampling or `textureLoad`, copy for readbacks). */
export function createFroxelTexture(device: GPUDevice, label: string, size: readonly [number, number, number], format: GPUTextureFormat, sampled = true): GPUTexture {
  return device.createTexture({
    label,
    size: [size[0], size[1], size[2]],
    dimension: "3d",
    format,
    usage: TEXTURE_USAGE.STORAGE_BINDING | TEXTURE_USAGE.COPY_SRC | (sampled ? TEXTURE_USAGE.TEXTURE_BINDING : 0),
  });
}
