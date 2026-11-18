/**
 * A recording fake of the WebGPU objects the pool touches (no real GPU). Buffers
 * hold real bytes: `queue.writeBuffer` lands at once, encoder copies land at
 * `queue.submit`, which is WebGPU's queue order. Every call is logged so tests can
 * assert exactly what the pool did. Validation mirrors the WebGPU rules the pool
 * must respect (alignment, bounds, destroyed buffers, same-buffer copies).
 */

export type FakeOp =
  | { kind: "create"; buffer: FakeBuffer }
  | { kind: "destroy"; buffer: FakeBuffer }
  | { kind: "write"; buffer: FakeBuffer; offset: number; size: number }
  | { kind: "copy"; src: FakeBuffer; srcOffset: number; dst: FakeBuffer; dstOffset: number; size: number }
  | { kind: "submit"; copies: number };

let nextBufferId = 1;

export class FakeBuffer {
  readonly id = nextBufferId++;
  readonly size: number;
  readonly usage: number;
  readonly label: string;
  readonly data: Uint8Array;
  destroyed = false;
  destroyCalls = 0;
  readonly #device: FakeDevice;

  constructor(device: FakeDevice, desc: { size: number; usage: number; label?: string }) {
    this.#device = device;
    this.size = desc.size;
    this.usage = desc.usage;
    this.label = desc.label ?? "";
    this.data = new Uint8Array(desc.size);
  }

  destroy(): void {
    this.destroyCalls++;
    if (!this.destroyed) this.#device.log.push({ kind: "destroy", buffer: this });
    this.destroyed = true;
  }

  words(): Uint32Array {
    return new Uint32Array(this.data.buffer, 0, this.size / 4);
  }
}

type Copy = Extract<FakeOp, { kind: "copy" }>;

class FakeEncoder {
  readonly copies: Copy[] = [];
  constructor(private readonly device: FakeDevice) {}

  copyBufferToBuffer(src: FakeBuffer, srcOffset: number, dst: FakeBuffer, dstOffset: number, size: number): void {
    if (src === dst) throw new Error("fake: copyBufferToBuffer with src === dst");
    if (srcOffset % 4 || dstOffset % 4 || size % 4) throw new Error("fake: copy offsets/size must be multiples of 4");
    if (srcOffset + size > src.size || dstOffset + size > dst.size) throw new Error("fake: copy out of bounds");
    const op: Copy = { kind: "copy", src, srcOffset, dst, dstOffset, size };
    this.copies.push(op);
    this.device.log.push(op);
  }

  finish(): { copies: Copy[] } {
    return { copies: this.copies };
  }
}

class FakeQueue {
  constructor(private readonly device: FakeDevice) {}

  writeBuffer(buffer: FakeBuffer, offset: number, data: ArrayBufferView): void {
    if (buffer.destroyed) throw new Error("fake: writeBuffer to a destroyed buffer");
    if (offset % 4 || data.byteLength % 4) throw new Error("fake: writeBuffer offset/size must be multiples of 4");
    if (offset + data.byteLength > buffer.size) {
      throw new Error(`fake: writeBuffer out of bounds (${offset} + ${data.byteLength} > ${buffer.size})`);
    }
    buffer.data.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    this.device.log.push({ kind: "write", buffer, offset, size: data.byteLength });
  }

  submit(commandBuffers: { copies: Copy[] }[]): void {
    let n = 0;
    for (const cb of commandBuffers) {
      for (const c of cb.copies) {
        if (c.src.destroyed || c.dst.destroyed) throw new Error("fake: submitted a copy touching a destroyed buffer");
        c.dst.data.set(c.src.data.subarray(c.srcOffset, c.srcOffset + c.size), c.dstOffset);
        n++;
      }
    }
    this.device.log.push({ kind: "submit", copies: n });
  }
}

export interface FakeLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxStorageBuffersPerShaderStage: number;
}

export class FakeDevice {
  readonly log: FakeOp[] = [];
  readonly buffers: FakeBuffer[] = [];
  readonly limits: FakeLimits;
  readonly queue = new FakeQueue(this);
  readonly lost = new Promise<never>(() => {});
  errorScopes = 0;

  constructor(limits: Partial<FakeLimits> = {}) {
    this.limits = {
      maxStorageBufferBindingSize: 128 << 20,
      maxBufferSize: 256 << 20,
      maxStorageBuffersPerShaderStage: 8,
      ...limits,
    };
  }

  createBuffer(desc: { size: number; usage: number; label?: string }): FakeBuffer {
    if (desc.size > this.limits.maxBufferSize) throw new Error("fake: buffer larger than maxBufferSize");
    if (desc.size % 4) throw new Error("fake: buffer size must be a multiple of 4");
    const b = new FakeBuffer(this, desc);
    this.buffers.push(b);
    this.log.push({ kind: "create", buffer: b });
    return b;
  }

  createCommandEncoder(): FakeEncoder {
    return new FakeEncoder(this);
  }

  pushErrorScope(): void {
    this.errorScopes++;
  }

  popErrorScope(): Promise<null> {
    this.errorScopes--;
    return Promise.resolve(null);
  }

  /** Ops logged since `mark` (an index from `log.length`). */
  since(mark: number): FakeOp[] {
    return this.log.slice(mark);
  }

  /** Runs `update` on a fresh encoder and submits it, like `flush`, returning the ops it caused. */
  updateAndSubmit(pool: { update(encoder: GPUCommandEncoder): void }): FakeOp[] {
    const mark = this.log.length;
    const encoder = this.createCommandEncoder();
    pool.update(encoder as unknown as GPUCommandEncoder);
    this.queue.submit([encoder.finish()]);
    return this.since(mark);
  }

  asGPU(): GPUDevice {
    return this as unknown as GPUDevice;
  }
}

/** Usage bits (WebGPU spec values). */
export const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 } as const;
