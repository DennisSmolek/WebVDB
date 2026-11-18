/**
 * A recording fake of the WebGPU objects the pool touches (no real GPU). Buffers
 * hold real bytes: `queue.writeBuffer` lands at once, encoder copies land at
 * `queue.submit`, which is WebGPU's queue order. Every call is logged so tests can
 * assert exactly what the pool did. Validation mirrors the WebGPU rules the pool
 * must respect (alignment, bounds, destroyed buffers, same-buffer copies).
 *
 * Error scopes, allocation failure and queue completion are modelled too:
 * `failCreate` makes the next matching `createBuffer` return an invalid buffer
 * and raise an `out-of-memory` error into the innermost matching scope; writes
 * to an invalid buffer are dropped and a command buffer touching one is skipped
 * at submit (both raise a validation error), as on a real device.
 * `onSubmittedWorkDone` resolves only when the test calls `completeWork()`, and
 * `holdScopes()` delays every `popErrorScope` until its release function runs.
 */

export type FakeOp =
  | { kind: "create"; buffer: FakeBuffer }
  | { kind: "destroy"; buffer: FakeBuffer }
  | { kind: "write"; buffer: FakeBuffer; offset: number; size: number }
  | { kind: "copy"; src: FakeBuffer; srcOffset: number; dst: FakeBuffer; dstOffset: number; size: number }
  | { kind: "submit"; copies: number }
  | { kind: "error"; filter: "out-of-memory" | "validation"; message: string; captured: boolean };

let nextBufferId = 1;

export class FakeBuffer {
  readonly id = nextBufferId++;
  readonly size: number;
  readonly usage: number;
  readonly label: string;
  readonly data: Uint8Array;
  destroyed = false;
  destroyCalls = 0;
  /** Created by a failed allocation (`FakeDevice.failCreate`): usable as a handle, holds nothing. */
  invalid = false;
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
    if (buffer.invalid) {
      this.device.raise("validation", `writeBuffer to invalid buffer "${buffer.label}"`);
      return;
    }
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
      }
      if (cb.copies.some((c) => c.src.invalid || c.dst.invalid)) {
        this.device.raise("validation", "submitted a command buffer that uses an invalid buffer; it was skipped");
        continue;
      }
      for (const c of cb.copies) {
        c.dst.data.set(c.src.data.subarray(c.srcOffset, c.srcOffset + c.size), c.dstOffset);
        n++;
      }
    }
    this.device.submits++;
    this.device.log.push({ kind: "submit", copies: n });
  }

  /** Resolves when the test calls `device.completeWork()`. */
  onSubmittedWorkDone(): Promise<void> {
    this.device.workDoneRequests++;
    return new Promise<void>((resolve) => this.device.workDone.push(resolve));
  }
}

type ErrorFilter = "out-of-memory" | "validation" | "internal";

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
  /** Open error scopes, innermost last. */
  readonly scopes: { filter: ErrorFilter; error: { message: string } | null }[] = [];
  /** Errors no scope captured. */
  readonly uncaptured: string[] = [];
  /** Pending `onSubmittedWorkDone` resolvers. */
  readonly workDone: (() => void)[] = [];
  workDoneRequests = 0;
  submits = 0;
  /** When set, the next `createBuffer` it matches fails (out of memory) and the predicate is cleared. */
  failCreate: ((desc: { size: number; label?: string }) => boolean) | null = null;
  #scopeGate: Promise<void> | null = null;

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
    const fail = this.failCreate?.(desc) ?? false;
    const b = new FakeBuffer(this, fail ? { ...desc, size: 0 } : desc);
    // An invalid buffer still reports its requested size, like a real GPUBuffer.
    if (fail) {
      Object.defineProperty(b, "size", { value: desc.size });
      b.invalid = true;
      this.failCreate = null;
    }
    this.buffers.push(b);
    this.log.push({ kind: "create", buffer: b });
    if (fail) this.raise("out-of-memory", `fake: out of memory allocating ${desc.size} bytes`);
    return b;
  }

  createCommandEncoder(): FakeEncoder {
    return new FakeEncoder(this);
  }

  /** Open error scopes (pushed and not yet popped). */
  get errorScopes(): number {
    return this.scopes.length;
  }

  pushErrorScope(filter: ErrorFilter): void {
    this.scopes.push({ filter, error: null });
  }

  popErrorScope(): Promise<{ message: string } | null> {
    const scope = this.scopes.pop();
    if (!scope) return Promise.reject(new Error("fake: popErrorScope with no scope"));
    const gate = this.#scopeGate;
    return gate ? gate.then(() => scope.error) : Promise.resolve(scope.error);
  }

  /** Routes an error to the innermost scope with its filter (first error wins), else `uncaptured`. */
  raise(filter: "out-of-memory" | "validation", message: string): void {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const s = this.scopes[i]!;
      if (s.filter !== filter) continue;
      s.error ??= { message };
      this.log.push({ kind: "error", filter, message, captured: true });
      return;
    }
    this.uncaptured.push(message);
    this.log.push({ kind: "error", filter, message, captured: false });
  }

  /** Delays every `popErrorScope` result until the returned function is called. */
  holdScopes(): () => Promise<void> {
    let release!: () => void;
    this.#scopeGate = new Promise<void>((r) => (release = r));
    return async () => {
      this.#scopeGate = null;
      release();
      await settle();
    };
  }

  /** Resolves every pending `onSubmittedWorkDone`, then lets their callbacks run. */
  async completeWork(): Promise<void> {
    for (const resolve of this.workDone.splice(0)) resolve();
    await settle();
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

/** Lets every queued promise callback run (one macrotask turn drains the microtask queue). */
export function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
