/**
 * Device checks and size limits for the pool (docs/design/K1-core.md §1.5).
 */

/**
 * `GPUBufferUsage` bits, restated as numbers so the core runs where the WebGPU
 * globals don't exist (Node tests, workers before a device). Values from the
 * WebGPU spec.
 */
export const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  STORAGE: 0x0080,
} as const;

/** Usage flags every pool-owned buffer has. */
export const POOL_BUFFER_USAGE = BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC;

/** Largest pool the 32-bit image addressing allows (§2), rounded down to the 32 B alignment. */
export const ADDRESS_LIMIT_BYTES = 2 ** 32 - 32;

/** The smallest buffer a pool can be: header + a one-record table, rounded to 32 B. */
export const MIN_POOL_BYTES = 256;

const lostDevices = new WeakSet<GPUDevice>();
const watchedDevices = new WeakSet<GPUDevice>();

/** Starts tracking `device.lost` so later checks can report a lost device synchronously. */
export function watchDevice(device: GPUDevice): void {
  if (watchedDevices.has(device)) return;
  watchedDevices.add(device);
  const lost: Promise<GPUDeviceLostInfo> | undefined = device.lost;
  void lost?.then(() => lostDevices.add(device));
}

/** True once a watched device has reported loss. */
export function isDeviceLost(device: GPUDevice): boolean {
  return lostDevices.has(device);
}

/** Largest pool buffer `device` can bind, clamped to the 32-bit addressing limit. */
export function deviceByteLimit(device: GPUDevice): number {
  const { maxStorageBufferBindingSize, maxBufferSize } = device.limits;
  return floor32(Math.min(maxStorageBufferBindingSize, maxBufferSize, ADDRESS_LIMIT_BYTES));
}

/** Rounds down to a multiple of 32. */
export function floor32(n: number): number {
  return Math.floor(n / 32) * 32;
}

/**
 * Why `device` can't hold a pool, or `null` when it can. Messages name the fix.
 * Checks: a device object with limits, not lost (tracked once the core has seen
 * the device), at least one storage buffer per stage, and room for a minimal pool.
 */
export function unsupportedReason(device: GPUDevice | null | undefined): string | null {
  if (!device) return "No GPUDevice was passed. Request one with adapter.requestDevice() and pass it in.";
  const limits = device.limits as GPUSupportedLimits | undefined;
  if (!limits) return "The object passed as a GPUDevice has no limits. Pass a real GPUDevice.";
  if (isDeviceLost(device)) return "The GPUDevice was lost. Request a new device and create a new pool on it.";
  if (!(limits.maxStorageBuffersPerShaderStage >= 1)) {
    return (
      `The device allows ${limits.maxStorageBuffersPerShaderStage} storage buffers per shader stage; the pool needs 1. ` +
      `Request a core (non-compatibility) device.`
    );
  }
  const limit = deviceByteLimit(device);
  if (limit < MIN_POOL_BYTES) {
    return (
      `The device can bind storage buffers of ${limit} bytes; the pool needs at least ${MIN_POOL_BYTES}. ` +
      `Request the device with VDBGridPool.requiredLimits(adapter).`
    );
  }
  return null;
}

/**
 * The limits to pass to `adapter.requestDevice({ requiredLimits })` so the pool
 * can grow as far as the adapter allows: the adapter's maxima for
 * `maxStorageBufferBindingSize` and `maxBufferSize`.
 */
export function requiredLimits(adapter: GPUAdapter): Record<string, number> {
  return {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  };
}
