import { describe, expect, it } from "vitest";
import { computeRequiredLimits, createVolumeRenderer, nextPow2 } from "../src/renderer.js";

/**
 * `createVolumeRenderer` needs a real GPUDevice, which Vitest's node
 * environment can't provide — so we test the pure parts directly:
 * `nextPow2`/`computeRequiredLimits` (the limits math, extracted so it's
 * unit-testable without a GPU), and the `navigator.gpu` absence error path,
 * which is exactly what happens when this runs under Node.
 */

describe("nextPow2", () => {
  it("returns 1 for n <= 1", () => {
    expect(nextPow2(0)).toBe(1);
    expect(nextPow2(1)).toBe(1);
  });

  it("returns the next power of two for non-power-of-two input", () => {
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(5)).toBe(8);
    expect(nextPow2(1_000_000)).toBe(1_048_576);
  });

  it("is the identity on exact powers of two", () => {
    expect(nextPow2(2)).toBe(2);
    expect(nextPow2(1024)).toBe(1024);
  });
});

describe("computeRequiredLimits", () => {
  const bigAdapter = { maxStorageBufferBindingSize: 4 * 1024 ** 3, maxBufferSize: 4 * 1024 ** 3 };

  it("floors at 128 MiB when gridBytes is small or omitted", () => {
    const limits = computeRequiredLimits(bigAdapter);
    expect(limits.maxStorageBufferBindingSize).toBe(128 * 1024 * 1024);
    expect(limits.maxBufferSize).toBe(128 * 1024 * 1024);

    const limitsSmallGrid = computeRequiredLimits(bigAdapter, 1024);
    expect(limitsSmallGrid.maxStorageBufferBindingSize).toBe(128 * 1024 * 1024);
  });

  it("rounds a grid larger than 128 MiB up to the next power of two", () => {
    const gridBytes = 200 * 1024 * 1024; // between 128 and 256 MiB
    const limits = computeRequiredLimits(bigAdapter, gridBytes);
    expect(limits.maxStorageBufferBindingSize).toBe(256 * 1024 * 1024);
    expect(limits.maxBufferSize).toBe(256 * 1024 * 1024);
  });

  it("clamps to the adapter's limit even if the grid wants more", () => {
    const smallAdapter = { maxStorageBufferBindingSize: 64 * 1024 * 1024, maxBufferSize: 64 * 1024 * 1024 };
    const limits = computeRequiredLimits(smallAdapter, 1024 * 1024 * 1024); // 1 GiB grid
    // Adapter only offers 64 MiB, which is below even the 128 MiB floor —
    // min(adapterLimit, wanted) must respect the adapter's ceiling.
    expect(limits.maxStorageBufferBindingSize).toBe(64 * 1024 * 1024);
    expect(limits.maxBufferSize).toBe(64 * 1024 * 1024);
  });

  it("allows independently-sized adapter limits per field", () => {
    const asymmetricAdapter = { maxStorageBufferBindingSize: 200 * 1024 * 1024, maxBufferSize: 4 * 1024 ** 3 };
    const limits = computeRequiredLimits(asymmetricAdapter, 300 * 1024 * 1024);
    // wanted = nextPow2(300 MiB) = 512 MiB, clamped per-field against each adapter limit.
    expect(limits.maxStorageBufferBindingSize).toBe(200 * 1024 * 1024);
    expect(limits.maxBufferSize).toBe(512 * 1024 * 1024);
  });
});

describe("createVolumeRenderer — navigator.gpu absence", () => {
  it("throws a clear error when navigator.gpu is unavailable (as in Node)", async () => {
    expect(typeof navigator === "undefined" || !navigator.gpu).toBe(true);
    await expect(createVolumeRenderer()).rejects.toThrowError(/WebGPU is unavailable/);
  });
});
