import { describe, expect, it } from "vitest";
import { checkWebGPU } from "../src/demos/webgpu-check";

describe("checkWebGPU", () => {
  it("reports a missing navigator.gpu", async () => {
    expect(await checkWebGPU(undefined)).toMatchObject({ ok: false, reason: "no-api" });
  });
  it("reports a null adapter", async () => {
    expect(await checkWebGPU({ requestAdapter: async () => null })).toMatchObject({ ok: false, reason: "no-adapter" });
  });
  it("reports a throwing requestAdapter", async () => {
    const r = await checkWebGPU({
      requestAdapter: async () => {
        throw new Error("boom");
      },
    });
    expect(r).toMatchObject({ ok: false, reason: "error" });
    expect(r.ok === false && r.detail).toContain("boom");
  });
  it("passes with an adapter", async () => {
    expect(await checkWebGPU({ requestAdapter: async () => ({}) })).toEqual({ ok: true });
  });
});
