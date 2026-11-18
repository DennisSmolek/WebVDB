/**
 * The compute-side WebGPU calls the core compute tools make (K2.3), added to
 * the recording FakeDevice: bind group / pipeline creation, compute passes and
 * buffer mapping. Kernels don't run, so outputs read back as the tool
 * initialized them.
 */
import { FakeBuffer, FakeDevice } from "./fake-gpu.js";

export interface FakePass {
  pipeline: { entry: string; code: string };
  groups: { entries: GPUBindGroupEntry[] }[];
  dispatch: number[];
}

const passesOf = new WeakMap<object, FakePass[]>();

/** A FakeDevice that also records compute passes (into `passes`, per device). */
export function computeFake(): { fake: FakeDevice; device: GPUDevice; passes: FakePass[] } {
  const fake = new FakeDevice();
  const passes: FakePass[] = [];
  passesOf.set(fake, passes);
  const f = fake as unknown as Record<string, unknown>;
  f.createBindGroupLayout = (d: GPUBindGroupLayoutDescriptor) => ({ ...d });
  f.createPipelineLayout = (d: GPUPipelineLayoutDescriptor) => ({ ...d });
  f.createShaderModule = (d: GPUShaderModuleDescriptor) => ({ code: d.code });
  f.createComputePipeline = (d: GPUComputePipelineDescriptor) => ({
    entry: d.compute.entryPoint,
    code: (d.compute.module as unknown as { code: string }).code,
  });
  f.createBindGroup = (d: GPUBindGroupDescriptor) => ({ entries: [...d.entries] });

  const encProto = Object.getPrototypeOf(fake.createCommandEncoder()) as Record<string, unknown>;
  encProto.beginComputePass = function (this: { device: object }) {
    const pass: FakePass = { pipeline: { entry: "", code: "" }, groups: [], dispatch: [] };
    passesOf.get(this.device)?.push(pass);
    return {
      setPipeline: (p: FakePass["pipeline"]) => (pass.pipeline = p),
      setBindGroup: (i: number, g: { entries: GPUBindGroupEntry[] }) => (pass.groups[i] = g),
      dispatchWorkgroups: (...xyz: number[]) => (pass.dispatch = xyz),
      end: () => {},
    };
  };
  const bufProto = FakeBuffer.prototype as unknown as Record<string, unknown>;
  bufProto.getMappedRange = function (this: FakeBuffer) {
    return this.data.buffer;
  };
  bufProto.unmap = () => {};
  bufProto.mapAsync = function (this: FakeBuffer) {
    return this.destroyed ? Promise.reject(new Error("fake: mapAsync on a destroyed buffer")) : Promise.resolve();
  };
  return { fake, device: fake.asGPU(), passes };
}
