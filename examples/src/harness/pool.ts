/**
 * K1.3b browser check (docs/dev/design/K1-core.md §8): the nine primitive fixtures
 * go into one VDBGridPool started at 4 KiB, so it grows several times on a real
 * device; then the whole buffer is read back and every image range must be
 * byte-identical to its file, with header and records pointing at it.
 * Results go to `window.__VDB_POOL_READBACK__` for e2e/pool-readback.spec.ts.
 */
import { VDBGridPool, VDB_POOL_RECORD_WORDS, VDB_POOL_TABLE_OFFSET, decodeHeader, decodeRecord } from "../../../packages/vdb/src/core/internal.js";
import { NvdbFile } from "@pmndrs/vdb/tools";

interface FixtureCheck {
  name: string;
  id: number;
  imageOffset: number;
  imageBytes: number;
  identical: boolean;
  firstDiff: number;
  recordOk: boolean;
}

interface PoolReadback {
  done: boolean;
  error?: string;
  missing?: string[];
  fixtures: FixtureCheck[];
  versions: number[];
  byteLength: number;
  headerOk: boolean;
  validationError?: string;
}

declare global {
  interface Window {
    __VDB_POOL_READBACK__?: PoolReadback;
  }
}

const NAMES = ["box_fog", "sphere_fog", "torus_fog"].flatMap((p) => ["float", "fp8", "fpn"].map((t) => `${p}_${t}`));
const root = document.getElementById("demo-root")!;

function log(text: string, cls = ""): void {
  const p = document.createElement("p");
  p.textContent = text;
  if (cls) p.className = cls;
  root.append(p);
}

async function run(): Promise<PoolReadback> {
  const result: PoolReadback = { done: false, fixtures: [], versions: [], byteLength: 0, headerOk: false };
  if (!navigator.gpu) return { ...result, done: true, error: "WebGPU unavailable (navigator.gpu undefined)." };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ...result, done: true, error: "No WebGPU adapter." };
  const device = await adapter.requestDevice({ requiredLimits: VDBGridPool.requiredLimits(adapter) });

  const files = await Promise.all(
    NAMES.map(async (name) => {
      const res = await fetch(`/fixtures/primitives/${name}.nvdb`);
      return res.ok ? NvdbFile.fromArrayBuffer(await res.arrayBuffer()).gridImage(0) : null;
    }),
  );
  const missing = NAMES.filter((_, k) => files[k] === null);
  if (missing.length > 0) return { ...result, done: true, missing };
  const images = files as Uint32Array[];

  device.pushErrorScope("validation");
  const pool = await VDBGridPool.create(device, { initialBytes: 4096, maxGrids: 4, label: "pool readback" });
  const ids: number[] = [];
  for (let k = 0; k < images.length; k++) {
    ids.push(pool.add(images[k]!, { label: NAMES[k]! }));
    if (k % 3 === 2) {
      pool.flush();
      result.versions.push(pool.version);
    }
  }
  pool.remove(ids[4]!);
  ids[4] = pool.add(images[4]!, { label: NAMES[4]! });
  pool.flush();
  result.versions.push(pool.version);
  // One more update destroys the last retired buffer on a real device.
  pool.flush();

  const size = pool.byteLength;
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(pool.buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const validation = await device.popErrorScope();
  const words = new Uint32Array(bytes.buffer);

  const header = decodeHeader(words);
  result.headerOk =
    header.magic === 0x50424456 && header.layoutVersion === pool.layoutVersion && header.gridCount === 9 &&
    header.gridCapacity === pool.gridCapacity && header.usedBytes === pool.usedBytes;
  result.byteLength = size;
  for (let k = 0; k < images.length; k++) {
    const info = pool.info(ids[k]!);
    const src = new Uint8Array(images[k]!.buffer, images[k]!.byteOffset, images[k]!.byteLength);
    const got = bytes.subarray(info.imageOffset!, info.imageOffset! + info.imageBytes);
    let firstDiff = got.length === src.length ? -1 : 0;
    for (let i = 0; i < src.length && firstDiff < 0; i++) if (got[i] !== src[i]) firstDiff = i;
    const rec = decodeRecord(words, VDB_POOL_TABLE_OFFSET / 4 + ids[k]! * VDB_POOL_RECORD_WORDS);
    result.fixtures.push({
      name: NAMES[k]!,
      id: ids[k]!,
      imageOffset: info.imageOffset!,
      imageBytes: info.imageBytes,
      identical: firstDiff < 0,
      firstDiff,
      recordOk: rec.live && rec.imageOffset === info.imageOffset && rec.imageBytes === info.imageBytes && rec.imageOffset % 32 === 0,
    });
  }
  pool.dispose();
  device.destroy();
  return { ...result, done: true, ...(validation ? { validationError: validation.message } : {}) };
}

root.textContent = "";
run()
  .catch((e: unknown): PoolReadback => ({
    done: true,
    error: e instanceof Error ? `${e.name}: ${e.message}` : String(e),
    fixtures: [],
    versions: [],
    byteLength: 0,
    headerOk: false,
  }))
  .then((r) => {
    window.__VDB_POOL_READBACK__ = r;
    if (r.error || r.validationError) log(`error: ${r.error ?? r.validationError}`, "fail");
    else if (r.missing) log(`fixtures missing: ${r.missing.join(", ")} (pnpm fixtures:bake)`, "fail");
    else {
      log(`pool ${r.byteLength} bytes, versions ${r.versions.join(" → ")}, header ${r.headerOk ? "ok" : "BAD"}`);
      for (const f of r.fixtures) {
        log(`${f.name}: id ${f.id} @ ${f.imageOffset} (${f.imageBytes} B) ${f.identical && f.recordOk ? "identical" : `MISMATCH at ${f.firstDiff}`}`, f.identical && f.recordOk ? "ok" : "fail");
      }
    }
  });
