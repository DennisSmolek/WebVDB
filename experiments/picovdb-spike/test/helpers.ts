import { convertToPicoVDB, type ConvertOptions, type ConvertResult } from "../src/convert.js";
import { loadAll, type Fixture } from "../src/fixtures.js";

let fixtures: Fixture[] | undefined;
/** Loaded once per worker (the teapot build alone is a few seconds). */
export function corpus(): Fixture[] {
  fixtures ??= loadAll();
  return fixtures;
}

const cache = new Map<string, ConvertResult>();
export function converted(f: Fixture, opts: ConvertOptions = {}): ConvertResult {
  const key = `${f.id}|${opts.valueType ?? "f32"}|${opts.layout ?? "native"}`;
  let r = cache.get(key);
  if (!r) {
    r = convertToPicoVDB([{ image: f.image, gridClassId: f.gridClassId }], opts);
    cache.set(key, r);
  }
  return r;
}

export function variants(f: Fixture): ("f32" | "u8")[] {
  return f.kind === "sdf" ? ["f32", "u8"] : ["f32"];
}
