// Records byte-level goldens from upstream's NATIVE converter (oracle (b)).
//
// For every spike fixture this writes the FLOAT .nvdb both converters consume
// to .out/inputs/, runs `picovdb convert --type f32` (and `--type u8` for SDF
// grids), and stores each output's SHA-256 + size in test/native-golden.json.
// The vitest suite then byte-compares src/convert.ts output against these
// hashes, so CI needs no Zig toolchain. Re-run this only when the pin moves.
//
// Requires a native build of upstream at the pinned commit:
//   PICOVDB_NATIVE=/path/to/zig-out/bin/picovdb node experiments/picovdb-spike/scripts/native-oracle.mjs
// (see docs/spikes/PICOVDB.md "Reproduction" for building it with Zig 0.16
// from the PyPI `ziglang` wheel when ziglang.org is unreachable).

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { SPIKE_DIR, tsLoader } from "./lib/load-ts.mjs";

const NATIVE = process.env.PICOVDB_NATIVE;
if (!NATIVE) {
  console.error("Set PICOVDB_NATIVE to the native picovdb binary (zig-out/bin/picovdb).");
  process.exit(2);
}

const OUT = path.join(SPIKE_DIR, ".out");
const sha = (b) => createHash("sha256").update(b).digest("hex");

const ts = await tsLoader();
try {
  const fx = await ts.load("src/fixtures.ts");
  const golden = {
    $meta: {
      native: "emcfarlane/picovdb@f5d22e5602a7fc7f206d33ec35328f58e8667c83 `picovdb convert`, Zig 0.16.0, OpenVDB v13.0.0 headers",
      recorded: new Date().toISOString().slice(0, 10),
    },
  };
  mkdirSync(path.join(OUT, "inputs"), { recursive: true });
  mkdirSync(path.join(OUT, "native"), { recursive: true });
  for (const f of fx.loadAll()) {
    const s = fx.stem(f);
    const input = path.join(OUT, "inputs", `${s}.nvdb`);
    writeFileSync(input, f.nvdbFloat);
    golden[f.id] = { inputSha256: sha(f.nvdbFloat) };
    for (const type of f.kind === "sdf" ? ["f32", "u8"] : ["f32"]) {
      const out = path.join(OUT, "native", `${s}.${type}.pvdb`);
      const t0 = performance.now();
      execFileSync(NATIVE, ["convert", "--type", type, input, out], { stdio: ["ignore", "ignore", "ignore"] });
      const ms = performance.now() - t0;
      const bytes = readFileSync(out);
      golden[f.id][type] = { sha256: sha(bytes), bytes: bytes.byteLength, nativeMs: Math.round(ms) };
      console.log(`${f.id} ${type}: ${bytes.byteLength} B  ${sha(bytes).slice(0, 16)}  (${Math.round(ms)} ms)`);
    }
  }
  writeFileSync(path.join(SPIKE_DIR, "test/native-golden.json"), JSON.stringify(golden, null, 2) + "\n");
  console.log("wrote test/native-golden.json");
} finally {
  await ts.close();
}
