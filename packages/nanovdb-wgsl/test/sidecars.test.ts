import { readFile, readdir } from "node:fs/promises";
import { describe, expect, it } from "vitest";

// The committed sidecars are the ground truth the Phase 2 WGSL traversal
// suite replays; this guards their schema and physical plausibility so a
// bad re-bake is caught at commit time. (.nvdb files are git-ignored —
// regenerate everything with `pnpm fixtures:bake`.)
const dirUrl = new URL("../../../fixtures/primitives/", import.meta.url);

interface Sample {
  ijk: [number, number, number];
  value: number;
  active: boolean;
}

async function loadSidecars(): Promise<Map<string, any>> {
  const names = (await readdir(dirUrl)).filter((n) => n.endsWith(".sidecar.json"));
  const map = new Map<string, any>();
  for (const n of names) {
    map.set(n, JSON.parse(await readFile(new URL(n, dirUrl), "utf8")));
  }
  return map;
}

const PRIMITIVES = ["sphere_fog", "torus_fog", "box_fog"];
const VARIANTS = ["float", "fp8", "fpn"];

describe("primitive fixture sidecars", () => {
  it("all nine primitive×variant sidecars are committed", async () => {
    const sidecars = await loadSidecars();
    for (const p of PRIMITIVES) {
      for (const v of VARIANTS) {
        expect(sidecars.has(`${p}_${v}.sidecar.json`), `${p}_${v}`).toBe(true);
      }
    }
  });

  it("each sidecar is a plausible fog volume", async () => {
    for (const [name, sc] of await loadSidecars()) {
      expect(sc.grid.class, name).toBe("FogVolume");
      expect(sc.grid.activeVoxelCount, name).toBeGreaterThan(0);
      expect(sc.samples.length, name).toBeGreaterThanOrEqual(73);
      const [mn, mx] = sc.grid.indexBBox;
      for (let a = 0; a < 3; a++) expect(mn[a], name).toBeLessThan(mx[a]);
      let positive = 0;
      for (const s of sc.samples as Sample[]) {
        if (s.active) {
          // Active fog density is in [0,1] — 0 occurs legitimately at the
          // outer edge of the narrow band (e.g. box corners on the surface).
          expect(s.value, `${name} active ${s.ijk}`).toBeGreaterThanOrEqual(0);
          expect(s.value, `${name} active ${s.ijk}`).toBeLessThanOrEqual(1);
          if (s.value > 0) positive++;
        } else {
          expect(s.value, `${name} inactive ${s.ijk}`).toBe(0);
        }
      }
      expect(positive, `${name} has non-degenerate density`).toBeGreaterThan(0);
    }
  });

  it("quantized variants agree with float within Fp8 half-quantum", async () => {
    const sidecars = await loadSidecars();
    for (const p of PRIMITIVES) {
      const f = sidecars.get(`${p}_float.sidecar.json`);
      for (const v of ["fp8", "fpn"]) {
        const q = sidecars.get(`${p}_${v}.sidecar.json`);
        expect(q.grid.activeVoxelCount, `${p}_${v}`).toBe(f.grid.activeVoxelCount);
        expect(q.grid.indexBBox, `${p}_${v}`).toEqual(f.grid.indexBBox);
        (f.samples as Sample[]).forEach((s, i) => {
          const qs = (q.samples as Sample[])[i]!;
          expect(qs.ijk, `${p}_${v} sample ${i}`).toEqual(s.ijk);
          expect(qs.active, `${p}_${v} sample ${i}`).toBe(s.active);
          expect(Math.abs(qs.value - s.value), `${p}_${v} ${s.ijk}`).toBeLessThanOrEqual(0.005);
        });
      }
    }
  });
});
