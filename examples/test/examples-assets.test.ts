import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { assetBase, fixtureUrl } from "../src/demos/assets";
// @ts-expect-error plain .mjs without types
import { parseLicensingTable, readManifest, renderAttribution, resolveSet, shippedGroups, writeTar } from "../../scripts/examples-assets.mjs";

describe("fixtureUrl", () => {
  it("is /fixtures/ in dev and follows BASE_URL on the hosted site", () => {
    expect(fixtureUrl("wdas/x.nvdb", assetBase({ BASE_URL: "/" }))).toBe("/fixtures/wdas/x.nvdb");
    expect(fixtureUrl("wdas/x.nvdb", assetBase({ BASE_URL: "/vdb/examples/" }))).toBe("/vdb/examples/fixtures/wdas/x.nvdb");
  });
  it("VITE_VDB_ASSET_BASE overrides it (one trailing slash)", () => {
    expect(fixtureUrl("a.nvdb", assetBase({ BASE_URL: "/vdb/examples/", VITE_VDB_ASSET_BASE: "https://assets.example.com/vdb//" }))).toBe("https://assets.example.com/vdb/a.nvdb");
  });
});

describe("examples asset manifest", () => {
  const manifest = readManifest();

  it("parses the licensing table's status per row", () => {
    const rows = parseLicensingTable("| Asset | Size | a | Source | Licence | Status |\n|---|---|---|---|---|---|\n| `x.jpg` (+ `-1K`) | 1 | g | | | pending |\n| `Pack/` | 1 | g | s | CC0 | ok |\n");
    expect(rows).toEqual({ "x.jpg": "pending", "Pack/": "ok" });
  });

  it("ships a gallery group only when every licensing row is ok", () => {
    const pending = shippedGroups(manifest, { "sky-midday-HDR.jpg": "pending", "CloudPackVDB_wvdb/": "ok" });
    expect(pending.shipped.some((g: { id: string }) => g.id === "jangafx-cloudpack")).toBe(true);
    expect(pending.shipped.some((g: { id: string }) => g.id === "gallery-sky-midday")).toBe(false);
    const ok = shippedGroups(manifest, { "sky-midday-HDR.jpg": "ok", "CloudPackVDB_wvdb/": "ok" });
    expect(ok.shipped.some((g: { id: string }) => g.id === "gallery-sky-midday")).toBe(true);
  });

  it("never lists the waterfall or a Stanford bunny derivative as shipped", () => {
    const all = manifest.groups.flatMap((g: { files: string[] }) => g.files);
    expect(all.filter((f: string) => /waterfall|bunny/.test(f))).toEqual([]);
  });

  it("every shipped group carries source and licence for the attribution page", () => {
    const html = renderAttribution(manifest, { "CloudPackVDB_wvdb/": "ok" }) as string;
    for (const g of manifest.groups.filter((x: { licensingRows?: string[] }) => !x.licensingRows)) {
      expect(g.source, g.id).toBeTruthy();
      expect(g.licence, g.id).toBeTruthy();
      expect(html).toContain(g.title.replace(/&/g, "&amp;").replace(/</g, "&lt;"));
    }
    expect(html).toContain("CC BY-SA 3.0");
    expect(html).toContain("Modified");
  });

  it("resolveSet enforces the file and total size limits and reports a missing file", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "ex-assets-"));
    const small = { limits: { maxFileMB: 1, maxTotalMB: 2 }, groups: [{ id: "g", files: ["a.bin", "b.bin"] }] };
    mkdirSync(dir, { recursive: true });
    writeFileSync(path.join(dir, "a.bin"), Buffer.alloc(10));
    expect(() => resolveSet({ manifest: small, src: dir, licensing: {} })).toThrow(/missing: b\.bin/);
    writeFileSync(path.join(dir, "b.bin"), Buffer.alloc(1024 * 1024));
    expect(() => resolveSet({ manifest: small, src: dir, licensing: {} })).toThrow(/b\.bin is 1\.0 MB, limit 1 MB/);
    writeFileSync(path.join(dir, "b.bin"), Buffer.alloc(900 * 1024));
    const set = resolveSet({ manifest: small, src: dir, licensing: {} });
    expect(set.files.map((f: { rel: string }) => f.rel)).toEqual(["a.bin", "b.bin"]);
  });

  it("writes a deterministic tar", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "ex-tar-"));
    writeFileSync(path.join(dir, "a.txt"), "hello");
    const files = [{ rel: "a.txt", abs: path.join(dir, "a.txt"), size: 5 }];
    const a = writeTar(files, path.join(dir, "1.tar"));
    const b = writeTar(files, path.join(dir, "2.tar"));
    expect(a).toBe(b);
    expect(readFileSync(path.join(dir, "1.tar")).subarray(257, 262).toString()).toBe("ustar");
  });
});
