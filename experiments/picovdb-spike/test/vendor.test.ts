import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

/** Guards the vendoring contract in vendor/VENDOR.md (verbatim files, pinned commit). */
const PINNED_COMMIT = "f5d22e5602a7fc7f206d33ec35328f58e8667c83";
const HASHES: Record<string, string> = {
  "picovdb.wgsl": "78f31ed0ef15f855a13e2a099505585665cfa47c0711c727989f6a633dcfbde0",
  "picovdb.ts": "c736e64f27e71cc157c03bf2c7c4ba4820b514dce1a85dac5ea0df7cbbee738e",
  LICENSE: "6076ff3d4be9f35cc29763cb350b1008efb58dfa826fca6dd3013376977cabfb",
};

const vendor = (name: string) => new URL(`../vendor/${name}`, import.meta.url);

describe("vendored picovdb files", () => {
  it.each(Object.entries(HASHES))("%s matches the hash recorded in VENDOR.md", async (name, hash) => {
    const bytes = await readFile(vendor(name));
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(hash);
    const doc = await readFile(vendor("VENDOR.md"), "utf8");
    expect(doc).toContain(hash);
    expect(doc).toContain(PINNED_COMMIT);
  });

  it("LICENSE is Apache-2.0 and NOTICE attributes upstream at the pin", async () => {
    expect(await readFile(vendor("LICENSE"), "utf8")).toContain("Apache License");
    const notice = await readFile(vendor("NOTICE"), "utf8");
    expect(notice).toContain("Edward McFarlane");
    expect(notice).toContain(PINNED_COMMIT);
  });
});
