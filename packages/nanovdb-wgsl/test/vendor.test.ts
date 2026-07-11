import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { pnanovdbWgslUrl } from "../src/index.js";

/**
 * Guards the D2 vendoring contract: the WGSL fork is pinned, attributed,
 * and never edited without updating vendor/VENDOR.md's hash + diff log.
 */
const PINNED_COMMIT = "265e8d825e4e4ab8752196a28cccad592d9b4262";
const VENDORED_SHA256 =
  "76021f6a76256cd009f22395af4430d9dbf538e4eb3b7eb32d62b6095054d7e6";

describe("vendored pnanovdb.wgsl", () => {
  it("exists and carries the Apache-2.0 / attribution header", async () => {
    const wgsl = await readFile(pnanovdbWgslUrl, "utf8");
    expect(wgsl).toContain("SPDX-License-Identifier: Apache-2.0");
    expect(wgsl).toContain("Copyright Contributors to the OpenVDB Project");
    expect(wgsl).toContain("Ported to WGSL by Edward McFarlane");
    expect(wgsl).toContain("fn pnanovdb_buf_read_uint32");
  });

  it("matches the hash recorded in VENDOR.md (edit both together)", async () => {
    const wgsl = await readFile(pnanovdbWgslUrl);
    const hash = createHash("sha256").update(wgsl).digest("hex");
    expect(hash).toBe(VENDORED_SHA256);

    const vendorDoc = await readFile(
      new URL("../vendor/VENDOR.md", import.meta.url),
      "utf8",
    );
    expect(vendorDoc).toContain(VENDORED_SHA256);
    expect(vendorDoc).toContain(PINNED_COMMIT);
  });

  it("ships NOTICE and LICENSE next to the WGSL", async () => {
    const notice = await readFile(
      new URL("../vendor/NOTICE", import.meta.url),
      "utf8",
    );
    expect(notice).toContain(PINNED_COMMIT);
    const license = await readFile(
      new URL("../vendor/LICENSE", import.meta.url),
      "utf8",
    );
    expect(license).toContain("Apache License");
  });
});
