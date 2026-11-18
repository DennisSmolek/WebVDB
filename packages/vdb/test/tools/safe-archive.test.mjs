import { spawnSync } from "node:child_process";
import { link, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  assertSafeArchiveMembers,
  validateExtractedTree,
} from "../../../../scripts/lib/safe-archive.mjs";

const cleanup = [];

async function tempDirectory() {
  const directory = await mkdtemp(path.join(tmpdir(), "webvdb-safe-archive-"));
  cleanup.push(directory);
  return directory;
}

afterEach(async () => {
  await Promise.all(cleanup.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe("EmberGen archive safety", () => {
  it("accepts an ordinary ZIP and rejects a symlink member before extraction", async () => {
    const directory = await tempDirectory();
    await writeFile(path.join(directory, "frame.vdb"), "fixture");
    const safeArchive = path.join(directory, "safe.zip");
    expect(spawnSync("zip", ["-q", safeArchive, "frame.vdb"], { cwd: directory }).status).toBe(0);
    expect(assertSafeArchiveMembers("zip", safeArchive)).toContain("frame.vdb");

    await symlink("frame.vdb", path.join(directory, "linked.vdb"));
    const linkedArchive = path.join(directory, "linked.zip");
    expect(
      spawnSync("zip", ["-q", "-y", linkedArchive, "linked.vdb"], { cwd: directory }).status,
    ).toBe(0);
    expect(() => assertSafeArchiveMembers("zip", linkedArchive)).toThrow(/member type|unsafe/i);
  });

  it("recursively rejects symlinks, hardlinks, and special files", async () => {
    const directory = await tempDirectory();
    const original = path.join(directory, "frame.vdb");
    await writeFile(original, "fixture");

    await symlink("frame.vdb", path.join(directory, "symlink.vdb"));
    await expect(validateExtractedTree(directory)).rejects.toThrow(/symbolic link/i);
    await rm(path.join(directory, "symlink.vdb"));

    await link(original, path.join(directory, "hardlink.vdb"));
    await expect(validateExtractedTree(directory)).rejects.toThrow(/hard-linked/i);
    await rm(path.join(directory, "hardlink.vdb"));

    const fifo = path.join(directory, "special.vdb");
    expect(spawnSync("mkfifo", [fifo]).status).toBe(0);
    await expect(validateExtractedTree(directory)).rejects.toThrow(/special file/i);
  });
});
