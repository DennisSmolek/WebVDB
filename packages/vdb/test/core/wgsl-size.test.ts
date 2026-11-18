import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { vdbModuleWgsl, type VDBModuleOptions } from "../../src/core/index.js";
// @ts-expect-error: repo-level ESM helper without type declarations
import { variantMatrix, wgslSizes } from "../../../../scripts/lib/wgsl-variants.mjs";

// Snapshot and budgets written by `pnpm wgsl:size --write` (scripts/wgsl-size.mjs).
const SNAPSHOT = JSON.parse(readFileSync(new URL("../../src/core/wgsl/size-snapshot.json", import.meta.url), "utf8")) as {
  unit: string;
  variants: Record<string, { gz: number; min: number }>;
};
const BUDGETS = JSON.parse(readFileSync(new URL("../../src/core/wgsl/size-budgets.json", import.meta.url), "utf8")) as {
  bytesPerUnit: number;
  profiles: Record<string, Record<"float" | "oneFp" | "floatFp8Fpn", number>>;
};

const VARIANTS: { id: string; options: VDBModuleOptions }[] = variantMatrix();

/** The §3.6 row for a variant (see `rule` in size-budgets.json), in bytes. */
function budgetBytes(options: VDBModuleOptions): number {
  const helpers = [...(options.helpers ?? [])];
  const key = helpers.includes("nodeAverage")
    ? `${options.profile}+nodeAverage`
    : helpers.length > 0
      ? `${options.profile}+helpers`
      : options.profile === "read" && options.isActive === false ? "read.isActive=false" : options.profile;
  const row = BUDGETS.profiles[key];
  if (!row) throw new Error(`no budget row for ${key}`);
  const fpCount = [...options.types].filter((t) => t !== "float").length;
  const column = fpCount === 0 ? "float" : fpCount === 1 ? "oneFp" : "floatFp8Fpn";
  return row[column] * BUDGETS.bytesPerUnit;
}

const changed = (id: string, was: number, now: number) =>
  `WGSL size changed for ${id}: ${was} -> ${now}. If intended, run \`pnpm wgsl:size --write\` and commit.`;

describe("WGSL variant sizes (§3.5 snapshot, §3.6 budgets)", () => {
  it("the snapshot covers exactly the variant matrix", () => {
    const matrixIds = VARIANTS.map((v) => v.id).sort();
    expect(Object.keys(SNAPSHOT.variants).sort()).toEqual(matrixIds);
  });

  it.each(VARIANTS)("$id: min bytes match the snapshot and fit the budget", ({ id, options }) => {
    const { min } = wgslSizes(vdbModuleWgsl(options));
    const snap = SNAPSHOT.variants[id];
    expect(snap, `no snapshot entry for ${id}; run \`pnpm wgsl:size --write\``).toBeDefined();
    expect(min, changed(id, snap!.min, min)).toBe(snap!.min);
    const budget = budgetBytes(options);
    expect(min, `${id}: ${min} B exceeds the §3.6 budget of ${budget} B (${budget / BUDGETS.bytesPerUnit} KiB)`).toBeLessThanOrEqual(budget);
  });
});
