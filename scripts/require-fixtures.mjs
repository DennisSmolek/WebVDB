// K1.9: fixture-backed gates skip locally when fixtures are absent, but must
// FAIL in CI. With WEBVDB_REQUIRE_FIXTURES=1 (set by the CI workflow) a missing
// fixture file throws here instead of letting `skipIf` / `test.skip` quietly
// turn the parity gates into no-ops. Env unset: no-op (local skip behaviour
// is unchanged). Used as a vitest globalSetup and a Playwright globalSetup.
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { VDB_SAMPLES as PINNED_VDB_SAMPLES } from "./lib/vdb-samples.mjs";

const root = new URL("../fixtures/", import.meta.url);

const PRIMITIVES = ["box", "sphere", "torus"].flatMap((s) =>
  ["float", "fp8", "fpn"].map((q) => `primitives/${s}_fog_${q}.nvdb`),
);
// Only what `pnpm fixtures:vdb-samples` can fetch in CI (the pinned manifest). Other
// samples (Stanford-bunny derivatives, which are excluded on purpose — see
// fixtures/ATTRIBUTION.md — plus explosion, fire, icosahedron, the point grids)
// are optional local assets; their tests keep skipping when absent.
const VDB_SAMPLES = Object.keys(PINNED_VDB_SAMPLES).map((n) => `vdb-samples/${n}`);

export const REQUIRED_FIXTURES = [...PRIMITIVES, ...VDB_SAMPLES];

export function assertRequiredFixtures(env = process.env) {
  if (env.WEBVDB_REQUIRE_FIXTURES !== "1") return;
  const missing = REQUIRED_FIXTURES.filter((f) => !existsSync(new URL(f, root)));
  if (missing.length > 0) {
    throw new Error(
      `WEBVDB_REQUIRE_FIXTURES=1 but ${missing.length} fixture(s) are missing under ${fileURLToPath(root)}:\n  ` +
        `${missing.join("\n  ")}\nRun pnpm fixtures:bake && pnpm fixtures:vdb-samples (fixture-backed tests would otherwise skip silently).`,
    );
  }
}

export default function setup() {
  assertRequiredFixtures();
}
