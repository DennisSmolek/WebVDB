/**
 * Where the demos fetch their volumes, skies and frames from (RELEASING section 4).
 *
 * Never write a `/fixtures/` literal in a demo: use `fixtureUrl("wdas/x.nvdb")`. The base is
 * `VITE_VDB_ASSET_BASE` when set (an absolute URL or path, e.g. object storage), else Vite's
 * `BASE_URL` + `fixtures/`: `/fixtures/` under `pnpm dev` (the `serveFixtures` plugin) and
 * `/vdb/examples/fixtures/` on the hosted site (`scripts/examples-assets.mjs` copies the set there).
 */

/** The asset base with exactly one trailing slash. */
export function assetBase(env: { BASE_URL?: string; VITE_VDB_ASSET_BASE?: string } = (import.meta as unknown as { env: Record<string, string | undefined> }).env): string {
  const override = env.VITE_VDB_ASSET_BASE;
  const base = override ? override : `${env.BASE_URL ?? "/"}fixtures`;
  return `${base.replace(/\/+$/, "")}/`;
}

/** The URL of a file under `fixtures/`, e.g. `fixtureUrl("primitives/sphere_fog_float.nvdb")`. */
export function fixtureUrl(path: string, base: string = assetBase()): string {
  return `${base}${path.replace(/^\/+/, "")}`;
}

/** True in a production build (the hosted site): demos drop notes that name local-only fixtures. */
export const HOSTED: boolean = Boolean((import.meta as unknown as { env?: { PROD?: boolean } }).env?.PROD);

/** The examples home (the gallery list): `/` in dev, `/vdb/examples/` on the hosted site. */
export const HOME_URL: string = (import.meta as unknown as { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";
