# Development docs

Living process docs for `@pmndrs/vdb`: [PHASES.md](./PHASES.md) (the plan), [TASKS.md](./TASKS.md) (open work),
[DECISIONS.md](./DECISIONS.md) (settled choices), [DIVERGES.md](./DIVERGES.md) (where we differ from a source),
[assets-licensing.md](./assets-licensing.md), [design/](./design/) and [research/](./research/).
Frozen history is in [../archive/](../archive/). Agent rules are in [CLAUDE.md](../../CLAUDE.md).

## Repo layout

- [`packages/vdb`](../../packages/vdb): the published package (`src/core`, `src/tools`, `src/three`, `vendor/`)
- [`docs/site`](../site): the pmndrs/docs MDX pages (snippets compiled by `pnpm docs:check`)
- [`examples`](../../examples): the Vite demo site (ten demos; the cloud, sequence, explorer, builder and the froxel
  renderer showcase, demo 10, are the place to start), plus a gallery of showcase pages:
  - Gallery 11 · sky clouds: the WDAS cloud at full resolution and three JangaFX clouds under an HDR sky
  - Gallery 12 · explosion: an EmberGen explosion sequence in one dense source, its flames lighting the ground and props
  - Gallery 13 · level sets: a sculpture gallery of OpenVDB level sets traced at full resolution (`VDBLevelSetMaterial`)
  - Gallery 14 · points: the 9.5 M-point waterfall decoded by our parser in a worker (`VDBPoints`)
  - Gallery 15 · full res vs 128³: sparse full resolution against a dense `.wvdb` bake, split by a divider
  - Gallery 16 · streaming: coarse-first `.nvdb` into the froxel renderer, a ~14 KiB preview then the full grid in place
- [`DECISIONS.md`](./DECISIONS.md), [`DIVERGES.md`](./DIVERGES.md), [`THIRD_PARTY.md`](../../THIRD_PARTY.md)

## Commands

```sh
pnpm install
pnpm typecheck       # tsc --build across the workspace
pnpm test            # Vitest (fixture-backed tests skip when fixtures are missing)
pnpm test:e2e        # Playwright against the examples app (needs a WebGPU browser)
pnpm dev             # examples dev server, http://localhost:5173
pnpm build           # the published dist/
pnpm docs:check      # docs snippets are current and compile
pnpm pack:dry        # what npm would publish
pnpm fixtures        # download test volumes into fixtures/ (git-ignored)
```

In-repo imports resolve to TypeScript source through tsconfig `paths` and Vite/Vitest aliases; `pnpm build` is only
needed for the `dist/` that ships. Fixtures are never committed; see [fixtures/README.md](../../fixtures/README.md).
Version 0.1.0 status: [TASKS.md](./TASKS.md) and [RELEASING.md](../../RELEASING.md).
