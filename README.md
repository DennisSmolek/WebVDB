# @pmndrs/vdb

Sparse NanoVDB volumes on WebGPU, **browser-first**: traverse and render sparse volumes directly in WGSL (three.js
`WebGPURenderer` + TSL), with pure-TypeScript CPU tooling for parsing, building and inspecting grids. No dense-bake
tricks, no WASM. Part of the pmndrs rendering initiative.

One package, three entry points:

| Import | What it is |
|---|---|
| `@pmndrs/vdb` | The engine-free WebGPU core: `VDBGridPool`, the generated `vdb*` WGSL module, GPU compute tools |
| `@pmndrs/vdb/tools` | CPU side: parse `.vdb`, build and quantize NanoVDB, read and write `.nvdb` and `.wvdb`, CPU reads |
| `@pmndrs/vdb/three` | The three.js (r187) adapter: volume material, froxel renderer, level sets, points, sequences |

```sh
npm install @pmndrs/vdb          # core + tools
npm install @pmndrs/vdb three    # to use @pmndrs/vdb/three
```

Start with the [package README](./packages/vdb/README.md), then the docs in [`docs/site`](./docs/site).

## Status

Version **0.1.0** is being prepared: the plan ([PHASES.md](./PHASES.md)) is built through K4 and K5 (release prep
and the docs site); the on-device gates passed. Open work is in [TASKS.md](./TASKS.md), current state in
[STATUS.md](./STATUS.md). To cut a release see [RELEASING.md](./RELEASING.md); changes are in
[packages/vdb/CHANGELOG.md](./packages/vdb/CHANGELOG.md).

## Repo layout

- [`packages/vdb`](./packages/vdb): the published package (`src/core`, `src/tools`, `src/three`, `vendor/`)
- [`docs/site`](./docs/site): the pmndrs/docs MDX pages (snippets compiled by `pnpm docs:check`)
- [`examples`](./examples): the Vite demo site (ten demos; the cloud, sequence, explorer, builder and the froxel
  renderer showcase, demo 10, are the place to start), plus a gallery of showcase pages:
  - Gallery 11 · sky clouds: the WDAS cloud at full resolution and three JangaFX clouds under an HDR sky
  - Gallery 12 · explosion: an EmberGen explosion sequence in one dense source, its flames lighting the ground and props
  - Gallery 13 · level sets: a sculpture gallery of OpenVDB level sets traced at full resolution (`VDBLevelSetMaterial`)
  - Gallery 14 · points: the 9.5 M-point waterfall decoded by our parser in a worker (`VDBPoints`)
  - Gallery 15 · full res vs 128³: sparse full resolution against a dense `.wvdb` bake, split by a divider
  - Gallery 16 · streaming: coarse-first `.nvdb` into the froxel renderer, a ~14 KiB preview then the full grid in place
- [`DECISIONS.md`](./DECISIONS.md) (settled choices, D1 onward), [`DIVERGES.md`](./DIVERGES.md) (where we differ
  from a source), [`THIRD_PARTY.md`](./THIRD_PARTY.md) (licences)

## Development

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
needed for the `dist/` that ships. Fixtures are never committed; see [fixtures/README.md](./fixtures/README.md).

## Licence

MIT (see [LICENSE](./LICENSE)). The vendored `pnanovdb.wgsl` is Apache-2.0 and the renderer and `.wvdb` code
derive from [mjurczyk/openvdb](https://github.com/mjurczyk/openvdb) (MIT); see [THIRD_PARTY.md](./THIRD_PARTY.md)
and the shipped `packages/vdb/THIRD_PARTY_NOTICES.md`.
