# @pmndrs/vdb

Sparse NanoVDB volumes on WebGPU, browser-first: traverse and render OpenVDB and NanoVDB volumes directly in WGSL,
with no dense-bake step and no WASM. Part of the pmndrs rendering initiative.

**In the box:** an engine-free WebGPU **core** (`VDBGridPool`, the generated `vdb*` WGSL module, GPU compute tools),
pure-TypeScript **tools** (parse `.vdb`, build and quantize NanoVDB, read and write `.nvdb` and `.wvdb`), and a
**three.js** adapter (volume material, froxel renderer, level sets, points, sequences).

| Import | What it is |
|---|---|
| `@pmndrs/vdb` | The engine-free WebGPU core |
| `@pmndrs/vdb/tools` | CPU side: parsing, building, quantizing, file I/O |
| `@pmndrs/vdb/three` | The three.js (r187) adapter |

## Install

```sh
npm i @pmndrs/vdb three
```

`three` is needed only for `@pmndrs/vdb/three`; `npm i @pmndrs/vdb` alone gives the core and tools. A first render
is in the docs: [First render](https://pmndrs.github.io/vdb/getting-started/first-render).

## Links

- [Docs](https://pmndrs.github.io/vdb/) and [live examples](https://pmndrs.github.io/vdb/examples/)
- [Package README](./packages/vdb/README.md) (the API tour) and [CHANGELOG](./packages/vdb/CHANGELOG.md)
- [Contributing and dev docs](./docs/dev/): plan, tasks, decisions, repo layout, how to run the tests
- [RELEASING.md](./RELEASING.md)

## Licence and credits

MIT (see [LICENSE](./LICENSE)). The renderer and `.wvdb` code derive from
[mjurczyk/openvdb](https://github.com/mjurczyk/openvdb) (MIT), and the vendored `pnanovdb.wgsl` and `PNanoVDB.h`
are Apache-2.0. Full credits and licences: [THIRD_PARTY.md](./THIRD_PARTY.md) and the shipped
`packages/vdb/THIRD_PARTY_NOTICES.md`.
