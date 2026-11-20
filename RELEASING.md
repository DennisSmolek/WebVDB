# Releasing `@pmndrs/vdb`

## 0.1 publish checklist

Work through it top to bottom. **[owner]** marks a step only Dennis can do: accounts, settings, licences, the
publish itself. **[agent]** steps get a task row in `docs/dev/TASKS.md` when they start. Tick a box when its check passes.
The npm steps at the end follow the procedure in [First publish](#first-publish-010-manual).

### 1. Code frozen for 0.1

- [x] G12a `VDBSequence` range and ping-pong.
- [x] G15a per-axis voxel size from dense.
- [ ] G16a froxel swap for coarse-first streaming. **[agent]**
- [ ] G-shell: one gallery shell, CPU ACES in readback. **[agent]**
- [x] Independent API review of everything since the K5 GO review (75cc1db): the G12a, G15a and G16a signatures,
      JSDoc, api-surface snapshot, CHANGELOG. **[agent, Opus]** GO-with-fixes; fixes in 0657edb.
- [ ] CI green on `pmndrs/vdb` `main` at the release commit.

### 2. Repo cleanup

- [x] Process docs move to `docs/dev/` (TASKS, PHASES, DECISIONS, DIVERGES, design, research, assets-licensing).
      Frozen history moves to `docs/archive/` (the v1 study, handoffs, spikes, the spec draft, STATUS). The root
      keeps README, LICENSE, THIRD_PARTY, CLAUDE.md and this file. **[agent]**
- [x] No broken relative links after the move: a link check over every tracked `.md`/`.mdx`, plus code comments
      that cite paths. **[agent]**
- [x] The root README reads as a public landing page: what it is, install, links to docs, examples and the
      changelog. **[agent]**

### 3. Docs site (pmndrs/docs)

The MDX in `docs/site` already follows the pmndrs/docs conventions (`title`, `description` and `nav` frontmatter;
`<Intro>`). pmndrs libraries build with `npx @pmndrs/docs@4 build <mdx> out --format website` (the reusable
workflow `pmndrs/docs/.github/workflows/build.yml@v4`) and host on GitHub Pages at `https://pmndrs.github.io/<repo>`.

- [ ] Pages workflow `.github/workflows/pages.yml`, triggered on push to `main` and by hand. It builds the docs
      (`mdx: docs/site`, `libname: vdb`, `home_redirect: /getting-started/introduction`, `github`) and the examples
      (§4) into one Pages artifact, then deploys with `actions/deploy-pages`. It needs its own build job, not the
      reusable workflow alone: that uploads the docs as the whole site, and a repo gets one Pages site. Written, not yet run
      on GitHub; its examples steps need the asset release (§4, owner). **[agent]**
- [x] Every page builds and renders (19 pages, no console errors or broken links, `llms-full.txt` emitted; checked 2026-11-20). Check locally with `npx @pmndrs/docs@4 build docs/site out --format website`
      and click through the nav; fix any component pmndrs/docs doesn't have. **[agent]**
- [ ] Icon and logo for the docs header (`icon`, `logo` inputs). **[owner]** picks them.
- [ ] Repo Settings → Pages → Source: **GitHub Actions**. **[owner]** Then turn on the `push: main` trigger in
      `pages.yml`; it runs only by hand until then, so pushes to `main` don't fail.
- [ ] Docs link to the live examples (an `<iframe>` or link per relevant page) and the examples link back.
      **[agent]**
- [ ] After the first deploy: a PR to `pmndrs/docs` adding `@pmndrs/vdb` to `src/libs.ts` so it lists on
      docs.pmnd.rs (`docs_url: https://pmndrs.github.io/vdb`; `llms_full: true` once `/llms-full.txt` returns 200).
      **[owner]** opens or approves it.

### 4. Examples site

The hosted site lives at `https://pmndrs.github.io/vdb/examples/`, in the same Pages artifact as the docs. In dev,
`examples/vite.config.ts` still serves the git-ignored `fixtures/` from disk at `/fixtures/`; the hosted build gets its
volumes from an asset set (`examples/assets.manifest.json`, `scripts/examples-assets.mjs`). Demos build every asset URL
with `fixtureUrl("wdas/x.nvdb")` (`examples/src/demos/assets.ts`): Vite `BASE_URL` + `fixtures/`, or
`VITE_VDB_ASSET_BASE` to point at object storage later.

- [x] Production asset pipeline: an asset base URL for the demos (`fixtureUrl`), and `scripts/examples-assets.mjs`,
      which copies the manifest's file set into the build and fails if any file is ≥ 95 MB or the examples dir passes
      900 MB (the Pages limits are 100 MB and 1 GB). `pnpm examples:build` builds and copies; the 0.1 set is
      327 files, 296.5 MB (298.7 MB site). **[agent]**
- [x] Asset set: everything the demos load **except** `waterfall_points.vdb` (149 MB, it never ships), the Stanford
      bunny derivatives and the pending gallery assets. Demo 14 hosted opens on `sphere_points` and describes the model
      on screen; it offers the waterfall only when present (local dev). Unused files (zips, `downloads/`, raw
      EmberGen packs, frames no demo plays) stay out. **[agent]**
- [x] Attribution page `examples/attribution.html`, generated from the manifest at build time (source, author,
      licence link, "modified" note per asset; WDAS CC BY-SA 3.0, openvdb.org samples CC BY-SA 4.0, EmberGen CC0),
      linked from the examples home and each showcase caption. **[agent]**
- [ ] Gallery assets (HDR skies, the car GLB, `boat_points.wvdb`): every row in `docs/dev/assets-licensing.md` set to `ok`,
      or the asset swapped. The manifest ships a gallery group only when its rows are `ok` (the JangaFX CloudPack
      already is); until then the hosted gallery uses its procedural skies and props. Also decide the Stanford bunny
      (Stanford's terms are not CC BY-SA; excluded for now, demos 10 and 13 skip it). **[owner]**
- [x] Vite `base` for the Pages sub-path (`/vdb/examples/`, `VDB_EXAMPLES_BASE` overrides); every demo loads from the
      built site. `pnpm examples:build && pnpm examples:smoke` serves the build under `/vdb/examples/` and runs
      `e2e/examples-site.spec.ts` (home and attribution links, all 16 demos reach their ready state without errors and
      request nothing outside `/vdb/`, no-WebGPU card). **[agent]**
- [x] A clear message on browsers without WebGPU, instead of a blank canvas: `examples/src/demos/webgpu-gate.ts`,
      imported first by every demo (`e2e/no-webgpu.spec.ts`). **[agent]**
- [ ] Publish the asset tarball (the CI cannot fetch WDAS and friends), then run the workflow by hand. **[owner]**
      ```sh
      node scripts/examples-assets.mjs --pack        # needs the local fixtures; writes artifacts/examples-assets/
      # put the printed sha256 into examples/assets.manifest.json (release.sha256) and commit it
      gh release create examples-assets-v1 artifacts/examples-assets/examples-assets.tar \
        --repo pmndrs/vdb --title "Examples assets v1" \
        --notes "Volumes for the hosted examples (scripts/examples-assets.mjs --pack). Licences: examples/attribution.html."
      ```
      `pages.yml` downloads `examples-assets.tar` from that release, checks its sha256 against the manifest and fails
      (no deploy) when the release, the asset or the checksum is missing. A new set means a new tag
      (`examples-assets-v2`) and the tag and sha256 in the manifest.
- [ ] Bandwidth: Pages has a soft 100 GB/month limit and the WDAS demos pull 30–90 MB per visit. Fine for launch;
      if traffic grows, move the volumes to object storage behind the same asset base URL
      (`VITE_VDB_ASSET_BASE` at build time). **[owner]** decides when.

### 5. Licences and credits

- [ ] `THIRD_PARTY.md` and `packages/vdb/THIRD_PARTY_NOTICES.md` current (vendored WGSL, NanoVDB, mjurczyk
      port, neural weights credit, Wyman CMF fit). **[agent]** checks.
- [ ] mjurczyk outreach: his LICENSE file, the neural weights' training data, the CIE table. Not blocking: his
      package metadata declares MIT and we credit him. **[owner]**

### 6. npm

- [ ] Your npm account can publish under the `@pmndrs` scope (ask a pmndrs npm org owner) and has 2FA on.
      `npm view @pmndrs/vdb` returns 404 (the name is free). **[owner]**
- [ ] `packages/vdb/README.md` links the docs and examples URLs; `homepage` in `package.json` points at the docs.
      **[agent]**
- [ ] CHANGELOG `Unreleased` lists every change since the last review, then gets dated in the release commit.
- [ ] First publish by hand, steps 1–4 below. **[owner]**
- [ ] Trusted publishing, step 5 below; later releases run `release.yml`. **[owner]**

### 7. After the publish

- [ ] `pmndrs/vdb` default branch is `main`; delete `v2` once nothing points at it. **[owner]**
- [ ] Docs and examples live; docs.pmnd.rs PR merged.
- [ ] Announce (pmndrs Discord, the rendering initiative). **[owner]**

## First publish (0.1.0): manual

`packages/vdb/package.json` is `0.0.0` until the first release commit bumps it (step 1).
`.github/workflows/release.yml` refuses to publish `0.0.0`. The release remote is `pmndrs` (github.com/pmndrs/vdb);
add it if needed: `git remote add pmndrs https://github.com/pmndrs/vdb.git`.

Run from a clean checkout of the commit you mean to release, logged in to npm with rights to the `@pmndrs` scope
(`npm whoami`).

1. Bump the version, date the CHANGELOG, and commit both (the CHANGELOG ships in the tarball, so this happens
   before publishing): in `packages/vdb/CHANGELOG.md` rename `## [Unreleased]` to `## [0.1.0] - <YYYY-MM-DD>` and add
   a fresh empty `## [Unreleased]` above it, then:
   ```sh
   cd packages/vdb && npm version 0.1.0 --no-git-tag-version && cd ../..
   git commit -am "chore(release): 0.1.0"
   ```
2. Run the gate from the repo root (fixtures must be present: `pnpm fixtures:bake && pnpm fixtures:vdb-samples`, and
   set `WEBVDB_REQUIRE_FIXTURES=1` so a missing fixture fails instead of skipping):
   ```sh
   pnpm install --frozen-lockfile && pnpm typecheck && pnpm docs:check && pnpm test && pnpm build \
     && pnpm wgsl:size && pnpm pack:dry && pnpm verify:packed-subpaths
   ```
3. Publish. Do not pass `--provenance`: it needs the CI OIDC identity and fails from a laptop.
   ```sh
   cd packages/vdb && npm publish --access public
   ```
4. Push the release commit and tag to `pmndrs/vdb` (later releases run `release.yml` from its `main`):
   ```sh
   git tag v0.1.0
   git push pmndrs HEAD:main && git push pmndrs v0.1.0
   ```
5. Set up npm trusted publishing for `@pmndrs/vdb` (npmjs.com, package Settings, Trusted Publisher): publisher
   GitHub Actions, organization `pmndrs`, repository `vdb`, workflow filename `release.yml`, no environment.

## Later releases: `release.yml`

Actions, Release, Run workflow, from `main`. Leave `version` empty to publish what `package.json` says, or give an
exact SemVer (`0.2.0`, `0.2.0-beta.1`). The workflow checks the tag and npm are free, refuses `0.0.0`, runs the full
gate (typecheck, docs:check, unit tests with fixtures required, build, wgsl:size, pack dry-run, packed-subpaths),
publishes with provenance over OIDC (no `NPM_TOKEN`), then pushes the `v<version>` tag. Prereleases publish under
their first identifier as the dist-tag (`beta`, `rc`), never `latest`. Update CHANGELOG before running it.
