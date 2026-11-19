# Releasing `@pmndrs/vdb`

## 0.1 publish checklist

Work through it top to bottom. **[owner]** marks a step only Dennis can do: accounts, settings, licences, the
publish itself. **[agent]** steps get a task row in `TASKS.md` when they start. Tick a box when its check passes.
The npm steps at the end follow the procedure in [First publish](#first-publish-010-manual).

### 1. Code frozen for 0.1

- [x] G12a `VDBSequence` range and ping-pong.
- [x] G15a per-axis voxel size from dense.
- [ ] G16a froxel swap for coarse-first streaming. **[agent]**
- [ ] G-shell: one gallery shell, CPU ACES in readback. **[agent]**
- [ ] Independent API review of everything since the K5 GO review (75cc1db): the G12a, G15a and G16a signatures,
      JSDoc, api-surface snapshot, CHANGELOG. **[agent, Opus]**
- [ ] CI green on `pmndrs/vdb` `main` at the release commit.

### 2. Repo cleanup

- [ ] Process docs move to `docs/dev/` (TASKS, PHASES, DECISIONS, DIVERGES, design, research, assets-licensing).
      Frozen history moves to `docs/archive/` (the v1 study, handoffs, spikes, the spec draft, STATUS). The root
      keeps README, LICENSE, THIRD_PARTY, CLAUDE.md and this file. **[agent]**
- [ ] No broken relative links after the move: a link check over every tracked `.md`/`.mdx`, plus code comments
      that cite paths. **[agent]**
- [ ] The root README reads as a public landing page: what it is, install, links to docs, examples and the
      changelog. **[agent]**

### 3. Docs site (pmndrs/docs)

The MDX in `docs/site` already follows the pmndrs/docs conventions (`title`, `description` and `nav` frontmatter;
`<Intro>`). pmndrs libraries build with `npx @pmndrs/docs@4 build <mdx> out --format website` (the reusable
workflow `pmndrs/docs/.github/workflows/build.yml@v4`) and host on GitHub Pages at `https://pmndrs.github.io/<repo>`.

- [ ] Pages workflow `.github/workflows/pages.yml`, triggered on push to `main` and by hand. It builds the docs
      (`mdx: docs/site`, `libname: vdb`, `home_redirect: /getting-started/introduction`, `github`) and the examples
      (§4) into one Pages artifact, then deploys with `actions/deploy-pages`. It needs its own build job, not the
      reusable workflow alone: that uploads the docs as the whole site, and a repo gets one Pages site. Written, not yet run
      on GitHub; its examples step is a placeholder (`if: false`) until §4. **[agent]**
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

The examples build ships no fixtures today: in dev, `examples/vite.config.ts` serves the git-ignored `fixtures/`
from disk. A hosted build needs an asset set.

- [ ] Production asset pipeline: an asset base URL for the demos, and a script that copies the allowed fixture
      subset into the build. Fail the build if any file is ≥ 100 MB (the GitHub Pages limit) or the site passes
      ~900 MB (the limit is 1 GB). **[agent]**
- [ ] Asset set: everything the demos load **except** `waterfall_points.vdb` (149 MB, too large). Demo 14 hosted
      opens on a smaller points asset and offers the waterfall only when it is present (local dev). Unused files
      (zips, `downloads/`, the full EmberGen pack beyond the frames a demo plays) stay out. **[agent]**
- [ ] Attribution page on the site, built from `fixtures/ATTRIBUTION.md` and `docs/dev/assets-licensing.md`. Every
      hosted asset is listed with source and licence (openvdb.org samples and the WDAS cloud are CC BY-SA and need
      it; EmberGen is CC0). **[agent]**
- [ ] Gallery assets (HDR skies, GLBs, `boat_points.wvdb`): every row in `docs/assets-licensing.md` set to `ok`,
      or the asset swapped. Until then the hosted gallery uses its procedural fallbacks. **[owner]**
- [ ] Vite `base` set for the Pages sub-path (docs at `/vdb/`, examples at `/vdb/examples/`); every demo loads from
      the built site, not only from `pnpm dev`. Check with `vite preview` plus an e2e smoke run against the build.
      **[agent]**
- [ ] A clear message on browsers without WebGPU, instead of a blank canvas. **[agent]**
- [ ] Bandwidth: Pages has a soft 100 GB/month limit and the WDAS demos pull 30–90 MB per visit. Fine for launch;
      if traffic grows, move the volumes to object storage behind the same asset base URL. **[owner]** decides when.

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
