# Releasing `@pmndrs/vdb`

`packages/vdb/package.json` is `0.0.0` until the first release commit bumps it (step 1).
`.github/workflows/release.yml` refuses to publish `0.0.0`. The release remote is `pmndrs` (github.com/pmndrs/vdb);
add it if needed: `git remote add pmndrs https://github.com/pmndrs/vdb.git`.

## First publish (0.1.0): manual

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
