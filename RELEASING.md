# Releasing `@pmndrs/vdb`

`packages/vdb/package.json` stays at `0.0.0` in the repo. `.github/workflows/release.yml` refuses to publish
`0.0.0`.

## First publish (0.1.0): manual

Run from a clean checkout of the commit you mean to release, logged in to npm with rights to the `@pmndrs` scope
(`npm whoami`).

1. Bump the version and commit it:
   ```sh
   cd packages/vdb && npm version 0.1.0 --no-git-tag-version
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
4. Tag and push to `pmndrs/vdb`:
   ```sh
   git tag v0.1.0 && git push --tags
   ```
5. In `packages/vdb/CHANGELOG.md`, rename `[Unreleased]` to `[0.1.0] - <date>`, add a fresh empty `[Unreleased]`
   above it, and commit.
6. Set up npm trusted publishing for `@pmndrs/vdb` (npmjs.com, package Settings, Trusted Publisher): publisher
   GitHub Actions, organization `pmndrs`, repository `vdb`, workflow filename `release.yml`, no environment.

## Later releases: `release.yml`

Actions, Release, Run workflow, from `main`. Leave `version` empty to publish what `package.json` says, or give an
exact SemVer (`0.2.0`, `0.2.0-beta.1`). The workflow checks the tag and npm are free, refuses `0.0.0`, runs the full
gate (typecheck, docs:check, unit tests with fixtures required, build, wgsl:size, pack dry-run, packed-subpaths),
publishes with provenance over OIDC (no `NPM_TOKEN`), then pushes the `v<version>` tag. Prereleases publish under
their first identifier as the dist-tag (`beta`, `rc`), never `latest`. Update CHANGELOG before running it.
