---
name: release
description: >-
  Cut a release of the `machine-learning` library — ship to npm and deploy the playground site, in
  the correct, safe order. Use this whenever the user wants to release, publish, "cut a version",
  "ship it", bump the version, tag a release, or "merge and publish" for this repo — even if they
  only say "publish" or "release the new chapter". Runs the full ordered flow (feature branch →
  commit → PR → CI → version bump → merge → tag → automated npm publish) and bakes in the ordering
  gotchas so a release can never half-ship.
---

# Releasing the machine-learning project

A release has **two independent automations**, and the whole trick is firing them in the right order:

- **Merging a PR to `master` deploys the site** (`deploy-site.yml`, on pushes touching `site/**` or `src/lib/**`).
- **Pushing a `vX.Y.Z` tag publishes to npm** (`publish.yml`: type-check → test → build → `npm publish`, using the `NPM_TOKEN` secret).

So: **merge → site, tag → npm.** You never run `npm publish` by hand, and you never deploy the site by hand. You drive git; CI does the publishing. The job of this skill is to get the commits, the version bump, and the tag onto `master` in an order where neither automation can fire against a half-finished state.

## The flow

Assumes there's unreleased work in the tree (a finished chapter/feature). Adapt if a PR already exists.

1. **Branch.** Never commit the release directly to `master` — protected, and PRs are how CI gates it. If the work is sitting on `master`, move it: `git checkout -b <feature-name>` carries uncommitted changes with you.
2. **Commit & push** the feature work, then **open a PR** into `master`:
   ```bash
   git add -A && git commit -m "<chapter/feature summary>"
   git push -u origin <feature-name>
   gh pr create --base master --head <feature-name> --title "…" --body "…"
   ```
3. **Bump the version *on the branch*** so it reaches `master` through the merge (avoids a separate direct-to-master commit). Edit `package.json` `version` and commit it as the release marker — this is also what makes the tag publishable:
   ```bash
   # bump package.json: e.g. 0.7.0 → 0.8.0
   git commit -am "Release X.Y.Z: <one-line summary>"
   git push
   ```
   Versioning convention (still 0.x, "early development"): a **minor bump per batch of new algorithms** — 0.5.0 k-means, 0.6.0 trees, 0.7.0 the unsupervised suite, 0.8.0 CNNs. A single new chapter is a minor bump.
4. **Wait for CI to pass**, then **merge** (this is an outward, hard-to-undo step — confirm with the user first):
   ```bash
   gh pr checks <pr-number> --watch    # ci.yml (typecheck+test) + site-ci.yml must go green
   gh pr merge <pr-number> --merge --delete-branch
   ```
   The merge auto-deploys the site. Nothing else to do for the site.
5. **Tag `master` and push** — this is what publishes to npm (also outward/irreversible — confirm first):
   ```bash
   git checkout master && git pull --ff-only
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```
6. **Verify** (see below).

If the user only wants part of this (e.g. "just merge", or the PR's already merged and they want to publish), do only the relevant steps — but keep the ordering rules.

## Ordering rules that matter (learned the hard way)

These exist because each one, skipped, produces a release that *looks* fine but is broken:

- **Tag only AFTER merging**, and tag a commit on `master`. A tag on the feature branch (or on a pre-merge commit) publishes the wrong tree.
- **The tagged commit must contain BOTH the bumped version AND `publish.yml`.** If `package.json` still says the previous version, `npm publish` fails (that version already exists). If `publish.yml` isn't in that commit's tree, the tag triggers nothing. (We once merged a PR *before* the `publish.yml` commit was pushed, so it never reached `master` and had to be re-added in a follow-up PR — and an early `v0.7.0` tag predated `publish.yml`, so 0.7.0 had to be published by hand.)
- **Don't merge the PR until every intended commit — including the version bump — is on the branch.** Merging mid-push orphans whatever landed late.
- **Merge and tag are outward and effectively irreversible** (a deployed site, a published npm version you can't overwrite). Confirm with the user before each unless they've clearly said "do the whole release".

## Verifying it landed

- **Site:** `gh run list --workflow="Deploy site" --branch master --limit 1` → success.
- **npm:** the publish run must be green: `gh run watch <run-id>` (find it via `gh run list --workflow=publish.yml --limit 1`).
- **Confirm the version is actually live — query it directly**, because `npm view <pkg> version` / `latest` can serve a stale cached value for a minute:
  ```bash
  npm view machine-learning@X.Y.Z version        # should echo X.Y.Z
  curl -s https://registry.npmjs.org/machine-learning | \
    node -e "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>console.log(JSON.parse(s)['dist-tags'].latest))"
  ```

## Repo reference

- `README.md` → **"Releasing"** section (the human-facing summary of this flow).
- `.github/workflows/publish.yml` — tag-triggered npm publish (needs `NPM_TOKEN` secret).
- `.github/workflows/deploy-site.yml` — master-push site deploy to GitHub Pages.
- `.github/workflows/ci.yml` — type-check + test (library), on PRs and master.
- `.github/workflows/site-ci.yml` — site build, on PRs touching `site/` or `src/lib/`.
- `package.json` — `version` (bump this) and `prepublishOnly: yarn build` / `files: ["dist/lib"]` (so `npm publish` ships only the compiled library).

> Commit-message convention for the bump: `Release X.Y.Z: <summary>`. Co-author git commits per the repo/global convention.
