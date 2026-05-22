# CI/CD Pipeline

This project currently uses GitHub Actions to publish release builds to PyPI.
The active workflow is:

- `.github/workflows/publish.yml`

## Local Validation

Run these checks before opening a pull request or publishing a release:

```bash
pip install -e ".[test]"
pytest -m "not slow"
python -m build
```

Use Python 3.11 or newer. `pytest -m "not slow"` validates the regular solver,
evaluator, graph transformation, and package export behavior while skipping
long-running tests. Run the slow tests separately only when release risk or
algorithm changes justify the extra runtime. `python -m build` verifies that
the source distribution and wheel can be produced from `pyproject.toml`.

## Release Publishing Flow

Publishing is triggered when a GitHub release is published.

1. Create and push a release tag that uses semantic version format with a `v`
   prefix, such as `v0.1.0`.
2. Publish a GitHub release for that tag.
3. GitHub Actions checks out the tagged revision.
4. The workflow validates that the release tag matches `vMAJOR.MINOR.PATCH`.
5. The workflow installs `build`, runs `python -m build`, and uploads the
   generated distribution packages to PyPI.

The package version is derived from Git tags through `hatch-vcs`, configured in
`pyproject.toml`. The workflow therefore publishes the version represented by
the release tag, without manually editing a version string.

## Permissions and Secrets

The publish job uses PyPI trusted publishing through GitHub Actions OIDC:

- `contents: read` allows the workflow to read the repository at the release
  tag.
- `id-token: write` allows the PyPI publish action to request an OIDC token.

Do not add PyPI API tokens or credentials to the repository. If publishing
fails because PyPI does not trust this repository or workflow, update the PyPI
project's trusted publisher configuration rather than hardcoding credentials.

## Artifact Policy

Build outputs are generated into `dist/` by `python -m build`. Treat `dist/` as
ephemeral output:

- do not edit files in `dist/` manually
- do not commit generated distributions unless a release process explicitly
  requires it
- rebuild artifacts from a clean tag when diagnosing publishing problems

## Future CI Checks

There is no pull-request CI workflow in this repository yet. If one is added,
it should mirror the local validation commands above and run at least:

```bash
pip install -e ".[test]"
pytest -m "not slow"
python -m build
```

Document any added workflow files here so contributors can keep local and CI
behavior aligned.
