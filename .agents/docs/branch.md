# Branch Workflow

## Why

Predictable branch names expose intent and issue linkage without relying on local context.

## Naming

Repository-specific naming takes precedence over the library default.
Feature branches use:

```text
feat/ISSUE-<issue-number>
```

Examples:

```text
feat/ISSUE-11
```

Issue number rules:
- digits only
- must reference the primary issue for the branch
- create or identify the issue before creating the branch

No repository-specific naming convention is currently defined for non-feature
branches. Do not invent one without explicit agreement.

## Lifecycle

- Branch from repository default branch unless project policy says otherwise.
- Keep one primary issue per branch.
- Sync with default branch before final validation when divergence matters.
- Never force-push a shared branch without coordination.
- Delete branch after merge when no follow-up work depends on it.
