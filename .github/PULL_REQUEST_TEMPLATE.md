<!--
Thanks for the PR! Please fill in the sections below.
If this is a docs-only or trivial change, feel free to delete sections
that don't apply.
-->

## Summary

<!-- One paragraph: what changed and why. -->

## Type of change

- [ ] Bug fix (no public API change)
- [ ] New feature (no breaking change)
- [ ] Breaking change (signature change, behavior change, removed export)
- [ ] Docs / tests / tooling only

## Checklist

- [ ] `uv run ruff check libs/ tests/` is clean
- [ ] `uv run ruff format --check libs/ tests/` is clean
- [ ] `uv run mypy libs/llmagpie` is clean
- [ ] `uv run pytest` passes (and coverage stays ≥ 75%)
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` (for user-visible changes)
- [ ] If fixing a bug: a regression test in `tests/test_basics.py`

## Notes for reviewers

<!-- Anything tricky, anything you're unsure about, anything to look at first. -->
