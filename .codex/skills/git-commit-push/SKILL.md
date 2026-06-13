---
name: git-commit-push
description: Local repository workflow for Git commits and pushes. Use when the user asks Codex to commit, push, prepare commits, split work into commits, or when a push request should include committing completed work first. Enforces committing by TODO/task unit and treating a push request as including any necessary commits.
---

# Git Commit Push

Use this skill for commit and push operations in this repository.

## Core Rules

- Split commits by TODO or task unit.
- If three independent TODOs were completed in one session, create three separate commits.
- Do not combine unrelated changes into one commit just because they were implemented together.
- When the user asks to push, include committing completed uncommitted work before pushing, unless the user explicitly says not to commit.
- If there is nothing to commit, report that clearly and push only if there are local commits not on the remote.
- Do not include user-made unrelated changes in your commits.

## Workflow

1. Inspect `git status --short` and relevant diffs.
2. Identify completed TODO/task units.
3. Map changed files or hunks to each task unit.
4. Stage and commit one task unit at a time.
5. Use Japanese commit messages unless the repository convention says otherwise.
6. If the user requested push, run `git push` after all required commits are created.
7. Report the created commits and push result.

## Commit Boundaries

Prefer one commit per meaningful completed item:

- Project setup
- Rule implementation
- Test addition
- Documentation update
- Refactor needed for a specific feature

Keep a test in the same commit as the behavior it verifies when they belong to the same TODO. Separate tests only when the TODO itself is specifically test-only.

## Safety

- Review diffs before staging.
- Use path-specific staging when possible.
- If a file contains both your changes and unrelated user changes, stage only the relevant hunks or ask before proceeding if safe staging is not practical.
- Never rewrite history, amend, reset, or force-push unless the user explicitly requests it.
- Before committing implementation changes, run the relevant tests when feasible. If tests are skipped, state why.
