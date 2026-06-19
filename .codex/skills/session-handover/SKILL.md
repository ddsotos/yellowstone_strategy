---
name: session-handover
description: Generate concise session handover notes for this repository. Use when the user asks to hand over work, create a handover note, end a session with notes, summarize the current work for the next session, or record what changed and what should be done next.
---

# Session Handover

Create a handover note at a work boundary so the next session can resume without repeating context gathering or design discussions.

## Workflow

1. Review the current session:
   - inspect recent conversation context;
   - check `git status --short`;
   - inspect relevant diffs, generated results, and test outcomes when available.
2. Ensure `.claude/handovers/` exists at the project root.
3. Create a Markdown file named `YYYY-MM-DD_HHmm.md` using the local current date and time.
4. If the target filename already exists, append `_2`, `_3`, etc. before `.md`.
5. Write the note using all required sections below.
6. Keep the note concise, factual, and mostly bullet-based.

## Required Sections

Include every section exactly once. If there is nothing relevant, write `なし`.

```markdown
# Handover YYYY-MM-DD HH:mm

## 今回やったこと

- ...

## 決定事項

- ...

## 捨てた選択肢と理由

- ...

## ハマりどころ

- ...

## 学び

- ...

## 次にやること

- ...

## 関連ファイル

- ...
```

## Content Rules

- Write in Japanese unless the user explicitly asks otherwise.
- Prefer concrete facts over interpretation.
- Include command results only as short summaries, not long logs.
- Mention uncommitted changes, generated artifacts, and test status when relevant.
- In `次にやること`, include priority signals such as `高`, `中`, or `低`.
- In `捨てた選択肢と理由`, record rejected approaches clearly so the next session does not repeat the same discussion.
- In `関連ファイル`, list the main paths touched or likely needed next; do not list every generated file unless it matters.

## File Creation Rules

- Use the project root as the base directory.
- Create `.claude/handovers/` if it does not exist.
- Use local time for the filename.
- Filename format:

```text
YYYY-MM-DD_HHmm.md
YYYY-MM-DD_HHmm_2.md
YYYY-MM-DD_HHmm_3.md
```

- Do not overwrite an existing handover note.
