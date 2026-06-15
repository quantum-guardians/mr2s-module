---
name: handoff
description: >
  Writes a session handoff document. Invoke when the user says "handoff", "/handoff",
  "wrap up", "session summary", "끝내기 전에 정리", or "핸드오프 작성".
---

Save a handoff so the next session can pick up without re-reading the conversation.

## Steps

1. Synthesize the session: what was done, what's the current state, what's next.
2. Write the handoff document content (using the template below).
3. Save by piping content to the project-local script:
   ```bash
   cat <<'EOF' | .claude/skills/handoff/handoff.sh
   # Handoff — ...
   ...
   EOF
   ```
   Writes to `<cwd>/.claude/handoffs/YYYY-MM-DD_HH-MM.md` and `<cwd>/.claude/handoffs/LATEST.md`.
4. Return the exact path printed by the script.

## Template

```markdown
# Handoff — {YYYY-MM-DD HH:MM}

## Summary
{1–3 sentences: goal and outcome}

## Done
- {specific file/function/command-level bullets}

## Current State
{what works, what's half-done, what's broken}

## Decisions
| Decision | Reason |
|----------|--------|

## Next Steps
1.
2.

## Blockers / Open Questions
{or "None"}
```
