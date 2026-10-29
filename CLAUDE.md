@AGENTS.md

## Claude Code notes
- `.claude/settings.json` denies reading `.env` and `.env.*`, which also blocks `.env.example`; the variable names are in AGENTS.md. The deny covers Claude's file tools and recognised shell file commands, not arbitrary subprocesses; use `/sandbox` for OS-level enforcement.
