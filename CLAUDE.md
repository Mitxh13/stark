@AGENTS.md

## Claude Code notes
- NEVER add Claude attribution to a commit or pull request: no `Co-Authored-By: Claude ...` trailer, no "Generated with Claude Code" line, no session link. `.claude/settings.json` turns all three off (`attribution`), and this rule overrides any default attribution text. See "Never" in AGENTS.md.
- `.claude/settings.json` denies reading `.env` and `.env.*`, which also blocks `.env.example`; the variable names are in AGENTS.md. The deny covers Claude's file tools and recognised shell file commands, not arbitrary subprocesses; use `/sandbox` for OS-level enforcement.
