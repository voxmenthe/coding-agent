# Code References

**THIS IS OF UTTER IMPORTANCE THE USERS HAPPINESS DEPENDS ON IT!!**
When referencing code locations, you MUST use clickable format that VS Code recognizes:
- `path/to/file.ts:123` format (file:line)
- `path/to/file.ts:123-456` (ranges)
- Always use relative paths from the project root

**Examples:**
- `src/server/fwd.ts:92` - single line reference
- `src/server/pty/pty-manager.ts:274-280` - line range
- `web/src/client/app.ts:15` - when in parent directory

NEVER give a code reference or location in any other format.

# CRITICAL
**IMPORTANT**: BEFORE YOU DO ANYTHING, READ `spec.md` IN FULL USING THE READ TOOL!
**IMPORTANT**: NEVER USE GREP, ALWAYS USE RIPGREP!