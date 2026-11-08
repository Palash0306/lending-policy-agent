#!/usr/bin/env bash
# Fails if a bank policy document (.pdf / .docx) is staged for commit or tracked in git.
# Policy documents are internal and must never be committed: tests build their own files on the fly.
# Used by .githooks/pre-commit and by scripts/ci.sh. Set ALLOW_DOCUMENTS=1 to skip (not recommended).
set -u
if [ "${ALLOW_DOCUMENTS:-0}" = "1" ]; then exit 0; fi
if ! git rev-parse --git-dir >/dev/null 2>&1; then exit 0; fi          # not a git checkout (e.g. a downloaded zip): nothing to check
mode="${1:-tracked}"
if [ "$mode" = "staged" ]; then
  files=$(git diff --cached --name-only --diff-filter=ACMR)
else
  files=$(git ls-files)
fi
found=$(printf '%s\n' "$files" | grep -i -E '\.(pdf|docx)$' || true)
if [ -n "$found" ]; then
  echo "ERROR: policy documents must not be committed:" >&2
  printf '  %s\n' $found >&2
  echo "Remove them from git (git rm --cached <file>) and keep them outside the repository." >&2
  exit 1
fi
exit 0
