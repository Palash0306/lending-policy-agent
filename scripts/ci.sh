#!/usr/bin/env bash
# One script any CI system (GitHub Actions, GitLab, Jenkins, Azure DevOps...) can run:  bash scripts/ci.sh
# It installs the test dependencies, refuses to continue if a policy document is tracked, lints, runs the tests with
# coverage, and finally runs a non-blocking dependency audit.
#   COV_MIN       minimum coverage in percent (default below)
#   SKIP_INSTALL  set to 1 if the dependencies are already installed
#   PYTHON        python command to use (default: python3)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
COV_MIN="${COV_MIN:-85}"

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements-ci.txt
fi

echo "== 1/4 no policy documents in git"
bash scripts/check_no_documents.sh tracked

echo "== 2/4 lint (ruff)"
"$PY" -m ruff check src tests scripts

echo "== 3/4 tests with coverage (minimum ${COV_MIN}%)"
"$PY" -m pytest --cov=src/lending_agent --cov-report=term-missing:skip-covered --cov-report=xml --cov-fail-under="${COV_MIN}"
if command -v node >/dev/null 2>&1; then
  echo "-- web page tests (node)"
  node tests/ui/run_all.js
fi

echo "== 4/4 dependency audit (does not fail the build)"
"$PY" -m pip_audit -r requirements-ci.txt || echo "pip-audit reported findings or could not run: review them, the build is not blocked."
echo "CI finished OK"
