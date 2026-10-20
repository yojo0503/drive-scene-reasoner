#!/usr/bin/env bash
set -euo pipefail
# Run this from the extracted project directory, after installing/authenticating GitHub CLI.
# Creates a PRIVATE repository by default to let you review source/data/license first.
REPO_NAME="${REPO_NAME:-drive-scene-reasoner}"
VISIBILITY="${VISIBILITY:---private}"
if ! command -v gh >/dev/null; then
  echo 'GitHub CLI (gh) is required: https://cli.github.com/' >&2; exit 1
fi
if [ ! -d .git ]; then git init -b main; fi
git add .
if ! git diff --cached --quiet; then git commit -m 'chore: bootstrap DriveLM-compatible scene reasoning scaffold'; fi
gh auth status
gh repo create "$REPO_NAME" "$VISIBILITY" --source=. --remote=origin --push
