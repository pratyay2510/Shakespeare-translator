#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./pull.sh
#   ./pull.sh origin main
#
# Behavior:
# - Detects repo root from script location.
# - Optionally auto-stashes local changes (including untracked files).
# - Pulls latest commits with rebase.
# - Restores stashed work after pull.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not a git repository: $REPO_ROOT"
  exit 1
fi

REMOTE="${1:-origin}"
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BRANCH="${2:-$CURRENT_BRANCH}"
if [[ "$BRANCH" == "HEAD" || -z "$BRANCH" ]]; then
  BRANCH="main"
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Error: remote '$REMOTE' is not configured."
  echo "Set it with: git remote add $REMOTE <repo-url>"
  exit 1
fi

STASHED=0
STASH_NAME="auto-stash pull.sh $(date '+%Y-%m-%d %H:%M:%S')"

# Auto-stash if there are local changes (tracked or untracked).
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Local changes detected. Stashing before pull..."
  git stash push -u -m "$STASH_NAME" >/dev/null
  STASHED=1
fi

echo "Fetching latest from $REMOTE/$BRANCH ..."
git fetch "$REMOTE" "$BRANCH"

if ! git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  echo "Error: remote branch '$REMOTE/$BRANCH' was not found."
  echo "Available remote branches:"
  git branch -r
  exit 1
fi

echo "Rebasing local branch '$BRANCH' onto $REMOTE/$BRANCH ..."
git pull --rebase "$REMOTE" "$BRANCH"

if [[ "$STASHED" -eq 1 ]]; then
  echo "Restoring stashed local changes..."
  if ! git stash pop >/dev/null; then
    echo "Warning: could not auto-apply stashed changes cleanly."
    echo "Your stash is still saved. Resolve conflicts and apply manually with:"
    echo "  git stash list"
    echo "  git stash apply <stash-ref>"
    exit 2
  fi
fi

echo "Done: local repository is up to date with $REMOTE/$BRANCH"
