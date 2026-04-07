#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./push.sh
#   ./push.sh "your commit message"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not a git repository: $REPO_ROOT"
  exit 1
fi

REMOTE="origin"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" == "HEAD" ]]; then
  BRANCH="main"
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Error: remote '$REMOTE' is not configured."
  echo "Set it with: git remote add origin <repo-url>"
  exit 1
fi

# Stage everything.
git add -A

# Commit only if there are staged changes.
if git diff --cached --quiet; then
  echo "No local changes to commit."
else
  COMMIT_MSG="${1:-auto: update $(date '+%Y-%m-%d %H:%M:%S')}"
  git commit -m "$COMMIT_MSG"
fi

# Sync with remote before push to reduce non-fast-forward failures.
git fetch "$REMOTE" "$BRANCH" || true
if git show-ref --verify --quiet "refs/remotes/$REMOTE/$BRANCH"; then
  git pull --rebase "$REMOTE" "$BRANCH"
fi

git push "$REMOTE" "$BRANCH"

echo "Done: pushed $BRANCH to $REMOTE"
