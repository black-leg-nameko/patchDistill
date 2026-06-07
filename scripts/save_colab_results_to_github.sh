#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_NAME="${ARCHIVE_NAME:-${1:-}}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-artifacts/colab_runs}"
RUNS_DIR="${RUNS_DIR:-runs}"
BRANCH="${BRANCH:-main}"
REMOTE="${REMOTE:-origin}"
REPO_URL="${REPO_URL:-https://github.com/black-leg-nameko/patchDistill.git}"
MAX_FILE_MB="${MAX_FILE_MB:-50}"

if [[ -z "${ARCHIVE_NAME}" ]]; then
  ARCHIVE_NAME="$(date -u +%Y%m%dT%H%M%SZ)"
fi

git config user.name "${GIT_AUTHOR_NAME:-colab-runner}"
git config user.email "${GIT_AUTHOR_EMAIL:-colab-runner@example.invalid}"

python -m patchdistill.cli collect-results \
  --runs "${RUNS_DIR}" \
  --out "${RUNS_DIR}/summary.json" \
  --markdown "${RUNS_DIR}/summary.md"

python -m patchdistill.cli archive-results \
  --runs "${RUNS_DIR}" \
  --archive-root "${ARCHIVE_ROOT}" \
  --name "${ARCHIVE_NAME}" \
  --max-file-mb "${MAX_FILE_MB}"

ARCHIVE_DIR="${ARCHIVE_ROOT}/${ARCHIVE_NAME}"
git add "${ARCHIVE_DIR}"

if git diff --cached --quiet; then
  echo "No new archived result files to commit."
else
  git commit -m "Add Colab results: ${ARCHIVE_NAME}"
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  AUTH_HEADER="$(python -c 'import base64, os; print("AUTHORIZATION: basic " + base64.b64encode(("x-access-token:" + os.environ["GITHUB_TOKEN"]).encode()).decode())')"
  git -c "http.https://github.com/.extraheader=${AUTH_HEADER}" push "${REMOTE}" "HEAD:${BRANCH}"
else
  echo "GITHUB_TOKEN is not set. Trying normal git push; this may fail on Colab."
  if ! git push "${REMOTE}" "HEAD:${BRANCH}"; then
    cat <<'EOF'
Push failed because Colab is not authenticated for GitHub.

Set a fine-grained GitHub token with Contents: Read and write, then rerun:

  import os, getpass
  os.environ["GITHUB_TOKEN"] = getpass.getpass("GitHub token: ")
  !ARCHIVE_NAME=my_run scripts/save_colab_results_to_github.sh

The archive commit remains local in the Colab runtime until push succeeds.
EOF
    exit 1
  fi
fi

echo "Saved results to GitHub archive: ${ARCHIVE_DIR}"

