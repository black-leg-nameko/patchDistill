# Result Archiving

Colab outputs under `runs/` are ignored by Git, so use the archive script after
each experiment. It copies metrics, summaries, predictions, and gzipped JSONL
feature/patch files into `artifacts/colab_runs/<ARCHIVE_NAME>/`, commits them,
and pushes the commit to GitHub.

## Colab Usage

Run this after the experiment cells finish:

```python
import os, getpass
os.environ["GITHUB_TOKEN"] = getpass.getpass("GitHub token: ")
```

Then:

```bash
!ARCHIVE_NAME=gpt2_a100_pilot_001 scripts/save_colab_results_to_github.sh
```

The token needs repository `Contents: Read and write` permission. The script
uses the token only for `git push`; it does not write the token into the repo.

If no token is set, the script still creates the archive and commit locally, then
attempts a normal `git push`.

## If Push Fails With 403

If the final cell prints something like:

```text
remote: Permission to black-leg-nameko/patchDistill.git denied
fatal: unable to access ... 403
```

then the archive commit was created in the Colab checkout, but it was not pushed
to GitHub. Most commonly, the token is missing `Contents: Read and write` for
this repository, or the token belongs to an account without write access.

Create or update a fine-grained token for `black-leg-nameko/patchDistill` with:

- Repository access: `Only select repositories` -> `patchDistill`
- Repository permissions: `Contents: Read and write`

Then rerun only the save-results cell. If you rerun the clone/setup cell first,
the notebook now resets the Colab checkout to `origin/main`, so the failed local
archive commit may be discarded. That is usually fine because the experiment can
be rerun, but use the existing Colab runtime's `runs/` directory first if you
need to preserve an exact failed archive.

## Archive Contents

Each archive contains:

- `manifest.json`: timestamp, git commit, copied/skipped files.
- `analysis.md`: compact metric summary.
- `summary.json`: machine-readable summary of result JSON files.
- copied `runs/**/*.json` and `runs/**/*.md` files.
- copied `runs/**/*.csv` files.
- gzipped `runs/**/*.jsonl` files, unless `--no-jsonl` is passed.
