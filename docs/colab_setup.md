# Colab Setup

If Colab shows:

```bash
!pwd
# /content
!ls
# sample_data
```

then the PatchDistill repository is not present on the Colab runtime. Opening
`notebooks/patchdistill_colab.ipynb` from Cursor does not automatically copy the
local folder to `/content`.

Use one of these routes first.

## Route A: GitHub

Push this folder to GitHub, then run in Colab:

```bash
!git clone https://github.com/black-leg-nameko/patchDistill.git /content/patchDistill
%cd /content/patchDistill
!python -m pip install -r requirements.txt
```

## Route B: Google Drive

Put the whole `patchDistill` folder on Drive, then run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/patchDistill
!python -m pip install -r requirements.txt
```

After setup, this should show project files:

```bash
!pwd
!ls
```

Expected files include `requirements.txt`, `patchdistill/`, `scripts/`, and
`notebooks/`.
