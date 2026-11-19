# CT Hemorrhage Classification — Starter

An educational computer-vision project for detecting intracranial hemorrhage and classifying the five RSNA hemorrhage subtypes from non-contrast head CT images.

**Scope:** this project is a research prototype. The RSNA labels support hemorrhage detection and subtype classification. They do not label surgical urgency, and the app must not make treatment decisions.

## 1. Install on Windows

Download and install:

1. [Python 3.12 for Windows](https://www.python.org/downloads/windows/) — during setup, check **Add Python to PATH**.
2. [Git for Windows](https://git-scm.com/download/win).
3. [Visual Studio Code](https://code.visualstudio.com/), then install the **Python** and **Jupyter** extensions from Microsoft.
4. Create a [Kaggle account](https://www.kaggle.com/) and accept the RSNA competition rules before accessing its files.

Open this folder in VS Code. In its PowerShell terminal, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If PowerShell blocks activation, run this once in that terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The CPU PyTorch install is for setup and small experiments. Full CT training can be slow on a CPU; use a GPU notebook (for example, Kaggle Notebooks) for model training if you don't have a supported NVIDIA GPU. For a CUDA install, use the current Windows/Pip/CUDA selector at [PyTorch Get Started](https://pytorch.org/get-started/locally/) rather than guessing a CUDA version.

## 2. Get the data

The RSNA challenge data is hosted on Kaggle. It contains over 25,000 exams and hundreds of thousands of CT slices, so **do not download or unzip the full image archive until you have checked free disk space**. Start in a Kaggle Notebook with the competition data attached, or use a small subset locally first. Kaggle requires accepting the competition rules before download.

For local Kaggle CLI access, create an API token from your Kaggle account settings and put `kaggle.json` at:

```text
C:\Users\YOUR_WINDOWS_USERNAME\.kaggle\kaggle.json
```

Keep this token private. It is ignored by Git if stored inside this project, but the recommended location above is outside the project. Install and check the CLI:

```powershell
pip install kaggle
kaggle competitions files -c rsna-intracranial-hemorrhage-detection
```

Download only after checking the listed file names and your storage:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p data\raw
```

The archive is intentionally not included here. Dataset access and use are subject to Kaggle/RSNA terms; review them before sharing data or model outputs.

## 3. Verify the starter setup

After placing the competition label CSV at `data/raw/train.csv` (or adjusting the path), run:

```powershell
python src\inspect_labels.py --csv data\raw\train.csv
```

The script checks the label-table shape and prints counts for the five subtype labels plus `any`. It does not train a model.

## Project plan

1. Inspect labels and understand how image-level rows map to exams.
2. Build a small, reproducible baseline before scaling up.
3. Split by patient/exam, not by individual slice, to avoid leakage.
4. Report per-subtype metrics and examine false negatives and false positives.
5. Add a demo only after evaluation; display model predictions as research outputs, never as diagnoses or surgical recommendations.

## References

- [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/RSNA-Intracranial-Hemorrhage-Detection-Challenge-2019)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [Kaggle API documentation](https://www.kaggle.com/docs/api)
