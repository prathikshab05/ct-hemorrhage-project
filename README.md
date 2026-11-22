# CT Hemorrhage Classification — Starter

## Open CT Slice Lab

From the project folder in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\viewer_app.py
```

The browser opens to a synthetic slice. Change the window preset, toggle the
amber highlight, and adjust its opacity. Select **Research DICOM** to view one
local research slice. You can upload a corresponding two-dimensional binary
NumPy mask and export the displayed overlay as a PNG. Mask validation checks
pixel dimensions and binary values. Verify slice identity and orientation
before interpreting overlays.

The viewer blends a pixel-aligned mask over the slice. A future segmentation
model or an annotated mask can supply those pixels. The RSNA classification
labels describe hemorrhage presence and subtypes, so precise overlays need
additional localization evidence. The initial viewer handles one slice.

## Dataset and evaluation plan

| Source | Role | Current status |
| --- | --- | --- |
| [RSNA head CT challenge](https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection/) | Primary detection and subtype cohort, over 25,000 exams | Data acquisition pending |
| [PhysioNet CT-ICH](https://physionet.org/content/ct-ich/1.0.0/) | Separate source of trauma head CTs and hemorrhage annotations, 82 scans | Volume and mask conversion pending |

Keep raw scans and annotations locally under the ignored `data/` directory.
Split by patient before model development, choose thresholds on validation data,
and keep external exams separate from training and threshold selection. Map
labels across sources before comparing results and report each cohort separately.

For a held-out CSV with `exam_id,label,probability` columns, run the evaluator
once per target and cohort. For example, with a threshold of 0.5 chosen on
validation data:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py predictions.csv --threshold 0.5
```

The evaluator reports AUROC, average precision, sensitivity, specificity, and
error counts at exam level. Future segmentation evaluation should also report
Dice overlap against annotated masks. Model training and performance results
are pending. The viewer is a research prototype.

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
