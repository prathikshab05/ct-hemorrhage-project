# CT Hemorrhage Classification — Starter

## Run the website

On Windows, double-click **run-viewer.cmd** and keep its window open. Visit
[CT Slice Lab](http://127.0.0.1:8503/). Run `setup.cmd` first if the virtual
environment is missing.

## Deploy to Streamlit Community Cloud

Set the branch to `main`, main file path to `src/viewer_app.py`, and Python
to 3.12. The hosted requirements install CPU PyTorch and the viewer uses the
bundled research weights. Example CT volumes and reference masks live in the
ignored `data/` folder. The hosted viewer can run inference on a de-identified
research upload and display the published teaching image.

## Open CT Slice Lab

### Run the Kaggle classifier locally

Download `ct-spatial-cnn.pt` from the private Kaggle notebook's Output
panel and place it at `models/ct-spatial-cnn.pt`. The model folder and
research data are ignored by Git. Run the app with the command below,
choose **Research DICOM** and upload one authorized research CT slice.
The app opens the separate **Scan analysis** page automatically.
Click **Run research classifier** there. The app displays six uncalibrated
slice scores. Annotation overlays are shown separately.

The checkpoint uses the small spatial CNN from the notebook. The separate
ResNet training script produces a different checkpoint format.

From the project folder in the VS Code PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\viewer_app.py
```

The browser opens to **Sample CT**, using the example bundled with pydicom
or the copy under `data/`.
For real scans, switch to **Annotated research case**. Install the updated packages
with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
At [PhysioNet CT-ICH v1.3.1](https://physionet.org/content/ct-ich/1.3.1/),
sign in and review the data use agreement. Choose a hemorrhage-positive case
from the labels table and download the scan and mask with the same case ID.
You can select both files directly in the viewer. To preload a case on startup, place the files in `data/annotated_case/` as `scan.nii.gz` and `mask.nii.gz`.
If a file ends in `.nii`, gzip-compress it before using the `.nii.gz` extension.
Open **Annotated research case**. Pick a plane and scroll with its slice
slider. Turn on **Show neighboring slices** for up to four views at once.
Turn that grid off for side-by-side comparison. Toggle
**Show supplied mask** to compare the raw CT with the annotated region.
The viewer starts on the slice with the most marked pixels.

Select **Research DICOM** to view one
local research slice. You can upload a corresponding two-dimensional binary
NumPy mask and export the displayed overlay as a PNG. Mask validation checks
pixel dimensions and binary values. Verify slice identity and orientation
before interpreting overlays.

The viewer blends a pixel-aligned mask over the slice. A future segmentation
model or an annotated mask can supply those pixels. The RSNA classification
labels describe hemorrhage presence and subtypes, so precise overlays need
additional localization evidence. The initial viewer handles one slice.

## Dataset and evaluation plan

### Train a segmentation baseline

Use paired research CT volumes and binary masks in NIfTI format. Put a
local CSV in `data/` with columns `patient_id,scan,mask`; paths are relative
to the CSV. Use the same patient ID for every scan from a patient.

```powershell
.\.venv\Scripts\python.exe src\train_segmentation.py data\segmentation.csv
```

The experimental trainer keeps each patient group in one split and samples
slices with and without annotated pixels at equal expected rates. It chooses
the score threshold using Dice on positive validation volumes at 128x128 and
reports how many negative volumes contain predicted pixels. Keep a separate
patient test manifest out of training and threshold selection, then measure
overlap on the original scan grid:

```powershell
.\.venv\Scripts\python.exe src\evaluate_segmentation.py data\segmentation-test.csv --checkpoint models\ct-segmenter.pt
```

To export an experimental prediction from a
trained checkpoint and open it as a supplied mask in the volume viewer:

```powershell
.\.venv\Scripts\python.exe src\predict_segmentation.py data\scan.nii.gz outputs\predicted-mask.nii.gz
```

The exported mask uses the original scan's dimensions and spatial grid.

### BHSD segmentation experiment

The [BHSD labeled archive](https://huggingface.co/datasets/Wendy-Fly/BHSD)
supplied 40 paired scans and reference masks. The expanded experiment used
25 scans for training, 7 for threshold selection, and 8 for testing. All
subtypes were merged into a single foreground mask. Splits used archive
identifiers. Separate patient identities could not be verified.

| Run | Training scans | Mean native-grid Dice on the same 8 held-out scans |
| --- | ---: | ---: |
| Original pilot | 8 | 0.000 |
| Balanced pilot | 8 | 0.179 |
| Balanced expanded, single slice | 25 | 0.354 |
| Balanced expanded, neighboring slices | 25 | 0.359 |

Expanded test Dice ranged from 0.002 to 0.790. Mean voxel precision was
0.409 and mean voxel recall was 0.394 across the eight scans. Every test
scan had annotated hemorrhage, so this experiment cannot estimate
false positives on scans without hemorrhage. The neighboring-slice run used
the same partition and eight training epochs. Its mean voxel precision was
0.413 and mean voxel recall was 0.354. The small Dice gain and lower recall
do not establish an improvement on this test set. The demo uses the
single-slice checkpoint.

### Generate an overlay on a research volume

Open **Upload** and select a de-identified 3D head CT in NIfTI format
(`.nii` or `.nii.gz`). The Scan Viewer and Model Results links appear after
the upload opens successfully. Click **Generate experimental overlay** to
compute a mask for the current scan. For a DICOM slice, the viewer runs the classifier when the slice opens.

### Review model overlays

**Model Results** shows only the active scan. For volumes, an optional aligned
NIfTI reference mask enables the reference panel and Dice comparison. Uploading
a replacement scan clears the previous mask and scores. The fixed test-set
metrics are documented under **Methodology**.

To test the neighboring-slice variant, add `--neighboring-slices` to the
training command. The model uses three adjacent axial slices to predict the
center slice. The first and last slices repeat the edge input as needed.
The same evaluator and export commands accept either checkpoint.

### Train the six-output CNN on an RSNA subset

Put the RSNA label CSV and a small set of labeled DICOM slices under the
ignored `data/` folder. Keep enough distinct patients and positive examples
for validation. The folder below should directly contain the selected slices.
From the project root, with PyTorch and TorchVision installed:

```powershell
.\.venv\Scripts\python.exe src\train_cnn.py data\stage_2_train.csv data\rsna_subset --epochs 2
```

The script groups slices by DICOM PatientID, uses three contrast windows,
and prints validation log loss and AUROC per output. AUROC is undefined
when a validation label contains just one class. The first run may need
network access for the pretrained weights. The checkpoint is written to
`outputs/rsna_resnet50.pt`. These outputs are slice classification scores.
Pixel overlays come from the supplied annotation.

### Train the first baseline from local files

`src/train_baseline.py` is a binary research baseline. It learns from
histograms of CT slice intensities, then uses the highest slice score per
exam for ranking. This score has not been calibrated as an exam probability.
The model does not produce a localization mask.

Create `data/manifest.csv` with columns `path,patient_id,exam_id,label,split`.
Use one row per DICOM slice, paths relative to the manifest, binary slice
labels, and explicit `train` or `test` values. Assign each patient to just
one split. Keep the manifest with the local data.

```powershell
.\.venv\Scripts\python.exe src\train_baseline.py data\manifest.csv outputs\predictions.csv
```

This reports ranking metrics for held-out exams and saves their scores.
Choose any threshold for sensitivity and specificity on separate validation
data before applying it to the held-out predictions.

| Source | Role | Current status |
| --- | --- | --- |
| [RSNA head CT challenge](https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection/) | Primary detection and subtype cohort, over 25,000 exams | Data acquisition pending |
| [PhysioNet CT-ICH v1.3.1](https://physionet.org/content/ct-ich/1.3.1/) | Independent cohort with and without hemorrhage, 75 released NIfTI scans | PhysioNet registration and signed data use agreement required |

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
error counts at exam level. The segmentation evaluator reports voxel overlap
against annotated masks on the original CT grid. The viewer is a research prototype.

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
