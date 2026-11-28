# CT Slice Lab

CT Slice Lab is a Streamlit research prototype for viewing head CT scans and
exploring two experimental models for intracranial hemorrhage. It supports
single-slice classification and binary hemorrhage segmentation of CT volumes.

[Live app](https://ct-slice-lab.streamlit.app/) · [Methodology](https://ct-slice-lab.streamlit.app/methodology)

## Use the app

| Input | What the app does |
| --- | --- |
| One CT DICOM slice (.dcm) | Opens the image and runs the PyTorch classifier. The viewer reports one overall hemorrhage score and five subtype scores: epidural, subdural, subarachnoid, intraparenchymal, and intraventricular. |
| A CT volume (.nii or .nii.gz) | Opens axial, coronal, and sagittal views with adjustable CT windowing. Select **Generate experimental overlay** to run the segmentation model. |
| An optional NIfTI reference mask | Upload it under **Model Results** for the same CT volume. The original CT, reference annotation, and prediction share a slice slider. Dice for that volume is calculated once both masks are available. |

Use de-identified research files. The hosted app processes uploads on its
server. Uploading a replacement scan clears the previous predictions. The
**Open teaching scan** panel on the Upload page shows a published example
without requiring a file.

## Run locally

Use Python 3.12 for the CPU environment below. From the repository root in
Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r src\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run src\viewer_app.py --server.port 8503
```

Open [localhost:8503](http://localhost:8503/). The package list includes CPU
PyTorch and both demo checkpoints are bundled under `assets/checkpoints/`.
For Streamlit Community Cloud use branch `main`, entry point `src/viewer_app.py`,
and Python 3.12.

## Data and models

- **Classification:** The bundled spatial CNN uses three CT contrast windows
  as input and returns six uncalibrated scores. It was trained on a subset
  of the [RSNA intracranial hemorrhage dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection).
- **Segmentation:** The original deployment used a small 2D CNN trained with
  binary masks from the [Brain Hemorrhage Segmentation Dataset](https://huggingface.co/datasets/Wendy-Fly/BHSD).
  Its expanded run used 25 training volumes, 7 validation volumes for cutoff
  selection, and 8 held-out test volumes. The masks combine hemorrhage subtypes
  into one foreground label.
- **Teaching image:** The multiplanar preview is adapted from
  [Spahr et al., Figure 1B](https://doi.org/10.3389/fnimg.2023.1157565)
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Segmentation evaluation

| Experiment | Training volumes | Mean Dice on eight held-out volumes |
| --- | ---: | ---: |
| Balanced pilot | 8 | 0.179 |
| Expanded single-slice CNN | 25 | 0.354 |
| Expanded CNN with adjacent slices | 25 | 0.359 |

Dice measures overlap with the reference mask, from 0 for no overlap to 1
for identical nonempty masks. The original deployed model averaged **40.9%
voxel precision** and **39.4% voxel recall** on the original CT grid. The
same eight hemorrhage-positive volumes were used for these comparisons.
The split uses archive identifiers because patient-level identity could not
be independently verified. The cohort cannot establish scan-level specificity
or performance across hospitals. The app is for research and education only.

## Train and evaluate a model

Keep scans and masks under the ignored `data/` directory. Prepare a CSV
with columns `patient_id,scan,mask` and paths relative to the CSV. Pair each
NIfTI volume with a binary mask on the same spatial grid. Use the same
patient identifier for every scan belonging to that patient and keep the
test manifest separate from training and cutoff selection.

```powershell
.\.venv\Scripts\python.exe src\train_segmentation.py data\development.csv --architecture context-unet --epochs 14 --output models\ct-segmenter-context-unet.pt
.\.venv\Scripts\python.exe src\evaluate_segmentation.py data\held-out.csv --checkpoint models\ct-segmenter-context-unet.pt
.\.venv\Scripts\python.exe src\predict_segmentation.py data\scan.nii.gz outputs\prediction.nii.gz --checkpoint models\ct-segmenter-context-unet.pt
```

The context U-Net predicts the center slice from three adjacent axial
slices. The trainer selects a mask cutoff on the validation partition. The
evaluator measures overlap against the annotations in the original voxel grid.

| Source file | Role |
| --- | --- |
| [viewer_app.py](src/viewer_app.py) | Scan upload, viewing, and result display |
| [research_model.py](src/research_model.py) | Bundled DICOM classifier and preprocessing |
| [train_segmentation.py](src/train_segmentation.py) | Baseline and context segmentation training |
| [predict_segmentation.py](src/predict_segmentation.py) | Mask inference and NIfTI export |
| [evaluate_segmentation.py](src/evaluate_segmentation.py) | Held-out segmentation metrics |

The separate [RSNA training script](src/train_cnn.py) is a ResNet experiment
and writes a different checkpoint format from the bundled spatial CNN.
