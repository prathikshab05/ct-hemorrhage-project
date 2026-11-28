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
PyTorch and demo checkpoints are bundled under `assets/checkpoints/`.
For Streamlit Community Cloud use branch `main`, entry point `src/viewer_app.py`,
and Python 3.12.

## Data and models

- **Classification:** The bundled spatial CNN uses three CT contrast windows
  as input and returns six uncalibrated scores. It was trained on a subset
  of the [RSNA intracranial hemorrhage dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection).
- **Segmentation:** The deployed 2D U-Net predicts the center axial slice
  from a stack of three. The [Brain Hemorrhage Segmentation Dataset](https://huggingface.co/datasets/Wendy-Fly/BHSD)
  supplied 94 scan and mask pairs: 59 training volumes, 15 for threshold
  selection, and 20 test volumes. Reference masks combine hemorrhage subtypes
  into one foreground label.
- **Teaching image:** The multiplanar preview is adapted from
  [Spahr et al., Figure 1B](https://doi.org/10.3389/fnimg.2023.1157565)
  under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Segmentation evaluation

| Experiment | Training volumes | Original holdout (8) | Fresh holdout (12) |
| --- | ---: | ---: | ---: |
| Balanced pilot | 8 | 0.179 | — |
| Previous deployed CNN | 25 | 0.354 | 0.267 |
| CNN with adjacent slices | 25 | 0.359 | — |
| Context U-Net, small split | 25 | 0.311 | — |
| **Context U-Net, expanded split** | **59** | **0.495** | **0.351** |

Dice measures overlap with the reference mask, from 0 for no overlap to 1
for identical nonempty masks. The expanded U-Net averaged 67.1% voxel
precision and 45.3% voxel recall on the original holdout, and 53.3% precision
and 33.1% recall on the fresh holdout. Metrics are averaged per volume on
its native CT grid. The new model trained for eight epochs; its 0.80 cutoff
was selected on validation data before evaluating either holdout.

Both holdouts contain hemorrhage-positive BHSD volumes. They are separate
by archive identifier, but patient identity could not be independently
verified. These results do not establish specificity on bleed-negative scans
or performance across hospitals. The app is for research and education only.

## Train and evaluate a model

Keep scans and masks under the ignored `data/` directory. Prepare a CSV
with columns `patient_id,scan,mask` and paths relative to the CSV. Pair each
NIfTI volume with a binary mask on the same spatial grid. Use the same
patient identifier for every scan belonging to that patient and keep the
test manifest separate from training and cutoff selection.

```powershell
.\.venv\Scripts\python.exe src\train_segmentation.py data\development.csv --architecture context-unet --epochs 8 --output models\ct-segmenter-context-expanded.pt
.\.venv\Scripts\python.exe src\evaluate_segmentation.py data\held-out.csv --checkpoint models\ct-segmenter-context-expanded.pt
.\.venv\Scripts\python.exe src\predict_segmentation.py data\scan.nii.gz outputs\prediction.nii.gz --checkpoint models\ct-segmenter-context-expanded.pt
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
