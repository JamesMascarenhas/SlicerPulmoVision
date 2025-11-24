# PulmoVision

PulmoVision is an end-to-end 3D Slicer extension for automated radiomic analysis of CT lung tumors. It ships a lightweight 3D U-Net baseline, reference preprocessing/post-processing utilities, and a scripted Slicer UI so researchers can run a full pipeline with a single button click. The backend lives entirely in Python (NumPy/SimpleITK/PyTorch) so pieces can also be reused outside of Slicer for experimentation or offline batch processing.

## At a Glance
- **Purpose:** Demonstrate a reproducible, standardized radiomics workflow for lung tumors, combining preprocessing, segmentation, cleanup, and feature extraction.
- **Inputs:** 3D CT volumes (NRRD/NRRDS via Slicer, NumPy arrays for Python usage).
- **Outputs:** Binary tumor masks plus optional radiomics feature dictionaries.
- **Segmentation:** Default placeholder heuristics or a 3D U-Net checkpoint trained on MSD Task06 Lung (when PyTorch is available).
- **Environment:** Tested inside 3D Slicer 5.x with the PyTorch extension; Python-only usage requires the dependencies in `requirements.txt` or `environment.yml`.

## Repository Layout
- `slicer_module/PulmoVision/` – Scripted Slicer module definition and CMake glue.
  - `PulmoVision.py` – UI logic that wires Slicer widgets to the backend pipeline.
  - `PulmoBackend/` – Pure-Python pipeline (preprocessing, segmentation, post-processing, radiomics, training utilities).
- `data/msd_example/` – Placeholder `dataset.json` showing the expected MSD Task06 Lung folder structure.
- `docs/` – Proposal, notes, and diagrams describing the project design.
- `tests/` – Minimal tests for backend components.
- `requirements.txt` / `environment.yml` – Python dependency definitions for non-Slicer use.
- `CMakeLists.txt` – CMake project file used by 3D Slicer to build/install the module.
  
## Installation
### 1) Python environment (for standalone backend use or checkpoint training)
Create an environment with PyTorch, SimpleITK, and the radiomics stack:

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate pulmovision

# or pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Install the PulmoVision module inside 3D Slicer
1. Clone or download this repository.
2. Open **3D Slicer**.
3. Go to **Edit → Application Settings → Modules**.
4. Add the folder `<repo-root>/slicer_module/PulmoVision` to the **Additional Module Paths** list.
5. Restart Slicer. The module should now appear under the *Modules* drop-down.

### 3) Enable PyTorch inside Slicer (required for UNet3D)
3D Slicer does not bundle PyTorch. Install the PyTorch extension so the UNet3D segmentation path can run:
1. Open **Extensions Manager** in Slicer.
2. Search for **“PyTorch”**.
3. Install the **PyTorch** extension.
4. Restart Slicer.
5. Verify inside the Python Interactor:

```python
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().torch
print("PyTorch version:", torch.__version__)
```

If the import succeeds, UNet3D inference becomes available. Without PyTorch, the module automatically falls back to HU-threshold segmentation so the pipeline still runs.

## Data: MSD Task06 Lung
- **Not included:** The official MSD Task06 Lung volumes must be downloaded separately from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/).
- **Expected layout:**
  ```
  Task06_Lung/
  ├── dataset.json
  ├── imagesTr/
  ├── imagesTs/
  └── labelsTr/
  ```
- **Where to place the data:** Keep the dataset outside the repo (for example `../Task06_Lung`). The backend resolves the path in this order:
  1. `--data-root` argument passed to training scripts.
  2. `MSD_LUNG_DATA_ROOT` environment variable.
  3. A default path returned by `get_default_msd_root()` in `msd_lung_dataset.py`.
- `data/msd_example/dataset.json` documents the metadata schema but contains no real volumes.

## Using PulmoVision in 3D Slicer
1. **Load a CT volume** (NRRD/NRRDS) into Slicer.
2. Open **PulmoVision** from the module list.
3. Choose your input volume and (optionally) adjust parameters:
   - Window center/width for preprocessing.
   - Segmentation method: HU threshold (default fallback), percentile heuristic, or UNet3D.
   - Post-processing options for mask cleanup.
   - Radiomics feature extraction toggle.
4. Click **Run Full Pipeline**. The module preprocesses the volume, performs segmentation, post-processes the mask, and computes radiomics features if requested.
5. **Inspect results:** tumor overlays appear in the slice viewers; feature tables are shown in the module panel and can be exported for downstream analysis.

## Using the Backend in Python (outside Slicer)
You can reuse the core pipeline without launching Slicer by importing the backend package:

```python
import numpy as np
from slicer_module.PulmoVision.PulmoBackend.pipeline import run_pulmo_pipeline

# volume is a 3D NumPy array in HU
mask = run_pulmo_pipeline(volume, segmentation_method="hu_threshold")

# Request radiomics features and metadata
outputs = run_pulmo_pipeline(
    volume,
    segmentation_method="unet3d",
    segmentation_kwargs={"weights_path": "/path/to/unet3d.ckpt"},
    compute_features=True,
    return_metadata=True,
)
print(outputs["features"], outputs["segmentation_metadata"])
```

### Training the 3D U-Net
`training.py` supports two entry points: quick synthetic training for demos and full MSD Task06 training when the dataset is available.

- **Synthetic demo training (fast, no external data):**
  ```bash
  python -m slicer_module.PulmoVision.PulmoBackend.training \
      --epochs 5 \
      --batch-size 2
  ```
  This produces a checkpoint (default `unet3d_synthetic.ckpt`) that can be loaded by UNet3D inference.

- **MSD Task06 training (requires dataset):**
  ```bash
  python -m slicer_module.PulmoVision.PulmoBackend.training \
      --train-msd \
      --data-root /path/to/Task06_Lung \
      --epochs 10 \
      --batch-size 1 \
      --patch-size 96 96 96 \
      --output unet3d_msd.ckpt
  ```
  You can omit `--data-root` if `MSD_LUNG_DATA_ROOT` is set to the dataset location.

Trained checkpoints can be referenced from Slicer via the UNet3D parameters or loaded directly in Python using the inference utilities in `PulmoBackend/inference.py`.

## Troubleshooting
- **Module not visible in Slicer:** Double-check that `<repo-root>/slicer_module/PulmoVision` is listed under **Application Settings → Modules** and restart Slicer.
- **UNet3D not selectable:** Install the PyTorch extension and restart Slicer. When PyTorch is absent, the module keeps running with HU-threshold fallback segmentation.
- **MSD training cannot find data:** Ensure `--data-root` or `MSD_LUNG_DATA_ROOT` points to the folder containing `Task06_Lung/dataset.json`.
- **Radiomics features are empty:** Confirm you enabled *Compute features* in the Slicer UI or passed `compute_features=True` when calling `run_pulmo_pipeline`.

## License
This project is released under the MIT License. See `LICENSE` for details.

## Authors
James Mascarenhas  
Jihyeon Park  
CISC 881: Medical Informatics  
Queen's University (2025)
