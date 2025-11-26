# PulmoVision

<img src="docs/PulmoVision.png" width="350"/>

PulmoVision is an end-to-end 3D Slicer extension for automated radiomic analysis of CT lung tumors. It provides a full pipeline—from preprocessing to segmentation to feature extraction—accessible through a one-click Slicer UI. The backend is written entirely in Python (NumPy, SimpleITK, PyTorch), enabling both Slicer-integrated execution and standalone usage for experimentation or batch processing.

---

## Example UI

Below is an example of the PulmoVision module running inside 3D Slicer:

<img src="docs/UI example.png" width="750"/>

---

## At a Glance

- **Purpose:** A reproducible, standardized radiomics workflow for lung tumors, integrating preprocessing, segmentation, cleanup, and feature extraction.
- **Inputs:** 3D CT volumes (NRRD/NRRDS for Slicer, NumPy arrays for Python-only use).
- **Outputs:** Binary tumor masks plus optional radiomics feature dictionaries.
- **Segmentation Options:**
  - Classical HU-thresholding
  - Percentile thresholding (debug)
  - 3D U-Net (trained on MSD Task06 Lung)
- **Environment Compatibility:**
  - Fully functional inside **3D Slicer 5.x**
  - Python backend usable independently via `requirements.txt` or `environment.yml`

---

## Repository Layout

    slicer_module/PulmoVision/         # Main Slicer module
      ├── PulmoVision.py               # Slicer UI logic
      ├── PulmoVision.ui               # Qt .ui layout
      └── PulmoBackend/                # Python backend
            ├── preprocessing.py
            ├── inference.py
            ├── pipeline.py
            ├── unet3d.py
            ├── training.py
            ├── radiomics.py
            └── postprocessing.py

    docs/                              # Project documentation, screenshots, diagrams
    data/msd_example/                  # Example metadata for MSD structure
    tests/                             # Backend unit tests
    requirements.txt                   # Python deps for standalone mode
    environment.yml                    # Conda env for training
    CMakeLists.txt                     # Slicer build definition

---

## Installation

### 1) Python Environment (Standalone / Training)

~~~bash
# Conda (recommended)
conda env create -f environment.yml
conda activate pulmovision

# Or using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

---

### 2) Install PulmoVision inside 3D Slicer

1. Clone or download this repository.  
2. Open **3D Slicer**.  
3. Go to: **Edit → Application Settings → Modules**  
4. Add the path:

       <repo-root>/slicer_module/PulmoVision

   to **Additional Module Paths**.  
5. Restart Slicer. PulmoVision should now appear in the Module list.

---

### 3) Enable PyTorch in Slicer (Required for UNet3D)

1. Open **Extensions Manager**.  
2. Search for **"PyTorch"**.  
3. Install the **PyTorch** extension.  
4. Restart Slicer.  
5. Verify in the Python interactor:

~~~python
import PyTorchUtils
torch = PyTorchUtils.PyTorchUtilsLogic().torch
print(torch.__version__)
~~~

If this works, UNet3D becomes available.  
If not, the module falls back to HU-threshold segmentation automatically.

---

## Data: MSD Task06 Lung

The official MSD Task06 Lung dataset is **not included** and must be downloaded separately from:

- http://medicaldecathlon.com/

Expected structure:

    Task06_Lung/
      ├── dataset.json
      ├── imagesTr/
      ├── labelsTr/
      └── imagesTs/

PulmoVision resolves the dataset location in this order:

1. `--data-root` argument passed to the training script.  
2. `MSD_LUNG_DATA_ROOT` environment variable.  
3. A default path defined in `msd_lung_dataset.py`.

`data/msd_example/dataset.json` documents the expected metadata schema but contains no real volumes.

---

## Using PulmoVision in 3D Slicer

1. Load a CT volume (NRRD/NRRDS) into Slicer.  
2. Open the **PulmoVision** module from the module list.  
3. Configure:

   - **Input CT volume**
   - **Window center / width** for HU windowing
   - **Segmentation method:**
     - HU Thresholding
     - Percentile threshold (debug)
     - UNet3D (requires PyTorch)
   - **Postprocessing options:**
     - Enable/disable cleanup
     - Keep largest connected component
   - **Radiomics:**
     - Enable/disable feature extraction

4. Click **Run Full Pipeline**.  
5. Inspect the results:

   - Mask overlay in the slice viewers
   - Radiomics feature table in the UI
   - Optional export of masks and tables for downstream analysis

---

## Backend-Only Usage (Python)

You can reuse the PulmoVision backend without launching Slicer:

~~~python
import numpy as np
from slicer_module.PulmoVision.PulmoBackend.pipeline import run_pulmo_pipeline

# volume: 3D NumPy array of HU values
volume = np.load("ct_volume.npy")

outputs = run_pulmo_pipeline(
    volume,
    segmentation_method="unet3d",   # or "hu_threshold"
    compute_features=True,
    return_metadata=True,
)

mask = outputs["mask"]
features = outputs["features"]
metadata = outputs["segmentation_metadata"]

print(mask.shape)
print(features)
print(metadata)
~~~

---

## 3D U-Net Training

`PulmoBackend/training.py` provides two main entry points: a fast synthetic demo mode and full MSD Task06 training.

### Synthetic Demo Training (No External Data)

~~~bash
python -m slicer_module.PulmoVision.PulmoBackend.training \
    --epochs 5 \
    --batch-size 2
~~~

This produces a small synthetic checkpoint (e.g., `unet3d_synthetic.pt`) mainly for debugging and demonstrations.

---

### MSD Task06 Lung Training

~~~bash
python -m slicer_module.PulmoVision.PulmoBackend.training \
    --train-msd \
    --data-root /path/to/Task06_Lung \
    --epochs 30 \
    --batch-size 1 \
    --patch-size 96 96 96 \
    --output unet3d_msd.pt
~~~

Checkpoints are written to:

    slicer_module/PulmoVision/PulmoBackend/checkpoints/

These weights are automatically loaded by the UNet3D inference path in Slicer (via the default checkpoint path defined in `inference.py`).

---

## Example Output: Rough 3D U-Net Segmentation

Below is a rough example of a UNet3D segmentation overlaid on a CT scan:

<img src="docs/rough 3D U-Net example.png" width="750"/>

This represents the expected baseline quality for a lightweight 3D U-Net trained on a small subset of MSD Task06 Lung. It is **not** intended for diagnostic use, but to demonstrate an integrated, end-to-end radiomics workflow.

---

## Troubleshooting

- **Module not visible in Slicer:**  
  Ensure `<repo-root>/slicer_module/PulmoVision` is listed under  
  `Edit → Application Settings → Modules`, then restart Slicer.

- **UNet3D not available:**  
  Install the **PyTorch** extension from the Extensions Manager and restart Slicer.  
  When PyTorch is unavailable, PulmoVision automatically falls back to HU-threshold segmentation.

- **MSD training script cannot find data:**  
  Confirm that either:
  - `--data-root /path/to/Task06_Lung` is passed, or  
  - `MSD_LUNG_DATA_ROOT` points to the folder containing `Task06_Lung/dataset.json`.

- **Radiomics table is empty:**  
  Make sure the **Compute radiomics** checkbox is enabled in the UI, or  
  pass `compute_features=True` when calling `run_pulmo_pipeline` from Python.

---

## License

This project is released under the **MIT License**.  
See `LICENSE` for details.

---

## Author

James Mascarenhas  
Queen’s University (2025)
