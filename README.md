# Object Detection

## Dataset Preparation & Preprocessing

### Overview

This section describes the data preparation pipeline used to build a unified dataset for training the object detection model.

The pipeline performs the following steps:

- Downloads datasets from Roboflow
- Merges multiple datasets into a single structure
- Normalizes class labels:
  - `0 → Human`
  - `1 → Tent`
- Converts all annotations into YOLO format
- Standardizes dataset splits (`train`, `val`, `test`)
- Generates a `data.yaml` file for training

The resulting dataset is fully compatible with YOLO-based training pipelines.

---

### Dataset Structure

After running the preprocessing pipeline, the dataset will be organized as follows:

```
data/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

---

### Setup

Create and activate a virtual environment (recommended):

```
python -m venv venv
venv\\Scripts\\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

### Environment Configuration

Create a `.env` file in the project root containing your Roboflow API key:

```
ROBOFLOW_API_KEY=your_api_key_here
```

An example file is provided:

```
.env.example
```

You can copy it using:

```
copy .env.example .env
```

---

### Running the Pipeline

To download and prepare the dataset, run in the project root directory:

```
python preproc_pipeline/data_init.py
```

This will:

- Download datasets into a temporary directory
- Merge and normalize them
- Save the final dataset in `./data/`
- Remove temporary files after completion

---

### Notes

- The `data/` directory is excluded from version control and must be generated locally
- Ensure you have access to the Roboflow workspace before running the script

---

### Preprocessing Strategy

The preprocessing pipeline consists of two stages:

1. **Offline preprocessing (this module):**
   - dataset merging
   - label normalization
   - structural standardization

2. **Online preprocessing (during training):**
   - data augmentation (rotation, scaling, flipping)
   - handled by the YOLO training framework

---

### Troubleshooting

- If download fails → verify your API key and workspace access
- If dataset is empty → check dataset names and splits
- If labels appear incorrect → inspect `.txt` files in `data/labels/`

