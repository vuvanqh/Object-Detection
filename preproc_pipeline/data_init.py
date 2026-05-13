from roboflow import Roboflow
import shutil
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Root directory for the final merged dataset
ROOT = "data"

# Temporary directory used for downloaded datasets
TMP = "tmp_datasets"

"""
Unified class mapping:
0 -> Human
1 -> Tent
"""
CLASS_MAPPING = {
    # HERIDAL dataset contains humans -> map class 0 to Human
    "heridal-lrbkc-keibe": {0: 0},

    # Tent datasets -> map class 0 to Tent
    "tent-models-fkihy": {0: 1},
    "sarso_tent-o6bqt": {0: 1},
}


def ensure_dirs():
    """
    Create the required YOLO directory structure:
    
    data/
        images/
            train/
            valid/
            test/
        labels/
            train/
            valid/
            test/
    """
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{ROOT}/labels/{split}", exist_ok=True)


def download_all(rf):
    """
    Download all datasets from Roboflow in YOLOv5 format.

    Returns:
        list[str]: Paths to downloaded datasets
    """

    datasets = [
        "heridal-lrbkc-keibe",
        "tent-models-fkihy",
        "sarso_tent-o6bqt",
    ]

    paths = []

    for name in datasets:
        # Access project from workspace
        project = rf.workspace("quoc-hoang-vu-van").project(name)

        # Download version 1 of the dataset
        version = project.version(1)

        # Output directory for downloaded dataset
        out_dir = os.path.join(TMP, name)

        # Download in YOLOv5 format
        version.download("yolov5", location=out_dir)

        paths.append(out_dir)

    return paths


def remap_label_line(line, mapping):
    """
    Remap class IDs from source dataset labels
    to the unified dataset class IDs.

    Example:
        Original: 0 0.5 0.5 0.2 0.3
        Remapped: 1 0.5 0.5 0.2 0.3
    """

    parts = line.strip().split()

    # Extract original class ID
    cls = int(parts[0])

    # Ignore labels not included in mapping
    if cls not in mapping:
        return None

    # Replace original class ID with unified class ID
    new_cls = mapping[cls]

    return " ".join([str(new_cls)] + parts[1:])


def merge_dataset(dataset_paths):
    """
    Merge all downloaded datasets into one unified dataset.

    Steps:
    1. Copy images into unified directory
    2. Rename files using dataset prefix to avoid collisions
    3. Remap annotation classes
    4. Save new labels
    """

    for ds_path in dataset_paths:

        # Dataset name used as file prefix
        prefix = os.path.basename(ds_path)

        for split in ["train", "valid", "test"]:

            img_dir = os.path.join(ds_path, split, "images")
            lbl_dir = os.path.join(ds_path, split, "labels")

            # Skip missing splits
            if not os.path.exists(img_dir):
                continue

            for file in os.listdir(img_dir):

                # Add dataset prefix to avoid duplicate filenames
                new_name = f"{prefix}_{file}"

                # -------------------------
                # Copy image
                # -------------------------
                shutil.copy(
                    os.path.join(img_dir, file),
                    os.path.join(ROOT, "images", split, new_name),
                )

                # -------------------------
                # Prepare label paths
                # -------------------------
                label_file = file.rsplit(".", 1)[0] + ".txt"

                label_src = os.path.join(lbl_dir, label_file)

                dst_label_path = os.path.join(
                    ROOT,
                    "labels",
                    split,
                    new_name.rsplit(".", 1)[0] + ".txt",
                )

                # Validate class mapping exists
                if prefix not in CLASS_MAPPING:
                    raise ValueError(
                        f"No class mapping defined for dataset: {prefix}"
                    )

                mapping = CLASS_MAPPING[prefix]

                # -------------------------
                # Process annotations
                # -------------------------
                if os.path.exists(label_src):

                    # Read original labels
                    with open(label_src, "r") as f:
                        lines = f.readlines()

                    new_lines = []

                    # Remap each annotation line
                    for line in lines:
                        remapped = remap_label_line(line, mapping)

                        # Keep only valid mapped labels
                        if remapped:
                            new_lines.append(remapped)

                    # Save remapped labels
                    with open(dst_label_path, "w") as f:
                        f.write("\n".join(new_lines))

                else:
                    # Create empty label file if no annotations exist
                    open(dst_label_path, "w").close()


def create_yaml():
    """
    Create YOLO dataset configuration file (data.yaml).
    """

    yaml = f"""
path: {ROOT}
train: images/train
val: images/valid
test: images/test

names:
  0: Human
  1: Tent
"""

    with open(f"{ROOT}/data.yaml", "w") as f:
        f.write(yaml.strip())


def dataset_exists():
    """
    Check whether the merged dataset already exists.
    Prevents unnecessary reprocessing.
    """

    required_paths = [
        f"{ROOT}/images/train",
        f"{ROOT}/images/valid",
        f"{ROOT}/images/test",
        f"{ROOT}/labels/train",
        f"{ROOT}/labels/valid",
        f"{ROOT}/labels/test",
        f"{ROOT}/data.yaml",
    ]

    return all(os.path.exists(path) for path in required_paths)


def main():

    # Skip preprocessing if dataset already exists
    if dataset_exists():
        print("Dataset already exists. Skipping preprocessing.")
        return

    # Remove old temporary directory if it exists
    if os.path.exists(TMP):
        shutil.rmtree(TMP)

    os.makedirs(TMP, exist_ok=True)

    # Load Roboflow API key from environment variables
    api_key = os.getenv("ROBOFLOW_API_KEY")

    # Initialize Roboflow client
    rf = Roboflow(api_key=api_key)

    # Create output directory structure
    ensure_dirs()

    # Download datasets
    paths = download_all(rf)

    # Merge datasets into unified structure
    merge_dataset(paths)

    # Generate YOLO configuration file
    create_yaml()

    print("Dataset ready in ./data")

    # Remove temporary downloaded files
    if os.path.exists(TMP):
        shutil.rmtree(TMP)

    print("Temporary files deleted.")


# Entry point of the script
if __name__ == "__main__":
    main()