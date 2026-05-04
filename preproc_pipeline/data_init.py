from roboflow import Roboflow
import shutil
import dotenv
import os

dotenv.load_dotenv()
ROOT = "data"
TMP = "tmp_datasets"

'''
0 -> Human
1 -> Tent
'''
CLASS_MAPPING = {
    "heridal-lrbkc-keibe": {0: 0},  
    "tent-models-fkihy": {0: 1},    
    "sarso_tent-o6bqt": {0: 1},     
}

def ensure_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{ROOT}/labels/{split}", exist_ok=True)


def download_all(rf):
    datasets = [
        "heridal-lrbkc-keibe",
        "tent-models-fkihy",
        "sarso_tent-o6bqt",
    ]

    paths = []

    for name in datasets:
        project = rf.workspace("quoc-hoang-vu-van").project(name)
        version = project.version(1)

        out_dir = os.path.join(TMP, name)
        version.download("yolov5", location=out_dir)

        paths.append(out_dir)

    return paths

def remap_label_line(line, mapping):
    parts = line.strip().split()
    cls = int(parts[0])

    if cls not in mapping:
        return None  

    new_cls = mapping[cls]
    return " ".join([str(new_cls)] + parts[1:])


def merge_dataset(dataset_paths):
    for ds_path in dataset_paths:
        prefix = os.path.basename(ds_path)

        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(ds_path, split, "images")
            lbl_dir = os.path.join(ds_path, split, "labels")

            if not os.path.exists(img_dir):
                continue

            for file in os.listdir(img_dir):
                new_name = f"{prefix}_{file}"

                # copy image
                shutil.copy(
                    os.path.join(img_dir, file),
                    os.path.join(ROOT, "images", split, new_name),
                )

                # copy label
                label_file = file.rsplit(".", 1)[0] + ".txt"
                label_src = os.path.join(lbl_dir, label_file)

                dst_label_path = os.path.join(
                    ROOT,
                    "labels",
                    split,
                    new_name.rsplit(".", 1)[0] + ".txt",
                )

                if prefix not in CLASS_MAPPING:
                    raise ValueError(f"No class mapping defined for dataset: {prefix}")
                
                mapping = CLASS_MAPPING[prefix]

                if os.path.exists(label_src):
                    with open(label_src, "r") as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        remapped = remap_label_line(line, mapping)
                        if remapped:
                            new_lines.append(remapped)

                    with open(dst_label_path, "w") as f:
                        f.write("\n".join(new_lines))
                else:
                    open(dst_label_path, "w").close()

def create_yaml():
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

def main():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)

    ensure_dirs()
    paths = download_all(rf)
    merge_dataset(paths)
    create_yaml()

    print("Dataset ready in ./data")
    
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    print("Temporary files deleted.")

if __name__ == "__main__":
    main()
                