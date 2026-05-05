from ultralytics import YOLO


data_path = "data/data.yaml"
hyp_path = "preproc_pipeline/hyp.yaml"

if __name__ == '__main__':
    model = YOLO("yolov5s.pt")   

    #training config
    train_config = {
        "data": data_path,
        "imgsz": 640,
        "batch": 16,
        "epochs": 50,
        "device": "cuda", 
        "workers": 4,
        "cfg": hyp_path,                       
        "project": "Object-Detection-Human-Tent", # Groups results in WandB
        "name": "yolov5s_v1"                   # Name of this specific experiment
    }

    model.train(**train_config)