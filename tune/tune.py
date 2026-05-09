from ultralytics import YOLO
import datetime
import os

data_path = "data/data.yaml"

if __name__ == '__main__':
    model = YOLO("yolov5s.pt")   

    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    group_name = f"yolov5s_tuning_{timestamp}"

    os.environ["WANDB_RUN_GROUP"] = group_name
    os.environ["WANDB_JOB_TYPE"] = "tuning"
    run_name = f"yolov5s_tuning_{timestamp}"

    tune_config = {
        "data": data_path,
        "imgsz": 640,
        "batch": 16,
        "epochs": 30,
        "iterations": 30,
        "device": "cuda",
        "workers": 8,     
        "project": "Object-Detection-Human-Tent", 
        "name": run_name,
        
        # Speed optimization flags for tuning
        "plots": False,
        "save": False,
        "val": False,
    }

    model.tune(**tune_config)