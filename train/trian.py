from ultralytics import YOLO

model = YOLO("yolov5s.pt")   

data_path = "data/data.yaml"
hyp_path = "hyp.yaml"

#training config
train_config = {
    "data": data_path,
    "imgsz": 640,
    "batch": 16,
    "epochs": 50,
    "device": "cuda", 
    "workers": 4,
    "hyp": hyp_path,
}