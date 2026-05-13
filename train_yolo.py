from ultralytics import YOLO

# Load pre-trained YOLO model (lightweight)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",   # path to your dataset config
    epochs=20,          # training rounds
    imgsz=640           # image size
)