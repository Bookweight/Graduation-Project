from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")
yamlpath=os.path.abspath("data.yaml")
results = model.train(data=yamlpath, epochs=30, workers=0)

results = model.val(data=yamlpath)