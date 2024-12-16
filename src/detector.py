
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    results = model(frame)
    return results.xyxy[0].cpu().numpy()
