
import cv2
from .detector import detect_objects
from .main_tracker import MainTracker
import numpy as np
from ..config_loader import CONFIG

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    main_tracker = MainTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)
        confidence_threshold = CONFIG['model']['confidence_threshold']
        filtered_detections = [det for det in detections if det[4] > confidence_threshold and det[5] == 0]

        if len(filtered_detections) == 0:
            out.write(frame)
            continue

        filtered_detections = np.array(filtered_detections)
        filtered_detections_coords = filtered_detections[:, :4]
        tracked_objects = main_tracker.update(filtered_detections_coords)

        for obj in tracked_objects:
            x_min, y_min, x_max, y_max, color = obj[1], obj[2], obj[3], obj[4], obj[5]
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            cv2.putText(frame, f"ID: {obj[0]}", (int(x_min), int(y_min)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

    cap.release()
    out.release()
