import numpy as np
from .object_tracker import ObjectTracker
from scipy.spatial.distance import cdist
from .iou_match import iou
import random
from ..config_loader import CONFIG

class MainTracker:
    def __init__(self):
        self.trackers = {}  # Dictionary to hold active trackers
        self.track_id = 0  # Unique ID for each tracker
        self.max_lost = CONFIG['tracker']['max_lost']  # Maximum frames a tracker can be "lost"
        self.iou_threshold = CONFIG['tracker']['iou_threshold']  # IOU threshold for matching
        self.colors = {}  # Colors for each tracker for visualization

    def update(self, detections):
        # Step 1: Predict the next state of all active trackers
        for tracker_id in list(self.trackers.keys()):
            self.trackers[tracker_id].predict()

        # Step 2: Match detections to existing trackers using IOU
        tracker_predictions = np.array([self.trackers[tracker_id].get_state()[0][:4] for tracker_id in self.trackers.keys()])
        
        matched_trackers = set()
        matched_detections = set()

        if len(detections) > 0 and len(tracker_predictions) > 0:
            iou_matrix = np.zeros((len(detections), len(tracker_predictions)))

            for i, det in enumerate(detections):
                for j, pred in enumerate(tracker_predictions):
                    iou_matrix[i, j] = iou(det, pred)

            while iou_matrix.size > 0:
                max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_iou_idx]

                if max_iou < self.iou_threshold:
                    break

                det_idx, tracker_idx = max_iou_idx
                tracker_id = list(self.trackers.keys())[tracker_idx]
                self.trackers[tracker_id].update(detections[det_idx])

                matched_trackers.add(tracker_id)
                matched_detections.add(det_idx)

                iou_matrix[det_idx, :] = -1  # Invalidate this detection
                iou_matrix[:, tracker_idx] = -1  # Invalidate this tracker

        # Step 3: Handle unmatched trackers (increment lost count)
        for tracker_id in list(self.trackers.keys()):
            if tracker_id not in matched_trackers:
                self.trackers[tracker_id].lost_count += 1

                # Remove tracker if it exceeds max_lost
                if self.trackers[tracker_id].lost_count > self.max_lost:
                    del self.trackers[tracker_id]

        # Step 4: Create new trackers for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detections:
                # Check if the detection matches any recently lost trackers
                matched_to_lost = False
                for tracker_id, tracker in self.trackers.items():
                    if tracker.lost_count > 0 and tracker.lost_count <= self.max_lost:
                        if iou(tracker.get_state()[0][:4], det) > self.iou_threshold:
                            tracker.update(det)
                            tracker.lost_count = 0
                            matched_to_lost = True
                            matched_trackers.add(tracker_id)
                            break

                if not matched_to_lost:
                    tracker = ObjectTracker(self.track_id, det)
                    self.trackers[self.track_id] = tracker
                    self.colors[self.track_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                    self.track_id += 1

        # Step 5: Collect tracked objects for output
        tracked_objects = [
            (
                tracker.tracker_id,
                tracker.get_state()[0][0],  # x1
                tracker.get_state()[0][1],  # y1
                tracker.get_state()[0][2],  # x2
                tracker.get_state()[0][3],  # y2
                self.colors[tracker.tracker_id]  # Color
            )
            for tracker in self.trackers.values()
            if tracker.lost_count == 0  # Only include active trackers
        ]

        return tracked_objects