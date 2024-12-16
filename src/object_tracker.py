
from .kalman_filter import KalmanFilter
import random

class ObjectTracker:
    def __init__(self, tracker_id, measurement):
        self.tracker_id = tracker_id
        self.kalman_filter = KalmanFilter()
        self.state, self.covariance = self.kalman_filter.initiate_tracker(measurement)
        self.lost_count = 0

    def update(self, measurement):
        self.state, self.covariance = self.kalman_filter.update_tracker(self.state, self.covariance, measurement)
        self.lost_count = 0

    def predict(self):
        self.state, self.covariance = self.kalman_filter.predict_state(self.state, self.covariance)
        self.lost_count += 1

    def get_state(self):
        return self.state, self.covariance
