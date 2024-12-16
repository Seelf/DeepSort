import numpy as np
import scipy.linalg
from ..config_loader import CONFIG

class KalmanFilter(object):
    def __init__(self):
        kalman_config = CONFIG["kalman_filter"]
        
        # Load parameters from the config
        state_dim = kalman_config["state_dim"]
        time_step = kalman_config["time_step"]
        self.position_std = kalman_config["position_std"]
        self.velocity_std = kalman_config["velocity_std"]
        self.position_scale = kalman_config["covariance_scaling"]["position"]
        self.velocity_scale = kalman_config["covariance_scaling"]["velocity"]
        self.innovation_factor = kalman_config["innovation_noise_factor"]["position"]

        # Initialize matrices
        self.motion_matrix = np.eye(2 * state_dim, 2 * state_dim)
        for i in range(state_dim):
            self.motion_matrix[i, state_dim + i] = time_step

        self.update_matrix = np.eye(state_dim, 2 * state_dim)

    def initiate_tracker(self, measurement):
        initial_position = measurement
        initial_velocity = np.zeros_like(initial_position)
        initial_state = np.r_[initial_position, initial_velocity]

        std_devs = [
            self.position_scale * self.position_std * measurement[3],
            self.position_scale * self.position_std * measurement[3],
            1e-2,
            self.position_scale * self.position_std * measurement[3],
            self.velocity_scale * self.velocity_std * measurement[3],
            self.velocity_scale * self.velocity_std * measurement[3],
            1e-5,
            self.velocity_scale * self.velocity_std * measurement[3]
        ]

        covariance_matrix = np.diag(np.square(std_devs))
        return initial_state, covariance_matrix

    def predict_state(self, state, covariance):
        position_noise = [
            self.position_std * state[3],
            self.position_std * state[3],
            1e-2,
            self.position_std * state[3]
        ]
        velocity_noise = [
            self.velocity_std * state[3],
            self.velocity_std * state[3],
            1e-5,
            self.velocity_std * state[3]
        ]
        motion_covariance = np.diag(np.square(np.r_[position_noise, velocity_noise]))

        predicted_state = np.dot(self.motion_matrix, state)
        predicted_covariance = np.linalg.multi_dot((self.motion_matrix, covariance, self.motion_matrix.T)) + motion_covariance

        return predicted_state, predicted_covariance

    def project_state(self, state, covariance):
        position_noise = [
            self.innovation_factor * self.position_std * state[3],
            self.innovation_factor * self.position_std * state[3],
            1e-1,
            self.innovation_factor * self.position_std * state[3]
        ]
        innovation_covariance = np.diag(np.square(position_noise))

        projected_state = np.dot(self.update_matrix, state)
        projected_covariance = np.linalg.multi_dot((self.update_matrix, covariance, self.update_matrix.T))
        return projected_state, projected_covariance + innovation_covariance

    def update_tracker(self, state, covariance, measurement):
        projected_state, projected_covariance = self.project_state(state, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_covariance, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.update_matrix.T).T,
            check_finite=False).T
        innovation = measurement - projected_state

        updated_state = state + np.dot(innovation, kalman_gain.T)
        updated_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_covariance, kalman_gain.T))
        return updated_state, updated_covariance

    def calculate_gating_distance(self, state, covariance, measurements, use_position_only=False):
        projected_state, projected_covariance = self.project_state(state, covariance)
        if use_position_only:
            projected_state, projected_covariance = projected_state[:2], projected_covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(projected_covariance)
        diff = measurements - projected_state
        z = scipy.linalg.solve_triangular(
            cholesky_factor, diff.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_mahalanobis = np.sum(z * z, axis=0)
        return squared_mahalanobis
