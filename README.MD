# DeepSORT - A Deep Learning-Based Object Tracking System

## Overview
------------

DeepSORT is a deep learning-based object tracking system designed for real-time video analysis and surveillance applications. It leverages the power of AI to detect, track, and identify objects in a scene with high accuracy and precision.

## Features
------------

*   **Real-time Object Detection**: DeepSORT uses a deep neural network (YOLOv5) to detect objects in the input video stream.
*   **Object Tracking**: The system tracks detected objects across frames using a Kalman filter-based approach.
*   WIP: **Re-Identification**: When an object re-enters the scene, DeepSORT employs a re-identification algorithm based on Siamese networks and a cosine similarity metric to identify it as the same object.
*   **Multiple Object Tracking**: The system can track multiple objects simultaneously.

## Configuration
-----------------

The DeepSORT system can be configured through a JSON configuration file (`config.json` by default). The following parameters can be adjusted:

```json
{
    "tracker": {
        "max_lost": 50, // Maximum number of frames an object can be lost before being considered as permanently lost (default: 50)
        "iou_threshold": 0.01 // IOU threshold for matching detections to tracklets (default: 0.01)
    },
    "model": {
        "confidence_threshold": 0.6 // Confidence threshold for object detection (default: 0.6)
    },
    "kalman_filter": {
        "state_dim": 4, // State dimensionality (default: 4) - typically used to model position and velocity
        "time_step": 1.0, // Time step of the Kalman filter (default: 1.0) - in seconds
        "position_std": 0.05, // Standard deviation for position uncertainty (default: 0.05)
        "velocity_std": 0.00625, // Standard deviation for velocity uncertainty (default: 0.00625)
        "covariance_scaling": {
            "position": 2, // Scaling factor for position covariance (default: 2)
            "velocity": 10 // Scaling factor for velocity covariance (default: 10)
        },
        "innovation_noise_factor": {
            "position": 0.1, // Noise factor for position innovation (default: 0.1)
            "velocity": 0.01 // Noise factor for velocity innovation (default: 0.01)
        }
    }
}```

## Usage
---------

### Before you start

Before running script run this command in project main directory: `export QT_QPA_PLATFORM=offscreen`

DO NOT touch `config_loader.py` or `config.json`, you can only modify values of each parameter in `config.json`

### Installation

1.  Clone the repository: `git clone https://github.com/Seelf/DeepSort.git`
2.  Install the required dependencies:
    *   `pip install -r requirements.txt` (includes PyTorch, OpenCV, and other dependencies)
3.  Create a conda environment for the project using `conda create --name deepsort python=3.8`

### Running the Code

1.  Prepare your video input (e.g., a CSV file containing bounding box coordinates)
2.  Run `python -m <folder_with_project_files_name>.main`
3.  Program will ask for path to video to analyze
4.  Program will save analyzed video in the same place as input video with extension '_output.avi'

## Requirements
-------------

*   Python 3.8 or later
*   PyTorch 1.9 or later
*   OpenCV 4.5 or later

## Example Use Cases
-------------------

*   **Surveillance Monitoring**: Deploy DeepSORT in a surveillance system to detect and track individuals across a scene.
*   **Traffic Analysis**: Utilize the system for traffic monitoring, tracking vehicles, pedestrians, and other objects.
*   **Human-Computer Interaction**: Apply DeepSORT in human-computer interaction applications, such as gesture recognition and hand tracking.

## Contributing
--------------

Feel free to contribute to this project by submitting pull requests or reporting issues. We welcome contributions from the community!

### Code of Conduct

We are committed to maintaining a welcoming and inclusive environment for all contributors. Please see our [Code of Conduct](./CODE_OF_CONDUCT.md) for more information.

## License
----------

DeepSORT is released under the MIT License. Please see [LICENSE](./LICENSE) for details.

## Contact
------------

For questions, suggestions, or collaborations, please reach out to us at <your email address>.

## Directory Structure
------------------------

```bash
DeepSort/
|---- README.md
|---- requirements.txt
|---- config_loader.py
|---- config.json
|---- main.py
|---- __init__.py
|---- src/
|       |---- detector.py
|       |---- iou_match.py
|       |---- kalman_filter.py
|       |---- main_tracker.py
|       |---- object_tracker.py
|       |---- utils.py
