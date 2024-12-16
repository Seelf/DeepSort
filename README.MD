Before running script run this command in project main directory: `export QT_QPA_PLATFORM=offscreen`

DO NOT touch `config_loader.py` or `config.json`, you can only modify values of each parameter in `config.json`

To run this project simply clone this repo `git clone https://github.com/Seelf/DeepSort` 
and the run `python -m <folder_with_project_files_name>.main`


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

## Usage
---------

### Installation

1.  Clone the repository: `git clone https://github.com/Seelf/DeepSort.git`
2.  Install the required dependencies:
    *   `pip install -r requirements.txt` (includes PyTorch, OpenCV, and other dependencies)
3.  Create a conda environment for the project using `conda create --name deepsort python=3.8`

### Running the Code

1.  Prepare your video input (e.g., a CSV file containing bounding box coordinates)
2.  Run the DeepSORT tracker using Python: `python main.py --video_file <input_video_file>`

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
|---- LICENSE
|---- requirements.txt
|---- conda.yml (optional)
|---- data/
|       |---- videos/
|       |---- bounding_box_data.csv
|---- src/
|       |---- main.py
|       |---- model.py
|       |---- utils.py
|       |---- yolo/
|              |---- yolov3.cfg
|              |---- yolov3.weights
|---- models/
|       |---- siamese_net.pth
|       |---- kalman_filter.pth
|---- results/
|       |---- video_tracking_results.csv
