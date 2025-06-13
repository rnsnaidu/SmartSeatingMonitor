# SmartSeatingMonitor
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/rnsnaidu/SmartSeatingMonitor.git)

## Overview

SmartSeatingMonitor is a Python-based project designed for real-time analysis of video feeds, such as those from seminar halls, classrooms, or public spaces. It utilizes computer vision techniques to detect and count people and seats, and to classify detected individuals by gender. The system provides on-screen statistics including total people, counts of males and females, total seats, and the number of unfilled seats.

## Features

*   **Real-time Person Detection:** Identifies and counts individuals in the video frame.
*   **Real-time Seat Detection:** Identifies and counts chairs/seats.
*   **Gender Classification:** Classifies detected persons as 'Male' or 'Female' and provides respective counts.
*   **Occupancy Monitoring:** Calculates and displays the number of unfilled seats.
*   **Visual Feedback:** Overlays bounding boxes, labels, and summary statistics directly onto the video feed.
*   **Flexible Input:** Supports both live webcam feeds and pre-recorded video files.

## Prerequisites

Before running the project, ensure you have the following installed:

*   Python 3.x
*   OpenCV (`opencv-python`)
*   NumPy (`numpy`)

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rnsnaidu/SmartSeatingMonitor.git
    cd SmartSeatingMonitor
    ```

2.  **Install Python Dependencies:**
    Navigate to the cloned directory and install the required Python libraries:
    ```bash
    pip install opencv-python numpy
    ```

3.  **Model Files:**
    This repository includes all the necessary pre-trained model files required by the `seminar_hall_analysis.py` script. These files should be present in the root directory of the project after cloning:
    *   `yolov3.weights`
    *   `yolov3.cfg`
    *   `coco.names`
    *   `gender_deploy.prototxt`
    *   `gender_net.caffemodel`

    The script will automatically check for these files upon execution.

## Usage

To run the seminar hall analysis, execute the `seminar_hall_analysis.py` script from the terminal:

```bash
python seminar_hall_analysis.py
```

**Video Source:**
*   **Webcam:** By default, the script uses the primary webcam (indexed as `0`).
*   **Video File:** To use a pre-recorded video file, you need to modify the `video_source` variable within the `seminar_hall_analysis.py` script. For example:
    ```python
    # In seminar_hall_analysis.py
    # video_source = 0 # Default: webcam
    video_source = "path/to/your/video.mp4" # Example: for a video file
    ```

**Exiting the Application:**
Press the 'q' key while the video feed window is active to stop the script and close the window.

## File Descriptions

*   `seminar_hall_analysis.py`: The main Python script that orchestrates the video processing, object detection (people and chairs), gender classification, and display of results.
*   `yolov3.weights`: Pre-trained weights for the YOLOv3 (You Only Look Once version 3) object detection model.
*   `yolov3.cfg`: Configuration file defining the architecture of the YOLOv3 model.
*   `coco.names`: A text file containing the list of object class names that the YOLOv3 model (trained on the COCO dataset) can detect. Used to map detected class IDs to human-readable names like 'person' or 'chair'.
*   `gender_deploy.prototxt`: Caffe model definition (prototxt) file describing the architecture of the neural network used for gender classification.
*   `gender_net.caffemodel`: Pre-trained weights for the gender classification Caffe model.
*   `deploy.prototxt`: Caffe model definition for an SSD (Single Shot MultiBox Detector) based face detector using a ResNet-10 backbone. (Note: This file is included in the repository but is not actively used by the `seminar_hall_analysis.py` script, which uses YOLO for person detection.)
*   `res10_300x300_ssd_iter_140000.caffemodel`: Pre-trained Caffe model weights for the SSD-based face detector. (Note: This file is included in the repository but is not actively used by the `seminar_hall_analysis.py` script.)

📊 Example Output
When running the script:

Bounding boxes around persons and chairs

Gender labels next to detected persons

Console output:
Total People: 5 (Males: 3, Females: 2)
Total Seats: 10 | Unfilled Seats: 5

📌 Features
✅ YOLOv3 for fast object detection
✅ Gender prediction using face crops and Caffe model
✅ Real-time seat analytics
✅ Works with webcam or video input


##OUTPUTs..

##BEFORE

![hall_photo](https://github.com/user-attachments/assets/27a06c05-54c6-413a-a443-a176e8bd96b5)
    
 ##AFTER

![output_sample](https://github.com/user-attachments/assets/f0d83791-0327-4c9f-b50e-d73597a4da68)


🤝 Acknowledgements
YOLOv3 from pjreddie/darknet

Gender models from learnopencv

OpenCV DNN Module

📮 Contact
Maintained by RNS Naidu

GitHub: https://github.com/rnsnaidu


