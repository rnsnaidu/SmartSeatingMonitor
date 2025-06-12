# seminar_hall_analysis.py
#
# A Python script to perform real-time analysis of a seminar hall video feed.
#
# Project Goals:
# 1. Detect and count the total number of people.
# 2. Detect and count the number of unfilled seats.
# 3. Classify and count the number of males and females.
#
# To run this project, you need:
# 1. Python installed.
# 2. OpenCV and NumPy libraries. Install them using pip:
#    pip install opencv-python numpy
#
# 3. Pre-trained Model Files:
#    You must download these files and place them in the same folder as this script.
#
#    a) YOLOv3 Model (for object detection):
#       - Weights: Download from https://pjreddie.com/media/files/yolov3.weights
#       - Config: Download from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
#       - Class Names: Download from https://github.com/pjreddie/darknet/blob/master/data/coco.names
#
#    b) Gender Classification Model:
#       - Prototxt: Download from https://github.com/spmallick/learnopencv/blob/master/Gender-Detection/gender_deploy.prototxt
#       - Caffemodel: Download from https://github.com/spmallick/learnopencv/blob/master/Gender-Detection/gender_net.caffemodel

import cv2
import numpy as np
import os

# --- Configuration and Model Loading ---

# Confidence threshold to filter out weak detections
CONFIDENCE_THRESHOLD = 0.5
# Non-Maximum Suppression threshold to remove overlapping boxes
NMS_THRESHOLD = 0.3

# Paths to the model files
# Ensure these files are in the same directory as the script
yolo_weights_path = "yolov3.weights"
yolo_config_path = "yolov3.cfg"
coco_names_path = "coco.names"

gender_prototxt_path = "gender_deploy.prototxt"
gender_model_path = "gender_net.caffemodel"

# --- Helper Function to Check for Model Files ---
def check_model_files():
    """Checks if all necessary model files exist."""
    files = [
        yolo_weights_path, yolo_config_path, coco_names_path,
        gender_prototxt_path, gender_model_path
    ]
    missing_files = [f for f in files if not os.path.exists(f)]
    if missing_files:
        print("ERROR: Missing model files!")
        print("Please download the following files and place them in the script's directory:")
        for f in missing_files:
            print(f"- {f}")
        return False
    return True

# --- Load Models ---
print("[INFO] Loading models...")

# Load YOLO object detector trained on COCO dataset
# This model can detect 80 different objects, including 'person' and 'chair'
net_obj = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# Get the names of all output layers in the YOLO network
layer_names = net_obj.getLayerNames()
output_layers = [layer_names[i - 1] for i in net_obj.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load Gender Classification model
# This model takes a face image and predicts 'Male' or 'Female'
gender_net = cv2.dnn.readNet(gender_model_path, gender_prototxt_path)
GENDER_LIST = ['Male', 'Female']
# The gender model expects a 227x227 input image
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# --- Video Input ---
# To use a webcam, set video_source = 0
# To use a video file, set video_source = "path/to/your/video.mp4"
video_source = 0 # Use the primary webcam
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {video_source}")
    exit()

print("[INFO] Starting video stream analysis... Press 'q' to quit.")

# --- Main Processing Loop ---
while True:
    # Read one frame from the video source
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video stream or error reading frame.")
        break

    # Get frame dimensions
    (H, W) = frame.shape[:2]

    # --- Object Detection (YOLO) ---
    # Create a blob from the image and perform a forward pass of YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net_obj.setInput(blob)
    layerOutputs = net_obj.forward(output_layers)

    # Initialize lists to store detection details
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # We are only interested in 'person' and 'chair'
            if classes[classID] in ["person", "chair"]:
                if confidence > CONFIDENCE_THRESHOLD:
                    # Scale bounding box coordinates back to the original image size
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Initialize counters for the current frame
    person_count = 0
    chair_count = 0
    male_count = 0
    female_count = 0

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            class_name = classes[classIDs[i]]
            
            # --- Analysis and Drawing ---
            if class_name == "person":
                person_count += 1
                color = (0, 255, 0) # Green for people
                
                # --- Gender Classification ---
                # Crop the person's region for gender analysis. Add padding.
                face_crop_box = [max(0, x - 20), max(0, y - 20), w + 40, h + 40]
                face = frame[face_crop_box[1]:face_crop_box[1]+face_crop_box[3], face_crop_box[0]:face_crop_box[0]+face_crop_box[2]]

                if face.shape[0] > 0 and face.shape[1] > 0:
                    # Create a blob from the face crop and classify
                    face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    gender_net.setInput(face_blob)
                    gender_preds = gender_net.forward()
                    gender = GENDER_LIST[gender_preds[0].argmax()]

                    if gender == "Male":
                        male_count += 1
                    else:
                        female_count += 1
                    
                    label = f"Person: {gender}"
                else:
                    label = "Person"

                # Draw bounding box and label for the person
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            elif class_name == "chair":
                chair_count += 1
                color = (255, 165, 0) # Blue-Orange for chairs
                label = "Seat"
                
                # Draw bounding box and label for the chair
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Display Information on Frame ---
    # Calculate unfilled seats. Ensure it's not negative.
    unfilled_seats = max(0, chair_count - person_count)

    # Create a black rectangle at the top for text overlay
    info_y = 30
    cv2.rectangle(frame, (0, 0), (W, 110), (0, 0, 0), -1)

    # Display counts
    info_text_1 = f"Total People: {person_count} (Males: {male_count}, Females: {female_count})"
    info_text_2 = f"Total Seats: {chair_count} | Unfilled Seats: {unfilled_seats}"
    
    cv2.putText(frame, info_text_1, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, info_text_2, (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, info_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the output frame
    cv2.imshow("Seminar Hall Real-Time Analysis", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("[INFO] Cleaning up and shutting down.")
cap.release()
cv2.destroyAllWindows()

# --- Final Check ---
if not check_model_files():
    # This check runs at the end to remind the user if files were missing
    print("\nPlease ensure the model files are downloaded before running again.")
else:
    print("[INFO] Script finished successfully.")
