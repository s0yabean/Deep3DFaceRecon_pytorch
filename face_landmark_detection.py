import os
import cv2
from mtcnn import MTCNN
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Facial Landmark Detection using MTCNN.")
parser.add_argument('--image_folder', type=str, required=True, help="Folder containing images to process")
args = parser.parse_args()

# Define paths
data_root = 'datasets'
image_folder = os.path.join(data_root, args.image_folder)
detections_folder = os.path.join(image_folder, 'detections')

# Create the detections folder if it doesn't exist
if not os.path.exists(detections_folder):
    os.makedirs(detections_folder)

# Initialize MTCNN detector
detector = MTCNN()

# Get all image files in the specified folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

# Process each image file
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    facial_landmarks = detector.detect_faces(img)

    print(f"File: {image_file} Points: {facial_landmarks}")

    # Find the landmark with the highest confidence
    if facial_landmarks:  # Ensure there is at least one detected face
        highest_confidence_landmark = max(facial_landmarks, key=lambda x: x['confidence'])

        # Save the keypoints to a file
        txt_file = os.path.join(detections_folder, os.path.splitext(image_file)[0] + '.txt')
        with open(txt_file, 'w') as f:
            keypoints = highest_confidence_landmark['keypoints']
            for key, value in keypoints.items():
                f.write(f"{value[0]} {value[1]}\n")

        print(f"Processed {image_file} and saved keypoints to {txt_file}")
    else:
        print(f"No face detected in {image_file}")

