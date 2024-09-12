# VideoReID
Detect, track and re-identify people across frames in a video.

This project implements a video-based person re-identification (Re-ID) system using computer vision and deep learning techniques. Here's a detailed explanation of what the project does:

## Project Overview

1. Setup and Initialization:
   - The script imports necessary libraries, including PyTorch, CLIP, OpenCV, and YOLO.
   - It sets up hyperparamters for the tracking algorithm.
   - The device (GPU or CPU) is configured for processing.
   - CLIP model is loaded for generating image embeddings.
   - YOLO model is loaded for person detection.

2. Utility Functions:
   - `get_clip_embedding()`: Generates CLIP embeddings for given images.
   - `cosine_similarity()`: Calculates similarity between two vectors.
   - `iou()`: Computes Intersection over Union for bounding boxes.

3. Person Class:
   - Represents a tracked person with attributes like ID, embedding history, and bounding box history.
   - Methods for updating person information and managing detection history.

4. Tracking Update Function:
   - `update_tracking()`: Matches new detections with existing tracked persons or creates new person entries.

5. Video Processing Function:
   - `process_video()`: Main function for processing the input video.
   - Steps:
     a. Open input video and prepare output video writer.
     b. For each frame:
        - Detect persons using YOLO.
        - For each detected person:
          * Extract person image from frame.
          * Generate CLIP embedding for the person.
          * Update tracking information.
          * Draw bounding box and ID on the frame.
        - Update and clean up tracking data.
        - Write processed frame to output video.

6. Main Execution:
   - When run as a script, it processes a video file named "input_video.mp4" and outputs "output_video.mp4".

## Detailed Process Flow

1. The script reads frames from the input video.
2. For each frame, it uses YOLO to detect people.
3. For each detected person:
   - It extracts the person's image from the frame.
   - Generates a CLIP embedding for the person's image.
   - Compares this new detection with all existing tracked persons:
     * Calculates appearance similarity using CLIP embeddings.
     * Calculates spatial similarity using IoU of bounding boxes.
     * Combines these similarities with predefined weights.
   - If the best match exceeds a threshold, it updates the existing person's data.
   - If no good match is found, it creates a new person entry.
4. The script then updates the tracking data:
   - Increments the "last seen" counter for all persons.
   - Removes persons not seen for a long time.
5. Finally, it draws bounding boxes and IDs on the frame and writes it to the output video.

This system allows for tracking and re-identifying persons across video frames, even if they temporarily leave and re-enter the frame. It uses a combination of appearance (CLIP embeddings) and spatial (bounding box) information for robust tracking.

## Output videos

The processed output videos, tracking people and reassigning them their original ids even after occlusion (hiding behind a table, moving out of the frame partially or completely, another person or a hand coming in front and eclipsing the person behind from view, etc.) generated using the above script can be found [here.](https://drive.google.com/drive/folders/17XVGBcbJRX85lA1cjXB1B32V5q_uGtAD?usp=sharing)

For videos where there was little to no occlusion, using bytetrack (code in the block below) worked just fine out of the box.
 ```python
from ultralytics import YOLO

model = YOLO('yolov10x.pt')
results = model.track(source="input_video.mp4", save=True, classes=[0], conf=0.1, tracker="bytetrack.yaml")
 ```
Output videos tracked using this can be found [here.](https://drive.google.com/drive/folders/1gWxWiGYwfJlBXrUHCym6J1Zk6gO6rauU?usp=sharing)
