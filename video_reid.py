# Import necessary libraries
import torch
import clip
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import deque

# Define constants for the tracking algorithm
MAX_FRAMES_UNSEEN = 50         # Maximum number of frames a person can be unseen before being removed
PERSON_CLASS_ID = 0            # Class ID for person in YOLO model
APPEARANCE_WEIGHT = 0.75       # Weight given to appearance similarity in matching
SPATIAL_WEIGHT = 0.25          # Weight given to spatial similarity in matching
COMBINED_THRESHOLD = 0.65      # Threshold for considering a match between detections
HISTORY_SIZE = 10              # Number of historical embeddings and boxes to keep for each person
USE_AVERAGE_EMBEDDING = False  # Whether to use average of historical clip embeddings or just the latest

# Set up the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model for generating image embeddings
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load the YOLO model for person detection
yolo_model = YOLO("yolov10x.pt")

def get_clip_embedding(image):
    """ Generate CLIP embedding for an image """
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Preprocess the image: convert to PIL Image, apply CLIP preprocessing, add batch dimension, and move to device
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        # Generate CLIP embedding for the image
        embedding = clip_model.encode_image(image)
        # Convert the embedding to numpy array and return
        return embedding.cpu().numpy()

def cosine_similarity(a, b):
    """ Calculate cosine similarity between two vectors """
    # Ensure inputs are 1D tensors
    a = torch.tensor(a).squeeze()
    b = torch.tensor(b).squeeze()
    # Calculate and return cosine similarity
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def iou(box1, box2):
    """ Calculate Intersection over Union (IoU) between two bounding boxes """
    # Calculate coordinates of intersection rectangle
    x1, y1 = np.maximum(box1[:2], box2[:2])
    x2, y2 = np.minimum(box1[2:], box2[2:])
    # Calculate area of intersection
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate and return IoU
    return intersection / (area1 + area2 - intersection)


class Person:
    """ Class to represent a tracked person """

    def __init__(self, id, embedding, box):
        self.id = id  # Unique identifier for the person
        # Keep a history of embeddings and boxes
        self.embedding_history = deque([embedding], maxlen=HISTORY_SIZE)
        self.box_history = deque([box], maxlen=HISTORY_SIZE)
        self.last_seen = 0  # Frames since last detection

    def update(self, embedding, box):
        """ Update the person with new detection """
        self.embedding_history.append(embedding)
        self.box_history.append(box)
        self.last_seen = 0  # Reset last_seen counter

    def increment_last_seen(self):
        """ Increment the counter for frames since last seen """
        self.last_seen += 1

    def get_comparison_embedding(self):
        """ Get the embedding to use for comparison """
        if USE_AVERAGE_EMBEDDING:
            return np.mean(self.embedding_history, axis=0)
        else:
            return self.embedding_history[-1]


def update_tracking(new_embedding, new_box, people, next_id):
    """ Update tracking based on new detection """

    best_matching_person = None
    max_combined_similarity = -float('inf')

    # Compare new detection with all existing people
    for person in people.values():
        current_box = person.box_history[-1]
        # Calculate appearance and spatial similarities
        appearance_similarity = cosine_similarity(new_embedding, person.get_comparison_embedding())
        spatial_similarity = iou(new_box, current_box)
        
        # Combine similarities using weights
        combined_similarity = (APPEARANCE_WEIGHT * appearance_similarity + 
                               SPATIAL_WEIGHT * spatial_similarity)

        # Update best match if this is the highest similarity so far
        if combined_similarity > max_combined_similarity:
            max_combined_similarity = combined_similarity
            best_matching_person = person

    # If similarity is above threshold, update existing person
    if max_combined_similarity > COMBINED_THRESHOLD:
        best_matching_person.update(new_embedding, new_box)
        return best_matching_person.id, next_id
    
    # Otherwise, create a new person
    new_person = Person(next_id, new_embedding, new_box)
    people[next_id] = new_person

    return next_id, next_id + 1


def process_video(video_path, output_path):
    """ Process video for person tracking """

    cap = cv2.VideoCapture(video_path)
    writer = None
    people = {}
    next_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect persons in the frame
        results = yolo_model(frame, classes=[PERSON_CLASS_ID])
        
        for box in results[0].boxes.xyxy:
            # Extract person from frame
            x1, y1, x2, y2 = map(int, box.tolist())
            person_img = frame[y1:y2, x1:x2]
            
            if person_img.size == 0:
                continue

            # Generate embedding and update tracking
            embedding = get_clip_embedding(person_img)
            person_id, next_id = update_tracking(embedding, (x1, y1, x2, y2), people, next_id)

            # Draw bounding box and ID on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update last_seen for all people and remove if unseen for too long
        for person in list(people.values()):
            person.increment_last_seen()
            if person.last_seen > MAX_FRAMES_UNSEEN:
                del people[person.id]
        
        # Initialize video writer if not already done
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 30,
                                    (frame.shape[1], frame.shape[0]), True)
        
        # Write frame to output video
        writer.write(frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Process video when script is run directly
    process_video("input_video.mp4", "output_video.mp4")
