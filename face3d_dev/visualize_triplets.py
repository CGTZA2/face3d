import cv2
import matplotlib.pyplot as plt
from data_manager import ImageManager
from landmark_utils import LandmarkDetector
import os

def visualize_subject(subject, detector):
    """
    Visualizes the triplet for a single subject using Matplotlib.
    """
    try:
        # Get paths from our data class
        paths = subject.get_paths()
        titles = ["Frontal", "3/4 View", "Profile"]
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Subject ID: {subject.subject_id}", fontsize=16)
        
        for i, path in enumerate(paths):
            if not os.path.exists(path):
                print(f"File missing: {path}")
                continue

            # Load image using OpenCV
            img = cv2.imread(path)
            if img is None:
                print(f"Error loading image data: {path}")
                continue
                
            # Detect landmarks
            landmarks = detector.get_landmarks(img)
            if landmarks is not None:
                # Draw small circles for each landmark
                for (x, y) in landmarks:
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            else:
                print(f"Warning: No landmarks detected for {titles[i]}")

            # Convert BGR (OpenCV default) to RGB (Matplotlib default)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(titles[i])
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing subject {subject.subject_id}: {e}")

if __name__ == "__main__":
    # Initialize manager pointing to your image directory
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()
    
    # Initialize the detector once
    detector = LandmarkDetector()
    
    subjects = manager.get_complete_subjects()
    
    if not subjects:
        print("No complete triplets found. Please check your 'images/bm' folder.")
    else:
        print(f"Found {len(subjects)} complete subjects. Displaying the first 3...")
        # Loop through the first 3 subjects found
        for subject in subjects[:3]:
            visualize_subject(subject, detector)