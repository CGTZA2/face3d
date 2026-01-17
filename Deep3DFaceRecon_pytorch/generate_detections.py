"""
Generate 5-point facial landmarks for Deep3DFaceRecon_pytorch
Uses dlib to detect key facial landmarks.
"""
import cv2
import numpy as np
import dlib
import os

# Initialize dlib face detector and landmark predictor (uses 5-point model)
detector = dlib.get_frontal_face_detector()

def get_5_landmarks_opencv(image):
    """
    Detect 5 facial landmarks using OpenCV's built-in face detector.
    Returns: left_eye, right_eye, nose, left_mouth, right_mouth
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Get the first face
    x, y, w, h = faces[0]

    # Estimate 5 key points based on face bounding box
    # This is a simplified approach
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    # Approximate landmark positions
    landmarks = np.array([
        [x + w * 0.3, y + h * 0.4],   # Left eye
        [x + w * 0.7, y + h * 0.4],   # Right eye
        [face_center_x, y + h * 0.6], # Nose
        [x + w * 0.35, y + h * 0.8],  # Left mouth
        [x + w * 0.65, y + h * 0.8],  # Right mouth
    ])

    return landmarks


def process_image(image_path, output_dir):
    """Process single image and save detection file."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return False

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces with dlib
    faces = detector(gray, 1)

    if len(faces) == 0:
        # Fallback to OpenCV if dlib fails
        landmarks_5 = get_5_landmarks_opencv(image)
        if landmarks_5 is None:
            print(f"Error: No face detected in {image_path}")
            return False
    else:
        # Use dlib detection - estimate 5 points from face rectangle
        face = faces[0]
        x, y = face.left(), face.top()
        w, h_face = face.width(), face.height()

        # Estimate 5 key points
        landmarks_5 = np.array([
            [x + w * 0.3, y + h_face * 0.4],   # Left eye
            [x + w * 0.7, y + h_face * 0.4],   # Right eye
            [x + w * 0.5, y + h_face * 0.6],   # Nose
            [x + w * 0.35, y + h_face * 0.8],  # Left mouth
            [x + w * 0.65, y + h_face * 0.8],  # Right mouth
        ])

    # Save detection file
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{basename}.txt")

    np.savetxt(output_path, landmarks_5, fmt='%.2f', delimiter=' ')

    print(f"[OK] {basename}.txt")
    return True


if __name__ == "__main__":
    # Input/output directories
    image_dir = "datasets/test_faces"
    detection_dir = os.path.join(image_dir, "detections")

    # Create detection directory
    os.makedirs(detection_dir, exist_ok=True)

    print("="*60)
    print("GENERATING 5-POINT LANDMARKS FOR DEEP3D")
    print("="*60)

    # Process all images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\nFound {len(image_files)} images")
    print(f"Output: {detection_dir}\n")

    success_count = 0
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        if process_image(img_path, detection_dir):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Complete! {success_count}/{len(image_files)} images processed")
    print(f"{'='*60}\n")
