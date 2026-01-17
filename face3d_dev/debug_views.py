"""
Debug script to check which views are being processed successfully.
"""
import cv2
import matplotlib.pyplot as plt
from data_manager import ImageManager
from landmark_utils import LandmarkDetector

def test_landmark_detection():
    """Test landmark detection on all three views."""
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()

    # Use same settings as generate_mesh.py
    detector = LandmarkDetector(min_confidence=0.1, refine_landmarks=False)

    subjects = manager.get_complete_subjects()

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Testing Subject: {subject.subject_id}")
        print(f"{'='*60}")

        views = [
            ("Frontal", subject.frontal_path),
            ("3/4 View", subject.three_quarter_path),
            ("Profile", subject.profile_path)
        ]

        results = []

        for view_name, path in views:
            print(f"\n{view_name}: {path}")

            if not path:
                print("  ❌ Path missing")
                results.append((view_name, None, "Missing"))
                continue

            img = cv2.imread(path)
            if img is None:
                print("  ❌ Failed to load image")
                results.append((view_name, None, "Load Failed"))
                continue

            print(f"  Image size: {img.shape}")

            # Try to detect landmarks
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_results = detector.face_mesh.process(img_rgb)

            if not face_results.multi_face_landmarks:
                print("  ❌ No face detected")
                results.append((view_name, img, "No Face Detected"))
            else:
                landmarks = face_results.multi_face_landmarks[0].landmark
                print(f"  ✅ Detected {len(landmarks)} landmarks")

                # Draw landmarks on image
                h, w, _ = img.shape
                img_with_landmarks = img.copy()
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img_with_landmarks, (x, y), 1, (0, 255, 0), -1)

                results.append((view_name, img_with_landmarks, "Success"))

        # Visualize results
        print(f"\n{'='*60}")
        print("Visualization:")
        print(f"{'='*60}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (view_name, img, status) in enumerate(results):
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
            else:
                axes[idx].text(0.5, 0.5, 'No Image',
                             ha='center', va='center', fontsize=20)
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)

            title = f"{view_name}\n{status}"
            color = 'green' if status == "Success" else 'red'
            axes[idx].set_title(title, color=color, fontsize=12, weight='bold')
            axes[idx].axis('off')

        plt.suptitle(f"Subject: {subject.subject_id}", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_landmark_detection()
