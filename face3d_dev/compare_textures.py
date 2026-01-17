"""
Compare the original frontal image with the final blended texture
to see if side views were incorporated.
"""
import cv2
import matplotlib.pyplot as plt
import os
from data_manager import ImageManager

def compare_textures():
    """Compare original frontal images with generated textures."""
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()
    subjects = manager.get_complete_subjects()

    output_dir = "output"

    for subject in subjects:
        print(f"\nComparing textures for {subject.subject_id}...")

        # Load original frontal image
        frontal_img = cv2.imread(subject.frontal_path)
        frontal_rgb = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2RGB)

        # Load 3/4 view
        quarter_img = cv2.imread(subject.three_quarter_path) if subject.three_quarter_path else None
        quarter_rgb = cv2.cvtColor(quarter_img, cv2.COLOR_BGR2RGB) if quarter_img is not None else None

        # Load profile view
        profile_img = cv2.imread(subject.profile_path) if subject.profile_path else None
        profile_rgb = cv2.cvtColor(profile_img, cv2.COLOR_BGR2RGB) if profile_img is not None else None

        # Load generated texture
        texture_path = os.path.join(output_dir, f"{subject.subject_id}_texture.jpg")
        if not os.path.exists(texture_path):
            print(f"  Texture not found: {texture_path}")
            continue

        texture_img = cv2.imread(texture_path)
        texture_rgb = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)

        # Create comparison visualization
        fig = plt.figure(figsize=(16, 8))

        # Original views
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(frontal_rgb)
        ax1.set_title("Original Frontal", fontsize=12, weight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(2, 3, 2)
        if quarter_rgb is not None:
            ax2.imshow(quarter_rgb)
            ax2.set_title("Original 3/4 View", fontsize=12, weight='bold')
        else:
            ax2.text(0.5, 0.5, 'No 3/4 View', ha='center', va='center')
            ax2.set_title("Original 3/4 View (Missing)", fontsize=12, color='red')
        ax2.axis('off')

        ax3 = plt.subplot(2, 3, 3)
        if profile_rgb is not None:
            ax3.imshow(profile_rgb)
            ax3.set_title("Original Profile", fontsize=12, weight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Profile', ha='center', va='center')
            ax3.set_title("Original Profile (Missing)", fontsize=12, color='red')
        ax3.axis('off')

        # Generated texture (large)
        ax4 = plt.subplot(2, 1, 2)
        ax4.imshow(texture_rgb)
        ax4.set_title("Generated Multi-View Blended Texture", fontsize=14, weight='bold', color='blue')
        ax4.axis('off')

        plt.suptitle(f"Subject: {subject.subject_id} - Texture Comparison", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()

        # Calculate difference between frontal and final texture
        # If they're identical, no side views were used
        frontal_resized = cv2.resize(frontal_img, (texture_img.shape[1], texture_img.shape[0]))
        diff = cv2.absdiff(frontal_resized, texture_img)
        diff_sum = diff.sum()

        print(f"  Texture difference from frontal: {diff_sum:,.0f}")
        if diff_sum < 1000:
            print("  ⚠️  WARNING: Texture is nearly identical to frontal - side views may not have been used!")
        else:
            print("  ✅ Texture differs from frontal - side views were likely incorporated")

if __name__ == "__main__":
    compare_textures()
