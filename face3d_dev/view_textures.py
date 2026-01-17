import cv2
import matplotlib.pyplot as plt
import os

def view_textures(output_dir="output"):
    """
    Displays the generated texture files to see which views were blended.
    """
    texture_files = [f for f in os.listdir(output_dir) if f.endswith("_texture.jpg")]

    if not texture_files:
        print("No texture files found in output folder.")
        return

    print(f"Found {len(texture_files)} texture files.")

    fig, axes = plt.subplots(1, len(texture_files), figsize=(15, 5))
    if len(texture_files) == 1:
        axes = [axes]

    for idx, filename in enumerate(texture_files):
        path = os.path.join(output_dir, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(img_rgb)
        axes[idx].set_title(filename)
        axes[idx].axis('off')

    plt.tight_layout()
    print("Displaying textures... Close window to exit.")
    plt.show()

if __name__ == "__main__":
    view_textures()
