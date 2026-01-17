"""
Simple OBJ viewer to check if Basel models have proper textures.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2
import os


def load_obj_with_texture(obj_path):
    """Load OBJ file and check texture mapping."""
    vertices = []
    texture_coords = []
    faces = []
    texture_file = None

    # Parse OBJ file
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                parts = line.split()
                texture_coords.append([float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                # Parse face with texture: f v1/vt1 v2/vt2 v3/vt3
                parts = line.split()[1:]
                face_v = []
                face_vt = []
                for p in parts:
                    indices = p.split('/')
                    face_v.append(int(indices[0]) - 1)
                    if len(indices) > 1 and indices[1]:
                        face_vt.append(int(indices[1]) - 1)
                faces.append((face_v, face_vt))

    # Load MTL file to find texture
    mtl_path = obj_path.replace('.obj', '.mtl')
    if os.path.exists(mtl_path):
        with open(mtl_path, 'r') as f:
            for line in f:
                if line.startswith('map_Kd'):
                    texture_file = line.split()[1]
                    break

    vertices = np.array(vertices)
    texture_coords = np.array(texture_coords) if texture_coords else None

    print(f"Loaded OBJ: {obj_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Texture coordinates: {len(texture_coords) if texture_coords is not None else 0}")
    print(f"  Faces: {len(faces)}")
    print(f"  Texture file: {texture_file}")

    # Load texture image
    texture_img = None
    if texture_file:
        texture_path = os.path.join(os.path.dirname(obj_path), texture_file)
        if os.path.exists(texture_path):
            texture_img = cv2.imread(texture_path)
            texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
            print(f"  Texture loaded: {texture_img.shape}")
        else:
            print(f"  WARNING: Texture file not found: {texture_path}")

    return vertices, texture_coords, faces, texture_img


def visualize_model(obj_path):
    """Visualize the Basel model."""
    vertices, tex_coords, faces, texture_img = load_obj_with_texture(obj_path)

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # 3D mesh view
    ax1 = fig.add_subplot(121, projection='3d')

    # Extract just vertex indices from faces
    face_vertices = [f[0] for f in faces[:1000]]  # Limit to first 1000 faces for speed

    # Create mesh collection
    mesh_faces = []
    for face_v in face_vertices:
        if len(face_v) >= 3:
            triangle = [vertices[face_v[0]], vertices[face_v[1]], vertices[face_v[2]]]
            mesh_faces.append(triangle)

    collection = Poly3DCollection(mesh_faces, alpha=0.7, facecolor='cyan', edgecolor='black', linewidths=0.1)
    ax1.add_collection3d(collection)

    # Set limits
    ax1.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax1.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax1.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Basel 3D Model (first 1000 faces)', fontsize=14, weight='bold')

    # Show texture image
    ax2 = fig.add_subplot(122)
    if texture_img is not None:
        ax2.imshow(texture_img)
        ax2.set_title('Texture Image', fontsize=14, weight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Texture Found', ha='center', va='center', fontsize=20)
        ax2.set_title('Texture Missing', fontsize=14, weight='bold', color='red')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        obj_path = sys.argv[1]
    else:
        # Find first Basel OBJ in output
        output_dir = "output"
        obj_files = [f for f in os.listdir(output_dir) if f.endswith('_basel.obj')]
        if obj_files:
            obj_path = os.path.join(output_dir, obj_files[0])
            print(f"Found Basel model: {obj_path}")
        else:
            print("No Basel models found in output folder.")
            print("Usage: python view_basel_obj.py [path/to/model.obj]")
            exit(1)

    visualize_model(obj_path)
