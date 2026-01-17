"""
Simplified Basel Face Model reconstruction.
Uses bounding box fitting instead of landmark correspondence optimization.
"""
import cv2
import numpy as np
import os
from data_manager import ImageManager
from landmark_utils import LandmarkDetector
from basel_face_model import BaselFaceModel

# Optional: trimesh for GLB export
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def fit_basel_to_face_bbox(bfm, landmarks_2d, img_shape):
    """
    Fit Basel model by scaling/positioning to match face bounding box.

    Args:
        bfm: BaselFaceModel instance
        landmarks_2d: MediaPipe 2D landmarks (N, 2)
        img_shape: (height, width, channels)

    Returns:
        vertices: Fitted Basel model vertices (N, 3)
    """
    h, w = img_shape[:2]

    # Calculate bounding box of detected face landmarks
    landmarks_array = np.array(landmarks_2d)
    face_min = landmarks_array.min(axis=0)
    face_max = landmarks_array.max(axis=0)
    face_center = (face_min + face_max) / 2
    face_size = face_max - face_min

    print(f"  Face bounding box:")
    print(f"    Center: ({face_center[0]:.1f}, {face_center[1]:.1f})")
    print(f"    Size: {face_size[0]:.1f} x {face_size[1]:.1f} pixels")

    # Generate average Basel face
    vertices = bfm.generate_face()

    # Calculate Basel model bounding box in XY plane
    model_min = vertices[:, :2].min(axis=0)
    model_max = vertices[:, :2].max(axis=0)
    model_center = (model_min + model_max) / 2
    model_size = model_max - model_min

    print(f"  Basel model (before fitting):")
    print(f"    Center: ({model_center[0]:.1f}, {model_center[1]:.1f})")
    print(f"    Size: {model_size[0]:.1f} x {model_size[1]:.1f}")

    # Calculate scale to match face size
    # Use the larger dimension to ensure full coverage
    scale_x = face_size[0] / model_size[0]
    scale_y = face_size[1] / model_size[1]
    scale = max(scale_x, scale_y) * 1.1  # 10% larger for safety

    print(f"  Scale factor: {scale:.2f}")

    # Scale all vertices uniformly
    vertices_scaled = vertices * scale

    # Recalculate center after scaling
    model_center_scaled = (vertices_scaled[:, :2].min(axis=0) + vertices_scaled[:, :2].max(axis=0)) / 2

    # Translate to align centers
    translation_xy = face_center - model_center_scaled
    vertices_scaled[:, 0] += translation_xy[0]
    vertices_scaled[:, 1] += translation_xy[1]

    # Set Z depth reasonably (negative = into screen in CV coordinates)
    # Put the face at a reasonable depth
    vertices_scaled[:, 2] = vertices_scaled[:, 2] * scale * 0.5

    print(f"  Translation: ({translation_xy[0]:.1f}, {translation_xy[1]:.1f})")
    print(f"  Final model positioned in image")

    return vertices_scaled


def project_texture_direct(bfm_vertices, bfm_faces, frontal_img):
    """
    Direct texture projection: map vertex XY positions to image pixels.

    Args:
        bfm_vertices: Basel vertices (N, 3)
        bfm_faces: Triangle faces (M, 3)
        frontal_img: Frontal face image

    Returns:
        texture_img: Texture image
        vertex_uvs: UV coordinates (N, 2)
    """
    h_img, w_img = frontal_img.shape[:2]

    # Map vertex XY positions directly to UV coordinates
    vertex_uvs = np.zeros((len(bfm_vertices), 2))
    vertex_uvs[:, 0] = bfm_vertices[:, 0] / w_img  # U
    vertex_uvs[:, 1] = bfm_vertices[:, 1] / h_img  # V

    # Clamp to [0, 1]
    vertex_uvs = np.clip(vertex_uvs, 0.0, 1.0)

    # Use frontal image as texture
    return frontal_img, vertex_uvs


def save_basel_obj(vertices, faces, texture_img, vertex_uvs, output_dir, subject_id):
    """
    Save Basel model as textured OBJ/MTL/GLB files.
    """
    obj_path = os.path.join(output_dir, f"{subject_id}_basel_simple.obj")
    mtl_path = os.path.join(output_dir, f"{subject_id}_basel_simple.mtl")
    texture_path = os.path.join(output_dir, f"{subject_id}_basel_simple_texture.jpg")

    # Save texture
    cv2.imwrite(texture_path, texture_img)
    print(f"  Saved texture: {texture_path}")

    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl basel_mat\n")
        f.write(f"Ka 1.0 1.0 1.0\n")
        f.write(f"Kd 1.0 1.0 1.0\n")
        f.write(f"Ks 0.0 0.0 0.0\n")
        f.write(f"illum 1\n")
        f.write(f"map_Kd {subject_id}_basel_simple_texture.jpg\n")

    print(f"  Saved material: {mtl_path}")

    # Write OBJ file
    with open(obj_path, 'w') as f:
        f.write(f"# Basel Face Model - Simple Reconstruction\n")
        f.write(f"# Subject: {subject_id}\n")
        f.write(f"mtllib {subject_id}_basel_simple.mtl\n\n")

        # Vertices (flip Y for OBJ convention)
        for v in vertices:
            f.write(f"v {v[0]:.6f} {-v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Texture coordinates (flip V for proper orientation)
        for uv in vertex_uvs:
            f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")

        f.write("\n")

        # Faces
        f.write(f"usemtl basel_mat\n")
        for face in faces:
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")

    print(f"  Saved OBJ: {obj_path}")

    # Convert to GLB
    if HAS_TRIMESH:
        glb_path = os.path.join(output_dir, f"{subject_id}_basel_simple.glb")
        try:
            mesh = trimesh.load(obj_path, process=False, force='mesh')
            mesh.export(glb_path, file_type='glb', include_normals=True)
            print(f"  Saved GLB: {glb_path}")
        except Exception as e:
            print(f"  GLB export failed: {e}")


def reconstruct_simple(subject, detector, bfm, output_dir="output"):
    """
    Simplified Basel reconstruction using bounding box fitting.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*60}")
    print(f"Basel Simple Reconstruction: {subject.subject_id}")
    print(f"{'='*60}")

    # Load frontal image
    if not subject.frontal_path or not os.path.exists(subject.frontal_path):
        print(f"Error: Missing frontal image")
        return

    frontal_img = cv2.imread(subject.frontal_path)
    if frontal_img is None:
        print(f"Error: Could not load frontal image")
        return

    h, w = frontal_img.shape[:2]
    print(f"Frontal image: {w}x{h}")

    # Detect landmarks
    frontal_rgb = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2RGB)
    results = detector.face_mesh.process(frontal_rgb)

    if not results.multi_face_landmarks:
        print(f"Error: No face detected")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_2d = [[lm.x * w, lm.y * h] for lm in landmarks]

    print(f"Detected {len(landmarks)} landmarks")

    # Fit Basel model to face bounding box
    print(f"\nFitting Basel model...")
    bfm_vertices = fit_basel_to_face_bbox(bfm, landmarks_2d, frontal_img.shape)

    # Create texture
    print(f"\nCreating texture...")
    texture, vertex_uvs = project_texture_direct(bfm_vertices, bfm.faces, frontal_img)

    # Save results
    print(f"\nSaving results...")
    save_basel_obj(bfm_vertices, bfm.faces, texture, vertex_uvs, output_dir, subject.subject_id)

    print(f"\n{'='*60}")
    print(f"[OK] Reconstruction complete!")
    print(f"  Vertices: {len(bfm_vertices):,}")
    print(f"  Faces: {len(bfm.faces):,}")
    print(f"  View in: https://gltf-viewer.donmccurdy.com/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("="*60)
    print("BASEL FACE MODEL - SIMPLIFIED APPROACH")
    print("="*60)

    model_path = "models/model2019_fullHead.h5"
    if not os.path.exists(model_path):
        print(f"\nError: Basel model not found at {model_path}")
        exit(1)

    print("\nLoading Basel Face Model...")
    bfm = BaselFaceModel(model_path)

    print("\nScanning for face images...")
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()
    subjects = manager.get_complete_subjects()

    if not subjects:
        print("No complete face triplets found!")
        exit(1)

    print(f"Found {len(subjects)} subjects")

    # Initialize detector
    detector = LandmarkDetector(min_confidence=0.1, refine_landmarks=False)

    # Process each subject
    for subject in subjects:
        reconstruct_simple(subject, detector, bfm)

    print("\n" + "="*60)
    print("ALL RECONSTRUCTIONS COMPLETE!")
    print("="*60)
    print(f"\nYour 3D models are in the output folder.")
    print(f"Files: *_basel_simple.obj/glb")
    print("="*60 + "\n")
