"""
Proper Basel Face Model Reconstruction with Correct Texture Mapping
Uses Basel's official UV coordinates for texture projection.
"""
import cv2
import numpy as np
import os
from data_manager import ImageManager
from landmark_utils import LandmarkDetector
from basel_face_model import BaselFaceModel
from basel_texture_projector import BaselTextureProjector

# Optional: trimesh for GLB export
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def fit_basel_bbox(bfm, landmarks_2d, img_shape):
    """
    Fit Basel model using bounding box alignment.
    Simple but effective for frontal faces.
    """
    h, w = img_shape[:2]

    # Calculate face bounding box
    landmarks_array = np.array(landmarks_2d)
    face_min = landmarks_array.min(axis=0)
    face_max = landmarks_array.max(axis=0)
    face_center = (face_min + face_max) / 2
    face_size = face_max - face_min

    print(f"  Face bbox: center=({face_center[0]:.0f}, {face_center[1]:.0f}), size={face_size[0]:.0f}x{face_size[1]:.0f}")

    # Generate average Basel face
    vertices = bfm.generate_face()

    # Calculate Basel model bounding box
    model_min = vertices[:, :2].min(axis=0)
    model_max = vertices[:, :2].max(axis=0)
    model_center = (model_min + model_max) / 2
    model_size = model_max - model_min

    # Scale to match face size
    scale = max(face_size[0] / model_size[0], face_size[1] / model_size[1]) * 1.1
    vertices_scaled = vertices * scale

    # Recalculate center and translate
    model_center_scaled = (vertices_scaled[:, :2].min(axis=0) + vertices_scaled[:, :2].max(axis=0)) / 2
    translation_xy = face_center - model_center_scaled

    vertices_scaled[:, 0] += translation_xy[0]
    vertices_scaled[:, 1] += translation_xy[1]
    vertices_scaled[:, 2] = vertices_scaled[:, 2] * 0.5  # Reduce Z depth

    print(f"  Scale: {scale:.2f}, Translation: ({translation_xy[0]:.0f}, {translation_xy[1]:.0f})")

    return vertices_scaled


def save_textured_basel(vertices, faces, texture_img, uv_coords, output_dir, subject_id):
    """
    Save Basel model with proper texture atlas.
    """
    obj_path = os.path.join(output_dir, f"{subject_id}_basel_proper.obj")
    mtl_path = os.path.join(output_dir, f"{subject_id}_basel_proper.mtl")
    texture_path = os.path.join(output_dir, f"{subject_id}_basel_proper_texture.jpg")

    # Save texture
    cv2.imwrite(texture_path, texture_img)
    print(f"  Saved texture: {os.path.basename(texture_path)}")

    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl basel_mat\n")
        f.write(f"Ka 1.0 1.0 1.0\n")
        f.write(f"Kd 1.0 1.0 1.0\n")
        f.write(f"Ks 0.0 0.0 0.0\n")
        f.write(f"illum 1\n")
        f.write(f"map_Kd {subject_id}_basel_proper_texture.jpg\n")

    print(f"  Saved material: {os.path.basename(mtl_path)}")

    # Write OBJ file
    with open(obj_path, 'w') as f:
        f.write(f"# Basel Face Model - Proper Reconstruction\n")
        f.write(f"# Subject: {subject_id}\n")
        f.write(f"mtllib {subject_id}_basel_proper.mtl\n\n")

        # Vertices (flip Y for OBJ convention)
        for v in vertices:
            f.write(f"v {v[0]:.6f} {-v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Texture coordinates (flip V)
        for uv in uv_coords:
            f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")

        f.write("\n")

        # Faces with texture indices
        f.write(f"usemtl basel_mat\n")
        for face in faces:
            # OBJ format: f v1/vt1 v2/vt2 v3/vt3
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")

    print(f"  Saved OBJ: {os.path.basename(obj_path)}")

    # Convert to GLB
    if HAS_TRIMESH:
        glb_path = os.path.join(output_dir, f"{subject_id}_basel_proper.glb")
        try:
            mesh = trimesh.load(obj_path, process=False, force='mesh')
            mesh.export(glb_path, file_type='glb', include_normals=True)
            print(f"  Saved GLB: {os.path.basename(glb_path)}")
        except Exception as e:
            print(f"  GLB export warning: {e}")


def reconstruct_proper(subject, detector, bfm, tex_projector, output_dir="output"):
    """
    Proper Basel reconstruction with correct texture mapping.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*60}")
    print(f"Basel Proper Reconstruction: {subject.subject_id}")
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
    print(f"Image: {w}x{h}")

    # Detect landmarks
    frontal_rgb = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2RGB)
    results = detector.face_mesh.process(frontal_rgb)

    if not results.multi_face_landmarks:
        print(f"Error: No face detected")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_2d = [[lm.x * w, lm.y * h] for lm in landmarks]
    print(f"Landmarks: {len(landmarks)}")

    # Fit Basel model
    print(f"\nFitting Basel model...")
    bfm_vertices = fit_basel_bbox(bfm, landmarks_2d, frontal_img.shape)

    # Create texture using proper projection
    print(f"\nCreating texture atlas...")
    texture_atlas, uv_coords = tex_projector.create_texture_atlas(
        bfm_vertices,
        frontal_img,
        texture_size=(2048, 2048)
    )

    # Save results
    print(f"\nSaving results...")
    save_textured_basel(bfm_vertices, bfm.faces, texture_atlas, uv_coords, output_dir, subject.subject_id)

    print(f"\n{'='*60}")
    print(f"[OK] Reconstruction complete!")
    print(f"  Vertices: {len(bfm_vertices):,}")
    print(f"  Faces: {len(bfm.faces):,}")
    print(f"  Texture: 2048x2048 atlas")
    print(f"  Files: {subject.subject_id}_basel_proper.*")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("="*60)
    print("BASEL FACE MODEL - PROPER TEXTURE MAPPING")
    print("="*60)

    # Check for Basel model
    model_path = "models/model2019_fullHead.h5"
    texture_mapping_path = "models/model2019_textureMapping.json"

    if not os.path.exists(model_path):
        print(f"\nError: Basel model not found: {model_path}")
        exit(1)

    if not os.path.exists(texture_mapping_path):
        print(f"\nError: Texture mapping not found: {texture_mapping_path}")
        print("Download from Basel Face Model 2019 distribution")
        exit(1)

    # Load Basel model
    print("\nLoading Basel Face Model...")
    bfm = BaselFaceModel(model_path)

    # Load texture projector
    print("\nLoading texture projector...")
    tex_projector = BaselTextureProjector(texture_mapping_path)

    # Load face images
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
        reconstruct_proper(subject, detector, bfm, tex_projector)

    print("\n" + "="*60)
    print("ALL RECONSTRUCTIONS COMPLETE!")
    print("="*60)
    print(f"\nFiles in output folder: *_basel_proper.*")
    print(f"View GLB files at: https://gltf-viewer.donmccurdy.com/")
    print("="*60 + "\n")
