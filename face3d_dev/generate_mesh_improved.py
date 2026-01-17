"""
Improved 3D Face Reconstruction with Better Edge Handling
- Multi-view depth fusion for better geometry
- Smart texture mapping that handles edge cases
- Face region masking to avoid stretched edges
"""
import cv2
import numpy as np
import os
from scipy.spatial import Delaunay
from data_manager import ImageManager
from landmark_utils import LandmarkDetector

# Optional: trimesh for GLB export
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def get_face_region_mask(landmarks_2d, img_shape, erosion_percent=0.05):
    """
    Creates a mask for the visible face region.
    Erodes the face boundary to avoid edge artifacts.
    """
    h, w = img_shape[:2]

    # Create mask from face convex hull
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(landmarks_2d, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # Erode slightly to avoid stretched edges
    erosion_size = int(min(w, h) * erosion_percent)
    if erosion_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        mask = cv2.erode(mask, kernel)

    return mask


def estimate_head_pose(landmarks_2d, img_shape):
    """
    Estimate rough head pose (rotation angles) from 2D landmarks.
    Returns: (yaw, pitch, roll) in degrees
    """
    h, w = img_shape[:2]

    if len(landmarks_2d) < 478:
        return 0, 0, 0

    # Nose tip
    nose = np.array(landmarks_2d[1])

    # Face center
    face_center_x = w / 2
    nose_x = nose[0]

    # Estimate yaw based on nose position
    yaw_normalized = (nose_x - face_center_x) / (w / 2)
    yaw_degrees = yaw_normalized * 45

    return yaw_degrees, 0, 0


def fuse_depth_from_multiple_views(frontal_landmarks, frontal_img_shape,
                                   side_landmarks_list, side_poses_list):
    """
    Fuses depth information from multiple views.
    """
    h, w = frontal_img_shape[:2]

    points_3d = []
    points_2d = []

    for lm in frontal_landmarks:
        x = lm.x * w
        y = lm.y * h
        z_frontal = -lm.z * w

        points_2d.append((x, y))
        points_3d.append([x, y, z_frontal])

    points_3d = np.array(points_3d)

    # Fuse with side views
    if side_landmarks_list:
        print(f"  Fusing depth from {len(side_landmarks_list)} additional view(s)...")

        for side_landmarks, (yaw, pitch, roll) in zip(side_landmarks_list, side_poses_list):
            side_weight = min(abs(yaw) / 45.0, 1.0)

            if side_weight < 0.1:
                continue

            print(f"    Side view with yaw={yaw:.1f}° (weight={side_weight:.2f})")

            for i, lm_side in enumerate(side_landmarks):
                if i >= len(points_3d):
                    break

                z_side = -lm_side.z * w
                z_current = points_3d[i, 2]
                z_adjusted = (1 - side_weight * 0.5) * z_current + (side_weight * 0.5) * z_side
                points_3d[i, 2] = z_adjusted

    return points_3d.tolist(), points_2d


def create_better_texture(frontal_img, landmarks_2d):
    """
    Creates a texture with better edge handling.
    Applies a soft mask to fade out stretched areas.
    """
    h, w = frontal_img.shape[:2]

    # Get face region mask
    face_mask = get_face_region_mask(landmarks_2d, frontal_img.shape, erosion_percent=0.02)

    # Create soft edge falloff
    face_mask_float = face_mask.astype(np.float32) / 255.0

    # Blur the mask for smooth edges
    blur_size = int(min(w, h) * 0.03)
    if blur_size % 2 == 0:
        blur_size += 1
    face_mask_blurred = cv2.GaussianBlur(face_mask_float, (blur_size, blur_size), 0)

    # Create neutral background (skin tone average)
    mask_region = frontal_img[face_mask > 0]
    if len(mask_region) > 0:
        avg_color = mask_region.mean(axis=0).astype(np.uint8)
    else:
        avg_color = np.array([180, 150, 130], dtype=np.uint8)  # Default skin tone

    background = np.full_like(frontal_img, avg_color)

    # Blend face with background
    face_mask_3c = np.dstack([face_mask_blurred] * 3)
    blended = (frontal_img * face_mask_3c + background * (1 - face_mask_3c)).astype(np.uint8)

    return blended


def filter_valid_faces(tri, points_2d, img_shape, max_edge_length_ratio=0.15):
    """
    Filter out triangular faces that are too large (likely artifacts at edges).
    """
    h, w = img_shape[:2]
    max_edge_length = max(w, h) * max_edge_length_ratio

    valid_faces = []
    for simplex in tri.simplices:
        p1 = np.array(points_2d[simplex[0]])
        p2 = np.array(points_2d[simplex[1]])
        p3 = np.array(points_2d[simplex[2]])

        # Check edge lengths
        edge1 = np.linalg.norm(p2 - p1)
        edge2 = np.linalg.norm(p3 - p2)
        edge3 = np.linalg.norm(p1 - p3)

        if max(edge1, edge2, edge3) < max_edge_length:
            valid_faces.append(simplex)

    return np.array(valid_faces)


def save_textured_obj(output_dir, subject_id, landmarks_3d, landmarks_2d, img_shape, texture_filename):
    """
    Saves a textured .obj file with filtered triangulation.
    """
    obj_path = os.path.join(output_dir, f"{subject_id}.obj")
    mtl_path = os.path.join(output_dir, f"{subject_id}.mtl")
    mtl_filename = f"{subject_id}.mtl"

    # Calculate Delaunay triangulation
    tri = Delaunay(landmarks_2d)

    # Filter out bad triangles
    faces = filter_valid_faces(tri, landmarks_2d, img_shape)
    print(f"  Filtered triangulation: {len(faces)}/{len(tri.simplices)} faces kept")

    # Write Material File
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl face_mat\n")
        f.write(f"Ka 1.0 1.0 1.0\n")
        f.write(f"Kd 1.0 1.0 1.0\n")
        f.write(f"Ks 0.2 0.2 0.2\n")  # Slight specularity for realism
        f.write(f"map_Kd {texture_filename}\n")

    # Write OBJ File
    h, w = img_shape[:2]
    with open(obj_path, 'w') as f:
        f.write(f"mtllib {mtl_filename}\n")

        # Vertices
        for x, y, z in landmarks_3d:
            f.write(f"v {x} {-y} {z}\n")

        # Texture Coordinates
        for x, y in landmarks_2d:
            u = x / w
            v = 1.0 - (y / h)
            f.write(f"vt {u} {v}\n")

        # Faces
        f.write(f"usemtl face_mat\n")
        for p1, p2, p3 in faces:
            f.write(f"f {p1+1}/{p1+1} {p2+1}/{p2+1} {p3+1}/{p3+1}\n")

    print(f"Saved textured model to: {obj_path}")

    # Convert to GLB
    if HAS_TRIMESH:
        glb_path = os.path.join(output_dir, f"{subject_id}.glb")
        try:
            mesh = trimesh.load(obj_path, process=False)
            mesh.export(glb_path)
            print(f"Saved GLB model to: {glb_path}")
        except Exception as e:
            print(f"Failed to create GLB: {e}")
    else:
        print(f"(Skipping GLB export - install trimesh with: pip install trimesh)")


def generate_face_model(subject, detector, output_dir="output"):
    """
    Generates a 3D model with improved edge handling.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not subject.frontal_path or not os.path.exists(subject.frontal_path):
        print(f"Skipping {subject.subject_id}: Missing frontal image.")
        return

    print(f"\nProcessing {subject.subject_id}...")
    print("=" * 60)

    frontal_img = cv2.imread(subject.frontal_path)
    if frontal_img is None:
        return

    frontal_rgb = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2RGB)
    frontal_results = detector.face_mesh.process(frontal_rgb)

    if not frontal_results.multi_face_landmarks:
        print(f"No face detected in frontal view for {subject.subject_id}")
        return

    frontal_landmarks = frontal_results.multi_face_landmarks[0].landmark
    h, w, _ = frontal_img.shape

    print(f"✓ Frontal view: {w}x{h}, {len(frontal_landmarks)} landmarks detected")

    # Process side views
    side_landmarks_list = []
    side_poses_list = []

    if subject.three_quarter_path and os.path.exists(subject.three_quarter_path):
        print("Processing 3/4 view...")
        quarter_img = cv2.imread(subject.three_quarter_path)
        quarter_rgb = cv2.cvtColor(quarter_img, cv2.COLOR_BGR2RGB)
        quarter_results = detector.face_mesh.process(quarter_rgb)

        if quarter_results.multi_face_landmarks:
            quarter_landmarks = quarter_results.multi_face_landmarks[0].landmark
            h_q, w_q, _ = quarter_img.shape
            quarter_2d = [(lm.x * w_q, lm.y * h_q) for lm in quarter_landmarks]
            yaw, pitch, roll = estimate_head_pose(quarter_2d, quarter_img.shape)
            print(f"  ✓ 3/4 view: {len(quarter_landmarks)} landmarks, estimated yaw={yaw:.1f}°")
            side_landmarks_list.append(quarter_landmarks)
            side_poses_list.append((yaw, pitch, roll))
        else:
            print("  ✗ No landmarks detected in 3/4 view")

    # Fuse depth
    print("\nFusing depth information...")
    points_3d, points_2d = fuse_depth_from_multiple_views(
        frontal_landmarks,
        frontal_img.shape,
        side_landmarks_list,
        side_poses_list
    )

    z_coords = [p[2] for p in points_3d]
    print(f"Depth range: {min(z_coords):.1f} to {max(z_coords):.1f}")

    # Create improved texture
    print("Creating texture with edge handling...")
    texture_img = create_better_texture(frontal_img, points_2d)

    texture_filename = f"{subject.subject_id}_texture.jpg"
    texture_path = os.path.join(output_dir, texture_filename)
    cv2.imwrite(texture_path, texture_img)
    print(f"Texture saved: {texture_filename}")

    # Save model
    save_textured_obj(output_dir, subject.subject_id, points_3d, points_2d, frontal_img.shape, texture_filename)
    print("=" * 60)


if __name__ == "__main__":
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()

    detector = LandmarkDetector(min_confidence=0.1, refine_landmarks=False)

    subjects = manager.get_complete_subjects()
    print(f"\n{'='*60}")
    print(f"IMPROVED Multi-View - Processing {len(subjects)} subjects")
    print(f"{'='*60}\n")

    for subj in subjects:
        generate_face_model(subj, detector)

    print(f"\n{'='*60}")
    print("All subjects processed!")
    print(f"{'='*60}")
