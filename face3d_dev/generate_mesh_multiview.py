"""
Multi-View Depth Fusion for 3D Face Reconstruction
Uses frontal and 3/4 views to improve 3D geometry depth estimation.
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


def estimate_head_pose(landmarks_2d, img_shape):
    """
    Estimate rough head pose (rotation angles) from 2D landmarks.
    Returns: (yaw, pitch, roll) in degrees
    """
    h, w = img_shape[:2]

    # Use key facial points to estimate pose
    # MediaPipe landmark indices (approximate):
    # 1: Nose tip
    # 33: Left eye outer corner
    # 263: Right eye outer corner
    # 61: Left mouth corner
    # 291: Right mouth corner

    if len(landmarks_2d) < 478:
        return 0, 0, 0  # Default to frontal

    # Nose tip
    nose = np.array(landmarks_2d[1])

    # Eyes
    left_eye = np.array(landmarks_2d[33])
    right_eye = np.array(landmarks_2d[263])

    # Calculate face center
    face_center_x = w / 2
    nose_x = nose[0]

    # Estimate yaw (left-right rotation) based on nose position relative to center
    # Normalized to -1 (full left) to 1 (full right)
    yaw_normalized = (nose_x - face_center_x) / (w / 2)
    yaw_degrees = yaw_normalized * 45  # Map to ±45 degrees

    # Eye distance can indicate pitch/roll, but we'll keep it simple
    pitch_degrees = 0
    roll_degrees = 0

    return yaw_degrees, pitch_degrees, roll_degrees


def fuse_depth_from_multiple_views(frontal_landmarks, frontal_img_shape,
                                   side_landmarks_list, side_poses_list):
    """
    Fuses depth information from multiple views to improve Z-coordinates.

    Args:
        frontal_landmarks: List of MediaPipe landmarks from frontal view
        frontal_img_shape: (height, width, channels) of frontal image
        side_landmarks_list: List of landmark sets from side views
        side_poses_list: List of (yaw, pitch, roll) tuples for each side view

    Returns:
        points_3d: List of (x, y, z) coordinates with fused depth
    """
    h, w = frontal_img_shape[:2]

    # Start with frontal view 3D points
    points_3d = []
    points_2d = []

    for lm in frontal_landmarks:
        x = lm.x * w
        y = lm.y * h
        z_frontal = -lm.z * w  # Frontal depth estimate

        points_2d.append((x, y))
        points_3d.append([x, y, z_frontal])

    points_3d = np.array(points_3d)

    # If we have side views, use them to refine depth
    if side_landmarks_list:
        print(f"  Fusing depth from {len(side_landmarks_list)} additional view(s)...")

        for side_landmarks, (yaw, pitch, roll) in zip(side_landmarks_list, side_poses_list):
            # Weight based on how much side view this is
            # More side view = more weight for depth correction
            side_weight = min(abs(yaw) / 45.0, 1.0)  # 0 (frontal) to 1 (45° side)

            if side_weight < 0.1:
                continue  # Skip nearly frontal views

            print(f"    Side view with yaw={yaw:.1f}° (weight={side_weight:.2f})")

            # For each landmark, compare Z-depth estimates
            for i, lm_side in enumerate(side_landmarks):
                if i >= len(points_3d):
                    break

                # Side view Z-depth (in its own coordinate system)
                z_side = -lm_side.z * w

                # The key insight:
                # - For frontal view, Z represents depth (nose sticks out)
                # - For side view, X represents depth (ear to nose distance)
                # We can use the side view's X-variation to validate/correct Z-depth

                # Landmark's X position in side view tells us about depth
                x_side_norm = lm_side.x  # Normalized 0-1

                # For a true side view (yaw ≈ 45°), landmarks closer to camera have higher X
                # We use this to adjust the frontal Z estimate

                # Simple fusion: weighted average of depth estimates
                z_current = points_3d[i, 2]
                z_adjusted = (1 - side_weight) * z_current + side_weight * z_side

                # Apply adjustment
                points_3d[i, 2] = z_adjusted

    return points_3d.tolist(), points_2d


def save_textured_obj(output_dir, subject_id, landmarks_3d, landmarks_2d, img_shape, texture_filename):
    """
    Saves a textured .obj file with a corresponding .mtl file.
    """
    obj_path = os.path.join(output_dir, f"{subject_id}.obj")
    mtl_path = os.path.join(output_dir, f"{subject_id}.mtl")
    mtl_filename = f"{subject_id}.mtl"

    # 1. Calculate Triangles (Delaunay Triangulation on 2D points)
    tri = Delaunay(landmarks_2d)
    faces = tri.simplices

    # 2. Write Material File (.mtl)
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl face_mat\n")
        f.write(f"Ka 1.0 1.0 1.0\n")
        f.write(f"Kd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {texture_filename}\n")

    # 3. Write OBJ File
    h, w = img_shape[:2]
    with open(obj_path, 'w') as f:
        f.write(f"mtllib {mtl_filename}\n")

        # Vertices (v)
        for x, y, z in landmarks_3d:
            f.write(f"v {x} {-y} {z}\n")

        # Texture Coordinates (vt) - Normalized 0..1
        for x, y in landmarks_2d:
            u = x / w
            v = 1.0 - (y / h)
            f.write(f"vt {u} {v}\n")

        # Faces (f)
        f.write(f"usemtl face_mat\n")
        for p1, p2, p3 in faces:
            f.write(f"f {p1+1}/{p1+1} {p2+1}/{p2+1} {p3+1}/{p3+1}\n")

    print(f"Saved textured model to: {obj_path}")

    # 4. Convert to GLB (GLTF Binary) for easier web use
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
    Generates a 3D model using multi-view depth fusion.
    Uses frontal view for texture and geometry base, with side views to refine depth.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load and Process Frontal Image
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

    # 2. Process Side Views for Depth Fusion
    side_landmarks_list = []
    side_poses_list = []

    # Try 3/4 view
    if subject.three_quarter_path and os.path.exists(subject.three_quarter_path):
        print("Processing 3/4 view...")
        quarter_img = cv2.imread(subject.three_quarter_path)
        quarter_rgb = cv2.cvtColor(quarter_img, cv2.COLOR_BGR2RGB)
        quarter_results = detector.face_mesh.process(quarter_rgb)

        if quarter_results.multi_face_landmarks:
            quarter_landmarks = quarter_results.multi_face_landmarks[0].landmark

            # Convert to 2D for pose estimation
            h_q, w_q, _ = quarter_img.shape
            quarter_2d = [(lm.x * w_q, lm.y * h_q) for lm in quarter_landmarks]

            yaw, pitch, roll = estimate_head_pose(quarter_2d, quarter_img.shape)
            print(f"  ✓ 3/4 view: {len(quarter_landmarks)} landmarks, estimated yaw={yaw:.1f}°")

            side_landmarks_list.append(quarter_landmarks)
            side_poses_list.append((yaw, pitch, roll))
        else:
            print("  ✗ No landmarks detected in 3/4 view")

    # Try profile view (likely to fail, but we'll try)
    if subject.profile_path and os.path.exists(subject.profile_path):
        print("Processing profile view...")
        profile_img = cv2.imread(subject.profile_path)
        profile_rgb = cv2.cvtColor(profile_img, cv2.COLOR_BGR2RGB)
        profile_results = detector.face_mesh.process(profile_rgb)

        if profile_results.multi_face_landmarks:
            profile_landmarks = profile_results.multi_face_landmarks[0].landmark

            h_p, w_p, _ = profile_img.shape
            profile_2d = [(lm.x * w_p, lm.y * h_p) for lm in profile_landmarks]

            yaw, pitch, roll = estimate_head_pose(profile_2d, profile_img.shape)
            print(f"  ✓ Profile view: {len(profile_landmarks)} landmarks, estimated yaw={yaw:.1f}°")

            side_landmarks_list.append(profile_landmarks)
            side_poses_list.append((yaw, pitch, roll))
        else:
            print("  ✗ No landmarks detected in profile view")

    # 3. Fuse Depth Information
    print("\nFusing depth information from multiple views...")
    points_3d, points_2d = fuse_depth_from_multiple_views(
        frontal_landmarks,
        frontal_img.shape,
        side_landmarks_list,
        side_poses_list
    )

    # Report depth statistics
    z_coords = [p[2] for p in points_3d]
    print(f"Depth range: {min(z_coords):.1f} to {max(z_coords):.1f} (span: {max(z_coords) - min(z_coords):.1f})")

    # 4. Use frontal image as texture (clean, no warping artifacts)
    texture_filename = f"{subject.subject_id}_texture.jpg"
    texture_path = os.path.join(output_dir, texture_filename)
    cv2.imwrite(texture_path, frontal_img)
    print(f"Texture saved: {texture_filename}")

    # 5. Save Textured OBJ
    save_textured_obj(output_dir, subject.subject_id, points_3d, points_2d, frontal_img.shape, texture_filename)
    print("=" * 60)


if __name__ == "__main__":
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()

    # Lower confidence to help detect profile views
    # Disable refine_landmarks (iris tracking) because it fails when one eye is hidden
    detector = LandmarkDetector(min_confidence=0.1, refine_landmarks=False)

    subjects = manager.get_complete_subjects()
    print(f"\n{'='*60}")
    print(f"Multi-View Depth Fusion - Processing {len(subjects)} subjects")
    print(f"{'='*60}\n")

    for subj in subjects:
        generate_face_model(subj, detector)

    print(f"\n{'='*60}")
    print("All subjects processed!")
    print(f"{'='*60}")
