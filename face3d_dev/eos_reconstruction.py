"""
3D Face Reconstruction using eos library
Fast, CPU-only Basel Face Model fitting with proper texture mapping.
"""
import cv2
import numpy as np
import eos
import os
import argparse
from data_manager import ImageManager
from landmark_utils import LandmarkDetector

# Optional: trimesh for GLB export
try:
    import trimesh
    from PIL import Image
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

def get_resources_dir():
    """Get the absolute path to the share directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'share')

def convert_mediapipe_to_ibug68(mp_landmarks):
    """
    Convert MediaPipe 468 landmarks to iBUG 68-point format required by eos.

    eos expects 68 facial landmarks in iBUG format. We'll map key MediaPipe
    landmarks to the closest iBUG equivalents.

    Returns: List of (x, y) tuples for 68 landmarks
    """
    # Mapping from MediaPipe indices to iBUG 68-point landmarks
    # This is an approximate mapping - eos can work with subset of landmarks
    mediapipe_to_ibug = {
        # --- JAWLINE (1-17) ---
        0: 234,  1: 93,   2: 132,  3: 58,   4: 172,  5: 136,  6: 150,  7: 149,
        8: 152,  # Chin Center
        9: 148,  10: 176, 11: 377, 12: 400, 13: 378, 14: 379, 15: 365, 16: 454,

        # --- EYEBROWS (18-27) ---
        # Left Eyebrow (Image Left, Subject Right) -> iBUG 18-22
        17: 70,  18: 63,  19: 105, 20: 66,  21: 107,
        
        # Right Eyebrow (Image Right, Subject Left) -> iBUG 23-27
        22: 336, 23: 296, 24: 334, 25: 293, 26: 300,

        # --- NOSE (28-36) ---
        # Nose Bridge (28-31)
        27: 168, 28: 6,   29: 197, 30: 195,
        
        # Nose Base (32-36)
        # Fixed: Use outer nostril points (alae) to prevent pinched nose
        31: 102, 32: 48,  33: 1,   34: 278, 35: 331,

        # --- EYES (37-48) ---
        # Left Eye (Image Left, Subject Right) -> iBUG 37-42
        # Points: Outer, Top-Outer, Top-Inner, Inner, Bottom-Inner, Bottom-Outer
        36: 33,  37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
        
        # Right Eye (Image Right, Subject Left) -> iBUG 43-48
        # Points: Inner, Top-Inner, Top-Outer, Outer, Bottom-Outer, Bottom-Inner
        42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,

        # --- MOUTH (49-68) ---
        # Outer Lip (49-60)
        48: 61,   # 49: Left Corner (Image Left)
        49: 185,  # 50: Upper Left
        50: 37,   # 51: Upper Left (closer to center)
        51: 0,    # 52: Upper Center
        52: 267,  # 53: Upper Right (closer to center)
        53: 270,  # 54: Upper Right
        54: 291,  # 55: Right Corner (Image Right)
        55: 409,  # 56: Lower Right
        56: 321,  # 57: Lower Right
        57: 17,   # 58: Lower Center
        58: 181,  # 59: Lower Left
        59: 84,   # 60: Lower Left
        
        # Inner Lip (61-68)
        60: 78,   # 61: Left Corner Inner
        61: 191,  # 62: Upper Left Inner
        62: 13,   # 63: Upper Center Inner
        63: 308,  # 64: Upper Right Inner
        64: 292,  # 65: Right Corner Inner (Approx)
        65: 402,  # 66: Lower Right Inner
        66: 14,   # 67: Lower Center Inner
        67: 87    # 68: Lower Left Inner
    }

    # Convert to list of (x, y) coordinates
    landmarks_68 = []
    for ibug_idx in range(68):
        mp_idx = mediapipe_to_ibug.get(ibug_idx, 1)  # Default to landmark 1 if missing
        lm = mp_landmarks[mp_idx]
        landmarks_68.append((lm[0], lm[1]))

    return landmarks_68


def post_process_texture(isomap):
    """
    Fill in white/transparent patches in the texture map.
    """
    # Check if image has alpha channel
    if isomap.shape[2] == 4:
        # Create mask of transparent/empty pixels (alpha=0)
        alpha = isomap[:, :, 3]
        mask = (alpha == 0).astype(np.uint8)
        
        # Convert to RGB for processing
        rgb = isomap[:, :, :3]
        
        # Simple inpainting to fill holes
        # We dilate the valid regions into the invalid ones
        if np.sum(mask) > 0:
            # Inpaint using Telea method
            filled_rgb = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
            return filled_rgb
            
    return isomap[:, :, :3] # Return RGB


def reconstruct_with_eos(subject, detector, output_dir="output"):
    """
    Reconstruct 3D face using eos library.

    Args:
        subject: FaceTriplet with image paths
        detector: LandmarkDetector instance
        output_dir: Output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*60}")
    print(f"eos Reconstruction: {subject.subject_id}")
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

    # Detect MediaPipe landmarks
    frontal_rgb = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2RGB)
    results = detector.face_mesh.process(frontal_rgb)

    if not results.multi_face_landmarks:
        print(f"Error: No face detected")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    landmarks_2d = [[lm.x * w, lm.y * h] for lm in landmarks]
    print(f"Detected {len(landmarks)} MediaPipe landmarks")

    # Convert to iBUG 68-point format
    print(f"Converting to iBUG 68-point format...")
    landmarks_68 = convert_mediapipe_to_ibug68(landmarks_2d)

    # Convert to eos.core.Landmark objects
    eos_landmarks = []
    for i, (x, y) in enumerate(landmarks_68):
        # eos expects 1-based indexing for landmarks (1 to 68)
        eos_landmarks.append(eos.core.Landmark(str(i + 1), [x, y]))

    # Perform fitting using eos
    print(f"Fitting 3D Morphable Model...")
    try:
        share_dir = get_resources_dir()
        model_path = os.path.join(share_dir, "sfm_shape_3448.bin")
        blendshapes_path = os.path.join(share_dir, "expression_blendshapes_3448.bin")
        mapping_path = os.path.join(share_dir, "ibug_to_sfm.txt")
        
        # Load eos model and blendshapes
        print(f"  Loading model from: {os.path.basename(model_path)}")
        shape_model = eos.morphablemodel.load_model(model_path)
        blendshapes = eos.morphablemodel.load_blendshapes(blendshapes_path)
        model = eos.morphablemodel.MorphableModel(
            shape_model.get_shape_model(),
            blendshapes,
            color_model=eos.morphablemodel.PcaModel(),
            vertex_definitions=None,
            texture_coordinates=shape_model.get_texture_coordinates()
        )

        # Load landmark mappings
        landmark_mapper = eos.core.LandmarkMapper(mapping_path)

        # Fit the model using simpler API (without contours for now)
        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(
            model,
            eos_landmarks,
            landmark_mapper,
            w, h
        )

        print(f"  Shape coefficients: {len(shape_coeffs)}")
        print(f"  Blendshape coefficients: {len(blendshape_coeffs)}")

        # Extract texture
        print(f"Extracting texture...")
        # eos requires BGRA (4 channels) input
        frontal_bgra = cv2.cvtColor(frontal_img, cv2.COLOR_BGR2BGRA)
        isomap = eos.render.extract_texture(mesh, pose, frontal_bgra)
        
        # Post-process to remove white patches
        final_texture = post_process_texture(isomap)

        # Save results
        print(f"Saving results...")
        obj_path = os.path.join(output_dir, f"{subject.subject_id}_eos.obj")
        texture_path = obj_path.replace(".obj", ".texture.png")

        eos.core.write_textured_obj(mesh, obj_path)
        cv2.imwrite(texture_path, final_texture)

        print(f"  Saved: {os.path.basename(obj_path)}")
        print(f"  Saved: {os.path.basename(texture_path)}")

        # Export GLB if trimesh is available
        if HAS_TRIMESH:
            try:
                print("Exporting to GLB...")
                glb_path = os.path.join(output_dir, f"{subject.subject_id}_eos.glb")
                
                # Load the OBJ we just wrote
                mesh_tm = trimesh.load(obj_path, process=False, force='mesh')
                
                # Load the texture image
                tex_image = Image.open(texture_path)
                
                # Create a texture visual
                # We need to check if UVs were loaded
                if hasattr(mesh_tm.visual, 'uv') and mesh_tm.visual.uv is not None and len(mesh_tm.visual.uv) > 0:
                    mesh_tm.visual = trimesh.visual.TextureVisuals(
                        uv=mesh_tm.visual.uv,
                        image=tex_image
                    )
                
                mesh_tm.export(glb_path)
                print(f"  Saved GLB: {os.path.basename(glb_path)}")
            except Exception as e:
                print(f"  GLB export failed: {e}")

        print(f"\n{'='*60}")
        print(f"[OK] Reconstruction complete!")
        print(f"  Vertices: {len(mesh.vertices):,}")
        print(f"  Faces: {len(mesh.tvi):,}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error during fitting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("EOS LIBRARY - 3D FACE RECONSTRUCTION")
    print("="*60)

    parser = argparse.ArgumentParser(description="EOS 3D Face Reconstruction")
    parser.add_argument("image", nargs="?", help="Path to input image (optional)")
    args = parser.parse_args()

    # Initialize detector
    detector = LandmarkDetector(min_confidence=0.1, refine_landmarks=False)

    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            exit(1)
            
        print(f"Processing single image: {args.image}")
        
        # Create a simple object to mimic the FaceTriplet structure
        class SingleSubject:
            def __init__(self, path):
                self.frontal_path = path
                self.subject_id = os.path.splitext(os.path.basename(path))[0]
        
        subject = SingleSubject(args.image)
        reconstruct_with_eos(subject, detector)
        
    else:
        # Batch mode
        print("\nScanning for face images...")
        manager = ImageManager(image_dir="images/bm")
        manager.scan_directory()
        subjects = manager.get_complete_subjects()

        if not subjects:
            print("No complete face triplets found!")
            exit(1)

        print(f"Found {len(subjects)} subjects")

        # Process each subject
        for subject in subjects:
            reconstruct_with_eos(subject, detector)
