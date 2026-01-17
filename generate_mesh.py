import cv2
import numpy as np
import os
import shutil
import trimesh
from scipy.spatial import Delaunay
from data_manager import ImageManager
from landmark_utils import LandmarkDetector

def warp_triangle(img1, img2, t1, t2):
    """
    Warps a triangular region from img1 to img2 based on coordinates t1 and t2.
    """
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])
    
    # Affine Transform
    warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    img2_rect = cv2.warpAffine(img1_rect, warp_mat, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect

def save_textured_obj(output_dir, subject_id, landmarks_3d, landmarks_2d, img_shape, texture_filename):
    """
    Saves a textured .obj file with a corresponding .mtl file.
    """
    obj_path = os.path.join(output_dir, f"{subject_id}.obj")
    mtl_path = os.path.join(output_dir, f"{subject_id}.mtl")
    mtl_filename = f"{subject_id}.mtl"

    # 1. Calculate Triangles (Delaunay Triangulation on 2D points)
    # This creates the solid surface
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
        # OBJ UV origin is bottom-left, Image is top-left, so we flip Y
        for x, y in landmarks_2d:
            u = x / w
            v = 1.0 - (y / h)
            f.write(f"vt {u} {v}\n")

        # Faces (f)
        f.write(f"usemtl face_mat\n")
        for p1, p2, p3 in faces:
            # OBJ is 1-indexed
            # Format: f v1/vt1 v2/vt2 v3/vt3
            f.write(f"f {p1+1}/{p1+1} {p2+1}/{p2+1} {p3+1}/{p3+1}\n")

    print(f"Saved textured model to: {obj_path}")
    
    # 4. Convert to GLB (GLTF Binary) for easier web use
    glb_path = os.path.join(output_dir, f"{subject_id}.glb")
    try:
        # Load the OBJ we just created (trimesh handles the texture via the MTL)
        mesh = trimesh.load(obj_path, process=False)
        mesh.export(glb_path)
        print(f"Saved GLB model to: {glb_path}")
    except Exception as e:
        print(f"Failed to create GLB: {e}")

def generate_face_model(subject, detector, output_dir="output"):
    """
    Generates a 3D model from the frontal view of the subject.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Frontal Image
    if not subject.frontal_path or not os.path.exists(subject.frontal_path):
        print(f"Skipping {subject.subject_id}: Missing frontal image.")
        return

    img = cv2.imread(subject.frontal_path)
    if img is None:
        return

    # 2. Get Landmarks (2D)
    # Note: MediaPipe returns normalized coordinates. We need to access the raw 3D relative coords.
    # For this simple script, we will use the detector's helper but we might need to expand it 
    # to get Z-depth in the future.
    
    # Accessing internal mesh for 3D points
    results = detector.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        print(f"No face detected for {subject.subject_id}")
        return

    # 3. Extract 3D Coordinates
    # MediaPipe Z is relative to the head center, roughly same scale as X.
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = img.shape
    
    points_3d = []
    points_2d = []
    points_2d_raw = [] # Keep raw float coordinates for warping
    
    for lm in landmarks:
        # Scale X and Y to pixel coordinates, Z is scaled by width to maintain aspect ratio
        # We invert Z (-lm.z) so the nose points OUT towards the camera
        points_3d.append((lm.x * w, lm.y * h, -lm.z * w))
        points_2d.append((lm.x * w, lm.y * h))
        points_2d_raw.append([lm.x * w, lm.y * h])

    # 4. Prepare Texture
    # Start with Frontal Image
    final_texture = img.copy()
    
    # Calculate triangulation once for the frontal mesh (destination)
    tri = Delaunay(points_2d_raw)
    
    # Helper to warp a side view onto the frontal geometry
    def process_side_view(path, view_name):
        if not path or not os.path.exists(path):
            return None
        
        print(f"  - Processing {view_name} view...")
        img_side = cv2.imread(path)
        
        # Detect landmarks
        results_side = detector.face_mesh.process(cv2.cvtColor(img_side, cv2.COLOR_BGR2RGB))
        if not results_side.multi_face_landmarks:
            print(f"    (Skipping {view_name}: No landmarks detected)")
            return None
            
        lm_side = results_side.multi_face_landmarks[0].landmark
        h_s, w_s, _ = img_side.shape
        points_2d_side = [[p.x * w_s, p.y * h_s] for p in lm_side]
        
        # Warp image piece-by-piece
        warped_side = np.zeros_like(final_texture)
        for simplex in tri.simplices:
            t_frontal = [points_2d_raw[simplex[0]], points_2d_raw[simplex[1]], points_2d_raw[simplex[2]]]
            t_side = [points_2d_side[simplex[0]], points_2d_side[simplex[1]], points_2d_side[simplex[2]]]
            warp_triangle(img_side, warped_side, t_side, t_frontal)
            
        return warped_side

    # Accumulate valid side textures
    side_textures = []
    
    # Try 3/4 View
    w_t = process_side_view(subject.three_quarter_path, "3/4")
    if w_t is not None: side_textures.append(w_t)

    # Try Profile View
    w_p = process_side_view(subject.profile_path, "Profile")
    if w_p is not None: side_textures.append(w_p)
    
    # Attempt to incorporate 3/4 view
    if side_textures:
        # Combine available side textures (Average them)
        combined_side = side_textures[0].astype(np.float32)
        if len(side_textures) > 1:
            for i in range(1, len(side_textures)):
                combined_side = cv2.addWeighted(combined_side, 0.5, side_textures[i].astype(np.float32), 0.5, 0)
        combined_side = combined_side.astype(np.uint8)

        # --- SMART BLENDING ---
        # Create a mask that protects the center of the face (keep it sharp/frontal)
        # and only blends the 3/4 view on the outer edges.
        
        # 1. Define Face Center Mask
        mask = np.zeros((h, w), dtype=np.float32)
        hull = cv2.convexHull(np.array(points_2d_raw, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        # 2. Erode mask to isolate the center (eyes/nose/mouth)
        # The erosion amount determines how far from the edge we start blending
        erosion_size = int(w * 0.15) # 15% of image width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        center_mask = cv2.erode(mask, kernel)
        
        # 3. Blur the mask for a smooth transition
        center_mask = cv2.GaussianBlur(center_mask, (0, 0), sigmaX=erosion_size * 0.5)
        center_mask_3c = np.dstack([center_mask] * 3)

        # 4. Composite
        # Center (1.0) -> Use Frontal Image (Sharp)
        # Edge (0.0)   -> Blend Frontal + Warped 3/4
        
        # Calculate the edge blend (e.g., 50% frontal, 50% 3/4)
        # Use 100% of the side texture for the edge blend to avoid ghosting from stretched frontal pixels
        edge_blend = combined_side
        
        # Combine: Center gets Frontal, Edges get Blend
        final_texture = (final_texture * center_mask_3c + edge_blend * (1.0 - center_mask_3c)).astype(np.uint8)

    texture_filename = f"{subject.subject_id}_texture.jpg"
    texture_path = os.path.join(output_dir, texture_filename)
    cv2.imwrite(texture_path, final_texture)
    
    # 5. Save Textured OBJ
    save_textured_obj(output_dir, subject.subject_id, points_3d, points_2d, img.shape, texture_filename)

if __name__ == "__main__":
    manager = ImageManager(image_dir="images/bm")
    manager.scan_directory()
    # Lower confidence to help detect profile views
    # Disable refine_landmarks (iris tracking) because it fails when one eye is hidden (profile view)
    detector = LandmarkDetector(min_confidence=0.1, refine_landmarks=False)
    
    subjects = manager.get_complete_subjects()
    print(f"Generating models for {len(subjects)} subjects...")
    
    for subj in subjects:
        generate_face_model(subj, detector)