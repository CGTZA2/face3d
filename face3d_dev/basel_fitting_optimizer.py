"""
Basel Face Model Fitting with Optimization
Properly fits shape parameters to match detected landmarks.
"""
import numpy as np
from scipy.optimize import minimize
import cv2


class BaselFittingOptimizer:
    """
    Optimizes Basel Face Model parameters to match detected landmarks.
    """

    def __init__(self, bfm):
        """
        Args:
            bfm: BaselFaceModel instance
        """
        self.bfm = bfm
        self.landmark_correspondences = self._get_landmark_correspondences()

    def _get_landmark_correspondences(self):
        """
        Map MediaPipe landmark indices to Basel Face Model vertex indices.

        This is a critical mapping. For production, you'd have a pre-computed file.
        For now, we'll use key landmarks that we can estimate.

        Returns:
            dict: {mediapipe_idx: bfm_vertex_idx}
        """
        # Approximate correspondences for Basel 2019 full head model
        # These are educated guesses based on anatomical positions

        correspondences = {
            # Nose region
            1: 8156,      # Nose tip
            4: 8177,      # Nose bridge
            5: 8190,      # Nose top

            # Left eye region
            33: 11442,    # Left eye outer corner
            133: 11469,   # Left eye inner corner
            159: 11450,   # Left eye top
            145: 11455,   # Left eye bottom

            # Right eye region
            263: 4520,    # Right eye outer corner
            362: 4547,    # Right eye inner corner
            386: 4528,    # Right eye top
            374: 4535,    # Right eye bottom

            # Mouth region
            61: 3742,     # Left mouth corner
            291: 6932,    # Right mouth corner
            0: 5175,      # Upper lip center
            17: 5198,     # Lower lip center
            13: 5180,     # Upper lip left
            14: 5185,     # Upper lip right

            # Chin and jaw
            152: 2067,    # Chin center
            172: 2100,    # Chin left
            397: 2150,    # Chin right

            # Face outline
            234: 1234,    # Left jaw
            454: 9876,    # Right jaw
            10: 7845,     # Forehead center
            151: 2050,    # Chin tip
        }

        return correspondences

    def fit_shape_to_landmarks(self, landmarks_2d, landmarks_3d, img_shape,
                               n_shape_components=30, use_3d=True):
        """
        Fit Basel model shape parameters to match detected landmarks.

        Args:
            landmarks_2d: MediaPipe 2D landmarks (N, 2) - pixel coordinates
            landmarks_3d: MediaPipe 3D landmarks (N, 3) - with Z depth
            img_shape: (height, width, channels)
            n_shape_components: Number of PCA components to optimize
            use_3d: Whether to use 3D landmark positions (vs 2D only)

        Returns:
            fitted_vertices: (M, 3) optimized Basel vertices
            shape_params: Optimized shape parameters
            transform: (scale, translation) for positioning
        """
        print(f"\nOptimizing Basel model fit...")
        print(f"  Using {n_shape_components} shape components")
        print(f"  Landmark correspondences: {len(self.landmark_correspondences)}")

        h, w = img_shape[:2]

        # Initial parameters: [shape_params, scale, tx, ty, tz, rotation_params]
        n_params = n_shape_components + 7  # shape + scale(1) + translation(3) + rotation(3)

        # Estimate initial scale based on image size and model size
        # Basel model is roughly in millimeters, need to scale to pixels
        initial_scale = min(w, h) * 0.4  # More reasonable initial guess

        initial_params = np.zeros(n_params)
        initial_params[n_shape_components] = initial_scale
        initial_params[n_shape_components + 1] = w / 2  # tx - center X
        initial_params[n_shape_components + 2] = h / 2  # ty - center Y
        initial_params[n_shape_components + 3] = 0.0    # tz - depth offset

        # Extract landmark positions for fitting
        target_landmarks = []
        bfm_landmark_indices = []

        for mp_idx, bfm_idx in self.landmark_correspondences.items():
            if mp_idx < len(landmarks_3d):
                if use_3d:
                    target_landmarks.append(landmarks_3d[mp_idx])
                else:
                    target_landmarks.append([landmarks_2d[mp_idx][0],
                                            landmarks_2d[mp_idx][1], 0])
                bfm_landmark_indices.append(bfm_idx)

        target_landmarks = np.array(target_landmarks)
        print(f"  Fitting to {len(target_landmarks)} landmark correspondences")

        # Define optimization objective
        def objective(params):
            """
            Compute error between Basel model landmarks and detected landmarks.
            """
            # Extract parameters
            shape_params = params[:n_shape_components]
            scale = params[n_shape_components]
            tx, ty, tz = params[n_shape_components + 1:n_shape_components + 4]
            rx, ry, rz = params[n_shape_components + 4:n_shape_components + 7]

            # Generate face with these shape parameters
            vertices = self.bfm.generate_face(shape_params)

            # Apply rotation (simple Euler angles)
            # For simplicity, just use small rotations
            cos_rx, sin_rx = np.cos(rx), np.sin(rx)
            cos_ry, sin_ry = np.cos(ry), np.sin(ry)
            cos_rz, sin_rz = np.cos(rz), np.sin(rz)

            # Rotation matrices (simplified - proper would use full 3D rotation)
            R_x = np.array([[1, 0, 0], [0, cos_rx, -sin_rx], [0, sin_rx, cos_rx]])
            R_y = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])
            R_z = np.array([[cos_rz, -sin_rz, 0], [sin_rz, cos_rz, 0], [0, 0, 1]])
            R = R_z @ R_y @ R_x

            # Transform vertices
            vertices_transformed = (vertices @ R.T) * scale
            vertices_transformed[:, 0] += tx
            vertices_transformed[:, 1] += ty
            vertices_transformed[:, 2] += tz

            # Extract corresponding landmark positions
            model_landmarks = vertices_transformed[bfm_landmark_indices]

            # Compute error
            if use_3d:
                error = np.sum((model_landmarks - target_landmarks) ** 2)
            else:
                # Only X, Y error (ignore Z)
                error = np.sum((model_landmarks[:, :2] - target_landmarks[:, :2]) ** 2)

            return error

        # Optimization bounds
        bounds = []
        # Shape parameters: reasonable range based on variance
        for i in range(n_shape_components):
            std = np.sqrt(self.bfm.shape_pca_variance[i]) if i < len(self.bfm.shape_pca_variance) else 1.0
            bounds.append((-3 * std, 3 * std))  # ±3 standard deviations

        # Scale: must be positive, reasonable range
        bounds.append((min(w, h) * 0.2, min(w, h) * 2.0))

        # Translation: within image bounds
        bounds.append((0, w))      # tx
        bounds.append((0, h))      # ty
        bounds.append((-500, 500)) # tz

        # Rotation: small angles
        bounds.append((-0.3, 0.3))  # rx (radians)
        bounds.append((-0.3, 0.3))  # ry
        bounds.append((-0.3, 0.3))  # rz

        print("  Running optimization...")

        # Run optimization
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'disp': False}
        )

        if result.success:
            print(f"  [OK] Optimization converged!")
            print(f"  Final error: {result.fun:.2f}")
        else:
            print(f"  [WARNING] Optimization finished (may not have fully converged)")
            print(f"  Final error: {result.fun:.2f}")

        # Extract optimized parameters
        opt_shape_params = result.x[:n_shape_components]
        opt_scale = result.x[n_shape_components]
        opt_tx, opt_ty, opt_tz = result.x[n_shape_components + 1:n_shape_components + 4]
        opt_rx, opt_ry, opt_rz = result.x[n_shape_components + 4:n_shape_components + 7]

        print(f"  Optimized parameters:")
        print(f"    Scale: {opt_scale:.2f}")
        print(f"    Translation: ({opt_tx:.1f}, {opt_ty:.1f}, {opt_tz:.1f})")
        print(f"    Rotation: ({np.degrees(opt_rx):.1f}°, {np.degrees(opt_ry):.1f}°, {np.degrees(opt_rz):.1f}°)")

        # Generate final fitted mesh
        fitted_vertices = self.bfm.generate_face(opt_shape_params)

        # Apply transformations
        cos_rx, sin_rx = np.cos(opt_rx), np.sin(opt_rx)
        cos_ry, sin_ry = np.cos(opt_ry), np.sin(opt_ry)
        cos_rz, sin_rz = np.cos(opt_rz), np.sin(opt_rz)

        R_x = np.array([[1, 0, 0], [0, cos_rx, -sin_rx], [0, sin_rx, cos_rx]])
        R_y = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])
        R_z = np.array([[cos_rz, -sin_rz, 0], [sin_rz, cos_rz, 0], [0, 0, 1]])
        R = R_z @ R_y @ R_x

        fitted_vertices = (fitted_vertices @ R.T) * opt_scale
        fitted_vertices[:, 0] += opt_tx
        fitted_vertices[:, 1] += opt_ty
        fitted_vertices[:, 2] += opt_tz

        transform = {
            'scale': opt_scale,
            'translation': (opt_tx, opt_ty, opt_tz),
            'rotation': (opt_rx, opt_ry, opt_rz),
            'rotation_matrix': R
        }

        return fitted_vertices, opt_shape_params, transform


if __name__ == "__main__":
    print("Basel Fitting Optimizer module")
    print("This module provides optimization-based fitting for Basel Face Model")
    print("Import and use with: from basel_fitting_optimizer import BaselFittingOptimizer")
