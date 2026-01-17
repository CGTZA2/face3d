"""
Basel Face Model (BFM) 3D Face Reconstruction
Fits a statistical 3D face model to detected landmarks for high-quality results.
"""
import numpy as np
import cv2
import os
import h5py
from scipy.optimize import minimize


class BaselFaceModel:
    """
    Wrapper for Basel Face Model 2017.
    Provides methods to load the model and fit it to landmarks.
    """

    def __init__(self, model_path):
        """
        Load Basel Face Model from HDF5 file.

        Args:
            model_path: Path to Basel model file
                       Supports: model2017-1_bfm_nomouth.h5 (full head, recommended)
                                model2017-1_face12_nomouth.h5 (face only)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Basel Face Model not found at: {model_path}\n"
                f"Please download it from: https://faces.dmi.unibas.ch/bfm/bfm2017.html\n"
                f"Recommended: model2017-1_bfm_nomouth.h5 (full head model)\n"
                f"See setup_basel.md for instructions."
            )

        print(f"Loading Basel Face Model from: {model_path}")
        self.model_file = h5py.File(model_path, 'r')

        # Load shape model (PCA basis for face shapes)
        self.shape_mean = np.array(self.model_file['shape']['model']['mean'])
        self.shape_pca_basis = np.array(self.model_file['shape']['model']['pcaBasis'])
        self.shape_pca_variance = np.array(self.model_file['shape']['model']['pcaVariance'])

        # Reshape based on the data dimensions
        # Mean shape: should be (n_vertices * 3,) -> reshape to (n_vertices, 3)
        n_vertices = len(self.shape_mean) // 3
        self.shape_mean = self.shape_mean.reshape(-1, 3)

        # PCA basis: (n_vertices * 3, n_components) -> (n_vertices, 3, n_components)
        n_components = self.shape_pca_basis.shape[1]
        self.shape_pca_basis = self.shape_pca_basis.reshape(n_vertices, 3, n_components)

        print(f"  Shape model: {n_vertices} vertices, {n_components} PCA components")

        # Load expression model (if available)
        if 'expression' in self.model_file:
            self.expression_mean = np.array(self.model_file['expression']['model']['mean'])
            self.expression_pca_basis = np.array(self.model_file['expression']['model']['pcaBasis'])
            self.expression_pca_variance = np.array(self.model_file['expression']['model']['pcaVariance'])

            # Reshape expression model
            self.expression_mean = self.expression_mean.reshape(n_vertices, 3)
            n_exp_components = self.expression_pca_basis.shape[1]
            self.expression_pca_basis = self.expression_pca_basis.reshape(n_vertices, 3, n_exp_components)
            print(f"  Expression model: {n_exp_components} components")
        else:
            self.expression_mean = None
            self.expression_pca_basis = None
            print("  No expression model in this version")

        # Load color/texture model
        if 'color' in self.model_file:
            self.color_mean = np.array(self.model_file['color']['model']['mean'])
            self.color_pca_basis = np.array(self.model_file['color']['model']['pcaBasis'])

            # Reshape color model
            self.color_mean = self.color_mean.reshape(n_vertices, 3)
            n_color_components = self.color_pca_basis.shape[1]
            self.color_pca_basis = self.color_pca_basis.reshape(n_vertices, 3, n_color_components)
            print(f"  Color model: {n_color_components} components")
        else:
            self.color_mean = None
            self.color_pca_basis = None

        # Face topology (triangles)
        # Some models store as 'cells', others as 'triangles'
        if 'cells' in self.model_file['shape']['representer']:
            self.faces = np.array(self.model_file['shape']['representer']['cells'])
        elif 'triangles' in self.model_file['shape']['representer']:
            self.faces = np.array(self.model_file['shape']['representer']['triangles'])
        else:
            raise ValueError("Could not find face topology in model file")

        # Faces might be stored as (3, n_faces) or (n_faces, 3)
        if self.faces.shape[0] == 3 and self.faces.shape[1] > 3:
            self.faces = self.faces.T  # Transpose to (n_faces, 3)

        # Convert to 0-indexed if needed (some models are 1-indexed)
        if self.faces.min() == 1:
            self.faces = self.faces - 1

        self.n_vertices = self.shape_mean.shape[0]
        self.n_shape_params = self.shape_pca_basis.shape[2]

        print(f"  Topology: {len(self.faces)} triangular faces")
        print(f"  Total: {self.n_vertices} vertices, {self.n_shape_params} shape parameters")

    def generate_face(self, shape_params=None, expression_params=None):
        """
        Generate a 3D face mesh from parameters.

        Args:
            shape_params: Shape coefficients (default: zeros = average face)
            expression_params: Expression coefficients (default: neutral)

        Returns:
            vertices: (N, 3) array of 3D vertex positions
        """
        vertices = self.shape_mean.copy()

        # Apply shape parameters
        if shape_params is not None:
            shape_params = np.array(shape_params)
            if len(shape_params) > self.n_shape_params:
                shape_params = shape_params[:self.n_shape_params]

            for i, param in enumerate(shape_params):
                vertices += param * self.shape_pca_basis[:, :, i]

        # Apply expression parameters (if model supports it)
        if expression_params is not None and self.expression_pca_basis is not None:
            expression_params = np.array(expression_params)
            n_exp_params = min(len(expression_params), self.expression_pca_basis.shape[2])

            for i in range(n_exp_params):
                vertices += expression_params[i] * self.expression_pca_basis[:, :, i]

        return vertices

    def get_landmark_indices(self):
        """
        Returns indices of BFM vertices that correspond to facial landmarks.
        These map to MediaPipe's 468 landmarks.

        Note: Full mapping requires landmark correspondence file.
        For now, we'll use key landmarks.
        """
        # This is a simplified mapping - ideally you'd have a pre-computed file
        # that maps MediaPipe landmarks to BFM vertex indices

        # For demo purposes, return some key vertex indices
        # In production, you'd load this from a correspondence file

        # Approximate key points (these need to be calibrated)
        landmark_indices = {
            'nose_tip': 8171,      # Approximate nose tip
            'left_eye': 11397,     # Left eye center
            'right_eye': 4542,     # Right eye center
            'left_mouth': 3825,    # Left mouth corner
            'right_mouth': 6849,   # Right mouth corner
            'chin': 3704,          # Chin center
        }

        return landmark_indices

    def save_obj(self, vertices, output_path, texture_img=None):
        """
        Save the BFM mesh as an OBJ file.

        Args:
            vertices: (N, 3) vertex positions
            output_path: Output .obj file path
            texture_img: Optional texture image
        """
        with open(output_path, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (triangles)
            for face in self.faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Saved BFM mesh to: {output_path}")

    def close(self):
        """Close the HDF5 file."""
        if hasattr(self, 'model_file'):
            self.model_file.close()


class BaselFitter:
    """
    Fits Basel Face Model to detected landmarks.
    """

    def __init__(self, bfm):
        """
        Args:
            bfm: BaselFaceModel instance
        """
        self.bfm = bfm

    def fit_to_landmarks(self, landmarks_2d, landmarks_3d, img_shape,
                        n_shape_components=30, n_expression_components=10):
        """
        Fit BFM to detected landmarks.

        Args:
            landmarks_2d: MediaPipe 2D landmarks (pixel coordinates)
            landmarks_3d: MediaPipe 3D landmarks (with estimated depth)
            img_shape: (height, width, channels) of image
            n_shape_components: Number of shape PCA components to use
            n_expression_components: Number of expression components

        Returns:
            fitted_vertices: (N, 3) array of fitted 3D vertices
            shape_params: Fitted shape parameters
        """
        print("\nFitting Basel Face Model to landmarks...")
        print(f"  Using {n_shape_components} shape components")

        # Initialize parameters (start with average face)
        shape_params = np.zeros(n_shape_components)

        if self.bfm.expression_pca_basis is not None:
            expression_params = np.zeros(n_expression_components)
        else:
            expression_params = None

        # For now, generate an initial face (will implement optimization later)
        # This is a placeholder - full implementation requires landmark correspondence

        print("  Note: Using average Basel face (full fitting requires landmark correspondence)")
        print("        This will be implemented in the next iteration")

        vertices = self.bfm.generate_face(shape_params, expression_params)

        # Scale to match image dimensions
        h, w = img_shape[:2]
        scale = min(w, h) * 0.8
        vertices_scaled = vertices * scale

        # Center the face in image coordinates
        vertices_scaled[:, 0] += w / 2
        vertices_scaled[:, 1] += h / 2

        return vertices_scaled, shape_params


def test_basel_model():
    """
    Test function to verify Basel Face Model is loaded correctly.
    """
    # Try to find any Basel model (2017, 2019, etc.)
    model_paths = [
        "models/model2019_fullHead.h5",             # BFM 2019 full head
        "models/model2017-1_bfm_nomouth.h5",        # BFM 2017 full head
        "models/model2017-1_face12_nomouth.h5",     # BFM 2017 face only
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Found model: {path}")
            break

    if model_path is None:
        model_path = model_paths[0]  # Use first as default for error message

    try:
        bfm = BaselFaceModel(model_path)

        print("\n" + "="*60)
        print("Basel Face Model Test")
        print("="*60)

        # Generate average face
        avg_vertices = bfm.generate_face()
        print(f"\nGenerated average face: {avg_vertices.shape}")
        print(f"  Vertex range X: [{avg_vertices[:, 0].min():.2f}, {avg_vertices[:, 0].max():.2f}]")
        print(f"  Vertex range Y: [{avg_vertices[:, 1].min():.2f}, {avg_vertices[:, 1].max():.2f}]")
        print(f"  Vertex range Z: [{avg_vertices[:, 2].min():.2f}, {avg_vertices[:, 2].max():.2f}]")

        # Test with random shape parameters
        shape_params = np.random.randn(10) * 0.5  # Small random variations
        modified_vertices = bfm.generate_face(shape_params)
        print(f"\nGenerated modified face with random parameters")

        # Save test mesh
        output_path = "output/test_basel_average_face.obj"
        bfm.save_obj(avg_vertices, output_path)

        bfm.close()

        print("\n" + "="*60)
        print("✓ Basel Face Model loaded successfully!")
        print("="*60)

        return True

    except FileNotFoundError as e:
        print("\n" + "="*60)
        print("✗ Basel Face Model NOT FOUND")
        print("="*60)
        print(str(e))
        print("\nPlease follow the instructions in setup_basel.md")
        return False


if __name__ == "__main__":
    test_basel_model()
