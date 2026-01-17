"""
Basel Face Model Texture Projection
Uses proper UV coordinates from Basel model to create texture atlas.
"""
import cv2
import numpy as np
import json
import os


class BaselTextureProjector:
    """
    Projects face images onto Basel Face Model texture atlas using proper UV coordinates.
    """

    def __init__(self, texture_mapping_path="models/model2019_textureMapping.json"):
        """
        Load Basel texture mapping (UV coordinates).

        Args:
            texture_mapping_path: Path to texture mapping JSON file
        """
        if not os.path.exists(texture_mapping_path):
            raise FileNotFoundError(
                f"Texture mapping file not found: {texture_mapping_path}\n"
                f"Download from Basel Face Model 2019 distribution"
            )

        print(f"Loading Basel texture mapping: {texture_mapping_path}")
        with open(texture_mapping_path, 'r') as f:
            data = json.load(f)

        # Extract UV coordinates for texture vertices
        self.uv_coords = np.array(data['textureMapping']['pointData'])

        # Texture triangles: indices into UV coordinate array
        self.texture_triangles = np.array(data['textureMapping']['triangles'])

        # Mesh triangles: indices into 3D mesh vertex array
        self.mesh_triangles = np.array(data['triangles'])

        print(f"  UV coordinates: {len(self.uv_coords)}")
        print(f"  Mesh triangles: {len(self.mesh_triangles)}")
        print(f"  Texture triangles: {len(self.texture_triangles)}")

    def create_texture_atlas(self, vertices_3d, image, texture_size=(2048, 2048)):
        """
        Create texture atlas by projecting image onto UV space using 3D vertex positions.

        Args:
            vertices_3d: Fitted Basel model vertices in image space (N, 3)
            image: Face image to project onto texture
            texture_size: Output texture resolution (width, height)

        Returns:
            texture_atlas: Texture image in UV space
        """
        h_img, w_img = image.shape[:2]
        w_tex, h_tex = texture_size

        # Create empty texture atlas
        texture_atlas = np.zeros((h_tex, w_tex, 3), dtype=np.uint8)
        weight_map = np.zeros((h_tex, w_tex), dtype=np.float32)

        print(f"  Creating {w_tex}x{h_tex} texture atlas...")

        # For each triangle, sample colors from the image
        for tri_idx in range(len(self.mesh_triangles)):
            # Get mesh vertex indices
            mesh_tri = self.mesh_triangles[tri_idx]

            # Get 3D positions in image space
            v0 = vertices_3d[mesh_tri[0]]
            v1 = vertices_3d[mesh_tri[1]]
            v2 = vertices_3d[mesh_tri[2]]

            # Get UV coordinates for this triangle
            tex_tri = self.texture_triangles[tri_idx]
            uv0 = self.uv_coords[tex_tri[0]]
            uv1 = self.uv_coords[tex_tri[1]]
            uv2 = self.uv_coords[tex_tri[2]]

            # Check if triangle is visible (front-facing and within image bounds)
            # Use XY positions to check if in image
            if not self._is_triangle_valid(v0, v1, v2, w_img, h_img):
                continue

            # Rasterize triangle in UV space
            self._rasterize_textured_triangle(
                texture_atlas, weight_map,
                uv0, uv1, uv2,
                v0[:2], v1[:2], v2[:2],  # Only XY for image sampling
                image, w_tex, h_tex, w_img, h_img
            )

        # Normalize by weights (handle areas sampled multiple times)
        mask = weight_map > 0
        texture_atlas[mask] = (texture_atlas[mask] / weight_map[mask, np.newaxis]).astype(np.uint8)

        # Fill holes with nearest neighbor interpolation
        texture_atlas = self._fill_texture_holes(texture_atlas, mask)

        print(f"  Texture coverage: {(mask.sum() / mask.size * 100):.1f}%")

        return texture_atlas, self.uv_coords

    def _is_triangle_valid(self, v0, v1, v2, w_img, h_img):
        """Check if triangle is within image bounds and visible."""
        # Check bounds
        for v in [v0, v1, v2]:
            if v[0] < 0 or v[0] >= w_img or v[1] < 0 or v[1] >= h_img:
                return False
        return True

    def _rasterize_textured_triangle(self, tex_atlas, weight_map,
                                     uv0, uv1, uv2, img_p0, img_p1, img_p2,
                                     image, w_tex, h_tex, w_img, h_img):
        """
        Rasterize a textured triangle: sample colors from image and place in UV space.

        Args:
            tex_atlas: Output texture atlas
            weight_map: Weight accumulation map
            uv0, uv1, uv2: UV coordinates of triangle vertices (0-1 range)
            img_p0, img_p1, img_p2: Corresponding image positions (pixel coords)
            image: Source image
            w_tex, h_tex: Texture atlas size
            w_img, h_img: Image size
        """
        # Convert UV to pixel coordinates
        uv0_px = np.array([uv0[0] * w_tex, uv0[1] * h_tex])
        uv1_px = np.array([uv1[0] * w_tex, uv1[1] * h_tex])
        uv2_px = np.array([uv2[0] * w_tex, uv2[1] * h_tex])

        # Compute bounding box in UV space
        min_x = int(max(0, min(uv0_px[0], uv1_px[0], uv2_px[0])))
        max_x = int(min(w_tex, max(uv0_px[0], uv1_px[0], uv2_px[0]) + 1))
        min_y = int(max(0, min(uv0_px[1], uv1_px[1], uv2_px[1])))
        max_y = int(min(h_tex, max(uv0_px[1], uv1_px[1], uv2_px[1]) + 1))

        # Iterate over bounding box
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # Compute barycentric coordinates in UV space
                bary = self._barycentric(np.array([x, y]), uv0_px, uv1_px, uv2_px)

                # Check if point is inside triangle
                if bary[0] >= 0 and bary[1] >= 0 and bary[2] >= 0:
                    # Interpolate image position using barycentric coordinates
                    img_pos = bary[0] * img_p0 + bary[1] * img_p1 + bary[2] * img_p2

                    # Sample color from image (with bounds checking)
                    img_x = int(np.clip(img_pos[0], 0, w_img - 1))
                    img_y = int(np.clip(img_pos[1], 0, h_img - 1))
                    color = image[img_y, img_x]

                    # Accumulate color and weight
                    tex_atlas[y, x] += color
                    weight_map[y, x] += 1.0

    def _barycentric(self, p, a, b, c):
        """Compute barycentric coordinates of point p in triangle abc."""
        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return np.array([-1, -1, -1])  # Degenerate triangle

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return np.array([u, v, w])

    def _fill_texture_holes(self, texture, mask):
        """Fill holes in texture using nearest neighbor interpolation."""
        # Find unfilled regions
        unfilled = ~mask

        if unfilled.sum() == 0:
            return texture

        # Use inpainting to fill holes
        unfilled_uint8 = unfilled.astype(np.uint8) * 255
        filled = cv2.inpaint(texture, unfilled_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return filled


if __name__ == "__main__":
    print("Basel Texture Projector module")
    print("Use to create proper texture atlases for Basel Face Model")
    print("Import and use with: from basel_texture_projector import BaselTextureProjector")
