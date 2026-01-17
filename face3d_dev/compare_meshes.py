"""
Compare meshes generated with different methods.
Visualizes depth differences between single-view and multi-view approaches.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def load_obj_vertices(filepath):
    """Load vertices from an OBJ file."""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)


def compare_depth_profiles(old_obj, new_obj):
    """
    Compare depth (Z-coordinate) profiles of two meshes.
    """
    if not os.path.exists(old_obj):
        print(f"Old mesh not found: {old_obj}")
        return

    if not os.path.exists(new_obj):
        print(f"New mesh not found: {new_obj}")
        return

    print("Loading meshes...")
    verts_old = load_obj_vertices(old_obj)
    verts_new = load_obj_vertices(new_obj)

    if len(verts_old) != len(verts_new):
        print(f"Warning: Vertex counts differ: {len(verts_old)} vs {len(verts_new)}")

    # Extract Z-coordinates (depth)
    z_old = verts_old[:, 2]
    z_new = verts_new[:, 2]

    # Calculate statistics
    print("\n" + "="*60)
    print("DEPTH COMPARISON")
    print("="*60)
    print(f"Single-view (old):")
    print(f"  Z range: {z_old.min():.1f} to {z_old.max():.1f}")
    print(f"  Z span:  {z_old.max() - z_old.min():.1f}")
    print(f"  Z mean:  {z_old.mean():.1f}")
    print(f"  Z std:   {z_old.std():.1f}")

    print(f"\nMulti-view (new):")
    print(f"  Z range: {z_new.min():.1f} to {z_new.max():.1f}")
    print(f"  Z span:  {z_new.max() - z_new.min():.1f}")
    print(f"  Z mean:  {z_new.mean():.1f}")
    print(f"  Z std:   {z_new.std():.1f}")

    print(f"\nDifference:")
    z_diff = np.abs(z_new - z_old)
    print(f"  Mean absolute difference: {z_diff.mean():.1f}")
    print(f"  Max difference:           {z_diff.max():.1f}")
    print(f"  % vertices changed >5:    {(z_diff > 5).sum() / len(z_diff) * 100:.1f}%")
    print("="*60 + "\n")

    # Visualization
    fig = plt.figure(figsize=(16, 10))

    # 3D scatter plots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    sc1 = ax1.scatter(verts_old[:, 0], verts_old[:, 1], verts_old[:, 2],
                     c=z_old, cmap='viridis', s=1)
    ax1.set_title('Single-View (Old Method)', fontsize=12, weight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Depth)')
    plt.colorbar(sc1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    sc2 = ax2.scatter(verts_new[:, 0], verts_new[:, 1], verts_new[:, 2],
                     c=z_new, cmap='viridis', s=1)
    ax2.set_title('Multi-View Depth Fusion (New)', fontsize=12, weight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z (Depth)')
    plt.colorbar(sc2, ax=ax2, shrink=0.5)

    # Depth difference visualization
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    sc3 = ax3.scatter(verts_new[:, 0], verts_new[:, 1], verts_new[:, 2],
                     c=z_diff, cmap='Reds', s=2)
    ax3.set_title('Depth Change Magnitude', fontsize=12, weight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(sc3, ax=ax3, shrink=0.5, label='|Z_new - Z_old|')

    # Depth histograms
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(z_old, bins=50, alpha=0.5, label='Single-View', color='blue')
    ax4.hist(z_new, bins=50, alpha=0.5, label='Multi-View', color='green')
    ax4.set_xlabel('Z Depth')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Depth Distribution Comparison')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Depth difference histogram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(z_diff, bins=50, color='red', alpha=0.7)
    ax5.set_xlabel('Absolute Depth Difference')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Depth Change Distribution')
    ax5.axvline(z_diff.mean(), color='darkred', linestyle='--',
               label=f'Mean = {z_diff.mean():.1f}')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Scatter: old vs new depth
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(z_old, z_new, s=1, alpha=0.5)
    ax6.plot([z_old.min(), z_old.max()], [z_old.min(), z_old.max()],
            'r--', label='No change line')
    ax6.set_xlabel('Single-View Z Depth')
    ax6.set_ylabel('Multi-View Z Depth')
    ax6.set_title('Depth Correlation')
    ax6.legend()
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        subject_id = sys.argv[1]
    else:
        subject_id = "nbm0042"  # Default

    old_mesh = os.path.join("output", f"{subject_id}.obj")
    new_mesh = os.path.join("output_multiview", f"{subject_id}.obj")

    # If new mesh is in same output folder (for testing)
    if not os.path.exists(new_mesh):
        new_mesh = os.path.join("output", f"{subject_id}_multiview.obj")

    print(f"Comparing meshes for subject: {subject_id}")
    print(f"Old: {old_mesh}")
    print(f"New: {new_mesh}")

    compare_depth_profiles(old_mesh, new_mesh)
