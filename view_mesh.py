import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import os
import sys

def view_obj(filepath):
    print(f"Loading {filepath}...")
    vertices = []
    edges = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # v x y z
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('l '):
                # l v1 v2
                parts = line.strip().split()
                # OBJ is 1-indexed, Python is 0-indexed
                v1 = int(parts[1]) - 1
                v2 = int(parts[2]) - 1
                edges.append((v1, v2))
                
    if not vertices:
        print("No vertices found.")
        return

    vertices = np.array(vertices)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices (dots)
    # s=size, c=color
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], s=2, c='green', depthshade=False)
    
    # Plot edges (wireframe lines)
    if edges:
        segments = []
        for v1, v2 in edges:
            segments.append([vertices[v1], vertices[v2]])
            
        line_collection = Line3DCollection(segments, colors='black', linewidths=0.3, alpha=0.3)
        ax.add_collection(line_collection)
        
    # Formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Force equal aspect ratio (Matplotlib 3D doesn't do this by default)
    # We create a bounding box around the face
    max_range = np.array([
        vertices[:,0].max()-vertices[:,0].min(), 
        vertices[:,1].max()-vertices[:,1].min(), 
        vertices[:,2].max()-vertices[:,2].min()
    ]).max() / 2.0

    mid_x = (vertices[:,0].max()+vertices[:,0].min()) * 0.5
    mid_y = (vertices[:,1].max()+vertices[:,1].min()) * 0.5
    mid_z = (vertices[:,2].max()+vertices[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.title(f"3D Viewer: {os.path.basename(filepath)}")
    print("Opening viewer window... (Use mouse to rotate)")
    plt.show()

if __name__ == "__main__":
    # Automatically find the first OBJ file in the output folder
    output_dir = "output"
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(".obj")]
        if files:
            view_obj(os.path.join(output_dir, files[0]))
        else:
            print("No .obj files found in 'output' folder.")
    else:
        print("Output directory not found.")