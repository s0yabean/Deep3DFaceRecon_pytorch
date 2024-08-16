import numpy as np
import pyvista as pv
import yaml
import matplotlib.pyplot as plt
from PIL import Image


# Path to the OBJ file
obj_file = "checkpoints/custom/results/examples/epoch_20_000000/000055.obj"

# Path to the image file
img_path = "checkpoints/custom/results/examples/epoch_20_000000/000055.png"


# Load the points from the YAML file
with open('selected_point_ids.yaml', 'r') as file:
    data = yaml.safe_load(file)
    point_ids = data['point_ids']

# Read the OBJ file
mesh = pv.read(obj_file)
print(mesh)

# Retrieve the coordinates of the selected points from the mesh
selected_points = mesh.points[point_ids]

# Assume the points are [nose_tip, left_eye, right_eye]
nose_tip = selected_points[0]
left_eye = selected_points[1]
right_eye = selected_points[2]

# Define the source and destination points
src_points = np.array([nose_tip, left_eye, right_eye])
dst_points = np.array([
    [0, 0, 0],       # Nose Tip
    [-0.1, 0.1, 0],  # Left Eye
    [0.1, 0.1, 0]    # Right Eye
])

def compute_transformation_matrix(src_points, dst_points):
    """
    Calculate the transformation matrix to align src_points with dst_points.
    
    Args:
    src_points (np.array): Coordinates of points in the original mesh.
    dst_points (np.array): Target coordinates for alignment.
    
    Returns:
    np.array: 4x4 transformation matrix.
    """
    # Calculate the translation
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)
    translation = dst_center - src_center
    
    # Calculate the rotation matrix
    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center
    H = np.dot(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)
    
    # Combine translation and rotation
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_mat
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix

def transform_mesh(mesh, transformation_matrix):
    """
    Apply the transformation matrix to the mesh points.
    
    Args:
    mesh (pv.PolyData): PyVista mesh object.
    transformation_matrix (np.array): 4x4 transformation matrix.
    
    Returns:
    pv.PolyData: Transformed mesh.
    """
    # Convert mesh points to homogeneous coordinates
    homog_coords = np.hstack([mesh.points, np.ones((mesh.points.shape[0], 1))])
    transformed = np.dot(transformation_matrix, homog_coords.T).T
    transformed_mesh = pv.PolyData(transformed[:, :3])
    return transformed_mesh

# Compute the transformation matrix
transformation_matrix = compute_transformation_matrix(src_points, dst_points)

# Transform the mesh
transformed_mesh = transform_mesh(mesh, transformation_matrix)
# transformed_mesh = transformed_mesh.reconstruct_surface()
print (transformed_mesh)

# Create a PyVista plotter object with a 1x2 subplot layout
plotter = pv.Plotter(shape=(1, 2))

# Add the original mesh to the first subplot
plotter.subplot(0, 0)
plotter.add_mesh(mesh, show_edges=True, color='blue', edge_color='black', line_width=1.0, label='Original Mesh')
plotter.add_title("Original Mesh")
plotter.add_axes()

plotter.set_viewup([0, 1, 0])
plotter.camera_position = 'xy'
plotter.camera.zoom(1.2)


# Add the transformed mesh to the second subplot
plotter.subplot(0, 1)
plotter.add_mesh(transformed_mesh, show_edges=True, color='red', edge_color='black', line_width=1.0, label='Transformed Mesh')
plotter.add_title("Transformed Mesh")
plotter.add_axes()

# Adjust camera position and zoom for both subplots
plotter.set_viewup([0, 1, 0])
plotter.camera_position = 'xy'
plotter.camera.zoom(1.2)

# Add a legend
plotter.add_legend()

# Open the image file using PIL
image = Image.open(img_path)

# Display the image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')  # Hide axes
plt.title('Image')
plt.show()

# Show the plotter
plotter.show()




