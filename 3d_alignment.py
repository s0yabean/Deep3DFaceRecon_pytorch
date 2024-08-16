import os
import argparse
import numpy as np
import pyvista as pv
import yaml
import matplotlib.pyplot as plt
from PIL import Image

def compute_transformation_matrix(src_points, dst_points):
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)
    translation = dst_center - src_center

    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center
    H = np.dot(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_mat
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

def transform_mesh(mesh, transformation_matrix):
    homog_coords = np.hstack([mesh.points, np.ones((mesh.points.shape[0], 1))])
    transformed = np.dot(transformation_matrix, homog_coords.T).T
    transformed_mesh = pv.PolyData(transformed[:, :3])
    return transformed_mesh

def rotation_matrix(rotation_angles):
    x_angle, y_angle, z_angle = np.radians(rotation_angles)
    
    # X-axis rotation
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle), 0],
        [0, np.sin(x_angle), np.cos(x_angle), 0],
        [0, 0, 0, 1]
    ])
    
    # Y-axis rotation
    Ry = np.array([
        [np.cos(y_angle), 0, np.sin(y_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(y_angle), 0, np.cos(y_angle), 0],
        [0, 0, 0, 1]
    ])
    
    # Z-axis rotation
    Rz = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0, 0],
        [np.sin(z_angle), np.cos(z_angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx

def find_image_path(obj_file):
    base_path = os.path.splitext(obj_file)[0]
    for ext in ['png', 'jpg', 'jpeg']:
        img_path = f"{base_path}.{ext}"
        if os.path.exists(img_path):
            return img_path
    return None

def main(obj_file, rotation_angles):
    # Find image path based on obj_file
    img_path = find_image_path(obj_file)
    if img_path:
        print(f"Image found at: {img_path}")
    else:
        print("Couldn't locate image, skipping rendering image.")

    # If the image is found, render it
    if img_path:
        image = Image.open(img_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title('Image')
        plt.show()

    # Load the points from the YAML file
    with open('selected_point_ids.yaml', 'r') as file:
        data = yaml.safe_load(file)
        point_ids = data['point_ids']

    # Read the OBJ file
    mesh = pv.read(obj_file)

    # Retrieve the coordinates of the selected points from the mesh
    selected_points = mesh.points[point_ids]

    nose_tip, left_eye, right_eye = selected_points

    # Define the source and destination points
    src_points = np.array([nose_tip, left_eye, right_eye])
    dst_points = np.array([
        [0, 0, 0],       
        [-0.1, 0.1, 0],  
        [0.1, 0.1, 0]
    ])

    # Compute the transformation matrix
    transformation_matrix = compute_transformation_matrix(src_points, dst_points)

    # Transform the mesh
    transformed_mesh = transform_mesh(mesh, transformation_matrix)

    # Apply the rotation matrix to the transformed mesh
    rot_matrix = rotation_matrix(rotation_angles)
    transformed_mesh_rotated = transform_mesh(transformed_mesh, rot_matrix)

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

    # Add the transformed and rotated mesh to the second subplot
    plotter.subplot(0, 1)
    plotter.add_mesh(transformed_mesh_rotated, show_edges=True, color='pink', edge_color='black', line_width=1.0, label='Rotated Mesh')
    plotter.add_title(f"Transformed and Rotated Mesh\n {rotation_angles} degrees",font_size=10)
    plotter.add_axes()
    plotter.set_viewup([0, 1, 0])
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.2)
    plotter.add_legend()

    # Show the plotter
    plotter.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Face Mesh Transformation and Rotation")
    parser.add_argument("obj_file", type=str, help="Path to the OBJ file")
    parser.add_argument("--rotation_angles",type=float,nargs=3,default=[40.0, 0.0, 0.0],
                        help="Rotation angles for x, y, and z axes (default: 40 0 0)"
)


    args = parser.parse_args()

    main(args.obj_file, args.rotation_angles)
