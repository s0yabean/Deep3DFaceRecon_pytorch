import pyvista as pv
import yaml

# Path to the OBJ file
obj_file = "checkpoints/custom/results/examples/epoch_20_000000/000002.obj"

# Read the OBJ file
mesh = pv.read(obj_file)

# List to store selected point IDs
selected_point_ids = []

def point_info(point, picker):
    # Get the ID of the selected point
    point_id = picker.GetPointId()
    print (picker.GetPickPosition ())
    # Append the point ID to the list
    if point_id != -1:  # If a valid point is selected
        selected_point_ids.append(point_id)
    
    # If the list contains more than 3 points, remove the oldest one
    if len(selected_point_ids) > 3:
        selected_point_ids.pop(0)
    # Save the selected point IDs to a YAML file once we have 3 points
    if len(selected_point_ids) == 3:
        with open('selected_point_ids.yaml', 'w') as file:
            yaml.dump({'point_ids': selected_point_ids}, file)
        print("Point IDs saved to 'selected_point_ids.yaml'")
        plotter.close()  # Close the plotter after saving the point IDs

# Create a Plotter instance
plotter = pv.Plotter()

# Add the mesh to the plotter
plotter.add_mesh(mesh, show_edges=True, color='white')

# Enable point picking with the callback function
plotter.enable_point_picking(callback=point_info, show_message=False, font_size=18, color='pink', point_size=15, use_picker=True, show_point=True)

# Show the plotter window
plotter.show()
