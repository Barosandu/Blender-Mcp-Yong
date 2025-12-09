# Blender MCP Server
1. File Management
	Load a Blender file
`load_file(filepath: str) -> str`
Load a .blend file into Blender.


	Save the current Blender file
`save_file(filepath: str) -> str`
Save the current scene to a specified path.


2. Object Creation
	Create a cube
`create_cube(name: str) -> str`


	Create a sphere
`create_sphere(name: str) -> str`


	List all objects
`list_objects() -> list[str]`
Returns names of all objects in the scene.



3. Object Manipulation
	Move an object
`move_object(object_id: str, x: float, y: float, z: float) -> str`


	Rotate an object
`rotate_object(object_id: str, rx: float, ry: float, rz: float) -> str`
Rotates the object by Euler angles (radians).


	Compute distance between two objects
`distance_between(obj1_id: str, obj2_id: str) -> float`

4. Export
	Export to glTF
`export_gltf(filepath: str, objects: list[str] = None) -> str`
Exports selected or all objects to a .glb/.gltf file.


	Export to point cloud
`gltf_to_pointcloud(gltf_path: str, ply_path: str, num_points: int = 10000, include_interior: bool = False) -> str`
Converts a GLTF/GLB mesh to a PLY point cloud.
	include_interior=True samples points inside the mesh volume.
	num_points controls the number of points sampled.


	Set camera focal length. Tweaks focal length of a camera
`set_camera_focal_length(focal_length_mm: float):`

	Renders the image to png
`def render_image(output_path: str):`
