from typing import Any
from typing_extensions import override
from mcp.server.fastmcp import FastMCP
import bpy
from mathutils import Vector
import os
import sys
import io
import logging
import open3d as o3d

log_path = os.path.expanduser("~/blender_mcp.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),        # console
        logging.FileHandler(log_path)  # file
    ]
)

logger = logging.getLogger("blender_mcp")

class Unbuffered(io.TextIOWrapper):
    def __init__(self, buffer):
        super().__init__(buffer, encoding='utf-8', line_buffering=True)
    
    @override
    def write(self, s):
        l = super().write(s)
        self.flush()
        return l

    @override
    def writelines(self, lines):
        for l in lines:
            self.write(l)

sys.stdout = Unbuffered(sys.stdout.buffer)
sys.stderr = Unbuffered(sys.stderr.buffer)
# Create MCP server
mcp: FastMCP[Any] = FastMCP("blender-mit-mcp", port=8000)


@mcp.tool()
def gltf_to_pointcloud(gltf_path: str, ply_path: str, num_points: int = 10000) -> str:
    """
    Load a GLTF/GLB file, sample points from its mesh, and save as a PLY point cloud.
    
    Parameters:
        gltf_path (str): Path to input GLTF/GLB file
        ply_path (str): Path to save output PLY point cloud
        num_points (int): Number of points to sample
    """
    logger.debug(f"gltf_to_pointcloud called: {gltf_path} -> {ply_path} with {num_points} points")
    
    try:
        mesh = o3d.io.read_triangle_mesh(gltf_path)
        if mesh.is_empty():
            return f"No mesh data found in {gltf_path}"
        
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        o3d.io.write_point_cloud(ply_path, pcd)
        
        logger.debug(f"Point cloud saved to {ply_path}")
        return f"Point cloud saved to '{ply_path}' ({num_points} points)"
    
    except Exception as e:
        logger.error(f"Failed to convert GLTF to point cloud: {e}")
        return f"Failed to convert GLTF to point cloud: {e}"

@mcp.tool()
def export_gltf(filepath: str, objects: list[str] | None = None) -> str:
    """
    Export Blender objects to a glTF file.
    - filepath: path to save the glTF file
    - objects: list of object names to export, None = all visible mesh objects
    """
    logger.debug(f"export_gltf called: {filepath}, objects={objects}")

    bpy.ops.object.select_all(action='DESELECT')

    if objects is None:
        objs_to_export = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.visible_get()]
    else:
        objs_to_export = []
        for name in objects:
            obj = bpy.data.objects.get(name)
            if obj and obj.type == 'MESH':
                objs_to_export.append(obj)

    if not objs_to_export:
        logger.error("No mesh objects found to export")
        return "No mesh objects found to export"

    for obj in objs_to_export:
        obj.select_set(True)

    try:
        bpy.ops.export_scene.gltf(
            filepath=filepath,
            use_selection=True,
            export_apply=True, 
            export_materials='EXPORT'
        )
        logger.debug(f"glTF exported to {filepath}")
        return f"glTF exported to '{filepath}' ({len(objs_to_export)} objects)"
    except Exception as e:
        logger.error(f"Failed to export glTF: {e}")
        return f"Failed to export glTF: {e}"

@mcp.tool()
def load_file(filepath: str) -> str:
    logger.debug(f"[DEBUG] load_file called with: {filepath}")
    if not os.path.exists(filepath):
        logger.debug(f"[DEBUG] File not found: {filepath}")
        return f"File '{filepath}' does not exist"
    bpy.ops.wm.open_mainfile(filepath=filepath)
    logger.debug(f"[DEBUG] File loaded: {filepath}")
    return f"File '{filepath}' loaded"


@mcp.tool()
def save_file(filepath: str) -> str:
    logger.debug(f"[DEBUG] save_file called with: {filepath}")
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    logger.debug(f"[DEBUG] File saved: {filepath}")
    return f"File saved to '{filepath}'"


@mcp.tool()
def create_cube(name: str) -> str:
    logger.debug(f"[DEBUG] create_cube called with: {name}")
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.name = name
    logger.debug(f"[DEBUG] Cube created: {obj.name}")
    return f"Cube '{name}' created"


@mcp.tool()
def create_sphere(name: str) -> str:
    logger.debug(f"[DEBUG] create_sphere called with: {name}")
    bpy.ops.mesh.primitive_uv_sphere_add()
    obj = bpy.context.active_object
    obj.name = name
    logger.debug(f"[DEBUG] Sphere created: {obj.name}")
    return f"Sphere '{name}' created"


@mcp.tool()
def move_object(object_id: str, x: float, y: float, z: float) -> str:
    logger.debug(f"[DEBUG] move_object called: {object_id} -> ({x}, {y}, {z})")
    obj = bpy.data.objects.get(object_id)
    if obj is None:
        logger.debug(f"[DEBUG] Object not found: {object_id}")
        return f"Object '{object_id}' not found"
    obj.location = (x, y, z)
    logger.debug(f"[DEBUG] Object '{object_id}' moved to: {obj.location}")
    return f"Object '{object_id}' moved to ({x}, {y}, {z})"


@mcp.tool()
def distance_between(obj1_id: str, obj2_id: str) -> float:
    logger.debug(f"[DEBUG] distance_between called: {obj1_id}, {obj2_id}")
    o1 = bpy.data.objects.get(obj1_id)
    o2 = bpy.data.objects.get(obj2_id)
    if o1 is None:
        logger.debug(f"[DEBUG] Object not found: {obj1_id}")
        raise ValueError(f"Object '{obj1_id}' not found")
    if o2 is None:
        logger.debug(f"[DEBUG] Object not found: {obj2_id}")
        raise ValueError(f"Object '{obj2_id}' not found")
    dist = (Vector(o1.location) - Vector(o2.location)).length
    logger.debug(f"[DEBUG] Distance between '{obj1_id}' and '{obj2_id}': {dist}")
    return dist


@mcp.tool()
def list_objects() -> list[dict[Any, Any]]:
    """
    List all objects in the current Blender scene with name, type, and location.
    """
    logger.debug("[DEBUG] list_objects called")
    objects_info = []
    for obj in bpy.data.objects:
        info = {
            "name": obj.name,
            "type": obj.type,
            "location": tuple(obj.location)
        }
        objects_info.append(info)
    logger.debug(f"[DEBUG] Found {len(objects_info)} objects")
    return objects_info


@mcp.tool()
def rotate_object(object_id: str, rot_x: float, rot_y: float, rot_z: float) -> str:
    """
    Rotate a Blender object by Euler angles (in radians).
    rot_x, rot_y, rot_z are rotation deltas to apply.
    """
    logger.debug(f"rotate_object called: {object_id} -> ({rot_x}, {rot_y}, {rot_z})")
    obj = bpy.data.objects.get(object_id)
    if obj is None:
        logger.error(f"Object not found: {object_id}")
        return f"Object '{object_id}' not found"

    # Apply rotation deltas
    obj.rotation_euler = (obj.rotation_euler[0] + rot_x, obj.rotation_euler[1] + rot_y, obj.rotation_euler[2]+rot_z)



    logger.debug(f"Object '{object_id}' rotated to: {tuple(obj.rotation_euler)}")
    return f"Object '{object_id}' rotated by ({rot_x}, {rot_y}, {rot_z}) radians"

if __name__ == "__main__":
    mcp.run(transport='stdio')

