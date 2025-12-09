from pathlib import Path
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
import numpy as np
import math

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
   
    gltf_path_as_path: Path = Path(gltf_path)
    ply_path_as_path: Path = Path(ply_path)

    try:
        mesh = o3d.io.read_triangle_mesh(gltf_path_as_path)
        if mesh.is_empty():
            return f"No mesh data found in {gltf_path_as_path}"
        
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        o3d.io.write_point_cloud(ply_path_as_path, pcd)
        
        logger.debug(f"Point cloud saved to {ply_path_as_path}")
        return f"Point cloud saved to '{ply_path_as_path}' ({num_points} points)"
    
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
def set_camera_focal_length(focal_length_mm: float):
    """
    Set camera focal length in millimeters.

    :param focal_length_mm: float, e.g. 35.0
    """

    bpy.data.cameras[0].lens = focal_length_mm

def get_camera_fov(degrees=True):
    """
    Returns (hfov, vfov) for the active or named camera.
    """

    scene = bpy.context.scene
    cam = bpy.data.cameras[0]

    if cam is None:
        raise ValueError("Camera not found")

    camd = cam

    if camd.type != 'PERSP':
        raise ValueError("Camera is not perspective")

    # Sensor dimensions (mm)
    sensor_width = camd.sensor_width
    sensor_height = camd.sensor_height

    # Focal length (mm)
    f = camd.lens

    # Render resolution & pixel aspect
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    px_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    # Correct sensor fit
    if camd.sensor_fit == 'VERTICAL':
        sensor_width = sensor_height * res_x / res_y / px_aspect
    else:  # HORIZONTAL or AUTO
        sensor_height = sensor_width * res_y / res_x * px_aspect

    # FOV formulas
    hfov = 2 * math.atan(sensor_width / (2 * f))
    vfov = 2 * math.atan(sensor_height / (2 * f))

    if degrees:
        hfov = math.degrees(hfov)
        vfov = math.degrees(vfov)

    return hfov, vfov

@mcp.tool()
def get_projection_matrix(clip_start: float | None = None, clip_end: float | None = None) -> np.ndarray:
    """ Returns the projection matrix, it allows to overwrite the current used values for the near and far
    clipping plane.

    :param clip_start: The distance between the camera pose and the near clipping plane.
    :param clip_end: The distance between the camera pose and the far clipping plane.
    :return: The 4x4 projection matrix of the current camera
    """

    cam = bpy.data.cameras[0]

    if clip_start is None:
        near = cam.clip_start
    else:
        near = clip_start
    if clip_end is None:
        far = cam.clip_end
    else:
        far = clip_end
    # get the field of view
    x_fov, y_fov = get_camera_fov()
    height, width = 1.0 / np.tan(y_fov * 0.5), 1. / np.tan(x_fov * 0.5)
    return np.array([[width, 0, 0, 0], [0, height, 0, 0],
                     [0, 0, -(near + far) / (far - near), -(2 * near * far) / (far - near)],
                     [0, 0, -1, 0]])



def set_camera_from_projection_matrix(
    P: np.ndarray,
    width_px: int,
    height_px: int,
):
    """
    Sets Blender camera intrinsics so that rendering behaves as if
    the given projection matrix P were used.

    P: 4x4 OpenGL-style projection matrix
    width_px, height_px: render resolution
    """

    scene = bpy.context.scene
    cam_obj = bpy.data.cameras[0] 
    cam = cam_obj

    # --- extract intrinsics from P ---
    fx = P[0, 0]
    fy = P[1, 1]

    # principal point in NDC
    cx_ndc = P[0, 2]
    cy_ndc = P[1, 2]

    # near / far
    A = P[2, 2]
    B = P[2, 3]

    near = B / (A - 1)
    far  = B / (A + 1)

    # --- set render resolution ---
    scene.render.resolution_x = width_px
    scene.render.resolution_y = height_px
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = fx / fy

    # --- camera model ---
    cam.type = 'PERSP'
    cam.sensor_fit = 'HORIZONTAL'

    # --- focal length (px → mm) ---
    cam.lens = fx * cam.sensor_width / width_px

    # --- principal point (NDC → Blender shift) ---
    cam.shift_x = cx_ndc / 2.0
    cam.shift_y = -cy_ndc / 2.0

    # --- clipping planes ---
    cam.clip_start = near
    cam.clip_end = far

@mcp.tool()
def set_camera_intrinsics_from_K(
    K,
    width_px,
    height_px,
    camera_name=None
):
    """
    Set Blender camera intrinsics from OpenCV camera matrix K.

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    width_px, height_px = image resolution
    """

    scene = bpy.context.scene

    # Get camera

    cam = bpy.data.cameras[0]

    cam_data = cam
    cam_data.type = 'PERSP'

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    # Set render resolution
    scene.render.resolution_x = width_px
    scene.render.resolution_y = height_px
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height

    cam_data.lens = fx * sensor_width / width_px

    cam_data.shift_x = (cx - width_px / 2) / width_px
    cam_data.shift_y = (height_px / 2 - cy) / height_px


@mcp.tool()
def render_image(output_path: str):
    """
    Render the current scene.

    :param output_path: str, path to output image (e.g. '//render.png')
    :param write_still: bool, save image to disk
    """
    scene = bpy.context.scene
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


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

