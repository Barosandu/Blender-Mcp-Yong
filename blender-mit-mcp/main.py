from typing import Any
from typing_extensions import override
from mcp.server.fastmcp import FastMCP
import bpy
from mathutils import Vector
import os
import sys
import io
import logging

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

if __name__ == "__main__":
    mcp.run(transport='stdio')

