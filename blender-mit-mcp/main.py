from typing import Any
from mcp.server.fastmcp import FastMCP
import bpy
from mathutils import Vector
import os

# Create MCP server
mcp: FastMCP[Any] = FastMCP("blender-mit-mcp", port=8000)

@mcp.tool()
def load_file(filepath: str) -> str:
    print(f"[DEBUG] load_file called with: {filepath}")
    if not os.path.exists(filepath):
        print(f"[DEBUG] File not found: {filepath}")
        return f"File '{filepath}' does not exist"
    bpy.ops.wm.open_mainfile(filepath=filepath)
    print(f"[DEBUG] File loaded: {filepath}")
    return f"File '{filepath}' loaded"


@mcp.tool()
def save_file(filepath: str) -> str:
    print(f"[DEBUG] save_file called with: {filepath}")
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"[DEBUG] File saved: {filepath}")
    return f"File saved to '{filepath}'"


@mcp.tool()
def create_cube(name: str) -> str:
    print(f"[DEBUG] create_cube called with: {name}")
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.name = name
    print(f"[DEBUG] Cube created: {obj.name}")
    return f"Cube '{name}' created"


@mcp.tool()
def create_sphere(name: str) -> str:
    print(f"[DEBUG] create_sphere called with: {name}")
    bpy.ops.mesh.primitive_uv_sphere_add()
    obj = bpy.context.active_object
    obj.name = name
    print(f"[DEBUG] Sphere created: {obj.name}")
    return f"Sphere '{name}' created"


@mcp.tool()
def move_object(object_id: str, x: float, y: float, z: float) -> str:
    print(f"[DEBUG] move_object called: {object_id} -> ({x}, {y}, {z})")
    obj = bpy.data.objects.get(object_id)
    if obj is None:
        print(f"[DEBUG] Object not found: {object_id}")
        return f"Object '{object_id}' not found"
    obj.location = (x, y, z)
    print(f"[DEBUG] Object '{object_id}' moved to: {obj.location}")
    return f"Object '{object_id}' moved to ({x}, {y}, {z})"


@mcp.tool()
def distance_between(obj1_id: str, obj2_id: str) -> float:
    print(f"[DEBUG] distance_between called: {obj1_id}, {obj2_id}")
    o1 = bpy.data.objects.get(obj1_id)
    o2 = bpy.data.objects.get(obj2_id)
    if o1 is None:
        print(f"[DEBUG] Object not found: {obj1_id}")
        raise ValueError(f"Object '{obj1_id}' not found")
    if o2 is None:
        print(f"[DEBUG] Object not found: {obj2_id}")
        raise ValueError(f"Object '{obj2_id}' not found")
    dist = (Vector(o1.location) - Vector(o2.location)).length
    print(f"[DEBUG] Distance between '{obj1_id}' and '{obj2_id}': {dist}")
    return dist


@mcp.tool()
def list_objects() -> list[dict[Any, Any]]:
    """
    List all objects in the current Blender scene with name, type, and location.
    """
    print("[DEBUG] list_objects called")
    objects_info = []
    for obj in bpy.data.objects:
        info = {
            "name": obj.name,
            "type": obj.type,
            "location": tuple(obj.location)
        }
        objects_info.append(info)
    print(f"[DEBUG] Found {len(objects_info)} objects")
    return objects_info

if __name__ == "__main__":
    mcp.run(transport='stdio')

