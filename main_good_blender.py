from typing import final
import cv2
from cv2.gapi.wip.draw import Image
import numpy as np
import torch
import torch.nn as nn

FOCAL_LENGTH = 1000.0

import bpy
import cv2
import numpy as np
import torch
import os
import mathutils

def render_verification_blender(image_path, real_w, real_h, focus, world, sensor_width_mm=36.0):
    # 1. Resetare scenă
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Preluăm dimensiunile imaginii
    img_cv = cv2.imread(image_path)
    img_h, img_w = img_cv.shape[:2]

    # 2. Creăm Dreptunghiul (Plane)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    rect = bpy.context.active_object
    rect.scale = (real_w , real_h, 1)
    
    # Material roșu care emite lumină
    mat = bpy.data.materials.new(name="GlowRed")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    node_emit = nodes.new(type='ShaderNodeEmission')
    node_emit.inputs['Color'].default_value = (1, 0, 0, 1)
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(node_emit.outputs['Emission'], node_out.inputs['Surface'])
    rect.data.materials.append(mat)

    # 3. Creăm Camera
    cam_data = bpy.data.cameras.new("OptimizedCam")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    
    # Asigură-te că world este în format numpy array
    if isinstance(world, torch.Tensor):
        world_np = world.numpy()
    elif isinstance(world, np.ndarray):
        world_np = world
    else:
        world_np = np.array(world)
    
    # Convertește la mathutils Matrix
    blender_mat = mathutils.Matrix(world_np)
    cam_obj.matrix_world = blender_mat
    cam_obj.rotation_mode = 'ZYX'
    bpy.context.scene.camera = cam_obj

    # 4. Setări Optică
    cam_data.lens = focus
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = sensor_width_mm
   
    bpy.context.scene.render.resolution_x = img_w
    bpy.context.scene.render.resolution_y = img_h
    bpy.context.scene.render.pixel_aspect_x = 1
    bpy.context.scene.render.pixel_aspect_y = 1
    
    # 5. SALVARE FIȘIER .BLEND
    blend_file_path = f"./blends/{Path(image_path).name}_render.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
    print(f"Fișierul proiectului a fost generat: {blend_file_path}")

    # 6. COMPUTE OPENCV CAMERA MATRIX
    cam = cam_data
    scene = bpy.context.scene
    
    # Validări
    assert scene.render.resolution_percentage == 100, "Resolution must be at 100%"
    assert cam.sensor_fit != 'VERTICAL', "Camera must use HORIZONTAL field of view"
    
    # Extragere parametri
    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    
    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    
    # Calcul focal length în pixeli
    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect
    
    # Principal point
    c_x = w * (0.5 - cam.shift_x)
    c_y = h * 0.5 + w * cam.shift_y
    
    # Camera matrix K
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0,   0,   1]], dtype=np.float32)
    
    print(f"Camera Matrix K:")
    print(K)
    print(f"Focal length: {focus:.2f}mm | Sensor: {sensor_width_mm}mm")
    print(f"f_x={f_x:.2f}, f_y={f_y:.2f}, c_x={c_x:.2f}, c_y={c_y:.2f}")
    print("Camera world matrix:", cam_obj.matrix_world)

    # 7. Randare efectivă imagine
    bpy.context.scene.render.filepath = f"./results/{Path(image_path).name}_render.png"

    bpy.ops.render.render(write_still=True)
    
    return K

def get_indexed_dots(image_path):
    # 1. Load and detect red dots
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range for red color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            dots.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    
    dots = np.array(dots)
    
    # 2. Indexing Clockwise
    center = dots.mean(axis=0)
    angles = np.arctan2(dots[:, 1] - center[1], dots[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    return torch.tensor(dots[sorted_indices], dtype=torch.float32)

def get_rotation_matrix_differentiable(angles):
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cr, sr = torch.cos(roll), torch.sin(roll)

    Rx = torch.stack([
        torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0)]),
        torch.stack([torch.tensor(0.0), cp, -sp]),
        torch.stack([torch.tensor(0.0), sp, cp])
    ])

    Ry = torch.stack([
        torch.stack([cy, torch.tensor(0.0), sy]),
        torch.stack([torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)]),
        torch.stack([-sy, torch.tensor(0.0), cy])
    ])

    Rz = torch.stack([
        torch.stack([cr, -sr, torch.tensor(0.0)]),
        torch.stack([sr, cr, torch.tensor(0.0)]),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)])
    ])

    return Rz @ Ry @ Rx

def optimize_camera_pose_mm(IDP, real_w, real_h, img_w, img_h, sensor_width_mm=36.0):
    half_w, half_h = real_w / 2, real_h / 2
    rect_3d = torch.tensor([
        [-half_w, -half_h, 0.0],
        [ half_w, -half_h, 0.0],
        [ half_w,  half_h, 0.0],
        [-half_w,  half_h, 0.0]
    ], dtype=torch.float32)

    # 1. Inițializăm parametrii
    cam_pos = nn.Parameter(torch.tensor([0.0, 0.0, -20.0]))
    cam_rot = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
    
    focal_mm_param = nn.Parameter(torch.tensor(32.0))
    
    optimizer = torch.optim.Adam([
        {'params': [cam_pos, cam_rot], 'lr': 0.01},
        {'params': [focal_mm_param], 'lr': 0.01} 
    ])

    for i in range(6000):
        optimizer.zero_grad()
        
        f_mm = torch.abs(focal_mm_param)
        
        # --- CONVERSIE MM -> PIXELI ---
        f_px = (f_mm * img_w) / sensor_width_mm
        
        R = get_rotation_matrix_differentiable(cam_rot)
        relative_points = (rect_3d - cam_pos) @ R.T
        
        z_coords = relative_points[:, 2]
        safe_z = torch.where(z_coords > 0.1, z_coords, torch.ones_like(z_coords) * 0.1)
        
        # Proiecție folosind f_px calculat din f_mm
        pdp_x = f_px * (relative_points[:, 0] / safe_z) + (img_w / 2)
        pdp_y = f_px * (relative_points[:, 1] / safe_z) + (img_h / 2)
        PDP = torch.stack([pdp_x, pdp_y], dim=1)

        # Calcul Loss
        all_rot_losses = []
        for s in range(4):
            shifted = torch.roll(PDP, shifts=s, dims=0)
            all_rot_losses.append(torch.sum(torch.norm(shifted - IDP, dim=1)))
        
        loss = torch.min(torch.stack(all_rot_losses))
        penalty = 0#torch.sum(torch.relu(0.5 - z_coords)) * 100
        
        total_loss = loss + penalty
        total_loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Iter {i} | Loss: {total_loss.item():.2f}")
            print(f"Lens: {f_mm.item():.2f}mm, Focus px: {f_px.item():.2f} | Pos Z: {cam_pos[2].item():.2f}")

    with torch.no_grad():
        R = get_rotation_matrix_differentiable(cam_rot)
        world_matrix = torch.eye(4)
        world_matrix[:3, :3] = R
        world_matrix[:3, 3] = cam_pos
    return cam_pos.detach(), cam_rot.detach(), focal_mm_param.detach(), world_matrix.detach(), total_loss.detach()

def get_wh(image_path: str):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    return (w, h)

def render_verification_mm(image_path, cam_pos, cam_rot, real_w, real_h, focus_mm, sensor_width_mm=36.0):
    img = cv2.imread(image_path)
    if img is None: 
        print("Eroare: Nu am găsit imaginea.")
        return
    h, w = img.shape[:2]
    
    # --- CONVERSIE MM -> PIXELI PENTRU RENDER ---
    focal_px = (focus_mm * w) / sensor_width_mm

    # Definirea colțurilor în 3D
    half_w, half_h = real_w / 2, real_h / 2
    rect_3d = torch.tensor([
        [-half_w, -half_h, 0], [half_w, -half_h, 0],
        [half_w, half_h, 0], [-half_w, half_h, 0]
    ], dtype=torch.float32)

    # Transformare în Camera Space
    R = get_rotation_matrix_differentiable(cam_rot)
    points_cam = (rect_3d - cam_pos) @ R.T

    pts_2d = []
    all_visible = True
    
    for p in points_cam:
        # Near clipping (z > 0.1)
        if p[2] <= 0.1:
            all_visible = False
            break
            
        # Proiecție folosind focal_px calculat din mm
        px = int(focal_px * (p[0] / p[2]) + (w / 2))
        py = int(focal_px * (p[1] / p[2]) + (h / 2))
        pts_2d.append([px, py])

    # Desenare
    if all_visible and len(pts_2d) == 4:
        pts_2d_np = np.array(pts_2d, np.int32)
        cv2.polylines(img, [pts_2d_np], isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Overlay info text
        info_text = f"Lens: {focus_mm:.2f}mm | Sensor: {sensor_width_mm}mm"
        cv2.putText(img, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for i, pt in enumerate(pts_2d):
            cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(img, "OBIECTUL E IN SPATELE CAMEREI", (50, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imwrite(f"./results/{Path(image_path).name}.png", img)
import math
def convert_to_blender_matrix(cam_pos, cam_rot):
    """
    Convertește camera pose din sistemul tău în sistemul Blender.
    
    Sistemul tău: 
      - Camera se uită spre +Z
      - Dreptunghiul la Z=0
      - Axele: X dreapta, Y sus
    
    Sistemul Blender:
      - Camera se uită spre -Z
      - Dreptunghiul la Z=0
      - Axele: X dreapta, Y sus
    """
    # 1. Extrage rotația curentă
    cam_rot = -cam_rot
    cam_rot[0] = -cam_rot[0]
    cam_rot[1] = -cam_rot[1]
    
    R = get_rotation_matrix_differentiable(cam_rot).numpy()
    
    # 2. În sistemul tău: camera se uită spre +Z, așa că o rotim 180° în jurul axei Y
    # pentru a se uita spre -Z (cum așteaptă Blender)
    R_y_180 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    
    # 3. Aplică rotația pentru Blender
    R_blender = R_y_180 @ R
    
    # 4. Transformă poziția camerei
    # În sistemul tău: camera este la pozitia cam_pos și se uită spre +Z
    # În Blender: camera trebuie să fie la aceeași poziție relativă,
    # dar orientată spre -Z, deci aplicăm aceeași transformare
    pos_np = cam_pos.numpy().copy()
    pos_np[2] = -pos_np[2]
    
    # 5. Creează matricea 4x4 pentru Blender
    blender_mat = np.eye(4)
    blender_mat[:3, :3] = R_blender
    blender_mat[:3, 3] = pos_np
    
    # 6. Blender folosește sistemul de coordonate cu axa Z în sus,
    # deci mai trebuie să facem o rotație
    # OpenCV/your system: X-right, Y-down, Z-forward
    # Blender: X-right, Y-up, Z-backward
    
    R_cv_to_blender = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    
    # Aplică transformarea sistemului de coordonate
    blender_mat[:3, :3] = R_cv_to_blender @ blender_mat[:3, :3]
    
    return mathutils.Matrix(blender_mat)



def convert_to_blender_matrix2(cam_pos, cam_rot):
    """
    Convertește camera pose din sistemul tău în sistemul Blender.
    
    Sistemul tău: 
      - Camera se uită spre +Z
      - Axele: X dreapta, Y sus (sau jos?), Z în față
      - Rotunghiul la Z=0
    
    Sistemul Blender:
      - Camera se uită spre -Z
      - Axele: X dreapta, Y sus, Z în sus
    """
    
    # 1. Extrage rotația din sistemul tău
    R_your_system = get_rotation_matrix_differentiable(cam_rot).numpy()
    
    # 2. Transformarea sistemului de coordonate:
    # Sistemul tău (presupunând Y în sus): X->X, Y->Y, Z->Z
    # Sistemul Blender: X->X, Y->-Z, Z->Y
    # Deci camera ta care se uită spre +Z înseamnă că se uită spre -Y în Blender
    
    # Matrice de transformare între sisteme
    # Dacă camera ta are axa Y în sus:
    # T = [[1, 0, 0],  # X -> X
    #      [0, 0, 1],  # Y -> Z (sus în Blender)
    #      [0, 1, 0]]  # Z -> Y (dar cu semn schimbat?)
    
    # Mai bine: fă-o pas cu pas
    
    # Pas 1: Camera ta se uită spre +Z, Blender se uită spre -Y
    # Deci rotim 90° în jurul axei X pentru a transforma Z->-Y
    R_x_90 = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Pas 2: Apoi mai rotim 180° în jurul noii axe Y (care era Z vechi)
    # pentru a transforma orientarea finală
    R_y_180 = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    
    # Combinația completă de rotație
    R_system_transform = R_y_180 @ R_x_90
    
    # Aplică transformarea sistemului la rotația camerei
    R_blender = R_system_transform @ R_your_system
    
    # Transformă poziția conform aceluiași sistem
    pos_np = cam_pos.numpy().copy()
    
    # Aplică aceeași transformare și poziției
    # Poziția ta: (x, y, z) -> Blender: (x, z, -y) sau similar
    # Să verificăm cu exemple:
    pos_blender = pos_np
    
    # Sau mai simplu, după transformarea sistemului:
    # x_blender = x_tau
    # y_blender = z_tau (pentru că Z devine Y în Blender)
    # z_blender = -y_tau (pentru că Y devine -Z în Blender)

    
    # 3. Creează matricea 4x4 pentru Blender
    blender_mat = np.eye(4)
    blender_mat[:3, :3] = R_blender
    blender_mat[:3, 3] = pos_blender
    
    return mathutils.Matrix(blender_mat)

def convert_to_blender_matrix3(cam_pos, cam_rot):
    """
    Convertește camera pose din sistemul tău în sistemul Blender.
    
    Sistemul tău: 
    - Camera se uită spre +Z
    - X dreapta, Y sus, Z înainte
    
    Sistemul Blender:
    - Camera se uită spre -Z local
    - X dreapta, Y sus, Z înapoi (în spațiul camerei)
    """
    
    # 1. Construiește matricea de rotație în sistemul tău
    # Presupunând că get_rotation_matrix_differentiable folosește convenția Euler XYZ

    cam_rot[0] = -cam_rot[0]
    cam_rot[1] = -cam_rot[1]
    cam_rot[2] = -cam_rot[2]
    R = get_rotation_matrix_differentiable(cam_rot).numpy()
    
    # 2. Poziția în Blender
    # Transformare din sistemul tău (Y sus, Z înainte) în Blender (Y sus, Z înapoi)
    pos_np = cam_pos.numpy().copy()
    pos_blender = np.array([pos_np[0], pos_np[1], pos_np[2]])
    
    # 3. Rotație: transformă din sistemul tău în Blender
    # Când camera ta se uită spre +Z, în Blender trebuie să se uite spre -Z
    # Aceasta înseamnă o rotație de 180° în jurul axei X (nu Y!)
    # pentru a inversa direcția Z menținând Y sus
    
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    # Aplică transformarea: mai întâi rotația camerei, apoi flip-ul
    R_blender = R_flip @ R
    
    # 4. Construiește matricea 4x4
    blender_mat = np.eye(4)
    blender_mat[:3, :3] = R_blender
    blender_mat[:3, 3] = pos_blender
    
    return mathutils.Matrix(blender_mat)
# # Example Usage:
def make_poses(name: str):
    idp = get_indexed_dots(name)
    w, h = get_wh(name)


    real_w1, real_h1 = (2 * 6.3, 2 * 4.29)
    pos1, rot1, focus1, world1, total_loss1 = optimize_camera_pose_mm(idp, real_w1, real_h1, w, h)

    real_h2, real_w2 = (2 * 6.3, 2 * 4.29)
    pos2, rot2, focus2, world2, total_loss2 = optimize_camera_pose_mm(idp, real_w2, real_h2, w, h)


    if (total_loss1 < total_loss2):
        pos, rot, focus, world, total_loss = pos1, rot1, focus1, world1, total_loss1 
        real_w, real_h = real_w1, real_h1
    else:
        pos, rot, focus, world, total_loss =  pos2, rot2, focus2, world2, total_loss2  
        real_w, real_h = real_w2, real_h2
    print(f"Optimizare finală:")
    print(f"Position: {pos}")
    print(f"Rotation (radians): {rot}")
    print(f"Focal length (mm): {focus.item():.2f}")

# Convert la grade pentru ușurință de înțeles
    import math
    rot_deg = [math.degrees(r) for r in rot]
    print(f"Rotation (degrees): {rot_deg}")

# Verificare cu randare OpenCV
    render_verification_mm(name, pos, rot, real_w, real_h, focus.item())

# Conversie la Blender
    blender_world = convert_to_blender_matrix3(pos, rot)

# Verificare cu randare Blender
    render_verification_blender(name, real_w, real_h, focus.item(), blender_world)

from pathlib import Path
def process_images_glob(folder_path: str, pattern: str = "*.jpeg"):
    """
    Folosește glob pattern pentru a selecta imaginile.
    
    Exemple de pattern:
    - "*.jpg" - doar JPG
    - "*.png" - doar PNG
    - "*.*" - toate fișierele
    - "img_*.jpg" - doar imaginile care încep cu "img_"
    """
    folder = Path(folder_path)
    
    for image_path in sorted(folder.glob(pattern)):
        name = image_path.stem
        print(f"Procesez: {name}")
        make_poses(str(image_path.resolve()))


process_images_glob("./poses/")
