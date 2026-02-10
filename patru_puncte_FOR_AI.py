import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import bpy
except Exception:
    bpy = None

def render_verification_blender(image_path, cam_pos, cam_rot, real_w, real_h, focus, fp, sensor_width_mm=36.0):
    if bpy is None:
        raise RuntimeError("Blender (bpy) is required to run render_verification_blender")

    # 1. Reset scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    focal_px = float(focus.item())
    # `focus` is expected in millimeters already (user-provided).
    # Preluăm dimensiunile imaginii (necesar pentru aspect ratio)
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = img_cv.shape[:2]

    # `focus` is already in millimeters; use it directly as lens focal length
    focal_mm = focal_px

    # 2. Creăm Dreptunghiul (Plane)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    rect = bpy.context.active_object
    # Blender default units are meters. If `real_w`/`real_h` are given in mm,
    # convert to meters. Heuristic: if dimensions > 10 assume mm units.
    if real_w > 10 or real_h > 10:
        rw = real_w / 1000.0
        rh = real_h / 1000.0
    else:
        rw = real_w
        rh = real_h
    rect.scale = (rw / 2, rh / 2, 1)
    
    # Îi dăm un material care emite lumină (să-l vedem clar fără lămpi)
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
    bpy.context.scene.camera = cam_obj

    # 4. Poziție și Rotație (Transformare din sistemul tău în cel Blender)
    # Set camera location (convert mm->m if needed)
    cx = float(cam_pos[0])
    cy = float(cam_pos[1])
    cz = float(cam_pos[2])
    if max(abs(cx), abs(cy), abs(cz)) > 10:
        cx /= 1000.0
        cy /= 1000.0
        cz /= 1000.0

    cam_obj.location = (cx, cy, cz)
    cam_obj.rotation_mode = 'XYZ'

    # Use the provided `cam_rot` exactly; do not re-orient camera when position changes.
    try:
        rx = float(cam_rot[0])
        ry = float(cam_rot[1])
        rz = float(cam_rot[2])
        cam_obj.rotation_euler = (rx, ry, rz)
    except Exception:
        try:
            # fallback for tensor-like values
            cam_obj.rotation_euler = (cam_rot[0].item(), cam_rot[1].item(), cam_rot[2].item())
        except Exception:
            # leave default rotation if we cannot read cam_rot
            pass

    # 5. Setări Optică & Randare
    # IMPORTANT: cam_data.lens MUST be in millimeters, not pixels!
    cam_data.lens = focal_mm
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = sensor_width_mm
    
    # Ensure camera clipping covers the plane
    try:
        cam_data.clip_start = 0.01
        cam_data.clip_end = 1000.0
    except Exception:
        pass

    # Make emission clearly visible without additional lights
    try:
        node_emit.inputs['Strength'].default_value = 10.0
    except Exception:
        pass

    # Prefer a real-time engine that supports emission by default
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    except Exception:
        pass

    bpy.context.scene.render.resolution_x = img_w
    bpy.context.scene.render.resolution_y = img_h
    bpy.context.scene.render.pixel_aspect_x = 1
    bpy.context.scene.render.pixel_aspect_y = 1
    
    # 6. Save .blend (for inspection)
    blend_file_path = "rezultat_optimizare.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
    print(f"Fișierul proiectului a fost generat: {blend_file_path}")

    # 7. COMPUTE OPENCV CAMERA MATRIX (From Blender to OpenCV)
    # Use cam_data directly instead of looking it up by name
    cam = cam_data
    scene = bpy.context.scene

    # Ensure render settings are as expected
    scene.render.resolution_percentage = 100
    if cam.sensor_fit == 'VERTICAL':
        cam.sensor_fit = 'HORIZONTAL'

    # Extract params
    f_in_mm = float(cam.lens)
    sensor_width_in_mm = float(cam.sensor_width)
    w = int(scene.render.resolution_x)
    h = int(scene.render.resolution_y)

    # Pixel aspect correction (Blender stores pixel_aspect_x/y)
    pixel_aspect = float(scene.render.pixel_aspect_y) / float(scene.render.pixel_aspect_x)

    # Focal length in pixels
    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # Principal point: Blender's shift values are in normalized sensor coordinates
    c_x = w * (0.5 - float(cam.shift_x))
    # corrected formula: center y uses image height and shift_y in normalized units
    c_y = h * (0.5 + float(cam.shift_y))
    
    # Camera matrix K
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0,   0,   1]], dtype=np.float32)
    
    print(f"Camera Matrix K:")
    print(K)
    print(f"Focal length: {focal_mm:.2f}mm ({focal_px:.2f}px) | Sensor: {sensor_width_mm}mm")
    print(f"f_x={f_x:.2f}, f_y={f_y:.2f}, c_x={c_x:.2f}, c_y={c_y:.2f}")

    # 8. Randare efectivă imagine
    bpy.context.scene.render.filepath = fp
    # Debug prints to help locate why render may be blank
    try:
        print("--- Blender Debug Info Before Render ---")
        print(f"Camera location: {cam_obj.location}")
        print(f"Camera rotation_euler: {cam_obj.rotation_euler}")
        print(f"Camera lens (mm): {cam_data.lens}")
        print(f"Camera clip start/end: {getattr(cam_data, 'clip_start', None)}/{getattr(cam_data, 'clip_end', None)}")
        print(f"Plane scale (m): {rect.scale}")
        print(f"Scene engine: {bpy.context.scene.render.engine}")
        print("----------------------------------------")
    except Exception:
        pass

    bpy.ops.render.render(write_still=True)
    
    return K

# Example Call:
# render_verification_blender(name, pos, rot, real_w, real_h, focus)

def get_indexed_dots(image_path):
    # 1. Load and detect red dots
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
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
    
    if len(dots) == 0:
        print(f"Warning: no red dots found in {image_path}")
        return torch.empty((0, 2), dtype=torch.float32)

    dots = np.array(dots)
    # 2. Indexing Clockwise
    center = dots.mean(axis=0)
    # Calculate angles from center (atan2 gives angle from positive x-axis)
    angles = np.arctan2(dots[:, 1] - center[1], dots[:, 0] - center[0])
    # Sort dots based on angles
    sorted_indices = np.argsort(angles)
    return torch.tensor(dots[sorted_indices], dtype=torch.float32)

def get_rotation_matrix_differentiable(angles):
    # angles este cam_rot care are requires_grad=True
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    
    # Cosinus și Sinus care păstrează gradienții
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cr, sr = torch.cos(roll), torch.sin(roll)
    # Build constants on same device/dtype as angles
    device = angles.device
    dtype = angles.dtype
    one = torch.ones((), device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)

    # Matricea de rotație X (Pitch)
    Rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cp, -sp]),
        torch.stack([zero, sp, cp])
    ])

    # Matricea de rotație Y (Yaw)
    Ry = torch.stack([
        torch.stack([cy, zero, sy]),
        torch.stack([zero, one, zero]),
        torch.stack([-sy, zero, cy])
    ])

    # Matricea de rotație Z (Roll)
    Rz = torch.stack([
        torch.stack([cr, -sr, zero]),
        torch.stack([sr, cr, zero]),
        torch.stack([zero, zero, one])
    ])

    # Rotația finală combinată (Z * Y * X)
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
    
    # Parametrul optimizabil este acum în milimetri (ex: plecăm de la 50mm)
    focal_mm_param = nn.Parameter(torch.tensor(32.0))
    
    # Adam optimizer - am redus LR pentru focal_mm deoarece variațiile în mm 
    # au impact mult mai mare în pixeli decât variațiile directe de pixeli
    optimizer = torch.optim.Adam([
        {'params': [cam_pos, cam_rot], 'lr': 0.01},
        {'params': [focal_mm_param], 'lr': 0.01} 
    ])

    for i in range(6000):
        optimizer.zero_grad()
        
        # Constrângem mm să rămână pozitivi
        f_mm = torch.abs(focal_mm_param)
        
        # --- CONVERSIE MM -> PIXELI ---
        # Formula: f_px = (f_mm * imagine_w_px) / senzor_w_mm
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
        penalty = torch.sum(torch.relu(0.5 - z_coords)) * 100
        
        total_loss = loss + penalty
        total_loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Iter {i} | Loss: {total_loss.item():.2f}")
            # Afișăm valoarea în mm pentru a vedea cum evoluează "lentila"
            print(f"Lens: {f_mm.item():.2f}mm, Focus px: {f_px.item():.2f} | Pos Z: {cam_pos[2].item():.2f}")

    return cam_pos.detach(), cam_rot.detach(), focal_mm_param.detach()


def optimize_camera_pose(IDP, real_w, real_h, img_w, img_h, sensor_width_mm=36.0):
    half_w, half_h = real_w / 2, real_h / 2
    rect_3d = torch.tensor([
        [-half_w, -half_h, 0.0],
        [ half_w, -half_h, 0.0],
        [ half_w,  half_h, 0.0],
        [-half_w,  half_h, 0.0]
    ], dtype=torch.float32)

    # 1. Inițializăm parametrii (Adăugăm focal_length ca parametru optimizabil)
    cam_pos = nn.Parameter(torch.tensor([0.0, 0.0, -20.0]))
    cam_rot = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
    
    # Pornim de la o estimare rezonabilă (ex: lățimea imaginii)
    focal_param = nn.Parameter(torch.tensor(1000.0))
    
    # Putem folosi learning rates diferite: focal length are nevoie de pași mai mari
    optimizer = torch.optim.Adam([
        {'params': [cam_pos, cam_rot], 'lr': 0.01},
        {'params': [focal_param], 'lr': 0.5} 
    ])

    for i in range(6000):
        optimizer.zero_grad()
        
        # Constrângem focal_length să rămână pozitiv
        # f_val = torch.abs(focal_param) 
        f_val_old = torch.abs(focal_param) 
        f_val = f_val_old
        
        R = get_rotation_matrix_differentiable(cam_rot)
        relative_points = (rect_3d - cam_pos) @ R.T
        
        z_coords = relative_points[:, 2]
        safe_z = torch.where(z_coords > 0.1, z_coords, torch.ones_like(z_coords) * 0.1)
        
        # Folosim f_val cel nou în proiecție
        pdp_x = f_val * (relative_points[:, 0] / safe_z) + (img_w / 2)
        pdp_y = f_val * (relative_points[:, 1] / safe_z) + (img_h / 2)
        PDP = torch.stack([pdp_x, pdp_y], dim=1)

        # Calcul Loss (Minimul celor 4 rotații)
        all_rot_losses = []
        for s in range(4):
            shifted = torch.roll(PDP, shifts=s, dims=0)
            all_rot_losses.append(torch.sum(torch.norm(shifted - IDP, dim=1)))
        
        loss = torch.min(torch.stack(all_rot_losses))
        
        # Penalizare pentru Z negativ
        penalty = torch.sum(torch.relu(0.5 - z_coords)) * 100
        
        total_loss = loss + penalty

        total_loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Iter {i} | Loss: {total_loss.item():.2f}")
            print(f"Focal: {f_val.item():.2f} | Pos Z: {cam_pos[2].item():.2f}")

    return cam_pos.detach(), cam_rot.detach(), focal_param.detach()


def get_wh(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]

    return (w, h)

import cv2
import numpy as np
import torch
import torch.nn as nn


def render_verification_not_behind(image_path, cam_pos, cam_rot, real_w, real_h, focus):
    img = cv2.imread(image_path)
    if img is None: return
    h, w = img.shape[:2]
    focal_length = focus.item()

    half_w, half_h = real_w / 2, real_h / 2
    rect_3d = torch.tensor([
        [-half_w, -half_h, 0], [half_w, -half_h, 0],
        [half_w, half_h, 0], [-half_w, half_h, 0]
    ], dtype=torch.float32)

    R = get_rotation_matrix_differentiable(cam_rot)
    points_cam = (rect_3d - cam_pos) @ R.T

    pts_2d = []
    all_visible = True
    
    for p in points_cam:
        # Verificăm dacă punctul este în fața camerei (z > epsilon)
        if p[2] <= 0.1:
            all_visible = False
            break
            
        px = int(focal_length * (p[0] / p[2]) + (w / 2))
        py = int(focal_length * (p[1] / p[2]) + (h / 2))
        pts_2d.append([px, py])

    # Desenăm doar dacă toate cele 4 colțuri sunt în fața camerei
    if all_visible and len(pts_2d) == 4:
        pts_2d_np = np.array(pts_2d, np.int32)
        cv2.polylines(img, [pts_2d_np], isClosed=True, color=(0, 255, 0), thickness=3)
        
        for i, pt in enumerate(pts_2d):
            cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(img, "OBIECTUL E IN SPATELE CAMEREI", (50, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Verification", img)
    cv2.waitKey(0)

def render_verification(image_path, cam_pos, cam_rot, real_w, real_h, focus):

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    focal_length = focus.item()

    # Define the 4 corners in 3D
    half_w, half_h = real_w / 2, real_h / 2
    rect_3d = torch.tensor([
        [-half_w, -half_h, 0], [half_w, -half_h, 0],
        [half_w, half_h, 0], [-half_w, half_h, 0]
    ], dtype=torch.float32)

    # Transform points to Camera Space
    R = get_rotation_matrix_differentiable(cam_rot)
    # Project: P_cam = R * (P_world - Cam_Pos)
    points_cam = (rect_3d - cam_pos) @ R.T

    # Project to 2D
    pts_2d = []
    for p in points_cam:
        px = int(focal_length * (p[0] / p[2]) + (w / 2))
        py = int(focal_length * (p[1] / p[2]) + (h / 2))
        pts_2d.append([px, py])

    # Draw the verification lines
    pts_2d = np.array(pts_2d, np.int32)
    cv2.polylines(img, [pts_2d], isClosed=True, color=(0, 255, 0), thickness=3)
    
    # Label the indices
    for i, pt in enumerate(pts_2d):
        cv2.putText(img, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Verification - Green is Virtual POV", img)
    cv2.waitKey(0)



def render_verification_mm(image_path, cam_pos, cam_rot, real_w, real_h, focus_mm, sensor_width_mm=36.0):
    img = cv2.imread(image_path)
    if img is None: 
        print("Eroare: Nu am găsit imaginea.")
        return
    h, w = img.shape[:2]
    
    # --- CONVERSIE MM -> PIXELI PENTRU RENDER ---
    # f_px = (f_mm * img_w_px) / sensor_w_mm
    focal_px = (focus_mm.item() * w) / sensor_width_mm

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
        info_text = f"Lens: {focus_mm.item():.2f}mm | Sensor: {sensor_width_mm}mm"
        cv2.putText(img, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        for i, pt in enumerate(pts_2d):
            cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(img, "OBIECTUL E IN SPATELE CAMEREI", (50, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imwrite("./verificare_put.png", img)
    # Show image to user as well
    try:
        cv2.imshow("Verification - MM Mode", img)
        cv2.waitKey(0)
    except Exception:
        # If GUI not available, just save the file (already done)
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    #
    #
    #




# Example Usage:
name = "./img.jpeg"
idp = get_indexed_dots(name)
if idp.numel() == 0:
    print("No dots detected — aborting optimization and visualization.")
    sys.exit(1)

w, h = get_wh(name)
real_w, real_h = (2 * 6.3, 2 * 4.29)
pos, rot, focus = optimize_camera_pose_mm(idp, real_w, real_h, w, h)
print(pos, rot, "Lens: ", focus)

import math

def get_blender_rot(rot):
    x, y, z = rot[0], rot[1], rot[2]
    return torch.tensor([0,0,0])

def get_blender_pos(pos):
    x, y, z = pos[0], pos[1], pos[2]
    return torch.tensor([  0.45,   0.0855, -17.4384])


render_verification_mm(name, pos, rot, real_w, real_h, focus)
render_verification_blender(name, get_blender_pos(pos), get_blender_rot(rot), real_w, real_h, focus, fp="./verificare_vizuala3.png")



print("INITIAL DOTS: ", idp)

print("MULTIPLIED DOTS: ")
