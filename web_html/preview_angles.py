"""
Preview tool: Renders a GLB at multiple angles so you can pick the best starting angle.
Saves your choice to a config file for use with scene_render_bpyrenderer.py

Usage:
  blender --background --python preview_angles.py -- <glb_file>

Example:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python preview_angles.py -- video1/PartCrafter-latest.glb
"""

import sys
import os

# Add user site-packages
user_site = os.path.expanduser("~/.local/lib/python3.11/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import json
import bpy
import math
from mathutils import Vector

# -------- CONFIG ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREVIEW_SIZE = 512
NUM_ANGLES = 12  # Preview at 0°, 30°, 60°, ... 330°
# --------------------------


def get_config_file(video_folder):
    """Get config file path for a video folder."""
    return os.path.join(SCRIPT_DIR, f"rotation_config_{video_folder}.json")


def get_preview_dir(video_folder):
    """Get preview directory for a video folder."""
    return os.path.join(SCRIPT_DIR, video_folder, "angle_previews")


def load_config(config_file):
    """Load existing rotation config."""
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


def save_config(config, config_file):
    """Save rotation config."""
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_file}")


def get_model_name(model_path):
    """Extract clean name from GLB file path."""
    basename = os.path.basename(model_path)
    name = os.path.splitext(basename)[0]
    name = name.replace(" ", "_").replace("(", "").replace(")", "")
    return name


def setup_scene(model_path):
    """Setup scene with model, camera, and lighting."""
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import model
    bpy.ops.import_scene.gltf(filepath=model_path)
    
    # Calculate bounding box
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No meshes found")
    
    min_corner = Vector((1e9, 1e9, 1e9))
    max_corner = Vector((-1e9, -1e9, -1e9))
    
    for obj in mesh_objs:
        for v in obj.bound_box:
            w = obj.matrix_world @ Vector(v)
            min_corner.x = min(min_corner.x, w.x)
            min_corner.y = min(min_corner.y, w.y)
            min_corner.z = min(min_corner.z, w.z)
            max_corner.x = max(max_corner.x, w.x)
            max_corner.y = max(max_corner.y, w.y)
            max_corner.z = max(max_corner.z, w.z)
    
    center = (min_corner + max_corner) / 2.0
    radius = (max_corner - min_corner).length / 2.0
    
    # Add camera
    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_obj = bpy.data.objects.new("PreviewCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Add sun light
    sun_data = bpy.data.lights.new(name="Sun", type='SUN')
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.8, 0.2, 0.5)
    
    # Setup render settings
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = PREVIEW_SIZE
    bpy.context.scene.render.resolution_y = PREVIEW_SIZE
    bpy.context.scene.render.film_transparent = True
    
    # White background
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (1, 1, 1, 1)
        bg_node.inputs["Strength"].default_value = 0.5
    
    return center, radius, cam_obj


def render_at_angle(center, radius, cam_obj, angle_deg, output_path):
    """Render the scene from a specific angle."""
    angle_rad = math.radians(angle_deg)
    distance = radius * 2.5
    height = radius * 0.6
    
    x = center.x + distance * math.cos(angle_rad)
    y = center.y + distance * math.sin(angle_rad)
    z = center.z + height
    
    cam_obj.location = (x, y, z)
    
    # Point camera at center
    direction = center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def create_preview_grid(model_path, video_folder):
    """Create a grid of preview images at different angles."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("ERROR: numpy and PIL required. Install with:")
        print("  pip install numpy pillow")
        return None
    
    model_name = get_model_name(model_path)
    preview_dir = get_preview_dir(video_folder)
    os.makedirs(preview_dir, exist_ok=True)
    
    print(f"\nGenerating angle previews for: {model_name}")
    print("=" * 50)
    
    # Setup scene
    center, radius, cam_obj = setup_scene(model_path)
    
    # Render at each angle
    preview_images = []
    angles = [i * (360 // NUM_ANGLES) for i in range(NUM_ANGLES)]
    
    for angle in angles:
        output_path = os.path.join(preview_dir, f"{model_name}_{angle:03d}.png")
        print(f"  Rendering angle {angle}°...")
        render_at_angle(center, radius, cam_obj, angle, output_path)
        preview_images.append((angle, output_path))
    
    # Create grid image
    images = [Image.open(path) for _, path in preview_images]
    
    # 4x3 grid for 12 angles
    cols, rows = 4, 3
    grid_width = cols * PREVIEW_SIZE
    grid_height = rows * PREVIEW_SIZE
    grid = Image.new('RGBA', (grid_width, grid_height), (255, 255, 255, 255))
    
    from PIL import ImageDraw, ImageFont
    print("  Creating preview grid...")
    
    for i, (angle, img) in enumerate(zip(angles, images)):
        row = i // cols
        col = i % cols
        x = col * PREVIEW_SIZE
        y = row * PREVIEW_SIZE
        
        # Paste image
        grid.paste(img, (x, y))
        
        # Add angle label with larger text
        draw = ImageDraw.Draw(grid)
        label = f"{angle}°"
        
        # Try to use a larger font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
            except:
                font = ImageFont.load_default()
        
        # Larger label background
        draw.rectangle([x, y, x + 80, y + 50], fill=(0, 0, 0, 220))
        draw.text((x + 10, y + 8), label, fill=(255, 255, 0), font=font)  # Yellow text
    
    grid_path = os.path.join(preview_dir, f"{model_name}_grid.png")
    grid.save(grid_path)
    print(f"\n✓ Preview grid saved: {grid_path}")
    print(f"  Open this image to pick the best starting angle!")
    
    # Clean up individual images
    for _, path in preview_images:
        os.remove(path)
    
    return grid_path


def main():
    # Get GLB path from command line args (after --)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("Usage: blender --background --python preview_angles.py -- <glb_file>")
        print("\nOr to set an angle directly:")
        print("  blender --background --python preview_angles.py -- <glb_file> <angle>")
        print("\nExamples:")
        print("  blender --python preview_angles.py -- video2/model.glb")
        print("  blender --python preview_angles.py -- video2/model.glb 90")
        return
    
    if len(argv) < 1:
        print("ERROR: Please provide a GLB file path")
        return
    
    glb_path = argv[0]
    if not os.path.isabs(glb_path):
        glb_path = os.path.join(SCRIPT_DIR, glb_path)
    
    if not os.path.exists(glb_path):
        print(f"ERROR: File not found: {glb_path}")
        return
    
    # Extract video folder from path (e.g., "video2" from "video2/model.glb")
    rel_path = os.path.relpath(glb_path, SCRIPT_DIR)
    video_folder = rel_path.split(os.sep)[0]  # First directory component
    
    model_name = get_model_name(glb_path)
    config_file = get_config_file(video_folder)
    preview_dir = get_preview_dir(video_folder)
    
    # If angle is provided, save it directly
    if len(argv) >= 2:
        try:
            angle = int(argv[1])
            config = load_config(config_file)
            config[model_name] = angle
            save_config(config, config_file)
            print(f"✓ Set {model_name} starting angle to {angle}°")
            return
        except ValueError:
            pass
    
    # Generate preview grid
    create_preview_grid(glb_path, video_folder)
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print(f"1. Open: {preview_dir}/{model_name}_grid.png")
    print("2. Find the angle that shows the 'front' of your model")
    print("3. Run this command to save your choice:")
    print(f"   /Applications/Blender.app/Contents/MacOS/Blender --background --python preview_angles.py -- {argv[0]} <ANGLE>")
    print("\n   Example (if 90° looks best):")
    print(f"   /Applications/Blender.app/Contents/MacOS/Blender --background --python preview_angles.py -- {argv[0]} 90")


if __name__ == "__main__":
    main()

