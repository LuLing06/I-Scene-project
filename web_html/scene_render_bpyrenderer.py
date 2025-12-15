import sys
import os

# Add user site-packages to path (for packages installed via Blender's pip)
user_site = os.path.expanduser("~/.local/lib/python3.11/site-packages")
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import json
from glob import glob

import imageio
import numpy as np

import bpy
from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.environment import set_background_color
from bpyrenderer.importer import load_file
from bpyrenderer.render_output import enable_color_output

# -------- CONFIG ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get video folder from command line: blender --python script.py -- video2
VIDEO_FOLDER = "video1"  # default
if "--" in sys.argv:
    args_after = sys.argv[sys.argv.index("--") + 1:]
    if args_after:
        VIDEO_FOLDER = args_after[0]

INPUT_DIR = os.path.join(SCRIPT_DIR, VIDEO_FOLDER)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, VIDEO_FOLDER, "bpyrenderer_output")
TEMP_RENDER_DIR = os.path.join(OUTPUT_DIR, "temp_frames")
ROTATION_CONFIG_FILE = os.path.join(SCRIPT_DIR, f"rotation_config_{VIDEO_FOLDER}.json")

WIDTH, HEIGHT = 1024, 1024

NUM_FRAMES = 120  # frames for 360° rotation
ELEVATION = 15    # camera elevation angle in degrees
FPS = 24
CAMERA_RADIUS = 1.8  # Distance from center (1.5 = close, 2.0 = far, gives more "padding")
# --------------------------


def load_rotation_config():
    """Load per-model rotation offsets from config file."""
    if os.path.exists(ROTATION_CONFIG_FILE):
        with open(ROTATION_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_model_name(model_path):
    """Extract clean name from GLB file path."""
    basename = os.path.basename(model_path)
    name = os.path.splitext(basename)[0]
    # Clean up problematic characters
    name = name.replace(" ", "_").replace("(", "").replace(")", "")
    return name


def setup_vertex_color_materials():
    """Create materials that use vertex colors for objects that have color attributes."""
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        
        mesh = obj.data
        
        # Check if mesh has color attributes (vertex colors)
        if not mesh.color_attributes:
            continue
        
        color_attr = mesh.color_attributes.active_color
        if not color_attr:
            continue
        
        # Create a new material that uses vertex colors
        mat_name = f"VertexColor_{obj.name}"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create nodes: Color Attribute -> Principled BSDF -> Material Output
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (300, 0)
        
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.location = (0, 0)
        
        color_attr_node = nodes.new(type='ShaderNodeVertexColor')
        color_attr_node.location = (-300, 0)
        color_attr_node.layer_name = color_attr.name
        
        # Link nodes
        links.new(color_attr_node.outputs['Color'], bsdf_node.inputs['Base Color'])
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # Assign material to object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def render_single_model(model_path, output_dir, rotation_config):
    """Render a single GLB model and output rgb/mask videos + metadata."""
    
    model_name = get_model_name(model_path)
    
    # Get per-model rotation offset (default 0)
    azimuth_offset = rotation_config.get(model_name, 0)
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # Create temp directory for frames
    temp_dir = os.path.join(output_dir, f"temp_{model_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. Init engine and scene manager
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)
    
    # 2. Import model
    load_file(model_path)
    
    # 2b. Setup vertex color materials (for models with vertex colors like PartCrafter)
    setup_vertex_color_materials()
    
    # 3. Normalize scene
    scene_manager.normalize_scene(1.0)
    
    # 4. Set environment (white background)
    set_background_color([1.0, 1.0, 1.0, 1.0])
    
    # 4b. Add lighting for proper material colors
    # Add a sun light
    sun_data = bpy.data.lights.new(name="Sun", type='SUN')
    sun_data.energy = 3.0
    sun_obj = bpy.data.objects.new(name="Sun", object_data=sun_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.8, 0.2, 0.5)  # Angle the sun
    
    # Add ambient light via world settings
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Strength"].default_value = 0.5  # Ambient light strength
    
    # 5. Prepare cameras on sphere (with per-model rotation offset)
    # Use the user's selected angle directly as the starting azimuth
    total_azimuth_offset = azimuth_offset
    print(f"  Using azimuth offset: {azimuth_offset}°")
    
    cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=CAMERA_RADIUS,
        elevations=[ELEVATION],
        num_camera_per_layer=NUM_FRAMES,
        azimuth_offset=total_azimuth_offset,
    )
    
    cameras = []
    for i, camera_mat in enumerate(cam_mats):
        camera = add_camera(camera_mat, add_frame=i < len(cam_mats) - 1)
        cameras.append(camera)
    
    # 6. Set render outputs
    enable_color_output(
        WIDTH,
        HEIGHT,
        temp_dir,
        mode="PNG",
        film_transparent=True,
    )
    
    # 7. Render all frames
    scene_manager.render()
    
    # 8. Convert rendered PNGs to RGB video
    render_files = sorted(glob(os.path.join(temp_dir, "render_*.png")))
    if render_files:
        rgb_video_path = os.path.join(output_dir, f"{model_name}_rgb.mp4")
        
        with imageio.get_writer(rgb_video_path, fps=FPS) as rgb_writer:
            for file in render_files:
                # Read RGBA image
                image = imageio.imread(file)
                
                # Composite onto white background
                white_bg = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
                alpha = image[:, :, 3:4] / 255.0
                rgb_image = image[:, :, :3] * alpha + white_bg * (1 - alpha)
                
                rgb_writer.append_data(rgb_image.astype(np.uint8))
                
                # Remove intermediate PNG
                os.remove(file)
        
        # Remove temp directory
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        print(f"  RGB video: {rgb_video_path}")
    
    # 9. Save camera metadata
    meta_info = {"width": WIDTH, "height": HEIGHT, "model": model_name, "locations": []}
    for i in range(len(cam_pos)):
        index = "{0:04d}".format(i)
        meta_info["locations"].append({
            "index": index,
            "projection_type": cameras[i].data.type,
            "ortho_scale": cameras[i].data.ortho_scale,
            "camera_angle_x": cameras[i].data.angle_x,
            "elevation": elevations[i],
            "azimuth": azimuths[i],
            "transform_matrix": cam_mats[i].tolist(),
        })
    
    meta_path = os.path.join(output_dir, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    print(f"  Metadata: {meta_path}")
    
    return model_name


# -------- MAIN ----------
if __name__ == "__main__":
    # Find all GLB files
    glb_files = glob(os.path.join(INPUT_DIR, "*.glb"))
    
    if not glb_files:
        print(f"No GLB files found in {INPUT_DIR}")
        sys.exit(1)
    
    print(f"Found {len(glb_files)} GLB files:")
    for f in glb_files:
        print(f"  - {os.path.basename(f)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load rotation config
    rotation_config = load_rotation_config()
    if rotation_config:
        print(f"\nLoaded rotation config from: {ROTATION_CONFIG_FILE}")
        for name, angle in rotation_config.items():
            print(f"  {name}: {angle}°")
    else:
        print(f"\nNo rotation config found. Using default angles.")
        print(f"  (Create {ROTATION_CONFIG_FILE} or use preview_angles.py to set offsets)")
    
    # Process each model
    processed = []
    for model_path in sorted(glb_files):
        try:
            name = render_single_model(model_path, OUTPUT_DIR, rotation_config)
            processed.append(name)
        except Exception as e:
            print(f"ERROR processing {model_path}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {len(processed)}/{len(glb_files)} models.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")
