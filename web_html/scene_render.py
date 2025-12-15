import bpy
import math
import os
from mathutils import Vector

# -------- CONFIG (change per GLB) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLB_PATH = os.path.join(SCRIPT_DIR, "video1/SceneGen-latest.glb")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "video1/scenegen.mp4")
FRAME_COUNT = 240    # 10s @24fps
FPS = 24

CAMERA_DISTANCE_FACTOR = 2.5
CAMERA_HEIGHT_FACTOR = 0.6
START_ANGLE = 0.0    # radians; 0 means starting on -Y axis
# ------------------------------------------

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    return bpy.context.scene

def compute_bbox_center_radius(scene):
    mesh_objs = [o for o in scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No meshes found.")

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
    return center, radius

def setup_orbit_camera(scene, center, radius):
    # target empty
    target = bpy.data.objects.new("TurntableTarget", None)
    scene.collection.objects.link(target)
    target.location = center

    # camera
    cam_data = bpy.data.cameras.new("TurntableCam")
    cam_obj = bpy.data.objects.new("TurntableCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    distance = radius * CAMERA_DISTANCE_FACTOR
    height   = radius * CAMERA_HEIGHT_FACTOR

    # frame / fps
    scene.frame_start = 1
    scene.frame_end   = FRAME_COUNT
    scene.render.fps  = FPS

    # Track-to so camera always looks at target
    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    # Animate camera position on a circle
    for f in range(scene.frame_start, scene.frame_end + 1):
        t = (f - scene.frame_start) / (scene.frame_end - scene.frame_start)
        angle = START_ANGLE + 2.0 * math.pi * t  # 0..2π

        x = center.x + distance * math.cos(angle)
        y = center.y + distance * math.sin(angle)
        z = center.z + height

        cam_obj.location = (x, y, z)
        cam_obj.keyframe_insert("location", frame=f)

    return cam_obj

def setup_render(scene):
    scene.render.filepath = OUTPUT_PATH
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

# ---------- run for this GLB ----------
clear_scene()
scene = import_glb(GLB_PATH)
center, radius = compute_bbox_center_radius(scene)
setup_orbit_camera(scene, center, radius)
setup_render(scene)

# Render the animation
bpy.ops.render.render(animation=True)
