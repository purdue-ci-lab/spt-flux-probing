"""
spt-photon-vis.py

3D visualization of photons flying toward an image sensor.

The sensor is a low-resolution (16x16) square. Its underlying intensity is a
circle in the middle; each pixel is Bernoulli-sampled from that intensity to
produce a single-quanta (binary) frame, so lit pixels are set to 1 and glow
white. Incoming photons are rendered as glowing spheres, each trailing a
tapered "comet" tail that points along its direction of travel toward the
sensor.

Run inside Blender:
    blender --python spt-photon-vis.py
or paste into Blender's scripting workspace and run.
Render with F12 (or `blender -b spt-photon-vis.py -o //render_ -f 1`).
"""

import math
import random

import bpy
from mathutils import Vector

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
SEED = 71                      # reproducible photon frame / positions
RESOLUTION = 16              # sensor is RESOLUTION x RESOLUTION pixels
SENSOR_SIZE = 4.0            # world-space width/height of the sensor square
PIXEL_GAP = 0.08            # fraction of a pixel left as a gap (grid look)

CIRCLE_RADIUS = 0.32        # circle radius as a fraction of the sensor half-size
CIRCLE_SOFTNESS = 0.12      # soft falloff width (fraction of half-size)
PEAK_PROB = 0.95            # per-pixel probability of firing at the circle center
BG_PROB = 0.2              # background per-pixel fire probability (dark-count noise)

N_PHOTONS = 28              # number of in-flight photons to draw
BG_PHOTON_FRAC = 0.3        # fraction of photons aimed anywhere on the sensor
PHOTON_RADIUS = 0.055       # sphere radius of a photon head
TRAIL_LENGTH = 0.6          # length of the comet trail
TRAIL_WIDTH = 0.045         # base radius of the comet trail
SPAWN_HEIGHT = 4.0          # how far above the sensor photons stream from

# Scene orientation:
#   "top"  - sensor lies flat, photons fall straight down onto it (original).
#   "side" - sensor stands upright, photons fly in horizontally from the side.
# The scene is always built in the "top" frame, then the whole rig is rotated
# to the chosen orientation and a matching camera framing is used.
ORIENTATION = "side"

VIEW_PRESETS = {
    "top": {
        # no rotation: keep the original flat layout
        "root_euler": (0.0, 0.0, 0.0),
        "cam_loc": (6.5, -7.5, 5.5),
        "cam_target": (0.0, 0.0, 1.2),
    },
    "side": {
        # tip the flat scene up 90 deg about X so the sensor becomes a wall
        "root_euler": (math.radians(90.0), 0.0, 0.0),
        # low, grazing view: the frame is seen more edge-on (from the side)
        # while the incoming photons streak across the image
        "cam_loc": (11.0, -4.5, 1.0),
        "cam_target": (0.0, -1.0, 0.4),
    },
}

# --------------------------------------------------------------------------- #
# Scene helpers
# --------------------------------------------------------------------------- #
def clear_scene():
    """Remove all objects and orphaned datablocks so re-runs are clean."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.curves,
                  bpy.data.lights, bpy.data.cameras):
        for datablock in list(block):
            if datablock.users == 0:
                block.remove(datablock)


def make_scene_root():
    """An empty that parents the whole rig so it can be reoriented at once."""
    root = bpy.data.objects.new("SceneRoot", None)
    root.rotation_euler = VIEW_PRESETS[ORIENTATION]["root_euler"]
    bpy.context.collection.objects.link(root)
    return root


def emission_material(name, color, strength):
    """A pure-emission material (glowing surface)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def transparent_emission_material(name, color, strength, alpha, gradient_to_black=False):
    """A weak emissive material that is also semi-transparent (alpha < 1).

    When gradient_to_black is True the emission color runs from `color` at the
    object's base (local -Z) to black at the tip (local +Z), giving a comet
    tail that fades out along its length.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength

    if gradient_to_black:
        # Generated coords: local Z runs 0 at the base to 1 at the tip.
        texcoord = nodes.new("ShaderNodeTexCoord")
        separate = nodes.new("ShaderNodeSeparateXYZ")
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (*color, 1.0)   # base: photon color
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)  # tip: black
        links.new(texcoord.outputs["Generated"], separate.inputs["Vector"])
        links.new(separate.outputs["Z"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], emission.inputs["Color"])

    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = alpha  # 0 = fully transparent, 1 = full emission
    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return mat


def matte_material(name, color):
    """A simple diffuse material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.85
    return mat


# --------------------------------------------------------------------------- #
# Sensor
# --------------------------------------------------------------------------- #
def circle_intensity(u, v):
    """Underlying intensity in [0, 1] for a circle centered on the sensor.

    u, v are pixel-center coordinates in [-1, 1] across the sensor.
    """
    r = math.hypot(u, v)
    # smooth edge: 1 inside the radius, falling to 0 across CIRCLE_SOFTNESS.
    edge0 = CIRCLE_RADIUS
    edge1 = CIRCLE_RADIUS + CIRCLE_SOFTNESS
    if r <= edge0:
        return 1.0
    if r >= edge1:
        return 0.0
    t = (r - edge0) / (edge1 - edge0)
    return 1.0 - (t * t * (3.0 - 2.0 * t))  # smoothstep falloff


def sample_binary_frame(rng):
    """Sample a RESOLUTION x RESOLUTION single-quanta (0/1) frame."""
    frame = [[0] * RESOLUTION for _ in range(RESOLUTION)]
    lit = []
    for j in range(RESOLUTION):
        for i in range(RESOLUTION):
            # pixel center mapped to [-1, 1]
            u = (i + 0.5) / RESOLUTION * 2.0 - 1.0
            v = (j + 0.5) / RESOLUTION * 2.0 - 1.0
            # background floor everywhere, rising to PEAK_PROB inside the circle
            prob = BG_PROB + (PEAK_PROB - BG_PROB) * circle_intensity(u, v)
            if rng.random() < prob:
                frame[j][i] = 1
                lit.append((i, j, u, v))
    return frame, lit


def build_sensor(rng):
    """Build the sensor as a grid of pixel tiles; lit pixels glow white."""
    frame, _ = sample_binary_frame(rng)

    white = emission_material("PixelOn", (1.0, 1.0, 1.0), 6.0)
    dark = matte_material("PixelOff", (0.02, 0.025, 0.035))
    backing = matte_material("SensorBacking", (0.01, 0.012, 0.02))

    pixel_pitch = SENSOR_SIZE / RESOLUTION
    tile = pixel_pitch * (1.0 - PIXEL_GAP)
    half = SENSOR_SIZE / 2.0

    parent = bpy.data.objects.new("Sensor", None)  # empty to group everything
    bpy.context.collection.objects.link(parent)

    # A backing plate just below the pixels.
    bpy.ops.mesh.primitive_plane_add(size=SENSOR_SIZE * 1.02, location=(0, 0, -0.01))
    plate = bpy.context.object
    plate.name = "SensorBacking"
    plate.data.materials.append(backing)
    plate.parent = parent

    for j in range(RESOLUTION):
        for i in range(RESOLUTION):
            x = -half + (i + 0.5) * pixel_pitch
            y = -half + (j + 0.5) * pixel_pitch
            bpy.ops.mesh.primitive_plane_add(size=tile, location=(x, y, 0.0))
            px = bpy.context.object
            px.name = f"px_{i:02d}_{j:02d}"
            px.data.materials.append(white if frame[j][i] else dark)
            px.parent = parent
    return parent


# --------------------------------------------------------------------------- #
# Photons
# --------------------------------------------------------------------------- #
def random_target_on_circle(rng):
    """A random landing point on the sensor plane.

    Most photons concentrate within the lit circle, but a fraction
    (BG_PHOTON_FRAC) land anywhere on the sensor so some strike the noisy
    background pixels.
    """
    half = SENSOR_SIZE / 2.0
    if rng.random() < BG_PHOTON_FRAC:
        # uniform anywhere on the sensor (leave a small margin from the edge)
        return Vector((rng.uniform(-half, half) * 0.95,
                       rng.uniform(-half, half) * 0.95, 0.0))
    # sample radius so points concentrate within the lit circle
    r = CIRCLE_RADIUS * math.sqrt(rng.random()) * 1.15
    theta = rng.uniform(0.0, 2.0 * math.pi)
    u = r * math.cos(theta)
    v = r * math.sin(theta)
    return Vector((u * half, v * half, 0.0))


def build_photon(index, rng, head_mat, trail_mat, root):
    """One photon: a glowing sphere head plus a tapered comet trail.

    Built in the "top" frame: photons travel straight down (perpendicular to the
    sensor plane), so the trail points vertically up (+Z) behind the head. The
    scene root rotation later carries this into the chosen orientation.
    """
    target = random_target_on_circle(rng)

    # Head hovers directly above its landing point, at a random height.
    head_z = SPAWN_HEIGHT * rng.uniform(0.15, 0.9)
    head_pos = Vector((target.x, target.y, head_z))

    direction = Vector((0.0, 0.0, -1.0))  # travel direction: straight down

    parent = bpy.data.objects.new(f"Photon_{index:02d}", None)
    parent.parent = root
    bpy.context.collection.objects.link(parent)

    # Head sphere.
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=PHOTON_RADIUS, segments=16, ring_count=8, location=head_pos)
    head = bpy.context.object
    head.name = f"Photon_{index:02d}_head"
    bpy.ops.object.shade_smooth()
    head.data.materials.append(head_mat)
    head.parent = parent

    # Comet trail: a cone whose wide base sits at the head and tip trails behind.
    # Cone default axis is +Z (tip at +Z). Align +Z with -direction so the tip
    # points backward, then shift so the base meets the head.
    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(-direction).to_euler()
    trail_center = head_pos - direction * (TRAIL_LENGTH / 2.0)
    bpy.ops.mesh.primitive_cone_add(
        radius1=TRAIL_WIDTH, radius2=0.0, depth=TRAIL_LENGTH,
        vertices=16, location=trail_center, rotation=rot)
    trail = bpy.context.object
    trail.name = f"Photon_{index:02d}_trail"
    bpy.ops.object.shade_smooth()
    trail.data.materials.append(trail_mat)
    trail.parent = parent
    return parent


def build_photons(rng, root):
    # light blue so photons stand out against the white pixels
    head_rgb = (35, 222, 222)
    head_rgb = tuple(c / 255.0 for c in head_rgb)
    trail_rgb = (107, 227, 223)
    trail_rgb = tuple(c / 255.0 for c in trail_rgb)
    # head: lit normally by the scene lights (no emission)
    head_mat = matte_material("PhotonHead", head_rgb)
    # trail: weak emission, semi-transparent
    trail_mat = transparent_emission_material(
        "PhotonTrail", trail_rgb, 1.5, 0.35, gradient_to_black=True)
    for k in range(N_PHOTONS):
        build_photon(k, rng, head_mat, trail_mat, root)


# --------------------------------------------------------------------------- #
# Camera, lights, render
# --------------------------------------------------------------------------- #
def setup_camera():
    preset = VIEW_PRESETS[ORIENTATION]
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = preset["cam_loc"]
    bpy.context.collection.objects.link(cam)

    # Aim the camera at the framing point for this orientation.
    target = bpy.data.objects.new("CamTarget", None)
    target.location = preset["cam_target"]
    bpy.context.collection.objects.link(target)
    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    bpy.context.scene.camera = cam
    return cam


def setup_lights():
    # Key light.
    key_data = bpy.data.lights.new("Key", type="AREA")
    key_data.energy = 400
    key_data.size = 6.0
    key = bpy.data.objects.new("Key", key_data)
    key.location = (5.0, -4.0, 8.0)
    key.rotation_euler = (math.radians(35), math.radians(15), 0)
    bpy.context.collection.objects.link(key)

    # Cool fill light from the opposite side.
    fill_data = bpy.data.lights.new("Fill", type="AREA")
    fill_data.energy = 120
    fill_data.size = 8.0
    fill_data.color = (0.6, 0.75, 1.0)
    fill = bpy.data.objects.new("Fill", fill_data)
    fill.location = (-6.0, -2.0, 4.0)
    fill.rotation_euler = (math.radians(60), math.radians(-20), 0)
    bpy.context.collection.objects.link(fill)


def setup_world():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.01, 0.012, 0.02, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def setup_render():
    scene = bpy.context.scene
    # Prefer whichever EEVEE identifier this Blender build exposes.
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1200
    scene.render.film_transparent = True   # background renders with alpha = 0
    scene.render.image_settings.color_mode = "RGBA"  # keep the alpha channel

    eevee = scene.eevee
    if hasattr(eevee, "use_bloom"):          # legacy EEVEE
        eevee.use_bloom = True
        eevee.bloom_intensity = 0.06
    for attr in ("use_raytracing", "use_gtao"):
        if hasattr(eevee, attr):
            setattr(eevee, attr, True)

    scene.view_settings.view_transform = "Filmic" if \
        "Filmic" in [v.name for v in scene.view_settings.bl_rna.properties[
            "view_transform"].enum_items] else scene.view_settings.view_transform


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
rng = random.Random(SEED)
clear_scene()
root = make_scene_root()
build_sensor(rng).parent = root
build_photons(rng, root)
setup_camera()
setup_lights()
setup_world()
setup_render()
print("spt-photon-vis: scene built.")
