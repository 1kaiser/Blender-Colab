#!/usr/bin/env python3
"""Single-process GPU-accelerated Blender 5.3 Multi-View April Sun-Path Renderer.

Renders four orthographic views ('top', 'front', 'side', 'isometric') in ONE
Blender run (`use_persistent_data = True`, Cycles OptiX GPU), computing true
world-space scene geometry bounds at `--frame` so all 4 cameras frame tightly
and cleanly, then stitches `multiview_april_sun.png` (< 1.8 MiB).

Can be run either via host Python:
    python scripts/sunpath_multiview.py --blender ... --blend ... --frame 15
or directly inside Blender:
    blender -b scene.blend -P scripts/sunpath_multiview.py -- --frame 15
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import bpy  # type: ignore
    from mathutils import Euler, Vector  # type: ignore
except ImportError:
    bpy = None
    Euler = None
    Vector = None

import numpy as np  # noqa: E402

import bake_and_render  # noqa: E402
import gpu_setup  # noqa: E402


MAX_PNG_BYTES = 1_750_000  # strictly < 1.8 MiB artifact guardrail
VIEW_ORDER = ("top", "front", "side", "isometric")
VIEW_LABELS = {
    "top": "Top View",
    "front": "Front View",
    "side": "Side View",
    "isometric": "Isometric View",
}
VIEW_ROT_DEG = {
    "top": (0.0, 0.0, 0.0),
    "front": (90.0, 0.0, 0.0),
    "side": (90.0, 0.0, 90.0),
    "isometric": (60.0, 0.0, 45.0),
}


def default_work_dir() -> Path:
    return Path("/content/work") if Path("/content").is_dir() else Path("./work").resolve()


def find_blender_executable(explicit: str = "") -> str:
    if explicit:
        return explicit
    candidates: list[str] = [
        "/content/work/blender-5.3.0/blender",
        "/content/work/blender-5.2.4/blender",
        "/content/work/blender-4.5.16/blender",
        "/usr/local/bin/blender",
    ]
    for pat in ("/content/work/blender-*/blender", "/content/blender-*/blender"):
        candidates.extend(sorted(glob.glob(pat), reverse=True))
    which_bl = shutil.which("blender")
    if which_bl:
        candidates.append(which_bl)
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "Could not locate Blender executable. Pass --blender /path/to/blender."
    )


def _blender_cli_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:] if bpy is None else []


def compute_solar_position(
    month: int = 4,
    day: int = 15,
    hour: float = 12.0,
    latitude_deg: float = 45.0,
) -> dict[str, float]:
    """Compute solar elevation & clockwise-from-North azimuth using Spencer's declination."""
    doy = datetime.date(2026, month, day).timetuple().tm_yday
    gamma = 2.0 * math.pi * (doy - 1) / 365.0
    decl_rad = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    )
    lat_rad = math.radians(latitude_deg)
    ha_rad = math.radians(15.0 * (hour - 12.0))

    sin_el = (
        math.sin(lat_rad) * math.sin(decl_rad)
        + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad)
    )
    sin_el = max(-1.0, min(1.0, sin_el))
    el_rad = math.asin(sin_el)

    y_term = math.sin(ha_rad)
    x_term = math.cos(ha_rad) * math.sin(lat_rad) - math.tan(decl_rad) * math.cos(lat_rad)
    az_deg = (math.degrees(math.atan2(y_term, x_term)) + 180.0) % 360.0

    return {
        "month": month,
        "day": day,
        "day_of_year": doy,
        "hour": hour,
        "latitude_deg": latitude_deg,
        "declination_deg": round(math.degrees(decl_rad), 3),
        "elevation_deg": round(math.degrees(el_rad), 3),
        "azimuth_deg": round(az_deg, 3),
    }


def _shrink_png_under_limit(png_path: Path, max_bytes: int = MAX_PNG_BYTES) -> int:
    if not png_path.is_file():
        return 0
    size = png_path.stat().st_size
    if size <= max_bytes:
        return size
    try:
        from PIL import Image  # type: ignore

        with Image.open(png_path) as im:
            im.save(png_path, format="PNG", optimize=True, compress_level=9)
            size = png_path.stat().st_size
            scale = 0.85
            while size > max_bytes and scale >= 0.45:
                nw = max(320, int(im.width * scale))
                nh = max(240, int(im.height * scale))
                resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
                resized.save(png_path, format="PNG", optimize=True, compress_level=9)
                size = png_path.stat().st_size
                scale -= 0.12
    except Exception:
        pass
    return png_path.stat().st_size


# ---------------------------------------------------------------------------
# Blender-side scene evaluation, sun placement & orthographic camera framing
# ---------------------------------------------------------------------------

def _place_sun_lamp(el_deg: float, az_deg: float, energy: float = 5.0) -> str:
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT" and getattr(obj.data, "type", "") == "SUN":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 20.0))
    sun = bpy.context.object
    sun.name = "AprilSun"
    sun.data.energy = float(energy)
    if hasattr(sun.data, "angle"):
        sun.data.angle = math.radians(1.5)

    el_rad = math.radians(el_deg)
    az_rad = math.radians(az_deg)
    # Blender Y=North, X=East, Z=Up: Sun(0,0,0) points -Z
    sun.rotation_euler = (math.pi / 2.0 - el_rad, 0.0, -az_rad + math.pi)
    return sun.name


def _is_visible_geometry_object(obj: Any, include_domain: bool = False) -> bool:
    if obj.type != "MESH":
        return False
    if obj.hide_render:
        return False
    if getattr(obj, "display_type", "TEXTURED") in {"WIRE", "BOUNDS"}:
        return False

    ff = getattr(obj, "flip_fluid", None)
    if ff is not None and getattr(ff, "is_active", False):
        ff_type = getattr(ff, "object_type", "TYPE_NONE")
        if ff_type in {"TYPE_INFLOW", "TYPE_OUTFLOW"}:
            return False
        if ff_type == "TYPE_DOMAIN" and not include_domain:
            return False
    return True


def _compute_world_bbox(
    depsgraph: Any,
    include_domain: bool = False,
) -> tuple[list[float], list[float], list[float], list[float], list[str]]:
    included: list[str] = []
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for obj in bpy.data.objects:
        if not _is_visible_geometry_object(obj, include_domain=include_domain):
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = getattr(obj_eval, "data", None)
        if mesh is None or not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
            continue

        n_verts = len(mesh.vertices)
        mat = np.array(obj_eval.matrix_world, dtype=np.float64)
        if n_verts <= 1_500_000:
            coords = np.empty(n_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", coords)
            pts = coords.reshape((-1, 3)).astype(np.float64)
            world_pts = pts @ mat[:3, :3].T + mat[:3, 3]
        else:
            corners = np.array([list(c) for c in obj_eval.bound_box], dtype=np.float64)
            world_pts = corners @ mat[:3, :3].T + mat[:3, 3]

        mins = np.minimum(mins, world_pts.min(axis=0))
        maxs = np.maximum(maxs, world_pts.max(axis=0))
        included.append(obj.name)

    if not included and not include_domain:
        return _compute_world_bbox(depsgraph, include_domain=True)

    if not included:
        mins = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        maxs = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    center = (mins + maxs) * 0.5
    size = np.maximum(maxs - mins, 1e-3)
    return (
        mins.round(6).tolist(),
        maxs.round(6).tolist(),
        center.round(6).tolist(),
        size.round(6).tolist(),
        included,
    )


def _compute_ortho_camera_spec(
    view_name: str,
    bmin: list[float],
    bmax: list[float],
    center: list[float],
    size: list[float],
    res_x: int,
    res_y: int,
    padding: float = 1.18,
) -> dict[str, Any]:
    rot_deg = VIEW_ROT_DEG[view_name]
    rot_rad = [math.radians(a) for a in rot_deg]
    cam_mat = Euler(rot_rad, "XYZ").to_matrix()

    fwd = cam_mat @ Vector((0.0, 0.0, -1.0))
    right = cam_mat @ Vector((1.0, 0.0, 0.0))
    up = cam_mat @ Vector((0.0, 1.0, 0.0))

    c_vec = Vector(center)
    corners = [
        Vector((x, y, z)) - c_vec
        for x in (bmin[0], bmax[0])
        for y in (bmin[1], bmax[1])
        for z in (bmin[2], bmax[2])
    ]
    r_vals = [v.dot(right) for v in corners]
    u_vals = [v.dot(up) for v in corners]
    proj_w = max(r_vals) - min(r_vals)
    proj_h = max(u_vals) - min(u_vals)

    aspect = float(res_x) / max(1.0, float(res_y))
    ortho_scale = padding * max(proj_w, proj_h * aspect)

    max_dim = max(size[0], size[1], size[2], 1.0)
    dist = max(20.0, max_dim * 4.0)
    loc = c_vec - fwd * dist

    return {
        "name": view_name,
        "location": [round(float(loc.x), 4), round(float(loc.y), 4), round(float(loc.z), 4)],
        "rotation_deg": list(rot_deg),
        "ortho_scale": round(float(ortho_scale), 4),
        "clip_start": 0.01,
        "clip_end": round(float(dist * 10.0), 2),
    }


def _ensure_ortho_camera(spec: dict[str, Any]) -> Any:
    name = f"SunPathCam_{spec['name']}"
    existing = bpy.data.objects.get(name)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    bpy.ops.object.camera_add(location=tuple(spec["location"]))
    cam = bpy.context.object
    cam.name = name
    cam.rotation_euler = [math.radians(a) for a in spec["rotation_deg"]]
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = float(spec["ortho_scale"])
    cam.data.sensor_fit = "AUTO"
    cam.data.clip_start = float(spec["clip_start"])
    cam.data.clip_end = float(spec["clip_end"])
    return cam


def _configure_sunpath_lighting(
    scene: Any,
    el_deg: float,
    az_deg: float,
    sun_energy: float,
    *,
    keep_scene_lights: bool = False,
    white_bg: bool = True,
    bg_strength: float = 1.0,
) -> str:
    """Configure April Sun lamp and clean directional daylight studio environment."""
    if not keep_scene_lights:
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                obj.hide_render = True

    sun_name = _place_sun_lamp(el_deg, az_deg, energy=sun_energy)

    if white_bg:
        if scene.world is None:
            scene.world = bpy.data.worlds.new("SunPathWorld")
        scene.world.use_nodes = True
        nt = scene.world.node_tree
        nt.nodes.clear()

        w_out = nt.nodes.new(type="ShaderNodeOutputWorld")
        w_out.location = (450, 0)

        mix = nt.nodes.new(type="ShaderNodeMixShader")
        mix.location = (220, 0)

        lp = nt.nodes.new(type="ShaderNodeLightPath")
        lp.location = (0, 220)

        bg_cam = nt.nodes.new(type="ShaderNodeBackground")
        bg_cam.location = (0, 40)
        bg_cam.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_cam.inputs[1].default_value = 3.5

        bg_env = nt.nodes.new(type="ShaderNodeBackground")
        bg_env.location = (0, -120)
        bg_env.inputs[0].default_value = (0.94, 0.97, 1.0, 1.0)
        bg_env.inputs[1].default_value = float(bg_strength)

        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg_env.outputs["Background"], mix.inputs[1])
        nt.links.new(bg_cam.outputs["Background"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], w_out.inputs["Surface"])

    return sun_name


def _render_four_views_in_blender(args: argparse.Namespace) -> dict[str, Any]:
    gpu_setup.ensure_flip_fluids_enabled(verbose=True)
    scene = bpy.context.scene

    _, dprops = bake_and_render._get_domain_props(scene)
    if dprops is None:
        bake_and_render.reactivate_dormant_flip_objects(scene, args.cache_dir)
        _, dprops = bake_and_render._get_domain_props(scene)
    if dprops is not None:
        if args.cache_dir:
            dprops.cache.cache_directory = os.path.abspath(args.cache_dir)
        if hasattr(dprops, "render") and hasattr(dprops.render, "viewport_display"):
            try:
                dprops.render.viewport_display = "DISPLAY_FINAL"
            except Exception:
                pass
        mcache = getattr(dprops, "mesh_cache", None)
        if mcache is not None and hasattr(mcache, "initialize_cache_objects"):
            try:
                mcache.initialize_cache_objects()
            except Exception:
                pass

    bake_and_render.hide_flip_helper_objects(scene)
    bake_and_render.repair_missing_materials(scene, dprops)

    gpu_setup.configure_render(
        scene,
        engine=args.engine,
        compute_type=args.compute_type,
        include_cpu=args.include_cpu,
        persistent_data=not args.no_persistent_data,
        use_denoising=not args.no_denoise,
        denoiser=args.denoiser,
        use_adaptive_sampling=not args.no_adaptive_sampling,
        samples=args.samples,
        adaptive_threshold=args.adaptive_threshold,
        use_texture_cache=not args.no_texture_cache,
        verbose=True,
    )

    solar = compute_solar_position(
        month=args.month,
        day=args.day,
        hour=args.hour,
        latitude_deg=args.latitude,
    )
    el_deg = float(args.elevation) if args.elevation is not None else solar["elevation_deg"]
    az_deg = float(args.azimuth) if args.azimuth is not None else solar["azimuth_deg"]
    solar["used_elevation_deg"] = el_deg
    solar["used_azimuth_deg"] = az_deg

    sun_name = _configure_sunpath_lighting(
        scene,
        el_deg,
        az_deg,
        args.sun_energy,
        keep_scene_lights=args.keep_scene_lights,
        white_bg=not args.keep_scene_world,
        bg_strength=args.bg_strength,
    )

    # Evaluate scene at target frame so FLIP fluid_surface mesh is loaded & textured
    scene.frame_set(int(args.frame))
    if dprops is not None and hasattr(dprops, "mesh_cache"):
        surf_cache = getattr(dprops.mesh_cache, "surface", None)
        if surf_cache is not None and hasattr(surf_cache, "load_frame"):
            try:
                surf_cache.load_frame(int(args.frame), True)
            except Exception:
                pass
    bake_and_render.ensure_flip_object_materials(scene)
    bake_and_render._tag_flip_meshes_updated(scene)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bmin, bmax, center, size, included = _compute_world_bbox(
        depsgraph, include_domain=args.include_domain
    )
    print(f"[sunpath] BBox objects ({len(included)}): {included}, center={center}, size={size}", flush=True)

    scene.render.resolution_x = int(args.res_x)
    scene.render.resolution_y = int(args.res_y)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    views_meta: dict[str, Any] = {}
    t_all0 = time.perf_counter()

    for view_name in VIEW_ORDER:
        spec = _compute_ortho_camera_spec(
            view_name,
            bmin,
            bmax,
            center,
            size,
            args.res_x,
            args.res_y,
            padding=args.padding,
        )
        cam = _ensure_ortho_camera(spec)
        scene.camera = cam
        out_png = out_dir / f"view_{view_name}.png"
        scene.render.filepath = str(out_png)

        t_v0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        dt_v = time.perf_counter() - t_v0
        out_bytes = _shrink_png_under_limit(out_png)

        spec["render_s"] = round(dt_v, 4)
        spec["output_png"] = str(out_png)
        spec["output_bytes"] = out_bytes
        views_meta[view_name] = spec
        print(
            f"[sunpath] Rendered {view_name} (ortho_scale={spec['ortho_scale']}) "
            f"in {dt_v:.2f}s -> {out_png} ({out_bytes} B)",
            flush=True,
        )

    total_render_s = round(time.perf_counter() - t_all0, 4)
    summary = {
        "frame": int(args.frame),
        "solar": solar,
        "sun_object": sun_name,
        "bbox": {
            "min": bmin,
            "max": bmax,
            "center": center,
            "size": size,
            "included_objects": included,
        },
        "resolution": [int(args.res_x), int(args.res_y)],
        "total_render_s": total_render_s,
        "views": views_meta,
    }
    meta_path = out_dir / "sunpath_metadata.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Host-side / fallback 1x4 PNG stitcher
# ---------------------------------------------------------------------------

def _write_rgb_png_pure_python(path: Path, rgb: np.ndarray) -> None:
    """Minimal stdlib+numpy PNG writer fallback when Pillow/matplotlib are absent."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw, level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png_bytes = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png_bytes)


def stitch_multiview_grid(out_dir: Path, summary: Optional[dict[str, Any]] = None) -> Path:
    """Stitch view_top/front/side/isometric into multiview_april_sun.png (< 1.8 MiB)."""
    if summary is None:
        meta_path = out_dir / "sunpath_metadata.json"
        if meta_path.is_file():
            summary = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            summary = {}

    solar = summary.get("solar", {})
    el = solar.get("used_elevation_deg", solar.get("elevation_deg", 54.5))
    az = solar.get("used_azimuth_deg", solar.get("azimuth_deg", 180.0))
    lat = solar.get("latitude_deg", 45.0)
    month = solar.get("month", 4)
    day = solar.get("day", 15)
    frame = summary.get("frame", 1)
    views_meta = summary.get("views", {})

    out_path = out_dir / "multiview_april_sun.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
        fig.patch.set_facecolor("#1a1a1a")
        month_name = datetime.date(2026, int(month), int(day)).strftime("%b %d")
        fig.suptitle(
            f"Blender 5.3 — Sun Path ({month_name}, lat {lat:.0f}°N)  |  "
            f"Elevation {el:.1f}° · Azimuth {az:.1f}° · Frame {frame}",
            fontsize=13,
            fontweight="bold",
            color="white",
            y=0.99,
        )

        for ax, view_name in zip(axes, VIEW_ORDER):
            img_path = out_dir / f"view_{view_name}.png"
            if img_path.is_file():
                ax.imshow(mpimg.imread(str(img_path)))
            else:
                ax.set_facecolor("#2a2a2a")
                ax.text(
                    0.5,
                    0.5,
                    f"{VIEW_LABELS[view_name]}\n(missing)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#aaaaaa",
                )
            ax.set_title(VIEW_LABELS[view_name], fontsize=12, color="white", pad=8)
            scale = views_meta.get(view_name, {}).get("ortho_scale")
            if scale is not None:
                ax.set_xlabel(f"ortho · scale {scale:.2f}", fontsize=9.5, color="#b0b0b0", labelpad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")

        plt.tight_layout(rect=[0, 0, 1, 0.93], pad=1.2)
        fig.savefig(
            str(out_path),
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        _shrink_png_under_limit(out_path)
        return out_path
    except ImportError:
        pass

    # Fallback if invoked strictly inside Blender's Python without matplotlib
    py_sys = shutil.which("python3")
    if bpy is not None and py_sys and Path(py_sys).resolve() != Path(sys.executable).resolve():
        proc = subprocess.run(
            [py_sys, str(Path(__file__).resolve()), "--stitch-only", "--out-dir", str(out_dir)],
            check=False,
        )
        if proc.returncode == 0 and out_path.is_file():
            return out_path

    # Pure-numpy horizontal stitch fallback
    panels: list[np.ndarray] = []
    for view_name in VIEW_ORDER:
        img_path = out_dir / f"view_{view_name}.png"
        if img_path.is_file():
            # Load via Blender's bpy.data.images if inside Blender
            if bpy is not None:
                img = bpy.data.images.load(str(img_path), check_existing=True)
                w, h = img.size[0], img.size[1]
                arr = np.asarray(img.pixels[:], dtype=np.float32).reshape((h, w, img.channels))
                arr = np.flipud(arr)[:, :, :3]
                panels.append(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
                bpy.data.images.remove(img)
    if panels:
        canvas = np.concatenate(panels, axis=1)
        _write_rgb_png_pure_python(out_path, canvas)
    return out_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    wdir = default_work_dir()
    parser = argparse.ArgumentParser(
        description="Single-process GPU Blender 5.3 multi-view April sun-path renderer."
    )
    parser.add_argument("--blender", default="", help="Path to Blender executable")
    parser.add_argument("--blend", default=str(wdir / "particles.blend"), help="Input .blend file")
    parser.add_argument("--cache-dir", default="", help="Optional FLIP Fluids cache directory")
    parser.add_argument("--out-dir", default=str(wdir / "sunpath_views"), help="Output PNG directory")
    parser.add_argument("--frame", type=int, default=15, help="Timeline frame to evaluate & render")

    # Solar geometry
    parser.add_argument("--month", type=int, default=4, help="Month 1..12 (default: 4)")
    parser.add_argument("--day", type=int, default=15, help="Day 1..31 (default: 15)")
    parser.add_argument("--hour", type=float, default=12.0, help="Solar hour 0..24 (default: 12.0)")
    parser.add_argument("--latitude", type=float, default=45.0, help="Latitude deg (default: 45.0)")
    parser.add_argument("--elevation", type=float, default=None, help="Optional solar elevation override")
    parser.add_argument("--azimuth", type=float, default=None, help="Optional solar azimuth override")
    parser.add_argument("--sun-energy", type=float, default=6.0, help="Sun lamp strength (default: 6.0)")

    # Camera / Render settings
    parser.add_argument("--res-x", type=int, default=960, help="Per-view width (default: 960)")
    parser.add_argument("--res-y", type=int, default=720, help="Per-view height (default: 720)")
    parser.add_argument("--padding", type=float, default=1.18, help="Ortho framing margin (default: 1.18)")
    parser.add_argument("--include-domain", action="store_true", help="Include FLIP domain box in bbox")
    parser.add_argument("--white-bg", action="store_true", help="Force clean white studio world background (default)")
    parser.add_argument("--keep-scene-lights", action="store_true", help="Keep pre-existing scene LIGHT objects active")
    parser.add_argument("--keep-scene-world", action="store_true", help="Keep pre-existing scene World shader untouched")
    parser.add_argument("--bg-strength", type=float, default=1.6)

    parser.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"])
    parser.add_argument("--compute-type", default="AUTO", choices=["AUTO", "OPTIX", "CUDA", "NONE"])
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--no-persistent-data", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--denoiser", default="AUTO", choices=["AUTO", "OPENIMAGEDENOISE", "OPTIX", "NONE"])
    parser.add_argument("--no-adaptive-sampling", action="store_true")
    parser.add_argument("--samples", type=int, default=64, help="Render samples per view (default: 64)")
    parser.add_argument("--adaptive-threshold", type=float, default=0.02)
    parser.add_argument("--no-texture-cache", action="store_true")

    parser.add_argument("--json-out", default="", help="Optional summary JSON output path")
    parser.add_argument("--stitch-only", action="store_true", help="Only stitch existing view_*.png")
    parser.add_argument("--internal-render", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(_blender_cli_argv())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stitch_only and bpy is None:
        stitched = stitch_multiview_grid(out_dir)
        print(f"[sunpath] Stitched -> {stitched} ({stitched.stat().st_size} B)", flush=True)
        return 0

    if bpy is not None:
        summary = _render_four_views_in_blender(args)
        if not args.internal_render:
            stitched = stitch_multiview_grid(out_dir, summary)
            summary["combined_png"] = str(stitched)
            summary["combined_bytes"] = stitched.stat().st_size if stitched.is_file() else 0
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    # Host Python orchestration: launch ONE Blender process, then stitch with matplotlib
    blender_bin = find_blender_executable(args.blender)
    blend_path = str(Path(args.blend).resolve())
    if not os.path.isfile(blend_path):
        raise FileNotFoundError(f"Blend file not found: {blend_path}")

    cmd = [
        blender_bin,
        "-b",
        blend_path,
        "-P",
        str(Path(__file__).resolve()),
        "--",
        "--internal-render",
        "--out-dir",
        str(out_dir),
        "--frame",
        str(args.frame),
        "--month",
        str(args.month),
        "--day",
        str(args.day),
        "--hour",
        str(args.hour),
        "--latitude",
        str(args.latitude),
        "--sun-energy",
        str(args.sun_energy),
        "--res-x",
        str(args.res_x),
        "--res-y",
        str(args.res_y),
        "--padding",
        str(args.padding),
        "--bg-strength",
        str(args.bg_strength),
        "--engine",
        args.engine,
        "--compute-type",
        args.compute_type,
        "--denoiser",
        args.denoiser,
        "--samples",
        str(args.samples),
        "--adaptive-threshold",
        str(args.adaptive_threshold),
    ]
    if args.cache_dir:
        cmd.extend(["--cache-dir", args.cache_dir])
    if args.elevation is not None:
        cmd.extend(["--elevation", str(args.elevation)])
    if args.azimuth is not None:
        cmd.extend(["--azimuth", str(args.azimuth)])
    if args.include_domain:
        cmd.append("--include-domain")
    if args.white_bg:
        cmd.append("--white-bg")
    if args.keep_scene_lights:
        cmd.append("--keep-scene-lights")
    if args.keep_scene_world:
        cmd.append("--keep-scene-world")
    if args.include_cpu:
        cmd.append("--include-cpu")
    if args.no_persistent_data:
        cmd.append("--no-persistent-data")
    if args.no_denoise:
        cmd.append("--no-denoise")
    if args.no_adaptive_sampling:
        cmd.append("--no-adaptive-sampling")
    if args.no_texture_cache:
        cmd.append("--no-texture-cache")

    t0 = time.perf_counter()
    print(f"[sunpath] Launching single-process Blender render: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    meta_path = out_dir / "sunpath_metadata.json"
    summary = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    stitched = stitch_multiview_grid(out_dir, summary)
    summary["combined_png"] = str(stitched)
    summary["combined_bytes"] = stitched.stat().st_size if stitched.is_file() else 0
    summary["wall_s"] = round(time.perf_counter() - t0, 4)

    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[sunpath] Complete in {summary['wall_s']:.2f}s -> {stitched} "
        f"({summary['combined_bytes']} B)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
