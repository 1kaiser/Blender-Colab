#!/usr/bin/env python3
"""Inspect a .blend scene inside headless Blender and emit structured JSON diagnostics.

Usage:
    blender -b <file.blend> -P scripts/inspect_scene.py -- --json-out <path>
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

import addon_utils
import bpy
from mathutils import Vector


WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]|^\\\\")


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Inspect Blender scene geometry, render settings, FLIP Fluids, and missing paths."
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output path to write JSON diagnostics",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Optional frame to evaluate before computing bounding boxes",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print compact completion message (still writes --json-out)",
    )
    return parser.parse_args(argv)


def is_windows_style_path(path_str: str) -> bool:
    if not path_str:
        return False
    return bool(WINDOWS_PATH_RE.match(path_str)) or ("\\" in path_str)


def resolve_blend_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    if is_windows_style_path(raw_path):
        return raw_path
    try:
        return os.path.abspath(bpy.path.abspath(raw_path))
    except Exception:
        return raw_path


def find_flip_fluids_addon(scene: bpy.types.Scene) -> dict:
    enabled_keys = [
        k for k in bpy.context.preferences.addons.keys()
        if k == "flip_fluids_addon" or k.endswith(".flip_fluids_addon")
    ]
    installed_modules = []
    for mod in addon_utils.modules():
        name = getattr(mod, "__name__", "")
        if name == "flip_fluids_addon" or name.endswith(".flip_fluids_addon"):
            installed_modules.append(name)

    active_mod_name = (
        "bl_ext.user_default.flip_fluids_addon"
        if "bl_ext.user_default.flip_fluids_addon" in enabled_keys
        else (enabled_keys[0] if enabled_keys else None)
    )

    info: dict = {
        "enabled": bool(active_mod_name),
        "module_name": active_mod_name,
        "enabled_modules": enabled_keys,
        "installed_modules_found": installed_modules,
        "bl_info_name": None,
        "bl_info_version": None,
        "bl_info_description": None,
        "installation_complete": None,
        "support_license": None,
        "is_addon_active_in_scene": None,
    }

    if active_mod_name:
        try:
            mod = sys.modules.get(active_mod_name) or importlib.import_module(active_mod_name)
            bl_info = {}
            try:
                bl_info = addon_utils.module_bl_info(mod) or {}
            except Exception:
                pass
            if not bl_info:
                bl_info = getattr(mod, "bl_info", {}) or {}

            manifest_data = {}
            addon_dir = os.path.dirname(os.path.realpath(mod.__file__))
            manifest_path = os.path.join(addon_dir, "blender_manifest.toml")
            if os.path.isfile(manifest_path):
                try:
                    import tomllib
                    with open(manifest_path, "rb") as mf:
                        manifest_data = tomllib.load(mf)
                except Exception:
                    pass

            ver = bl_info.get("version")
            if not ver and manifest_data.get("version"):
                ver = [int(x) for x in str(manifest_data["version"]).split(".") if x.isdigit()]
            info["bl_info_name"] = bl_info.get("name") or manifest_data.get("name")
            info["bl_info_version"] = list(ver) if ver else None
            info["bl_info_description"] = bl_info.get("description") or manifest_data.get("tagline")
            if manifest_data:
                info["manifest"] = {
                    "id": manifest_data.get("id"),
                    "version": manifest_data.get("version"),
                    "name": manifest_data.get("name"),
                    "tagline": manifest_data.get("tagline"),
                    "blender_version_min": manifest_data.get("blender_version_min"),
                }
            inst_utils = getattr(getattr(mod, "utils", None), "installation_utils", None)
            if inst_utils is not None:
                info["installation_complete"] = bool(inst_utils.is_installation_complete())
                info["support_license"] = inst_utils.get_support_license_label()
                info["is_addon_active_in_scene"] = bool(inst_utils.is_addon_active())
        except Exception as exc:
            info["error"] = str(exc)

    return info


def compute_object_world_bbox(obj: bpy.types.Object, depsgraph) -> dict:
    obj_eval = obj.evaluated_get(depsgraph)
    vertex_count = 0
    polygon_count = 0
    is_empty_mesh = False
    if obj_eval.type == "MESH" and obj_eval.data is not None:
        vertex_count = len(obj_eval.data.vertices)
        polygon_count = len(obj_eval.data.polygons)
        is_empty_mesh = vertex_count == 0

    if obj_eval.type == "MESH" and vertex_count > 0:
        mat = obj_eval.matrix_world
        pts = [mat @ Vector(corner) for corner in obj_eval.bound_box]
    elif hasattr(obj_eval, "bound_box") and obj_eval.type not in {"CAMERA", "LIGHT", "EMPTY"}:
        mat = obj_eval.matrix_world
        pts = [mat @ Vector(corner) for corner in obj_eval.bound_box]
    else:
        pos = obj_eval.matrix_world.translation
        pts = [pos]

    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]
    bmin = [round(min(xs), 6), round(min(ys), 6), round(min(zs), 6)]
    bmax = [round(max(xs), 6), round(max(ys), 6), round(max(zs), 6)]
    center = [round((a + b) * 0.5, 6) for a, b in zip(bmin, bmax)]
    size = [round(b - a, 6) for a, b in zip(bmin, bmax)]
    return {
        "min": bmin,
        "max": bmax,
        "center": center,
        "size": size,
        "vertex_count": vertex_count,
        "polygon_count": polygon_count,
        "is_empty_mesh": is_empty_mesh,
    }


def merge_bboxes(items: list[tuple[str, dict]]) -> dict | None:
    if not items:
        return None
    bmin = [
        min(b["min"][i] for _, b in items)
        for i in range(3)
    ]
    bmax = [
        max(b["max"][i] for _, b in items)
        for i in range(3)
    ]
    center = [round((a + b) * 0.5, 6) for a, b in zip(bmin, bmax)]
    size = [round(b - a, 6) for a, b in zip(bmin, bmax)]
    return {
        "min": [round(v, 6) for v in bmin],
        "max": [round(v, 6) for v in bmax],
        "center": center,
        "size": size,
        "object_count": len(items),
        "included_objects": [name for name, _ in items],
    }


def idprop_to_python(val, depth: int = 0):
    if depth > 6:
        return str(val)
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if hasattr(val, "to_dict"):
        return {str(k): idprop_to_python(v, depth + 1) for k, v in val.to_dict().items()}
    if hasattr(val, "to_list"):
        lst = val.to_list()
        return [idprop_to_python(x, depth + 1) for x in lst[:64]]
    if isinstance(val, dict):
        return {str(k): idprop_to_python(v, depth + 1) for k, v in val.items()}
    if hasattr(val, "__iter__") and not isinstance(val, (bytes, bytearray)):
        items = list(val)
        return [idprop_to_python(x, depth + 1) for x in items[:64]]
    return str(val)


def get_flip_object_type(obj: bpy.types.Object) -> str:
    ff = getattr(obj, "flip_fluid", None)
    if ff is None:
        return "UNAVAILABLE"
    try:
        if not getattr(ff, "is_active", False):
            return "TYPE_NONE"
        return str(getattr(ff, "object_type", "TYPE_NONE"))
    except Exception:
        return "TYPE_NONE"


def inspect_objects_and_bounds(scene: bpy.types.Scene, depsgraph) -> tuple[list[dict], dict]:
    objects_info: list[dict] = []
    mesh_bboxes: list[tuple[str, dict]] = []
    mesh_non_domain_bboxes: list[tuple[str, dict]] = []
    all_bboxes: list[tuple[str, dict]] = []

    for obj in scene.objects:
        bbox_info = compute_object_world_bbox(obj, depsgraph)
        ff_type = get_flip_object_type(obj)
        mat_names = [
            slot.material.name if slot.material is not None else None
            for slot in obj.material_slots
        ]
        mods = [
            {
                "name": m.name,
                "type": m.type,
                "fluid_type": getattr(m, "fluid_type", None) if m.type == "FLUID" else None,
            }
            for m in obj.modifiers
        ]
        raw_ff = obj.get("flip_fluid")
        has_raw_ff = raw_ff is not None

        entry = {
            "name": obj.name,
            "type": obj.type,
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
            "display_type": getattr(obj, "display_type", None),
            "location": [round(v, 6) for v in obj.location],
            "rotation_euler_deg": [round(math.degrees(v), 4) for v in obj.rotation_euler],
            "scale": [round(v, 6) for v in obj.scale],
            "dimensions": [round(v, 6) for v in obj.dimensions],
            "vertex_count": bbox_info["vertex_count"],
            "polygon_count": bbox_info["polygon_count"],
            "is_empty_mesh": bbox_info["is_empty_mesh"],
            "flip_fluid_type": ff_type,
            "has_raw_flip_fluid_idprop": has_raw_ff,
            "modifiers": mods,
            "materials": mat_names,
            "world_bbox": {
                "min": bbox_info["min"],
                "max": bbox_info["max"],
                "center": bbox_info["center"],
                "size": bbox_info["size"],
            },
        }
        objects_info.append(entry)
        all_bboxes.append((obj.name, bbox_info))

        if obj.type == "MESH" and not bbox_info["is_empty_mesh"]:
            mesh_bboxes.append((obj.name, bbox_info))
            if ff_type != "TYPE_DOMAIN" and not obj.hide_render:
                mesh_non_domain_bboxes.append((obj.name, bbox_info))

    world_bbox = {
        "meshes_only": merge_bboxes(mesh_bboxes),
        "meshes_renderable_excluding_domain": merge_bboxes(mesh_non_domain_bboxes),
        "all_objects": merge_bboxes(all_bboxes),
    }
    return objects_info, world_bbox


def inspect_cameras(scene: bpy.types.Scene) -> dict:
    def camera_summary(obj: bpy.types.Object) -> dict:
        cam = obj.data
        return {
            "name": obj.name,
            "location": [round(v, 6) for v in obj.location],
            "rotation_euler_rad": [round(v, 6) for v in obj.rotation_euler],
            "rotation_euler_deg": [round(math.degrees(v), 4) for v in obj.rotation_euler],
            "type": getattr(cam, "type", None),
            "lens": getattr(cam, "lens", None),
            "ortho_scale": getattr(cam, "ortho_scale", None),
            "sensor_width": getattr(cam, "sensor_width", None),
            "clip_start": getattr(cam, "clip_start", None),
            "clip_end": getattr(cam, "clip_end", None),
        }

    all_cams = [camera_summary(o) for o in scene.objects if o.type == "CAMERA"]
    active = camera_summary(scene.camera) if scene.camera is not None else None
    return {
        "active_camera": active,
        "all_cameras": all_cams,
    }


def inspect_lights_and_world(scene: bpy.types.Scene) -> dict:
    lights = []
    for obj in scene.objects:
        if obj.type != "LIGHT" or obj.data is None:
            continue
        ld = obj.data
        lights.append({
            "name": obj.name,
            "light_type": ld.type,
            "energy": getattr(ld, "energy", None),
            "color": [round(float(c), 4) for c in getattr(ld, "color", (1.0, 1.0, 1.0))],
            "angle_deg": round(math.degrees(ld.angle), 4) if hasattr(ld, "angle") else None,
            "size": getattr(ld, "size", None),
            "shadow_soft_size": getattr(ld, "shadow_soft_size", None),
            "location": [round(v, 6) for v in obj.location],
            "rotation_euler_deg": [round(math.degrees(v), 4) for v in obj.rotation_euler],
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
        })

    world_info: dict = {"name": None, "has_node_tree": False, "nodes": []}
    if scene.world is not None:
        w = scene.world
        world_info["name"] = w.name
        nt = getattr(w, "node_tree", None)
        world_info["has_node_tree"] = nt is not None
        if nt is not None:
            for node in nt.nodes:
                n_info = {"name": node.name, "type": node.type, "bl_idname": node.bl_idname}
                if node.type == "BACKGROUND":
                    n_info["color"] = [round(float(c), 4) for c in node.inputs[0].default_value]
                    n_info["strength"] = float(node.inputs[1].default_value)
                elif node.type == "TEX_ENVIRONMENT" and getattr(node, "image", None) is not None:
                    img = node.image
                    n_info["image_name"] = img.name
                    n_info["image_filepath"] = img.filepath
                world_info["nodes"].append(n_info)

    return {"lights": lights, "world": world_info}


def inspect_render_settings(scene: bpy.types.Scene) -> dict:
    r = scene.render
    info: dict = {
        "engine": r.engine,
        "resolution_x": r.resolution_x,
        "resolution_y": r.resolution_y,
        "resolution_percentage": r.resolution_percentage,
        "fps": r.fps,
        "fps_base": r.fps_base,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "frame_step": scene.frame_step,
        "frame_current": scene.frame_current,
        "filepath": r.filepath,
        "file_format": r.image_settings.file_format,
        "use_persistent_data": bool(getattr(r, "use_persistent_data", False)),
    }

    if hasattr(scene, "cycles"):
        c = scene.cycles
        info["cycles"] = {
            "device": getattr(c, "device", None),
            "samples": getattr(c, "samples", None),
            "preview_samples": getattr(c, "preview_samples", None),
            "use_adaptive_sampling": getattr(c, "use_adaptive_sampling", None),
            "adaptive_threshold": getattr(c, "adaptive_threshold", None),
            "use_denoising": getattr(c, "use_denoising", None),
            "denoiser": getattr(c, "denoiser", None),
            "denoising_input_passes": getattr(c, "denoising_input_passes", None),
            "max_bounces": getattr(c, "max_bounces", None),
            "diffuse_bounces": getattr(c, "diffuse_bounces", None),
            "glossy_bounces": getattr(c, "glossy_bounces", None),
            "transmission_bounces": getattr(c, "transmission_bounces", None),
            "transparent_max_bounces": getattr(c, "transparent_max_bounces", None),
            "volume_bounces": getattr(c, "volume_bounces", None),
            "use_texture_cache": getattr(c, "use_texture_cache", None),
        }

    vs = getattr(scene, "view_settings", None)
    ds = getattr(scene, "display_settings", None)
    info["color_management"] = {
        "display_device": getattr(ds, "display_device", None) if ds else None,
        "view_transform": getattr(vs, "view_transform", None) if vs else None,
        "look": getattr(vs, "look", None) if vs else None,
        "exposure": getattr(vs, "exposure", None) if vs else None,
        "gamma": getattr(vs, "gamma", None) if vs else None,
    }
    return info


def inspect_flip_fluids_scene(scene: bpy.types.Scene) -> dict:
    ff_scene = getattr(scene, "flip_fluid", None)
    if ff_scene is None:
        return {"available": False, "has_domain": False}

    domain_obj = ff_scene.get_domain_object()
    result: dict = {
        "available": True,
        "has_domain": domain_obj is not None,
        "domain_object_name": domain_obj.name if domain_obj is not None else None,
        "num_domain_objects": ff_scene.get_num_domain_objects(),
        "domain_properties": None,
        "fluids": [],
        "inflows": [],
        "obstacles": [],
        "outflows": [],
        "force_fields": [],
        "cache_objects": [],
    }

    if domain_obj is not None:
        dprops = ff_scene.get_domain_properties()
        sim = dprops.simulation
        cache = dprops.cache
        surf = dprops.surface
        ww = dprops.whitewater
        mats = dprops.materials
        ren = dprops.render

        try:
            resolved_start, resolved_end = sim.get_frame_range()
        except Exception:
            resolved_start, resolved_end = scene.frame_start, scene.frame_end

        cache_raw = getattr(cache, "cache_directory", "")
        cache_is_win = is_windows_style_path(cache_raw)
        cache_posix = cache_raw.replace("\\", "/") if cache_raw else ""
        try:
            cache_abspath = cache.get_cache_abspath() if not cache_is_win else cache_raw
        except Exception:
            cache_abspath = cache_raw

        cache_exists = bool(not cache_is_win and cache_abspath and os.path.isdir(cache_abspath))
        bakefiles_dir = os.path.join(cache_abspath, "bakefiles") if cache_exists else ""
        bakefiles_exists = bool(bakefiles_dir and os.path.isdir(bakefiles_dir))

        result["domain_properties"] = {
            "resolution": int(sim.resolution),
            "preview_resolution": int(sim.preview_resolution),
            "auto_preview_resolution": bool(sim.auto_preview_resolution),
            "lock_cell_size": bool(sim.lock_cell_size),
            "voxel_grid": {
                "current_isize": int(sim.current_isize),
                "current_jsize": int(sim.current_jsize),
                "current_ksize": int(sim.current_ksize),
                "current_dx": float(sim.current_dx),
            },
            "frame_range_mode": str(sim.frame_range_mode),
            "frame_range_custom": [
                int(sim.frame_range_custom.value_min),
                int(sim.frame_range_custom.value_max),
            ],
            "frame_start": int(resolved_start),
            "frame_end": int(resolved_end),
            "raw_frame_start": int(sim.frame_start),
            "raw_frame_end": int(sim.frame_end),
            "frame_rate_mode": str(sim.frame_rate_mode),
            "frame_rate": float(sim.get_frame_rate()) if hasattr(sim, "get_frame_rate") else scene.render.fps,
            "cache_directory_raw": cache_raw,
            "cache_directory_posix": cache_posix,
            "cache_directory_abspath": cache_abspath,
            "cache_directory_is_windows_style": cache_is_win,
            "cache_directory_exists": cache_exists,
            "bakefiles_exists": bakefiles_exists,
            "is_cache_directory_set": bool(cache.is_cache_directory_set),
            "cache_status": {
                "bakefiles_exist": bool(cache.cache_status.bakefiles_exist),
                "particle_bakefiles_exist": bool(cache.cache_status.particle_bakefiles_exist),
            },
            "surface": {
                "enable_surface_mesh_generation": bool(surf.enable_surface_mesh_generation),
                "subdivisions": int(surf.subdivisions),
                "compute_chunk_mode": str(surf.compute_chunk_mode),
                "meshing_volume_mode": str(surf.meshing_volume_mode),
                "remove_mesh_near_domain": bool(surf.remove_mesh_near_domain),
                "smoothing_value": float(getattr(surf, "smoothing_value", 0.0)),
                "smoothing_iterations": int(getattr(surf, "smoothing_iterations", 0)),
                "enable_velocity_vector_attribute": bool(getattr(surf, "enable_velocity_vector_attribute", False)),
                "enable_speed_attribute": bool(getattr(surf, "enable_speed_attribute", False)),
            },
            "whitewater": {
                "enable_whitewater_simulation": bool(ww.enable_whitewater_simulation),
                "enable_foam": bool(ww.enable_foam),
                "enable_bubbles": bool(ww.enable_bubbles),
                "enable_spray": bool(ww.enable_spray),
                "enable_dust": bool(ww.enable_dust),
                "whitewater_render_display": str(ren.whitewater_render_display),
                "render_foam_pct": int(ren.render_foam_pct),
                "render_bubble_pct": int(ren.render_bubble_pct),
                "render_spray_pct": int(ren.render_spray_pct),
                "foam_particle_scale": float(ren.foam_particle_scale),
                "bubble_particle_scale": float(ren.bubble_particle_scale),
                "spray_particle_scale": float(ren.spray_particle_scale),
            },
            "materials": {
                "surface_material": str(mats.surface_material),
                "fluid_particles_material": str(mats.fluid_particles_material),
                "whitewater_foam_material": str(mats.whitewater_foam_material),
                "whitewater_bubble_material": str(mats.whitewater_bubble_material),
                "whitewater_spray_material": str(mats.whitewater_spray_material),
                "whitewater_dust_material": str(mats.whitewater_dust_material),
            },
            "render_settings": {
                "render_display": str(ren.render_display),
                "viewport_display": str(ren.viewport_display),
                "simulation_playback_mode": str(ren.simulation_playback_mode),
            },
        }

    for obj in ff_scene.get_fluid_objects():
        fp = obj.flip_fluid.fluid
        result["fluids"].append({
            "name": obj.name,
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
            "fluid_velocity_mode": str(fp.fluid_velocity_mode),
            "viscosity": float(fp.viscosity),
            "density": float(fp.density),
            "export_animated_mesh": bool(fp.export_animated_mesh),
        })

    for obj in ff_scene.get_inflow_objects():
        ip = obj.flip_fluid.inflow
        result["inflows"].append({
            "name": obj.name,
            "is_enabled": bool(ip.is_enabled),
            "substep_emissions": int(ip.substep_emissions),
            "inflow_velocity_mode": str(ip.inflow_velocity_mode),
            "inflow_speed": float(ip.inflow_speed),
            "constrain_fluid_velocity": bool(ip.constrain_fluid_velocity),
            "export_animated_mesh": bool(ip.export_animated_mesh),
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
        })

    for obj in ff_scene.get_obstacle_objects():
        op = obj.flip_fluid.obstacle
        result["obstacles"].append({
            "name": obj.name,
            "is_enabled": bool(op.is_enabled),
            "is_inversed": bool(op.is_inversed),
            "friction": float(op.friction),
            "velocity_scale": float(op.velocity_scale),
            "whitewater_influence": float(op.whitewater_influence),
            "mesh_expansion": float(op.mesh_expansion),
            "export_animated_mesh": bool(op.export_animated_mesh),
            "hide_viewport": bool(obj.hide_viewport),
            "hide_render": bool(obj.hide_render),
        })

    for obj in ff_scene.get_outflow_objects():
        result["outflows"].append({"name": obj.name})

    for obj in ff_scene.get_force_field_objects():
        result["force_fields"].append({"name": obj.name})

    for cname in (
        "fluid_surface",
        "fluid_particles",
        "whitewater_foam",
        "whitewater_bubble",
        "whitewater_spray",
        "whitewater_dust",
    ):
        obj = bpy.data.objects.get(cname)
        if obj is not None:
            result["cache_objects"].append({
                "name": obj.name,
                "in_scene": obj.name in scene.objects,
                "hide_render": bool(obj.hide_render),
                "materials": [s.material.name if s.material else None for s in obj.material_slots],
                "modifiers": [m.type for m in obj.modifiers],
            })

    inactive_flip_objects = []
    mantaflow_modifiers = []
    for obj in scene.objects:
        for m in obj.modifiers:
            if m.type in {"FLUID", "PARTICLE_SYSTEM"}:
                mantaflow_modifiers.append({
                    "object": obj.name,
                    "modifier": m.name,
                    "type": m.type,
                    "fluid_type": getattr(m, "fluid_type", None) if m.type == "FLUID" else None,
                })
        raw_ff = obj.get("flip_fluid")
        ff_rna = getattr(obj, "flip_fluid", None)
        is_active = bool(getattr(ff_rna, "is_active", False)) if ff_rna is not None else False
        if raw_ff is not None and not is_active:
            raw_dict = idprop_to_python(raw_ff)
            inactive_flip_objects.append({
                "name": obj.name,
                "is_active": is_active,
                "rna_object_type": str(getattr(ff_rna, "object_type", "TYPE_NONE")) if ff_rna else None,
                "raw_flip_fluid": raw_dict,
            })

    result["inactive_flip_objects"] = inactive_flip_objects
    result["mantaflow_or_particle_modifiers"] = mantaflow_modifiers
    return result


def scan_dict_paths(obj, prefix: str, out: list[dict]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            scan_dict_paths(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            scan_dict_paths(item, f"{prefix}[{idx}]", out)
    elif isinstance(obj, str) and obj:
        key_lower = prefix.lower()
        looks_like_path = (
            is_windows_style_path(obj)
            or obj.startswith("//")
            or ("/" in obj and any(tok in key_lower for tok in ("path", "dir", "file")))
        )
        if looks_like_path:
            win = is_windows_style_path(obj)
            abspath = resolve_blend_path(obj)
            exists = bool(not win and abspath and os.path.exists(abspath))
            out.append({
                "key": prefix,
                "value": obj,
                "is_windows_path": win,
                "exists_on_disk": exists,
            })


def inspect_external_paths(scene: bpy.types.Scene, flip_info: dict) -> dict:
    libraries = []
    for lib in bpy.data.libraries:
        raw = lib.filepath
        abspath = resolve_blend_path(raw)
        win = is_windows_style_path(raw)
        exists = bool(not win and abspath and os.path.exists(abspath))
        linked_items = []
        for coll_name in ("materials", "objects", "meshes", "node_groups", "images"):
            coll = getattr(bpy.data, coll_name, None)
            if coll is None:
                continue
            for db in coll:
                if getattr(db, "library", None) == lib:
                    linked_items.append(f"{coll_name}:{db.name}")
        libraries.append({
            "name": lib.name,
            "filepath_raw": raw,
            "filepath_abspath": abspath,
            "is_windows_path": win,
            "exists_on_disk": exists,
            "linked_datablocks": linked_items,
        })

    images = []
    for img in bpy.data.images:
        if img.source in {"GENERATED", "VIEWER"}:
            continue
        raw = img.filepath
        if not raw and img.packed_file is None:
            continue
        abspath = resolve_blend_path(raw)
        win = is_windows_style_path(raw)
        packed = img.packed_file is not None
        exists = packed or bool(not win and abspath and os.path.exists(abspath))
        images.append({
            "name": img.name,
            "source": img.source,
            "packed": packed,
            "filepath_raw": raw,
            "filepath_abspath": abspath,
            "is_windows_path": win,
            "exists_on_disk": exists,
        })

    sounds = []
    for snd in bpy.data.sounds:
        raw = snd.filepath
        abspath = resolve_blend_path(raw)
        win = is_windows_style_path(raw)
        packed = snd.packed_file is not None
        exists = packed or bool(not win and abspath and os.path.exists(abspath))
        sounds.append({
            "name": snd.name,
            "filepath_raw": raw,
            "is_windows_path": win,
            "exists_on_disk": exists,
        })

    cache_files = []
    for cf in bpy.data.cache_files:
        raw = cf.filepath
        abspath = resolve_blend_path(raw)
        win = is_windows_style_path(raw)
        exists = bool(not win and abspath and os.path.exists(abspath))
        cache_files.append({
            "name": cf.name,
            "filepath_raw": raw,
            "is_windows_path": win,
            "exists_on_disk": exists,
        })

    missing_summary = []
    for lib in libraries:
        if not lib["exists_on_disk"]:
            missing_summary.append({
                "kind": "library",
                "name": lib["name"],
                "filepath": lib["filepath_raw"],
                "linked_datablocks": lib["linked_datablocks"],
            })
    for img in images:
        if not img["exists_on_disk"]:
            missing_summary.append({
                "kind": "image",
                "name": img["name"],
                "filepath": img["filepath_raw"],
            })
    for snd in sounds:
        if not snd["exists_on_disk"]:
            missing_summary.append({
                "kind": "sound",
                "name": snd["name"],
                "filepath": snd["filepath_raw"],
            })
    for cf in cache_files:
        if not cf["exists_on_disk"]:
            missing_summary.append({
                "kind": "cache_file",
                "name": cf["name"],
                "filepath": cf["filepath_raw"],
            })

    dprops = flip_info.get("domain_properties") or {}
    if dprops and (dprops.get("cache_directory_is_windows_style") or not dprops.get("cache_directory_exists")):
        missing_summary.append({
            "kind": "flip_cache_directory",
            "name": flip_info.get("domain_object_name"),
            "filepath": dprops.get("cache_directory_raw"),
        })

    flip_idprop_paths: list[dict] = []
    for obj in scene.objects:
        raw_ff = obj.get("flip_fluid")
        if raw_ff is not None:
            scan_dict_paths(idprop_to_python(raw_ff), f"objects[{obj.name}].flip_fluid", flip_idprop_paths)
    raw_scene_ff = scene.get("flip_fluid")
    if raw_scene_ff is not None:
        scan_dict_paths(idprop_to_python(raw_scene_ff), "scene.flip_fluid", flip_idprop_paths)

    for entry in flip_idprop_paths:
        if entry["is_windows_path"] or not entry["exists_on_disk"]:
            missing_summary.append({
                "kind": "flip_idprop_path",
                "name": entry["key"],
                "filepath": entry["value"],
                "is_windows_path": entry["is_windows_path"],
            })

    return {
        "libraries": libraries,
        "images": images,
        "sounds": sounds,
        "cache_files": cache_files,
        "flip_idprop_paths": flip_idprop_paths,
        "missing_items": missing_summary,
    }


def inspect_materials(scene: bpy.types.Scene) -> dict:
    materials_list = []
    for mat in bpy.data.materials:
        lib = getattr(mat, "library", None)
        lib_raw = lib.filepath if lib else None
        lib_missing = False
        if lib_raw:
            lib_abs = resolve_blend_path(lib_raw)
            lib_missing = is_windows_style_path(lib_raw) or not os.path.exists(lib_abs)

        nt = getattr(mat, "node_tree", None)
        use_nodes = bool(getattr(mat, "use_nodes", nt is not None))
        node_types = []
        undefined_nodes = []
        missing_group_trees = []
        broken_image_nodes = []
        has_output_node = False

        if nt is not None:
            for node in nt.nodes:
                node_types.append(node.bl_idname)
                if node.bl_idname == "NodeUndefined" or (node.type == "CUSTOM" and not node.bl_idname):
                    undefined_nodes.append(node.name)
                if node.type == "GROUP" and getattr(node, "node_tree", None) is None:
                    missing_group_trees.append(node.name)
                if node.type == "TEX_IMAGE":
                    img = getattr(node, "image", None)
                    if img is None:
                        broken_image_nodes.append({"node": node.name, "reason": "no_image"})
                    elif img.source not in {"GENERATED", "VIEWER"} and img.packed_file is None:
                        img_abs = resolve_blend_path(img.filepath)
                        if is_windows_style_path(img.filepath) or not os.path.exists(img_abs):
                            broken_image_nodes.append({
                                "node": node.name,
                                "image": img.name,
                                "filepath": img.filepath,
                                "reason": "missing_file",
                            })
                if node.type == "OUTPUT_MATERIAL":
                    has_output_node = True

        if lib_missing:
            status = "missing_linked_library"
        elif undefined_nodes or missing_group_trees or broken_image_nodes:
            status = "broken_nodes"
        elif nt is None:
            status = "no_node_tree"
        elif not has_output_node:
            status = "no_output_node"
        else:
            status = "ok"

        materials_list.append({
            "name": mat.name,
            "users": int(mat.users),
            "is_linked": lib is not None,
            "library_filepath": lib_raw,
            "library_missing": lib_missing,
            "use_nodes": use_nodes,
            "has_node_tree": nt is not None,
            "node_count": len(node_types),
            "node_types": sorted(set(node_types)),
            "undefined_nodes": undefined_nodes,
            "missing_group_trees": missing_group_trees,
            "broken_image_nodes": broken_image_nodes,
            "has_output_node": has_output_node,
            "status": status,
        })

    slot_issues = []
    for obj in scene.objects:
        for idx, slot in enumerate(obj.material_slots):
            if slot.material is None:
                slot_issues.append({
                    "object": obj.name,
                    "slot_index": idx,
                    "issue": "empty_material_slot",
                })
            else:
                mat_entry = next((m for m in materials_list if m["name"] == slot.material.name), None)
                if mat_entry and mat_entry["status"] != "ok":
                    slot_issues.append({
                        "object": obj.name,
                        "slot_index": idx,
                        "material": slot.material.name,
                        "issue": mat_entry["status"],
                    })

    return {
        "materials": materials_list,
        "object_slot_issues": slot_issues,
    }


def main() -> int:
    args = parse_args()
    scene = bpy.context.scene

    if args.frame is not None:
        scene.frame_set(args.frame)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    flip_addon = find_flip_fluids_addon(scene)
    objects_info, world_bbox = inspect_objects_and_bounds(scene, depsgraph)
    camera_info = inspect_cameras(scene)
    light_world_info = inspect_lights_and_world(scene)
    render_info = inspect_render_settings(scene)
    flip_scene = inspect_flip_fluids_scene(scene)
    external_paths = inspect_external_paths(scene, flip_scene)
    material_diag = inspect_materials(scene)

    report = {
        "blender": {
            "version": list(bpy.app.version),
            "version_string": bpy.app.version_string,
            "blend_filepath": bpy.data.filepath,
            "blend_file_version": list(bpy.data.version),
        },
        "flip_fluids_addon": flip_addon,
        "render_settings": render_info,
        "active_camera": camera_info["active_camera"],
        "all_cameras": camera_info["all_cameras"],
        "lights": light_world_info["lights"],
        "world": light_world_info["world"],
        "world_bbox": world_bbox,
        "object_counts": {
            "scene_objects": len(scene.objects),
            "data_objects": len(bpy.data.objects),
        },
        "objects": objects_info,
        "flip_fluids_scene": flip_scene,
        "missing_external_paths": external_paths,
        "material_diagnostics": material_diag,
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        print(json.dumps(report, indent=2))
    else:
        print(f"[inspect_scene] Inspected {bpy.data.filepath} -> {args.json_out or 'stdout'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
