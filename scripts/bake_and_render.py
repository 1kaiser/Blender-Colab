#!/usr/bin/env python3
"""FLIP Fluids v1.8.9 + Blender 5.3 Simulation Bake & GPU Persistent-Data Render Orchestrator.

Modes:
  - sanitize          : Fix stale Windows paths / missing linked materials, save working .blend
  - bake              : Headless FLIP Fluids simulation bake (`bake_fluid_simulation_cmd`)
  - render_persistent : Single-process GPU OptiX render (`use_persistent_data=True`), supports sharding & resume
  - interleaved       : Overlapping CPU FLIP bake + GPU persistent render watching `finished######.txt`
  - sequential        : Run bake then render_persistent back-to-back

Also supports `--assemble-video` (default on for full runs) to build `animation.mp4`
and a compact `< 1.5 MiB` `animation_preview.gif` via ffmpeg.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import bpy  # type: ignore
except ImportError:
    bpy = None

import gpu_setup  # noqa: E402


WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]|^\\\\")


def default_work_dir() -> Path:
    return Path("/content/work") if Path("/content").is_dir() else Path("./work").resolve()


def find_blender_executable(explicit: str = "") -> str:
    """Resolve Blender executable path if --blender is omitted."""
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
    return sys.argv[1:]


# ---------------------------------------------------------------------------
# Blender-side helpers (only run when bpy is available)
# ---------------------------------------------------------------------------

def _infer_dormant_flip_type(obj: Any) -> Optional[str]:
    raw_ff = obj.get("flip_fluid") if hasattr(obj, "get") else None
    if raw_ff is not None:
        raw_ot = raw_ff.get("object_type") if hasattr(raw_ff, "get") else None
        if isinstance(raw_ot, str) and raw_ot.startswith("TYPE_") and raw_ot != "TYPE_NONE":
            return raw_ot
        if isinstance(raw_ot, int) and raw_ot != 0:
            ff_prop = getattr(obj, "flip_fluid", None)
            if ff_prop is not None and hasattr(ff_prop, "bl_rna"):
                for item in ff_prop.bl_rna.properties["object_type"].enum_items:
                    if item.value == raw_ot and item.identifier != "TYPE_NONE":
                        return item.identifier
        for key, mapped in (
            ("domain", "TYPE_DOMAIN"),
            ("inflow", "TYPE_INFLOW"),
            ("outflow", "TYPE_OUTFLOW"),
            ("obstacle", "TYPE_OBSTACLE"),
            ("force_field", "TYPE_FORCE_FIELD"),
        ):
            if key in raw_ff:
                return mapped

    low = (obj.name or "").lower()
    if "water boundary" in low or "fluid_domain" in low or low == "domain":
        return "TYPE_DOMAIN"
    if "inlet" in low or "inflow" in low:
        return "TYPE_INFLOW"
    if "outlet" in low or "outflow" in low:
        return "TYPE_OUTFLOW"
    if "culvert" in low or "obstacle" in low:
        return "TYPE_OBSTACLE"
    return None


def hide_flip_helper_objects(scene: Any) -> list[str]:
    """Ensure FLIP domain, inflow, and outflow helper meshes are hidden from renders."""
    hidden: list[str] = []
    for obj in bpy.data.objects:
        ff = getattr(obj, "flip_fluid", None)
        if ff is None:
            continue
        if getattr(ff, "is_active", False) and getattr(ff, "object_type", "") in {
            "TYPE_DOMAIN",
            "TYPE_INFLOW",
            "TYPE_OUTFLOW",
        }:
            if not obj.hide_render:
                obj.hide_render = True
                hidden.append(obj.name)
    return hidden


def reactivate_dormant_flip_objects(scene: Any, cache_dir: str = "") -> list[str]:
    """Reactivate dormant FLIP Fluids objects (e.g. particles.blend with Mantaflow modifiers)."""
    ff_scene = getattr(scene, "flip_fluid", None)
    if ff_scene is not None:
        try:
            if ff_scene.get_domain_object() is not None:
                hide_flip_helper_objects(scene)
                return []
        except Exception:
            pass

    reactivated: list[str] = []
    for obj in bpy.data.objects:
        if not hasattr(obj, "flip_fluid"):
            continue
        target_type = _infer_dormant_flip_type(obj)
        if not target_type:
            continue

        # Remove conflicting Mantaflow FLUID / PARTICLE_SYSTEM modifiers if present
        for mod in list(obj.modifiers):
            if mod.type in {"FLUID", "PARTICLE_SYSTEM"}:
                try:
                    obj.modifiers.remove(mod)
                except Exception:
                    pass

        try:
            obj.flip_fluid.is_active = True
            obj.flip_fluid.object_type = target_type
            if target_type in {"TYPE_DOMAIN", "TYPE_INFLOW", "TYPE_OUTFLOW"}:
                obj.hide_render = True
            if target_type == "TYPE_DOMAIN":
                dprops = getattr(obj.flip_fluid, "domain", None)
                if dprops is not None:
                    if cache_dir:
                        dprops.cache.cache_directory = os.path.abspath(cache_dir)
                    if hasattr(dprops, "bake") and hasattr(dprops.bake, "export_filepath"):
                        raw_exp = getattr(dprops.bake, "export_filepath", "") or ""
                        if WINDOWS_PATH_RE.match(raw_exp) or "\\" in raw_exp:
                            dprops.bake.export_filepath = ""
            reactivated.append(f"{obj.name}->{target_type}")
        except Exception as exc:
            print(f"[sanitize] Could not reactivate {obj.name}: {exc}", flush=True)

    hide_flip_helper_objects(scene)
    if reactivated:
        print(f"[sanitize] Reactivated dormant FLIP objects: {reactivated}", flush=True)
    return reactivated


def _get_domain_props(scene: Any) -> tuple[Optional[Any], Optional[Any]]:
    ff_scene = getattr(scene, "flip_fluid", None)
    if ff_scene is not None:
        try:
            dom_obj = ff_scene.get_domain_object()
            dprops = ff_scene.get_domain_properties()
            if dom_obj is not None and dprops is not None:
                return dom_obj, dprops
        except Exception:
            pass
    for obj in bpy.data.objects:
        ff = getattr(obj, "flip_fluid", None)
        if ff is not None and getattr(ff, "is_active", False) and getattr(ff, "object_type", "") == "TYPE_DOMAIN":
            return obj, getattr(ff, "domain", None)
    return None, None


def _apply_frame_range_and_res(
    scene: Any,
    dprops: Optional[Any],
    start_frame: Optional[int],
    end_frame: Optional[int],
    resolution: int,
    cache_dir: str,
    bake_threads: int = 0,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if start_frame is not None:
        scene.frame_start = int(start_frame)
    if end_frame is not None:
        scene.frame_end = int(end_frame)
    info["frame_start"] = int(scene.frame_start)
    info["frame_end"] = int(scene.frame_end)

    if dprops is None:
        return info

    if cache_dir:
        abs_cache = os.path.abspath(cache_dir)
        os.makedirs(abs_cache, exist_ok=True)
        dprops.cache.cache_directory = abs_cache
        info["cache_directory"] = abs_cache

    sim = getattr(dprops, "simulation", None)
    if sim is not None:
        if start_frame is not None or end_frame is not None:
            if hasattr(sim, "frame_range_custom"):
                try:
                    sim.frame_range_custom.value_min = int(scene.frame_start)
                    sim.frame_range_custom.value_max = int(scene.frame_end)
                except Exception:
                    pass
            if hasattr(sim, "frame_range_mode"):
                try:
                    sim.frame_range_mode = "FRAME_RANGE_TIMELINE"
                except Exception:
                    pass
        if resolution and int(resolution) > 0 and hasattr(sim, "resolution"):
            sim.resolution = int(resolution)
        info["resolution"] = getattr(sim, "resolution", None)

    if bake_threads and int(bake_threads) > 0 and hasattr(dprops, "advanced"):
        adv = dprops.advanced
        if hasattr(adv, "threading_mode"):
            try:
                adv.threading_mode = "THREADING_MODE_FIXED"
            except Exception:
                pass
        if hasattr(adv, "num_threads_fixed"):
            try:
                adv.num_threads_fixed = int(bake_threads)
                info["bake_threads"] = int(bake_threads)
            except Exception:
                pass

    return info


def _set_bsdf_input(bsdf: Any, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        sock = bsdf.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return


def _classify_material_preset(name: str) -> str:
    low = (name or "").lower()
    if "bubble" in low:
        return "bubble"
    if "foam" in low:
        return "foam"
    if "spray" in low:
        return "spray"
    if "dust" in low:
        return "dust"
    return "water"


def _populate_principled_material(mat: Any, preset: str) -> None:
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out_node = nt.nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (300, 0)

    bsdf = nt.nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    if preset == "bubble":
        _set_bsdf_input(bsdf, ("Base Color",), (0.92, 0.97, 1.0, 1.0))
        _set_bsdf_input(bsdf, ("Roughness",), 0.06)
        _set_bsdf_input(bsdf, ("IOR",), 1.333)
        _set_bsdf_input(bsdf, ("Transmission Weight", "Transmission"), 0.85)
    elif preset == "foam":
        _set_bsdf_input(bsdf, ("Base Color",), (0.95, 0.97, 0.99, 1.0))
        _set_bsdf_input(bsdf, ("Roughness",), 0.45)
        _set_bsdf_input(bsdf, ("Subsurface Weight", "Subsurface"), 0.20)
    elif preset == "spray":
        _set_bsdf_input(bsdf, ("Base Color",), (0.90, 0.96, 1.0, 1.0))
        _set_bsdf_input(bsdf, ("Roughness",), 0.08)
        _set_bsdf_input(bsdf, ("IOR",), 1.333)
        _set_bsdf_input(bsdf, ("Transmission Weight", "Transmission"), 0.75)
    elif preset == "dust":
        _set_bsdf_input(bsdf, ("Base Color",), (0.85, 0.88, 0.90, 1.0))
        _set_bsdf_input(bsdf, ("Roughness",), 0.65)
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
        return
    else:  # water / fluid surface fallback: cyan refractive Principled + bright shadow ray pass
        _set_bsdf_input(bsdf, ("Base Color",), (0.55, 0.88, 0.96, 1.0))
        _set_bsdf_input(bsdf, ("Roughness",), 0.03)
        _set_bsdf_input(bsdf, ("IOR",), 1.333)
        _set_bsdf_input(bsdf, ("Transmission Weight", "Transmission"), 0.94)

        transp = nt.nodes.new(type="ShaderNodeBsdfTransparent")
        transp.location = (0, -260)
        transp.inputs["Color"].default_value = (0.85, 0.96, 1.0, 1.0)

        lp = nt.nodes.new(type="ShaderNodeLightPath")
        lp.location = (-240, 220)

        mix = nt.nodes.new(type="ShaderNodeMixShader")
        mix.location = (180, 0)

        nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
        nt.links.new(bsdf.outputs["BSDF"], mix.inputs[1])
        nt.links.new(transp.outputs["BSDF"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out_node.inputs["Surface"])
        return

    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])


def _is_legacy_black_water_material(mat: Any) -> bool:
    """Detect bare Transparent-only water shaders (particles.blend) that render black in Cycles."""
    if mat is None or not getattr(mat, "use_nodes", False) or mat.node_tree is None:
        return False
    if mat.name.lower() not in {"water", "ff water"}:
        return False
    node_types = {n.type for n in mat.node_tree.nodes}
    return ("BSDF_TRANSPARENT" in node_types) and not bool(
        node_types & {"BSDF_PRINCIPLED", "BSDF_GLASS", "BSDF_REFRACTION"}
    )


def _is_library_missing(lib: Any) -> bool:
    raw = getattr(lib, "filepath", "") or ""
    if not raw:
        return True
    if WINDOWS_PATH_RE.match(raw) or "\\" in raw:
        return True
    try:
        resolved = bpy.path.abspath(raw)
        return not os.path.exists(resolved)
    except Exception:
        return True


def _is_material_broken(mat: Any, missing_libs: set[Any]) -> bool:
    if mat is None:
        return True
    if getattr(mat, "library", None) is not None and mat.library in missing_libs:
        return True
    if not getattr(mat, "use_nodes", False) or mat.node_tree is None:
        return mat.name.startswith(("FF ", "FF_")) or getattr(mat, "library", None) in missing_libs
    nodes = mat.node_tree.nodes
    if len(nodes) == 0:
        return True
    has_output = False
    for node in nodes:
        if node.type == "GROUP" and getattr(node, "node_tree", None) is None:
            return True
        if node.type == "OUTPUT_MATERIAL":
            surf = node.inputs.get("Surface")
            if surf is not None and surf.is_linked:
                has_output = True
    return not has_output


def _get_or_create_valid_material(preferred_names: tuple[str, ...], fallback_name: str, preset: str) -> Any:
    missing_libs = {lib for lib in bpy.data.libraries if _is_library_missing(lib)}
    for name in preferred_names:
        mat = bpy.data.materials.get(name)
        if mat is not None and not _is_material_broken(mat, missing_libs):
            return mat
    mat = bpy.data.materials.get(fallback_name)
    if mat is None:
        mat = bpy.data.materials.new(name=fallback_name)
    if _is_material_broken(mat, missing_libs):
        _populate_principled_material(mat, preset)
    return mat


def ensure_flip_object_materials(scene: Any) -> list[str]:
    """Ensure FLIP cache mesh objects have valid materials even if dprops enum is MATERIAL_NONE."""
    assigned: list[str] = []
    targets = [
        ("fluid_surface", ("Water", "FF Water", "FF Water Fallback"), "FF Water Fallback", "water"),
        ("whitewater_foam", ("FF Foam", "Foam"), "FF Foam", "foam"),
        ("whitewater_bubble", ("FF Bubble", "Bubble"), "FF Bubble", "bubble"),
        ("whitewater_spray", ("FF Spray", "Spray"), "FF Spray", "spray"),
        ("whitewater_dust", ("FF Dust", "Dust"), "FF Dust", "dust"),
    ]
    for obj_prefix, pref_names, fallback_name, preset in targets:
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj.data is None:
                continue
            if not obj.name.lower().startswith(obj_prefix):
                continue
            has_valid_slot = any(m is not None for m in obj.data.materials)
            if has_valid_slot:
                continue
            mat = _get_or_create_valid_material(pref_names, fallback_name, preset)
            if len(obj.data.materials) == 0:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat
            assigned.append(f"{obj.name}={mat.name}")
    return assigned


def repair_missing_materials(scene: Any, dprops: Optional[Any]) -> dict[str, Any]:
    """Repair broken linked external materials (FF Bubble/Foam/Spray/Water) while keeping valid ones."""
    missing_libs = {lib for lib in bpy.data.libraries if _is_library_missing(lib)}
    repaired: list[str] = []
    preserved: list[str] = []

    for mat in list(bpy.data.materials):
        if _is_legacy_black_water_material(mat):
            _populate_principled_material(mat, "water")
            repaired.append(f"{mat.name} (upgraded-bare-transparent->water)")
            continue
        if not _is_material_broken(mat, missing_libs):
            preserved.append(mat.name)
            continue

        preset = _classify_material_preset(mat.name)
        orig_name = mat.name
        if mat.library is not None:
            new_mat = bpy.data.materials.new(name=f"{orig_name}_Local")
            _populate_principled_material(new_mat, preset)
            mat.user_remap(new_mat)
            new_mat.name = orig_name
            repaired.append(f"{orig_name} (linked->{preset})")
        else:
            _populate_principled_material(mat, preset)
            repaired.append(f"{orig_name} (rebuilt->{preset})")

    # Ensure FLIP Fluids material slots use valid enum strings (never Material objects)
    if dprops is not None and hasattr(dprops, "materials"):
        mprops = dprops.materials
        mat_slots = [
            ("surface_material", ("Water", "FF Water", "FF Water Fallback"), "FF Water Fallback", "water"),
            ("whitewater_foam_material", ("FF Foam", "Foam"), "FF Foam", "foam"),
            ("whitewater_bubble_material", ("FF Bubble", "Bubble"), "FF Bubble", "bubble"),
            ("whitewater_spray_material", ("FF Spray", "Spray"), "FF Spray", "spray"),
            ("whitewater_dust_material", ("FF Dust", "Dust"), "FF Dust", "dust"),
        ]
        for attr, pref_names, fallback_name, preset in mat_slots:
            if not hasattr(mprops, attr):
                continue
            cur = getattr(mprops, attr, "")
            target_mat = _get_or_create_valid_material(pref_names, fallback_name, preset)
            if not cur or cur == "MATERIAL_NONE":
                try:
                    setattr(mprops, attr, target_mat.name)
                    repaired.append(f"dprops.materials.{attr}={target_mat.name}")
                except (TypeError, ValueError):
                    pass

    assigned_objs = ensure_flip_object_materials(scene)
    repaired.extend(assigned_objs)

    # Clean up broken external libraries with zero remaining users
    removed_libs: list[str] = []
    for lib in list(missing_libs):
        raw_fp = getattr(lib, "filepath", "")
        if lib.users == 0:
            try:
                bpy.data.libraries.remove(lib)
                removed_libs.append(raw_fp)
            except Exception:
                pass

    return {
        "repaired": repaired,
        "preserved_count": len(preserved),
        "removed_missing_libraries": removed_libs,
    }


def _tag_flip_meshes_updated(scene: Any) -> None:
    """Force Cycles Persistent Data to notice dynamic FLIP Fluids cache mesh changes."""
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        low = obj.name.lower()
        is_flip_mesh = any(
            tok in low
            for tok in (
                "fluid_surface",
                "fluid_particles",
                "whitewater_",
                "ff_surface",
            )
        )
        if is_flip_mesh:
            try:
                obj.data.update()
                obj.update_tag()
            except Exception:
                pass
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


def _initialize_flip_mesh_cache(dprops: Optional[Any]) -> None:
    if dprops is None:
        return
    mcache = getattr(dprops, "mesh_cache", None)
    if mcache is not None and hasattr(mcache, "initialize_cache_objects"):
        try:
            mcache.initialize_cache_objects()
        except Exception as exc:
            print(f"[bake_and_render] mesh_cache init note: {exc}", flush=True)


def _copy_atomic(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_dir / f".{src.name}.tmp"
    final_dst = dst_dir / src.name
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, final_dst)


def _write_json(path_str: str, data: dict[str, Any]) -> None:
    if not path_str:
        return
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _blender_internal_sanitize(args: argparse.Namespace) -> int:
    gpu_setup.ensure_flip_fluids_enabled(verbose=True)
    scene = bpy.context.scene
    reactivated = reactivate_dormant_flip_objects(scene, args.cache_dir)
    _, dprops = _get_domain_props(scene)

    cfg_info = _apply_frame_range_and_res(
        scene,
        dprops,
        args.start_frame,
        args.end_frame,
        args.resolution,
        args.cache_dir,
        args.bake_threads,
    )
    mat_info = repair_missing_materials(scene, dprops)

    out_blend = Path(args.sanitized_blend).resolve()
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))

    summary = {
        "action": "sanitize",
        "source_blend": bpy.data.filepath,
        "sanitized_blend": str(out_blend),
        "reactivated_objects": reactivated,
        "scene_config": cfg_info,
        "materials": mat_info,
    }
    print(f"[sanitize] Saved sanitized scene -> {out_blend}", flush=True)
    print(f"[sanitize] Materials repaired: {mat_info['repaired']}", flush=True)
    _write_json(args.json_out, summary)
    return 0


def _blender_internal_bake(args: argparse.Namespace) -> int:
    gpu_setup.ensure_flip_fluids_enabled(verbose=True)
    scene = bpy.context.scene
    _, dprops = _get_domain_props(scene)
    if dprops is None:
        reactivate_dormant_flip_objects(scene, args.cache_dir)
        _, dprops = _get_domain_props(scene)
    if dprops is None:
        raise RuntimeError("No FLIP Fluids domain object found in scene.")

    cfg_info = _apply_frame_range_and_res(
        scene,
        dprops,
        args.start_frame,
        args.end_frame,
        args.resolution,
        args.cache_dir,
        args.bake_threads,
    )

    if args.reset_bake:
        bakefiles_dir = Path(dprops.cache.cache_directory) / "bakefiles"
        if bakefiles_dir.is_dir():
            shutil.rmtree(bakefiles_dir, ignore_errors=True)
        try:
            bpy.ops.flip_fluid_operators.reset_bake()
        except Exception:
            pass

    t0 = time.perf_counter()
    print(
        f"[bake] Starting FLIP Fluids bake: frames {scene.frame_start}..{scene.frame_end}, "
        f"res={cfg_info.get('resolution')}, cache={dprops.cache.cache_directory}",
        flush=True,
    )
    bpy.ops.flip_fluid_operators.bake_fluid_simulation_cmd()
    bake_s = time.perf_counter() - t0

    bakefiles_dir = Path(dprops.cache.cache_directory) / "bakefiles"
    finished_files = sorted(bakefiles_dir.glob("finished*.txt")) if bakefiles_dir.is_dir() else []
    summary = {
        "action": "bake",
        "wall_s": round(bake_s, 4),
        "cache_directory": dprops.cache.cache_directory,
        "finished_count": len(finished_files),
        "scene_config": cfg_info,
    }
    print(
        f"[bake] Completed in {bake_s:.2f}s ({len(finished_files)} finished frame markers)",
        flush=True,
    )
    _write_json(args.json_out, summary)
    return 0


def _setup_blender_render_env(args: argparse.Namespace) -> tuple[Any, Optional[Any]]:
    gpu_setup.ensure_flip_fluids_enabled(verbose=True)
    scene = bpy.context.scene
    _, dprops = _get_domain_props(scene)
    if dprops is None:
        reactivate_dormant_flip_objects(scene, args.cache_dir)
        _, dprops = _get_domain_props(scene)

    if dprops is not None and args.cache_dir:
        dprops.cache.cache_directory = os.path.abspath(args.cache_dir)
    hide_flip_helper_objects(scene)
    repair_missing_materials(scene, dprops)
    _initialize_flip_mesh_cache(dprops)
    ensure_flip_object_materials(scene)

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
    scene.render.image_settings.file_format = "PNG"
    return scene, dprops


def _frame_output_path(out_dir: Path, prefix: str, frame: int) -> Path:
    return out_dir / f"{prefix}{frame:06d}.png"


def _render_single_frame_in_session(
    scene: Any,
    frame: int,
    out_png: Path,
    sync_dir: Optional[Path],
) -> float:
    t0 = time.perf_counter()
    scene.frame_set(int(frame))
    ensure_flip_object_materials(scene)
    _tag_flip_meshes_updated(scene)
    scene.render.filepath = str(out_png)
    bpy.ops.render.render(write_still=True)
    dt = time.perf_counter() - t0
    if not out_png.is_file() or out_png.stat().st_size == 0:
        raise RuntimeError(f"Render produced missing/empty file for frame {frame}: {out_png}")
    if sync_dir is not None:
        _copy_atomic(out_png, sync_dir)
    return dt


def _blender_internal_render_persistent(args: argparse.Namespace) -> int:
    scene, _ = _setup_blender_render_env(args)
    start_f = int(args.start_frame if args.start_frame is not None else scene.frame_start)
    end_f = int(args.end_frame if args.end_frame is not None else scene.frame_end)
    all_frames = list(range(start_f, end_f + 1))

    if args.num_workers < 1 or args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(
            f"Invalid worker sharding: worker_id={args.worker_id}, num_workers={args.num_workers}"
        )

    assigned = [f for idx, f in enumerate(all_frames) if idx % args.num_workers == args.worker_id]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_dir = Path(args.sync_dir).resolve() if args.sync_dir else None

    t_total0 = time.perf_counter()
    rendered: list[dict[str, Any]] = []
    skipped: list[int] = []

    for f in assigned:
        out_png = _frame_output_path(out_dir, args.file_prefix, f)
        if args.skip_existing and out_png.is_file() and out_png.stat().st_size > 1024:
            skipped.append(f)
            print(f"[render] Skip existing frame {f} -> {out_png.name}", flush=True)
            continue
        dt = _render_single_frame_in_session(scene, f, out_png, sync_dir)
        rendered.append({"frame": f, "wall_s": round(dt, 4), "path": str(out_png)})
        print(
            f"[render] Frame {f}/{end_f} rendered in {dt:.2f}s -> {out_png.name}",
            flush=True,
        )

    total_s = time.perf_counter() - t_total0
    summary = {
        "action": "render_persistent",
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "start_frame": start_f,
        "end_frame": end_f,
        "assigned_frames": assigned,
        "rendered_count": len(rendered),
        "skipped_count": len(skipped),
        "wall_s": round(total_s, 4),
        "frames": rendered,
    }
    _write_json(args.json_out, summary)
    return 0


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_bake_sentinel(sentinel_path: Path, bake_pid: int) -> tuple[bool, Optional[int]]:
    if sentinel_path.is_file():
        try:
            data = json.loads(sentinel_path.read_text(encoding="utf-8"))
            if data.get("done"):
                return True, data.get("returncode")
        except Exception:
            pass
    if bake_pid > 0 and not _is_pid_alive(bake_pid):
        return True, None
    return False, None


def _blender_internal_render_interleaved(args: argparse.Namespace) -> int:
    scene, dprops = _setup_blender_render_env(args)
    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else Path(dprops.cache.cache_directory).resolve()
    )
    bakefiles_dir = cache_dir / "bakefiles"
    sentinel_path = Path(args.bake_sentinel).resolve()

    start_f = int(args.start_frame if args.start_frame is not None else scene.frame_start)
    end_f = int(args.end_frame if args.end_frame is not None else scene.frame_end)
    all_frames = list(range(start_f, end_f + 1))
    assigned = [f for idx, f in enumerate(all_frames) if idx % args.num_workers == args.worker_id]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sync_dir = Path(args.sync_dir).resolve() if args.sync_dir else None

    t_total0 = time.perf_counter()
    last_activity = t_total0
    pending = list(assigned)
    rendered: list[dict[str, Any]] = []
    skipped: list[int] = []
    bake_rc: Optional[int] = None

    while pending:
        did_work = False
        for f in list(pending):
            marker = bakefiles_dir / f"finished{f:06d}.txt"
            if not marker.is_file():
                continue

            out_png = _frame_output_path(out_dir, args.file_prefix, f)
            if args.skip_existing and out_png.is_file() and out_png.stat().st_size > 1024:
                skipped.append(f)
                pending.remove(f)
                last_activity = time.perf_counter()
                did_work = True
                continue

            # Tiny settle delay to ensure filesystem metadata is visible
            time.sleep(0.02)
            dt = _render_single_frame_in_session(scene, f, out_png, sync_dir)
            rendered.append({"frame": f, "wall_s": round(dt, 4), "path": str(out_png)})
            pending.remove(f)
            last_activity = time.perf_counter()
            did_work = True
            print(
                f"[interleaved-render] Frame {f} baked & rendered in {dt:.2f}s "
                f"({len(pending)} remaining)",
                flush=True,
            )

        if not pending:
            break

        bake_done, bake_rc = _read_bake_sentinel(sentinel_path, args.bake_pid)
        if bake_done:
            # Final pass after bake exit in case remaining markers landed simultaneously
            for f in list(pending):
                marker = bakefiles_dir / f"finished{f:06d}.txt"
                if marker.is_file():
                    out_png = _frame_output_path(out_dir, args.file_prefix, f)
                    if args.skip_existing and out_png.is_file() and out_png.stat().st_size > 1024:
                        skipped.append(f)
                    else:
                        dt = _render_single_frame_in_session(scene, f, out_png, sync_dir)
                        rendered.append({"frame": f, "wall_s": round(dt, 4), "path": str(out_png)})
                    pending.remove(f)
            break

        if not did_work:
            if time.perf_counter() - last_activity > args.stall_timeout:
                print(
                    f"[interleaved-render] ERROR: Timed out after {args.stall_timeout}s "
                    f"waiting for frames {pending[:5]}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            time.sleep(args.poll_interval)

    total_s = time.perf_counter() - t_total0
    missing = list(pending)
    summary = {
        "action": "render_interleaved",
        "start_frame": start_f,
        "end_frame": end_f,
        "rendered_count": len(rendered),
        "skipped_count": len(skipped),
        "missing_frames": missing,
        "bake_returncode": bake_rc,
        "wall_s": round(total_s, 4),
        "frames": rendered,
    }
    _write_json(args.json_out, summary)
    if missing:
        print(
            f"[interleaved-render] Exiting with unrendered frames: {missing}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Host Python orchestration & FFmpeg assembly
# ---------------------------------------------------------------------------

def _forward_gpu_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--engine",
        args.engine,
        "--compute-type",
        args.compute_type,
        "--denoiser",
        args.denoiser,
        "--adaptive-threshold",
        str(args.adaptive_threshold),
    ]
    if args.include_cpu:
        flags.append("--include-cpu")
    if args.no_persistent_data:
        flags.append("--no-persistent-data")
    if args.no_denoise:
        flags.append("--no-denoise")
    if args.no_adaptive_sampling:
        flags.append("--no-adaptive-sampling")
    if args.samples is not None:
        flags.extend(["--samples", str(args.samples)])
    if args.no_texture_cache:
        flags.append("--no-texture-cache")
    return flags


def _common_scene_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--cache-dir",
        args.cache_dir,
        "--out-dir",
        args.out_dir,
        "--file-prefix",
        args.file_prefix,
        "--resolution",
        str(args.resolution),
        "--bake-threads",
        str(args.bake_threads),
        "--worker-id",
        str(args.worker_id),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.start_frame is not None:
        flags.extend(["--start-frame", str(args.start_frame)])
    if args.end_frame is not None:
        flags.extend(["--end-frame", str(args.end_frame)])
    if args.sync_dir:
        flags.extend(["--sync-dir", args.sync_dir])
    if args.skip_existing:
        flags.append("--skip-existing")
    if args.reset_bake:
        flags.append("--reset-bake")
    return flags


def _run_blender_subcommand(
    blender_bin: str,
    blend_path: str,
    internal_action: str,
    extra_args: list[str],
) -> int:
    cmd = [
        blender_bin,
        "-b",
        blend_path,
        "-P",
        str(Path(__file__).resolve()),
        "--",
        "--internal-action",
        internal_action,
        *extra_args,
    ]
    print(f"[orchestrator] Launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def assemble_video_and_gif(
    frames_dir: Path,
    file_prefix: str,
    mp4_path: Path,
    gif_path: Path,
    fps: int = 24,
    max_gif_bytes: int = 1_500_000,
) -> dict[str, Any]:
    """Assemble rendered PNG frames into H.264 MP4 and < 1.5 MiB preview GIF via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"assembled": False, "reason": "ffmpeg not found in PATH"}

    pngs = sorted(frames_dir.glob(f"{file_prefix}*.png"))
    if not pngs:
        return {"assembled": False, "reason": f"No PNG frames found in {frames_dir}"}

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        concat_list = Path(tf.name)
        frame_dur = 1.0 / max(1, fps)
        for p in pngs:
            escaped = str(p.resolve()).replace("'", "'\\''")
            tf.write(f"file '{escaped}'\n")
            tf.write(f"duration {frame_dur:.6f}\n")
        # ffmpeg concat demuxer requires the last file repeated without duration
        escaped_last = str(pngs[-1].resolve()).replace("'", "'\\''")
        tf.write(f"file '{escaped_last}'\n")

    try:
        mp4_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ]
        subprocess.run(mp4_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Build compact preview GIF strictly under max_gif_bytes
        gif_profiles = [
            (12, 640, 256),
            (10, 480, 192),
            (8, 400, 128),
            (6, 320, 96),
        ]
        palette_path = gif_path.with_suffix(".palette.png")
        try:
            for g_fps, g_width, g_colors in gif_profiles:
                vf_palette = (
                    f"fps={g_fps},scale={g_width}:-1:flags=lanczos,"
                    f"palettegen=max_colors={g_colors}:stats_mode=diff"
                )
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_list),
                        "-vf",
                        vf_palette,
                        str(palette_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_list),
                        "-i",
                        str(palette_path),
                        "-lavfi",
                        f"fps={g_fps},scale={g_width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
                        str(gif_path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if gif_path.is_file() and gif_path.stat().st_size <= max_gif_bytes:
                    break
        finally:
            if palette_path.exists():
                palette_path.unlink()
    finally:
        if concat_list.exists():
            concat_list.unlink()

    return {
        "assembled": True,
        "frame_count": len(pngs),
        "mp4_path": str(mp4_path),
        "mp4_bytes": mp4_path.stat().st_size if mp4_path.is_file() else 0,
        "gif_path": str(gif_path),
        "gif_bytes": gif_path.stat().st_size if gif_path.is_file() else 0,
    }


def run_interleaved_host(
    blender_bin: str,
    working_blend: str,
    args: argparse.Namespace,
    render_json: Path,
    bake_json: Path,
) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    bakefiles_dir = cache_dir / "bakefiles"
    if args.reset_bake and bakefiles_dir.is_dir():
        shutil.rmtree(bakefiles_dir, ignore_errors=True)

    sentinel_fd, sentinel_tmp = tempfile.mkstemp(prefix="flip_bake_sentinel_", suffix=".json")
    os.close(sentinel_fd)
    sentinel_path = Path(sentinel_tmp)
    _write_json(str(sentinel_path), {"done": False, "returncode": None, "started_at": time.time()})

    bake_extra = _common_scene_flags(args)
    # Remove --reset-bake from bake child since orchestrator already cleared cleanly above
    bake_extra = [x for x in bake_extra if x != "--reset-bake"]
    bake_extra.extend(["--json-out", str(bake_json)])

    bake_cmd = [
        blender_bin,
        "-b",
        working_blend,
        "-P",
        str(Path(__file__).resolve()),
        "--",
        "--internal-action",
        "bake",
        *bake_extra,
    ]
    t0 = time.perf_counter()
    print(f"[interleaved] Starting CPU bake: {' '.join(bake_cmd)}", flush=True)
    bake_proc = subprocess.Popen(bake_cmd)

    render_extra = [
        *_common_scene_flags(args),
        *_forward_gpu_flags(args),
        "--bake-sentinel",
        str(sentinel_path),
        "--bake-pid",
        str(bake_proc.pid),
        "--poll-interval",
        str(args.poll_interval),
        "--stall-timeout",
        str(args.stall_timeout),
        "--json-out",
        str(render_json),
    ]
    render_cmd = [
        blender_bin,
        "-b",
        working_blend,
        "-P",
        str(Path(__file__).resolve()),
        "--",
        "--internal-action",
        "render_interleaved",
        *render_extra,
    ]
    print(f"[interleaved] Starting GPU persistent renderer: {' '.join(render_cmd)}", flush=True)
    render_proc = subprocess.Popen(render_cmd)

    bake_rc: Optional[int] = None
    render_rc: Optional[int] = None
    bake_wall_s: Optional[float] = None
    try:
        while True:
            if bake_rc is None:
                polled_b = bake_proc.poll()
                if polled_b is not None:
                    bake_rc = int(polled_b)
                    bake_wall_s = time.perf_counter() - t0
                    _write_json(
                        str(sentinel_path),
                        {
                            "done": True,
                            "returncode": bake_rc,
                            "finished_at": time.time(),
                        },
                    )
                    print(
                        f"[interleaved] CPU bake exited rc={bake_rc} at {bake_wall_s:.2f}s",
                        flush=True,
                    )

            polled_r = render_proc.poll()
            if polled_r is not None:
                render_rc = int(polled_r)
                break

            time.sleep(0.2)

        if bake_rc is None:
            # Render exited before bake finished (error or premature exit)
            if render_rc != 0:
                print(
                    f"[interleaved] Render worker exited early rc={render_rc}; terminating bake",
                    file=sys.stderr,
                    flush=True,
                )
                bake_proc.terminate()
                bake_rc = bake_proc.wait(timeout=15)
            else:
                bake_rc = bake_proc.wait()
                bake_wall_s = time.perf_counter() - t0
    finally:
        for proc in (render_proc, bake_proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if sentinel_path.exists():
            sentinel_path.unlink()

    total_wall_s = time.perf_counter() - t0
    return {
        "mode": "interleaved",
        "wall_s": round(total_wall_s, 4),
        "bake_wall_s": round(bake_wall_s, 4) if bake_wall_s is not None else None,
        "bake_returncode": bake_rc,
        "render_returncode": render_rc,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    wdir = default_work_dir()
    parser = argparse.ArgumentParser(
        description="Orchestrate FLIP Fluids v1.8.9 bake and Blender 5.3 GPU persistent render."
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=[
            "bake",
            "bake_only",
            "render_persistent",
            "render_only",
            "interleaved",
            "sequential",
            "sanitize",
        ],
        help="Execution mode (default: sequential)",
    )
    parser.add_argument("--blender", default="", help="Path to Blender executable")
    parser.add_argument(
        "--blend",
        default=str(wdir / "particles.blend"),
        help="Input .blend scene file",
    )
    parser.add_argument(
        "--sanitized-blend",
        default="",
        help="Working .blend output path when --sanitize is enabled",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize legacy paths, cache directory, and missing materials before running",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(wdir / "flip_fluid_cache"),
        help="Fast local SSD directory for FLIP Fluids cache",
    )
    parser.add_argument(
        "--out-dir",
        default=str(wdir / "frames"),
        help="Directory for rendered PNG frames",
    )
    parser.add_argument(
        "--sync-dir",
        default="",
        help="Optional directory to atomically copy each rendered frame into",
    )
    parser.add_argument(
        "--file-prefix",
        default="frame_",
        help="Output frame filename prefix (default: frame_ -> frame_000001.png)",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Override start frame")
    parser.add_argument("--end-frame", type=int, default=None, help="Override end frame")
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="FLIP Fluids domain resolution override (0 keeps scene default)",
    )
    parser.add_argument("--bake-threads", type=int, default=0, help="Fixed CPU bake threads (0=auto)")
    parser.add_argument("--reset-bake", action="store_true", help="Clear cache/bakefiles before bake")

    # Render / GPU options
    parser.add_argument(
        "--engine",
        default="CYCLES",
        choices=["CYCLES", "EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
    )
    parser.add_argument(
        "--compute-type",
        default="AUTO",
        choices=["AUTO", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "NONE"],
    )
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--no-persistent-data", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument(
        "--denoiser",
        default="AUTO",
        choices=["AUTO", "OPENIMAGEDENOISE", "OPTIX", "NONE"],
    )
    parser.add_argument("--no-adaptive-sampling", action="store_true")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--adaptive-threshold", type=float, default=0.02)
    parser.add_argument("--no-texture-cache", action="store_true")

    # Distributed sharding & resume
    parser.add_argument("--worker-id", type=int, default=0, help="0-based worker shard index")
    parser.add_argument("--num-workers", type=int, default=1, help="Total worker shards")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing PNG frames")

    # Interleaved polling
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--stall-timeout", type=float, default=900.0)

    # Video / GIF assembly
    parser.add_argument("--video-out", default="", help="Output MP4 path (default: <out-dir>/animation.mp4)")
    parser.add_argument("--gif-out", default="", help="Output GIF path (default: <out-dir>/animation_preview.gif)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--no-video", action="store_true", help="Skip ffmpeg MP4/GIF assembly")

    parser.add_argument("--json-out", default="", help="Write structured run summary JSON")

    # Internal Blender worker options
    parser.add_argument("--internal-action", default="", help=argparse.SUPPRESS)
    parser.add_argument("--bake-sentinel", default="", help=argparse.SUPPRESS)
    parser.add_argument("--bake-pid", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(_blender_cli_argv())

    # Inside Blender subprocess mode
    if bpy is not None and args.internal_action:
        if args.internal_action == "sanitize":
            return _blender_internal_sanitize(args)
        if args.internal_action == "bake":
            return _blender_internal_bake(args)
        if args.internal_action == "render_persistent":
            return _blender_internal_render_persistent(args)
        if args.internal_action == "render_interleaved":
            return _blender_internal_render_interleaved(args)
        raise ValueError(f"Unknown internal action: {args.internal_action}")

    # Host Python orchestration mode
    mode = args.mode
    if mode == "bake_only":
        mode = "bake"
    elif mode == "render_only":
        mode = "render_persistent"

    blender_bin = find_blender_executable(args.blender)
    src_blend = str(Path(args.blend).resolve())
    if not os.path.isfile(src_blend):
        raise FileNotFoundError(f"Blend file not found: {src_blend}")

    sanitized_blend = (
        str(Path(args.sanitized_blend).resolve())
        if args.sanitized_blend
        else str(Path(src_blend).with_name(f"{Path(src_blend).stem}_sanitized.blend"))
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="blender_colab_run_"))
    overall: dict[str, Any] = {
        "mode": mode,
        "blender_bin": blender_bin,
        "source_blend": src_blend,
    }
    t_start = time.perf_counter()

    try:
        working_blend = src_blend
        if args.sanitize or mode == "sanitize":
            san_json = tmp_dir / "sanitize.json"
            rc = _run_blender_subcommand(
                blender_bin,
                src_blend,
                "sanitize",
                [
                    *_common_scene_flags(args),
                    "--sanitized-blend",
                    sanitized_blend,
                    "--json-out",
                    str(san_json),
                ],
            )
            if rc != 0:
                return rc
            working_blend = sanitized_blend
            if san_json.is_file():
                overall["sanitize"] = json.loads(san_json.read_text(encoding="utf-8"))
            if mode == "sanitize":
                overall["wall_s"] = round(time.perf_counter() - t_start, 4)
                _write_json(args.json_out, overall)
                return 0

        overall["working_blend"] = working_blend

        if mode == "bake":
            bake_json = tmp_dir / "bake.json"
            rc = _run_blender_subcommand(
                blender_bin,
                working_blend,
                "bake",
                [*_common_scene_flags(args), "--json-out", str(bake_json)],
            )
            if bake_json.is_file():
                overall["bake"] = json.loads(bake_json.read_text(encoding="utf-8"))
            overall["wall_s"] = round(time.perf_counter() - t_start, 4)
            _write_json(args.json_out, overall)
            return rc

        if mode == "render_persistent":
            render_json = tmp_dir / "render.json"
            rc = _run_blender_subcommand(
                blender_bin,
                working_blend,
                "render_persistent",
                [*_common_scene_flags(args), *_forward_gpu_flags(args), "--json-out", str(render_json)],
            )
            if render_json.is_file():
                overall["render"] = json.loads(render_json.read_text(encoding="utf-8"))
            if rc != 0:
                overall["wall_s"] = round(time.perf_counter() - t_start, 4)
                _write_json(args.json_out, overall)
                return rc

        elif mode == "sequential":
            bake_json = tmp_dir / "bake.json"
            render_json = tmp_dir / "render.json"
            rc_bake = _run_blender_subcommand(
                blender_bin,
                working_blend,
                "bake",
                [*_common_scene_flags(args), "--json-out", str(bake_json)],
            )
            if bake_json.is_file():
                overall["bake"] = json.loads(bake_json.read_text(encoding="utf-8"))
            if rc_bake != 0:
                overall["wall_s"] = round(time.perf_counter() - t_start, 4)
                _write_json(args.json_out, overall)
                return rc_bake

            rc_render = _run_blender_subcommand(
                blender_bin,
                working_blend,
                "render_persistent",
                [*_common_scene_flags(args), *_forward_gpu_flags(args), "--json-out", str(render_json)],
            )
            if render_json.is_file():
                overall["render"] = json.loads(render_json.read_text(encoding="utf-8"))
            if rc_render != 0:
                overall["wall_s"] = round(time.perf_counter() - t_start, 4)
                _write_json(args.json_out, overall)
                return rc_render

        elif mode == "interleaved":
            bake_json = tmp_dir / "bake.json"
            render_json = tmp_dir / "render.json"
            res = run_interleaved_host(blender_bin, working_blend, args, render_json, bake_json)
            overall["interleaved"] = res
            if bake_json.is_file():
                overall["bake"] = json.loads(bake_json.read_text(encoding="utf-8"))
            if render_json.is_file():
                overall["render"] = json.loads(render_json.read_text(encoding="utf-8"))
            if res["bake_returncode"] != 0 or res["render_returncode"] != 0:
                overall["wall_s"] = round(time.perf_counter() - t_start, 4)
                _write_json(args.json_out, overall)
                return res["render_returncode"] or res["bake_returncode"] or 1

        if not args.no_video and args.num_workers == 1:
            out_dir = Path(args.out_dir).resolve()
            mp4_path = Path(args.video_out).resolve() if args.video_out else out_dir / "animation.mp4"
            gif_path = Path(args.gif_out).resolve() if args.gif_out else out_dir / "animation_preview.gif"
            overall["video"] = assemble_video_and_gif(
                out_dir,
                args.file_prefix,
                mp4_path,
                gif_path,
                fps=args.fps,
            )

        overall["wall_s"] = round(time.perf_counter() - t_start, 4)
        _write_json(args.json_out, overall)
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
