#!/usr/bin/env python3
"""GPU / Render Engine Configuration, FLIP Fluids Bootstrap & EEVEE Backend Probe.

Works both as:
  1. An importable helper module (inside Blender Python or host Python for EEVEE probing)
  2. A standalone Blender startup script:
       blender -b scene.blend -P scripts/gpu_setup.py -- --engine CYCLES --samples 64
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

try:
    import addon_utils  # type: ignore
    import bpy  # type: ignore
except ImportError:
    addon_utils = None
    bpy = None


def _blender_cli_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:] if bpy is None else []


def ensure_flip_fluids_enabled(verbose: bool = True) -> Optional[str]:
    """Ensure FLIP Fluids addon (Blender 5.x extension or legacy) is registered and initialized.

    Returns the enabled addon module name (e.g. 'bl_ext.user_default.flip_fluids_addon'
    or 'flip_fluids_addon'), or None if not installed.
    """
    if bpy is None:
        raise RuntimeError("ensure_flip_fluids_enabled() requires Blender's bpy module")

    prefs = bpy.context.preferences
    candidate_names: list[str] = [
        "bl_ext.user_default.flip_fluids_addon",
        "flip_fluids_addon",
    ]

    if addon_utils is not None:
        for mod in addon_utils.modules():
            mod_name = getattr(mod, "__name__", "")
            if mod_name == "flip_fluids_addon" or mod_name.endswith(".flip_fluids_addon"):
                if mod_name not in candidate_names:
                    candidate_names.append(mod_name)

    for key in list(prefs.addons.keys()):
        if key == "flip_fluids_addon" or key.endswith(".flip_fluids_addon"):
            if key not in candidate_names:
                candidate_names.insert(0, key)

    enabled_mod_name: Optional[str] = None
    for name in candidate_names:
        if name in prefs.addons:
            enabled_mod_name = name
            break

    if enabled_mod_name is None:
        for name in candidate_names:
            try:
                if addon_utils is not None:
                    res = addon_utils.enable(name, default_set=True, persistent=True)
                    if res is not None or name in prefs.addons:
                        enabled_mod_name = name
                        break
            except Exception:
                pass
            try:
                bpy.ops.preferences.addon_enable(module=name)
                if name in prefs.addons:
                    enabled_mod_name = name
                    break
            except Exception:
                pass

    if enabled_mod_name is None:
        if verbose:
            print("[gpu_setup] WARNING: FLIP Fluids addon is not installed.", flush=True)
        return None

    # Complete installation if marker is missing (enables engine & handlers cleanly)
    needs_complete = False
    try:
        mod = sys.modules.get(enabled_mod_name) or importlib.import_module(enabled_mod_name)
        inst_utils = getattr(getattr(mod, "utils", None), "installation_utils", None)
        if inst_utils is not None and not inst_utils.is_installation_complete():
            needs_complete = True
    except Exception:
        pass

    flip_ops = getattr(bpy.ops, "flip_fluid_operators", None)
    if flip_ops is not None and hasattr(flip_ops, "complete_installation"):
        try:
            if needs_complete or flip_ops.complete_installation.poll():
                flip_ops.complete_installation()
                if verbose:
                    print("[gpu_setup] Completed FLIP Fluids installation bootstrap.", flush=True)
        except Exception as exc:
            if verbose:
                print(f"[gpu_setup] complete_installation note: {exc}", flush=True)

    if verbose:
        has_cmd = flip_ops is not None and hasattr(flip_ops, "bake_fluid_simulation_cmd")
        print(
            f"[gpu_setup] FLIP Fluids enabled: module={enabled_mod_name}, "
            f"bake_cmd_ready={has_cmd}",
            flush=True,
        )
    return enabled_mod_name


def resolve_eevee_engine_id(scene: Optional[Any] = None) -> str:
    """Resolve the active EEVEE engine identifier across Blender 3.x / 4.2-4.5 / 5.0+."""
    if bpy is None:
        return "BLENDER_EEVEE"

    target_scene = scene or bpy.context.scene
    enum_ids: set[str] = set()
    try:
        enum_ids = {
            item.identifier
            for item in target_scene.render.bl_rna.properties["engine"].enum_items
        }
    except Exception:
        pass

    ver = tuple(bpy.app.version)
    if ver >= (5, 0, 0) or ver < (4, 2, 0):
        preferred = ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"]
    else:
        preferred = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]

    for candidate in preferred:
        if not enum_ids or candidate in enum_ids:
            return candidate
    return preferred[0]


def _configure_cycles_texture_cache(
    scene: Any, enable: bool = True, verbose: bool = True
) -> dict[str, Any]:
    """Configure Blender 5.2+ Cycles Texture Cache properties if available (hasattr guarded)."""
    configured: dict[str, Any] = {}
    targets = [
        ("render", scene.render, ("use_texture_cache", "use_auto_generate_texture_cache")),
        (
            "cycles",
            getattr(scene, "cycles", None),
            ("use_texture_cache", "use_texture_cache_auto_generate"),
        ),
    ]
    for scope_name, obj, attrs in targets:
        if obj is None:
            continue
        for attr in attrs:
            if hasattr(obj, attr):
                try:
                    setattr(obj, attr, bool(enable))
                    configured[f"{scope_name}.{attr}"] = bool(enable)
                except Exception as exc:
                    configured[f"{scope_name}.{attr}"] = f"error:{exc}"
    if verbose and configured:
        print(f"[gpu_setup] Texture Cache settings: {configured}", flush=True)
    return configured


def _configure_cycles_denoiser(
    scene: Any,
    use_denoising: bool,
    denoiser: str,
    gpu_active: bool,
    verbose: bool = True,
) -> dict[str, Any]:
    cscene = scene.cycles
    result: dict[str, Any] = {"use_denoising": bool(use_denoising)}
    if hasattr(cscene, "use_denoising"):
        cscene.use_denoising = bool(use_denoising)

    if not use_denoising:
        return result

    available_denoisers: set[str] = set()
    if hasattr(cscene, "bl_rna") and "denoiser" in cscene.bl_rna.properties:
        available_denoisers = {
            item.identifier for item in cscene.bl_rna.properties["denoiser"].enum_items
        }

    if denoiser == "AUTO":
        order = ["OPENIMAGEDENOISE", "OPTIX"]
    elif denoiser == "NONE":
        if hasattr(cscene, "use_denoising"):
            cscene.use_denoising = False
        result["use_denoising"] = False
        return result
    else:
        order = [denoiser, "OPENIMAGEDENOISE", "OPTIX"]

    chosen = None
    for cand in order:
        if available_denoisers and cand not in available_denoisers:
            continue
        try:
            cscene.denoiser = cand
            chosen = cand
            break
        except (TypeError, ValueError):
            continue

    if chosen is None and not getattr(cscene, "denoiser", ""):
        if hasattr(cscene, "use_denoising"):
            cscene.use_denoising = False
        result["use_denoising"] = False
        result["denoiser"] = None
        result["note"] = "No compiled denoiser backend available; disabled denoising"
        if verbose:
            print(f"[gpu_setup] Denoiser config: {result}", flush=True)
        return result

    result["denoiser"] = chosen or getattr(cscene, "denoiser", None)
    if result["denoiser"] == "OPENIMAGEDENOISE" and hasattr(cscene, "denoising_use_gpu"):
        try:
            cscene.denoising_use_gpu = bool(gpu_active)
            result["denoising_use_gpu"] = bool(gpu_active)
        except Exception as exc:
            result["denoising_use_gpu_error"] = str(exc)

    if verbose:
        print(f"[gpu_setup] Denoiser config: {result}", flush=True)
    return result


def configure_cycles_gpu(
    scene: Optional[Any] = None,
    *,
    compute_type: str = "AUTO",
    include_cpu: bool = False,
    persistent_data: bool = True,
    use_denoising: bool = True,
    denoiser: str = "AUTO",
    use_adaptive_sampling: bool = True,
    samples: Optional[int] = None,
    adaptive_threshold: Optional[float] = 0.02,
    use_texture_cache: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Configure Cycles GPU rendering cleanly on Blender 4.5-5.3.

    - Tries OPTIX -> CUDA -> NONE when compute_type == 'AUTO'
    - Refreshes & queries Cycles preferences devices
    - Selects GPUs strictly via `device.type != 'CPU'` (never name regex)
    - Keeps CPU disabled alongside GPU by default (`include_cpu=False`)
    """
    if bpy is None:
        raise RuntimeError("configure_cycles_gpu() requires Blender's bpy module")

    target_scene = scene or bpy.context.scene
    target_scene.render.engine = "CYCLES"

    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get("cycles")
    if cycles_addon is None:
        raise RuntimeError("Cycles addon is not registered in Blender preferences")
    cprefs = cycles_addon.preferences

    requested = compute_type.upper()
    if requested == "AUTO":
        candidates = ["OPTIX", "CUDA", "NONE"]
    elif requested in {"OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "NONE"}:
        candidates = [requested] if requested == "NONE" else [requested, "NONE"]
    else:
        raise ValueError(f"Unsupported Cycles compute_type: {compute_type}")

    selected_type = "NONE"
    active_gpu_names: list[str] = []
    all_device_records: list[dict[str, Any]] = []

    for cand in candidates:
        try:
            cprefs.compute_device_type = cand
        except (TypeError, ValueError):
            continue

        if hasattr(cprefs, "refresh_devices"):
            try:
                cprefs.refresh_devices()
            except Exception:
                pass
        cprefs.get_devices()

        devices = list(cprefs.devices)
        gpu_devs = [d for d in devices if d.type != "CPU"]
        if cand == "NONE":
            selected_type = "NONE"
            break

        # Verify at least one real GPU device matches the backend (or non-CPU device exists)
        matching_gpus = [d for d in gpu_devs if d.type == cand] or gpu_devs
        if matching_gpus:
            selected_type = cand
            break

    gpu_active = selected_type != "NONE"
    target_scene.cycles.device = "GPU" if gpu_active else "CPU"

    for device in cprefs.devices:
        if device.type != "CPU":
            device.use = bool(gpu_active)
            if device.use:
                active_gpu_names.append(f"{device.name} ({device.type})")
        else:
            device.use = bool(include_cpu) or (not gpu_active)

        all_device_records.append(
            {
                "name": device.name,
                "type": device.type,
                "use": bool(device.use),
            }
        )

    if hasattr(target_scene.render, "use_persistent_data"):
        target_scene.render.use_persistent_data = bool(persistent_data)
    if hasattr(target_scene.render, "use_lock_interface"):
        target_scene.render.use_lock_interface = True

    cscene = target_scene.cycles
    if hasattr(cscene, "use_adaptive_sampling"):
        cscene.use_adaptive_sampling = bool(use_adaptive_sampling)
    if (
        use_adaptive_sampling
        and adaptive_threshold is not None
        and hasattr(cscene, "adaptive_threshold")
    ):
        cscene.adaptive_threshold = float(adaptive_threshold)
    if samples is not None and int(samples) > 0:
        cscene.samples = int(samples)

    # Prevent multi-layer transparent/refractive fluid splashes from hitting low bounce limits
    for bounce_attr, min_val in (
        ("transparent_max_bounces", 64),
        ("transmission_bounces", 16),
        ("max_bounces", 16),
        ("glossy_bounces", 8),
        ("volume_bounces", 2),
    ):
        if hasattr(cscene, bounce_attr):
            try:
                cur_val = int(getattr(cscene, bounce_attr))
                if cur_val < min_val:
                    setattr(cscene, bounce_attr, min_val)
            except Exception:
                pass

    denoise_info = _configure_cycles_denoiser(
        target_scene,
        use_denoising=use_denoising,
        denoiser=denoiser,
        gpu_active=gpu_active,
        verbose=verbose,
    )
    tex_cache_info = _configure_cycles_texture_cache(
        target_scene,
        enable=use_texture_cache,
        verbose=verbose,
    )

    summary = {
        "engine": target_scene.render.engine,
        "blender_version": list(bpy.app.version),
        "requested_compute_type": requested,
        "selected_compute_type": selected_type,
        "cycles_device": target_scene.cycles.device,
        "include_cpu": bool(include_cpu),
        "active_gpus": active_gpu_names,
        "devices": all_device_records,
        "use_persistent_data": bool(getattr(target_scene.render, "use_persistent_data", False)),
        "use_adaptive_sampling": bool(getattr(cscene, "use_adaptive_sampling", False)),
        "adaptive_threshold": getattr(cscene, "adaptive_threshold", None),
        "samples": getattr(cscene, "samples", None),
        "denoising": denoise_info,
        "texture_cache": tex_cache_info,
    }
    if verbose:
        print(
            f"[gpu_setup] Cycles configured: backend={selected_type}, "
            f"device={target_scene.cycles.device}, gpus={active_gpu_names}, "
            f"include_cpu={include_cpu}, persistent_data={summary['use_persistent_data']}, "
            f"samples={summary['samples']}, adaptive={summary['use_adaptive_sampling']}",
            flush=True,
        )
    return summary


def configure_render(
    scene: Optional[Any] = None,
    *,
    engine: str = "CYCLES",
    compute_type: str = "AUTO",
    include_cpu: bool = False,
    persistent_data: bool = True,
    use_denoising: bool = True,
    denoiser: str = "AUTO",
    use_adaptive_sampling: bool = True,
    samples: Optional[int] = None,
    adaptive_threshold: Optional[float] = 0.02,
    use_texture_cache: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Configure either CYCLES or EEVEE cleanly on the target Blender scene."""
    if bpy is None:
        raise RuntimeError("configure_render() requires Blender's bpy module")

    target_scene = scene or bpy.context.scene
    eng_upper = engine.upper()
    if eng_upper in {"EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        eevee_id = resolve_eevee_engine_id(target_scene)
        target_scene.render.engine = eevee_id
        if hasattr(target_scene.render, "use_persistent_data"):
            target_scene.render.use_persistent_data = bool(persistent_data)
        if hasattr(target_scene.render, "use_lock_interface"):
            target_scene.render.use_lock_interface = True
        eevee_props = getattr(target_scene, "eevee", None)
        applied_samples = None
        if eevee_props is not None and samples is not None and int(samples) > 0:
            for attr in ("taa_render_samples", "samples"):
                if hasattr(eevee_props, attr):
                    setattr(eevee_props, attr, int(samples))
                    applied_samples = int(samples)
                    break
        summary = {
            "engine": eevee_id,
            "blender_version": list(bpy.app.version),
            "samples": applied_samples,
            "use_persistent_data": bool(getattr(target_scene.render, "use_persistent_data", False)),
        }
        if verbose:
            print(f"[gpu_setup] EEVEE configured: {summary}", flush=True)
        return summary

    return configure_cycles_gpu(
        target_scene,
        compute_type=compute_type,
        include_cpu=include_cpu,
        persistent_data=persistent_data,
        use_denoising=use_denoising,
        denoiser=denoiser,
        use_adaptive_sampling=use_adaptive_sampling,
        samples=samples,
        adaptive_threshold=adaptive_threshold,
        use_texture_cache=use_texture_cache,
        verbose=verbose,
    )


def detect_graphics_environment() -> dict[str, Any]:
    """Inspect host Linux Vulkan ICDs, GLVND EGL vendors, and display server availability."""
    vulkan_icds: list[str] = []
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"):
        if os.path.isdir(d):
            vulkan_icds.extend(sorted(glob.glob(os.path.join(d, "*.json"))))

    egl_vendors: list[str] = []
    for d in ("/usr/share/glvnd/egl_vendor.d", "/etc/glvnd/egl_vendor.d"):
        if os.path.isdir(d):
            egl_vendors.extend(sorted(glob.glob(os.path.join(d, "*.json"))))

    return {
        "display": os.environ.get("DISPLAY", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "vulkan_icds": vulkan_icds,
        "egl_vendors": egl_vendors,
        "vk_icd_filenames_env": os.environ.get("VK_ICD_FILENAMES", ""),
        "xvfb_run": shutil.which("xvfb-run"),
        "vulkaninfo": shutil.which("vulkaninfo"),
        "eglinfo": shutil.which("eglinfo"),
    }


def _extract_error_summary(stdout: str, stderr: str) -> str:
    combined = (stderr or "") + "\n" + (stdout or "")
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
    interesting = [
        ln
        for ln in lines
        if any(
            kw in ln.lower()
            for kw in (
                "error",
                "failed",
                "unable",
                "cannot",
                "vulkan",
                "opengl",
                "egl",
                "display",
                "unsupported",
                "segfault",
                "warning",
            )
        )
    ]
    picked = interesting[-8:] if interesting else lines[-8:]
    return " | ".join(picked) if picked else "No diagnostic output"


def probe_eevee_backends(
    blender_bin: str,
    blend_path: Optional[str] = None,
    *,
    frame: int = 1,
    out_dir: Optional[str] = None,
    samples: int = 16,
    resolution_x: int = 640,
    resolution_y: int = 480,
    backends: tuple[str, ...] = ("vulkan", "opengl"),
    timeout: int = 120,
) -> dict[str, Any]:
    """Probe Blender EEVEE headless rendering across GPU backends ('vulkan', 'opengl')."""
    env_info = detect_graphics_environment()
    probe_root = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="eevee_probe_"))
    probe_root.mkdir(parents=True, exist_ok=True)

    probe_py = probe_root / "_eevee_probe_worker.py"
    probe_py.write_text(
        "\n".join(
            [
                "import sys, time, bpy",
                "argv = sys.argv[sys.argv.index('--') + 1:]",
                "out_png, frame_s, samples_s, rx_s, ry_s = argv[:5]",
                "scene = bpy.context.scene",
                "ver = tuple(bpy.app.version)",
                "enum_ids = {i.identifier for i in scene.render.bl_rna.properties['engine'].enum_items}",
                "pref = 'BLENDER_EEVEE' if (ver >= (5, 0, 0) or ver < (4, 2, 0)) else 'BLENDER_EEVEE_NEXT'",
                "eng = pref if pref in enum_ids else ('BLENDER_EEVEE' if 'BLENDER_EEVEE' in enum_ids else 'BLENDER_EEVEE_NEXT')",
                "scene.render.engine = eng",
                "scene.render.resolution_x = int(rx_s)",
                "scene.render.resolution_y = int(ry_s)",
                "scene.render.resolution_percentage = 100",
                "scene.render.image_settings.file_format = 'PNG'",
                "scene.render.filepath = out_png",
                "if hasattr(scene, 'eevee'):",
                "    for a in ('taa_render_samples', 'samples'):",
                "        if hasattr(scene.eevee, a):",
                "            setattr(scene.eevee, a, int(samples_s))",
                "            break",
                "if scene.camera is None:",
                "    bpy.ops.object.camera_add(location=(5.0, -5.0, 4.0))",
                "    cam = bpy.context.object",
                "    cam.rotation_euler = (1.1, 0.0, 0.785)",
                "    scene.camera = cam",
                "if not any(o.type == 'LIGHT' for o in bpy.data.objects):",
                "    bpy.ops.object.light_add(type='SUN', location=(0.0, 0.0, 10.0))",
                "scene.frame_set(int(frame_s))",
                "t0 = time.perf_counter()",
                "bpy.ops.render.render(write_still=True)",
                "dt = time.perf_counter() - t0",
                "print(f'EEVEE_PROBE_OK engine={eng} render_s={dt:.4f}', flush=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    results: dict[str, Any] = {
        "environment": env_info,
        "backends": {},
    }

    for backend in backends:
        out_png = probe_root / f"eevee_{backend}.png"
        if out_png.exists():
            out_png.unlink()

        cmd = [blender_bin, "-b"]
        if blend_path and os.path.isfile(blend_path):
            cmd.append(blend_path)
        else:
            cmd.append("--factory-startup")
        cmd.extend(
            [
                "--gpu-backend",
                backend,
                "-P",
                str(probe_py),
                "--",
                str(out_png),
                str(frame),
                str(samples),
                str(resolution_x),
                str(resolution_y),
            ]
        )

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            wall_s = round(time.perf_counter() - t0, 4)
            ok = proc.returncode == 0 and out_png.is_file() and out_png.stat().st_size > 100
            render_s = None
            engine_used = None
            for line in (proc.stdout or "").splitlines():
                if line.startswith("EEVEE_PROBE_OK "):
                    for tok in line.split()[1:]:
                        k, _, v = tok.partition("=")
                        if k == "render_s":
                            render_s = float(v)
                        elif k == "engine":
                            engine_used = v

            results["backends"][backend] = {
                "supported": bool(ok),
                "status": "ok" if ok else "unsupported_or_failed",
                "returncode": proc.returncode,
                "wall_s": wall_s,
                "render_s": render_s,
                "engine": engine_used,
                "output_png": str(out_png) if ok else None,
                "output_bytes": out_png.stat().st_size if out_png.is_file() else 0,
                "diagnostic": (
                    "Headless EEVEE render succeeded"
                    if ok
                    else _extract_error_summary(proc.stdout, proc.stderr)
                ),
            }
        except subprocess.TimeoutExpired:
            results["backends"][backend] = {
                "supported": False,
                "status": "timeout",
                "returncode": -1,
                "wall_s": round(time.perf_counter() - t0, 4),
                "render_s": None,
                "engine": None,
                "output_png": None,
                "output_bytes": 0,
                "diagnostic": f"Timed out after {timeout}s",
            }
        except Exception as exc:
            results["backends"][backend] = {
                "supported": False,
                "status": "error",
                "returncode": -1,
                "wall_s": round(time.perf_counter() - t0, 4),
                "render_s": None,
                "engine": None,
                "output_png": None,
                "output_bytes": 0,
                "diagnostic": str(exc),
            }

    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Configure Cycles GPU / EEVEE and FLIP Fluids inside headless Blender."
    )
    parser.add_argument(
        "--engine",
        default="CYCLES",
        choices=["CYCLES", "EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"],
        help="Target render engine (default: CYCLES)",
    )
    parser.add_argument(
        "--compute-type",
        default="AUTO",
        choices=["AUTO", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL", "NONE"],
        help="Cycles compute backend preference (default: AUTO -> OPTIX then CUDA then NONE)",
    )
    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Keep CPU enabled alongside GPU devices (default: False)",
    )
    parser.add_argument(
        "--no-persistent-data",
        action="store_true",
        help="Disable scene.render.use_persistent_data",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable Cycles denoising",
    )
    parser.add_argument(
        "--denoiser",
        default="AUTO",
        choices=["AUTO", "OPENIMAGEDENOISE", "OPTIX", "NONE"],
        help="Cycles denoiser preference (default: AUTO)",
    )
    parser.add_argument(
        "--no-adaptive-sampling",
        action="store_true",
        help="Disable Cycles adaptive sampling",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override render sample count",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=0.02,
        help="Cycles adaptive sampling noise threshold (default: 0.02)",
    )
    parser.add_argument(
        "--no-texture-cache",
        action="store_true",
        help="Disable Blender 5.2+ Cycles Texture Cache",
    )
    parser.add_argument(
        "--ensure-flip",
        action="store_true",
        help="Ensure FLIP Fluids addon is enabled and installation completed",
    )
    parser.add_argument(
        "--probe-eevee",
        action="store_true",
        help="Run EEVEE Vulkan/OpenGL headless probe (when invoked with --blender outside Blender)",
    )
    parser.add_argument(
        "--blender",
        default="blender",
        help="Blender executable path when running --probe-eevee from host Python",
    )
    parser.add_argument(
        "--blend",
        default="",
        help="Optional .blend file for --probe-eevee",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write JSON configuration / probe summary",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(_blender_cli_argv())
    if bpy is None:
        if args.probe_eevee:
            res = probe_eevee_backends(
                args.blender,
                blend_path=args.blend or None,
                samples=args.samples or 16,
            )
            text = json.dumps(res, indent=2)
            print(text, flush=True)
            if args.json_out:
                Path(args.json_out).write_text(text, encoding="utf-8")
            return 0
        print(
            json.dumps(
                {
                    "mode": "host_python_env_check",
                    "graphics_env": detect_graphics_environment(),
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    flip_mod = ensure_flip_fluids_enabled(verbose=True) if args.ensure_flip else None
    summary = configure_render(
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
    if flip_mod is not None:
        summary["flip_fluids_module"] = flip_mod

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    rc = main()
    if bpy is None or rc != 0:
        raise SystemExit(rc)
