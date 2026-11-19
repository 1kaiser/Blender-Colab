# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: BlenderColab.ipynb
#     provenance: []
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Blender-Colab — Blender 5.3.0 + FLIP Fluids v1.8.9 (Unwatermarked)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/Blender-Colab/blob/master/BlenderColab.ipynb)
#
# > **GPU-accelerated Blender 5.3.0 rendering and unwatermarked FLIP Fluids v1.8.9 simulation on Google Colab or local Papermill.**
# > Supports Cycles OptiX/CUDA (`use_persistent_data=True`), overlapping CPU-bake || GPU-render interleaving, automated Windows-path/shader sanitization, and single-process 4-view April Sun Path rendering.
#
# <details>
# <summary><b>Architecture & Upgrades in v2.0 (Click to expand)</b></summary>
#
# - **Blender 5.3.0** default (with `5.2.4` LTS and `4.5.16` LTS options) extracted cleanly with mirror fallbacks (`download.blender.org`, `mirrors.dotsrc.org`, `ftp.nluug.nl`) and minimal runtime packages—no legacy `libtcmalloc-minimal4` / `LD_PRELOAD` hacks.
# - **FLIP Fluids v1.8.9 (Clean Unwatermarked `v2.0.0` Prebuilt + Source Fallback)**: Uses unwatermarked `rlguy/Blender-FLIP-Fluids` tag `v1.8.9` (`force_link_glibc_2.5.h` patched, `blender_manifest.toml.in` tagline shortened to `<= 64` chars, `_vertx/_trix` 3D Demo watermark stripped from `src/engine/fluidsimulation.cpp`). Defaults to downloading the prebuilt `v2.0.0` extension zip (`prebuilt_release`) in seconds and automatically falls back to compiling from source (`build_from_source`) on ABI/download errors.
# - **OptiX Persistent-Data & Interleaved Pipeline**: Runs Cycles GPU-only (`OPTIX` -> `CUDA` -> `NONE`, `use_persistent_data=True`, `transparent_max_bounces >= 64`) in one persistent Blender session, optionally overlapping CPU FLIP simulation baking with GPU OptiX frame rendering by polling `bakefiles/finished######.txt`.
# - **Standalone Colab Badge Compatibility**: Works directly from a fresh Google Colab VM without cloning beforehand—automatically detects local `scripts/` or fetches `1kaiser/Blender-Colab` into `/content/work`.
# - **Dual Colab Forms + Papermill**: Uses `# @param` UI controls inside a `tags=["parameters"]` cell, with opt-in `save_to_drive = False` so headless `colab exec` and `papermill` runs never block on Drive OAuth.
#
# </details>

# %% [markdown]
# ## 1. Hardware Telemetry & Runtime Detection
#
# > Detects Google Colab vs. local workstation execution, inspects NVIDIA GPU/CPU/RAM/disk resources, and resolves `WORK_DIR` and `SCRIPTS_DIR` (auto-fetching `scripts/` if opened standalone from a Colab badge).
#
# <details>
# <summary><b>Runtime & Standalone Bootstrap Details (Click to expand)</b></summary>
#
# 1. **Working directory (`WORK_DIR`)**: Uses `/content/work` when running on Google Colab (faster local VM SSD) or `./work` when executing locally via Papermill/Jupyter.
# 2. **Scripts resolution (`SCRIPTS_DIR`)**: Checks `./scripts`, `/content/work/scripts`, and `/content/Blender-Colab/scripts` first; if none contain the full `scripts/` suite, clones `https://github.com/1kaiser/Blender-Colab.git` (`--depth 1`) with raw GitHub download fallback into `/content/work/scripts`.
# 3. **Opt-in Google Drive**: Drive mounting is intentionally deferred until Section 2 parameters are parsed (`save_to_drive = False` default) so automated runs never hang on interactive OAuth prompts.
#
# </details>

# %%
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

IN_COLAB = ("google.colab" in sys.modules) or Path("/content").is_dir()
REPO_CLONE_URL = "https://github.com/1kaiser/Blender-Colab.git"
REPO_RAW_BASE = "https://raw.githubusercontent.com/1kaiser/Blender-Colab/master/scripts"
REQUIRED_SCRIPTS = (
    "build_flip_fluids.sh",
    "install_blender_flip.py",
    "inspect_scene.py",
    "gpu_setup.py",
    "bake_and_render.py",
    "sunpath_multiview.py",
    "benchmark_colab.py",
)

WORK_DIR = Path("/content/work") if Path("/content").is_dir() else (Path.cwd() / "work").resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _has_all_scripts(candidate: Path) -> bool:
    return candidate.is_dir() and all((candidate / name).is_file() for name in REQUIRED_SCRIPTS)


def resolve_scripts_dir() -> Path:
    candidates = [
        Path.cwd() / "scripts",
        WORK_DIR / "scripts",
        Path("/content/Blender-Colab/scripts"),
        WORK_DIR / "Blender-Colab" / "scripts",
    ]
    for cand in candidates:
        if _has_all_scripts(cand):
            return cand.resolve()

    repo_dir = WORK_DIR / "Blender-Colab"
    if shutil.which("git"):
        if (repo_dir / ".git").is_dir():
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=False)
        else:
            print(f"[bootstrap] Cloning {REPO_CLONE_URL} -> {repo_dir} ...", flush=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_CLONE_URL, str(repo_dir)],
                check=False,
            )
        if _has_all_scripts(repo_dir / "scripts"):
            return (repo_dir / "scripts").resolve()

    # Fallback: fetch individual script files directly from GitHub raw
    target = WORK_DIR / "scripts"
    target.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_SCRIPTS:
        dst = target / name
        if not dst.is_file():
            url = f"{REPO_RAW_BASE}/{name}"
            print(f"[bootstrap] Fetching {url} -> {dst}", flush=True)
            urllib.request.urlretrieve(url, dst)

    if not _has_all_scripts(target):
        missing = [n for n in REQUIRED_SCRIPTS if not (target / n).is_file()]
        raise FileNotFoundError(f"Missing required scripts in {target}: {missing}")
    return target.resolve()


SCRIPTS_DIR = resolve_scripts_dir()
build_sh = SCRIPTS_DIR / "build_flip_fluids.sh"
if build_sh.is_file():
    os.chmod(build_sh, os.stat(build_sh).st_mode | 0o755)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def run_streaming(cmd: list[str], *, check: bool = True) -> int:
    """Run subprocess while streaming combined output cleanly to Jupyter/Colab cell logs."""
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")
    return rc


def print_runtime_telemetry() -> None:
    ram_gib = "unknown"
    if os.path.isfile("/proc/meminfo"):
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                ram_gib = f"{int(line.split()[1]) / (1024.0 * 1024.0):.2f} GiB"
                break

    du = shutil.disk_usage(str(WORK_DIR))
    free_gib = du.free / (1024.0 ** 3)
    total_gib = du.total / (1024.0 ** 3)

    print("=" * 72)
    print("Blender-Colab Runtime Telemetry")
    print("=" * 72)
    print(f"  IN_COLAB    : {IN_COLAB}")
    print(f"  OS          : {platform.platform()}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  CPU Cores   : {os.cpu_count()}")
    print(f"  System RAM  : {ram_gib}")
    print(f"  Disk Free   : {free_gib:.1f} GiB / {total_gib:.1f} GiB ({WORK_DIR})")
    print(f"  SCRIPTS_DIR : {SCRIPTS_DIR}")
    print(f"  Tools       : git={bool(shutil.which('git'))}, cmake={bool(shutil.which('cmake'))}, "
          f"g++={bool(shutil.which('g++'))}, ffmpeg={bool(shutil.which('ffmpeg'))}")
    print("=" * 72)

    if shutil.which("nvidia-smi"):
        run_streaming(["nvidia-smi"], check=False)
    else:
        print("[telemetry] nvidia-smi not found (CPU-only runtime).")


print_runtime_telemetry()

# %% [markdown]
# ## 2. Configuration Parameters
#
# > Configure Blender version, FLIP Fluids addon source, scene URL, pipeline mode, sampling quality, multi-view sun path, and export options via Colab Forms or Papermill (`-p`).
#
# <details>
# <summary><b>Parameter Reference & Papermill Overrides (Click to expand)</b></summary>
#
# | Parameter | Default | Description |
# |---|---|---|
# | `blender_version` | `"5.3.0"` | Blender release (`5.3.0`, `5.2.4` LTS, or `4.5.16` LTS). |
# | `addon_source` | `"prebuilt_release"` | `prebuilt_release` (unwatermarked v2.0.0 asset, fast), `build_from_source`, `custom_url`, or `official_demo`. |
# | `custom_addon_url` | `""` | Custom zip URL or local path when `addon_source` is `custom_url`. |
# | `blend_file_url` | `particles.blend` | Direct HTTP/HTTPS URL, Google Drive share link, or local `.blend` path. |
# | `pipeline_mode` | `"interleaved"` | `interleaved` (CPU bake \|\| GPU render), `sequential`, `render_only`, or `bake_only`. |
# | `render_engine` | `"CYCLES"` | `CYCLES` (OptiX/CUDA GPU) or `EEVEE` (`BLENDER_EEVEE`). |
# | `start_frame` / `end_frame` | `1` / `25` | Frame range (`1..25` demo takes ~7 min on L4; `1..250` for full timeline). |
# | `cycles_samples` | `64` | Samples per frame for Cycles (or EEVEE samples). |
# | `use_denoising` | `True` | Enable OpenImageDenoise / OptiX denoiser. |
# | `use_adaptive_sampling` | `True` | Enable Cycles adaptive sampling (`threshold=0.02`). |
# | `flip_resolution_override` | `192` | FLIP domain voxel resolution (`0` keeps scene default; `>=192` recommended for `particles.blend`). |
# | `enable_sunpath_multiview` | `True` | Render 4 orthographic views (`top`, `front`, `side`, `isometric`) under April noon sun. |
# | `run_benchmarks` | `False` | Execute full 6-mode GPU/pipeline benchmark suite (`~7-8 min` on L4). |
# | `save_to_drive` | `False` | Opt-in Google Drive copy to `MyDrive/BlenderColab_Outputs` (skipped safely if headless). |
# | `worker_id` / `num_workers` | `0` / `1` | Multi-worker modulo frame sharding (`frame_idx % num_workers == worker_id`). |
#
# Example Papermill execution:
# ```bash
# papermill BlenderColab.ipynb out.ipynb -p end_frame 10 -p cycles_samples 32 -p run_benchmarks True
# ```
#
# </details>

# %% tags=["parameters"]
# @markdown ### Blender & FLIP Fluids Installation
blender_version = "5.3.0"  # @param ["5.3.0", "5.2.4", "4.5.16"] {allow-input: true}
addon_source = "prebuilt_release"  # @param ["prebuilt_release", "build_from_source", "custom_url", "official_demo"]
custom_addon_url = ""  # @param {type:"string"}

# @markdown ### Scene & Simulation Pipeline
blend_file_url = "https://github.com/1kaiser/Blender-Colab/releases/download/v1.0.0/particles.blend"  # @param {type:"string"}
pipeline_mode = "interleaved"  # @param ["interleaved", "sequential", "render_only", "bake_only"]
render_engine = "CYCLES"  # @param ["CYCLES", "EEVEE"]
start_frame = 1  # @param {type:"integer"}
end_frame = 25  # @param {type:"integer"}
cycles_samples = 64  # @param {type:"integer"}
use_denoising = True  # @param {type:"boolean"}
use_adaptive_sampling = True  # @param {type:"boolean"}
flip_resolution_override = 192  # @param {type:"integer"}

# @markdown ### Sun Path, Benchmarks & Export
enable_sunpath_multiview = True  # @param {type:"boolean"}
run_benchmarks = False  # @param {type:"boolean"}
save_to_drive = False  # @param {type:"boolean"}
worker_id = 0  # @param {type:"integer"}
num_workers = 1  # @param {type:"integer"}

# %%
start_frame = int(start_frame)
end_frame = int(end_frame)
cycles_samples = int(cycles_samples)
flip_resolution_override = int(flip_resolution_override)
worker_id = int(worker_id)
num_workers = int(num_workers)

if end_frame < start_frame:
    raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")
if num_workers < 1 or worker_id < 0 or worker_id >= num_workers:
    raise ValueError(f"Invalid worker sharding: worker_id={worker_id}, num_workers={num_workers}")
if cycles_samples < 1:
    raise ValueError(f"cycles_samples ({cycles_samples}) must be >= 1")
if flip_resolution_override < 0:
    raise ValueError(f"flip_resolution_override ({flip_resolution_override}) must be >= 0")

DRIVE_OUTPUT_DIR = None
if save_to_drive and IN_COLAB:
    mounted_root = Path("/content/drive/MyDrive")
    if mounted_root.is_dir():
        DRIVE_OUTPUT_DIR = mounted_root / "BlenderColab_Outputs"
        DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[drive] Reusing mounted Google Drive: {DRIVE_OUTPUT_DIR}")
    else:
        try:
            # Guard interactive OAuth so headless `colab exec` fails gracefully without hanging
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("Interactive Google Drive mount timed out (headless session?).")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(45)
            try:
                from google.colab import drive  # type: ignore

                drive.mount("/content/drive")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            if mounted_root.is_dir():
                DRIVE_OUTPUT_DIR = mounted_root / "BlenderColab_Outputs"
                DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[drive] Mounted Google Drive: {DRIVE_OUTPUT_DIR}")
        except Exception as exc:
            print(f"[drive] Skipping Google Drive mount ({exc}); outputs remain in {WORK_DIR}.")
elif save_to_drive and not IN_COLAB:
    print("[drive] Not running inside Google Colab; save_to_drive ignored.")

print(
    f"[config] Blender={blender_version} | addon={addon_source} | mode={pipeline_mode} | "
    f"engine={render_engine} | frames={start_frame}..{end_frame} | samples={cycles_samples} | "
    f"res_override={flip_resolution_override} | worker={worker_id}/{num_workers}"
)

# %% [markdown]
# ## 3. Install Blender 5.3.0 + FLIP Fluids v1.8.9
#
# > Downloads and verifies Blender, installs minimal runtime dependencies (via `ldd` inspection), fetches the verified unwatermarked v2.0.0 FLIP Fluids v1.8.9 package (falling back cleanly to `build_from_source` if needed), registers the Blender extension, and verifies `libffengine.so`.
#
# <details>
# <summary><b>Installation & Unwatermarked Source Build Details (Click to expand)</b></summary>
#
# - `scripts/install_blender_flip.py` defaults to `--addon-source prebuilt_release`, fetching `flip_fluids_addon_1.8.9_linux_x64_ext.zip` from `Blender-Colab` `v2.0.0` in seconds and automatically falling back to `build_from_source` if validation or `libffengine.so` loading fails on another Linux distro. It inspects `ldd` shared libraries and installs only missing X11/OpenGL/Vulkan runtime libraries (`libxkbcommon0`, `libsm6`, `libxi6`, `libxxf86vm1`, `libgl1`, `libegl1`, `libvulkan1`), plus `cmake`/`g++`/`make` when compiling from source.
# - `scripts/build_flip_fluids.sh` clones `rlguy/Blender-FLIP-Fluids` tag `v1.8.9`, removes `-include .../force_link_glibc_2.5.h`, shortens `blender_manifest.toml.in` `tagline` to 47 chars, strips the embedded `_vertx/_trix` 3D Demo watermark mesh from `src/engine/fluidsimulation.cpp`, validates with `blender --command extension validate`, and installs via `blender --command extension install-file -r user_default --enable`.
# - A headless bootstrap calls `bpy.ops.flip_fluid_operators.complete_installation()` and `bpy.ops.wm.save_userpref()`, then runs a fresh self-check verifying `bl_ext.user_default.flip_fluids_addon` and `FluidSimulation(10,10,10,0.1).get_version()`.
#
# </details>

# %%
INSTALL_JSON = WORK_DIR / "install_summary.json"

install_cmd = [
    sys.executable,
    str(SCRIPTS_DIR / "install_blender_flip.py"),
    "--blender-version",
    str(blender_version),
    "--work-dir",
    str(WORK_DIR),
    "--addon-source",
    str(addon_source),
    "--symlink",
    "--json-out",
    str(INSTALL_JSON),
]
if str(addon_source) == "custom_url" and str(custom_addon_url).strip():
    install_cmd.extend(["--custom-addon-url", str(custom_addon_url).strip()])

run_streaming(install_cmd, check=True)

install_summary = json.loads(INSTALL_JSON.read_text(encoding="utf-8"))
BLENDER_BIN = install_summary["blender_binary"]
print("\n[install] Verified installation summary:")
print(json.dumps({
    "blender_version": install_summary.get("blender_version_string"),
    "blender_binary": BLENDER_BIN,
    "addon_source_effective": install_summary.get("addon_source_effective"),
    "enabled_module": install_summary.get("enabled_module"),
    "bl_info_version": install_summary.get("bl_info_version"),
    "support_license": install_summary.get("support_license"),
    "installation_complete": install_summary.get("installation_complete"),
    "libffengine_version": install_summary.get("libffengine_version"),
    "load_post_registered": install_summary.get("load_post_registered"),
}, indent=2))

# %% [markdown]
# ## 4. Download, Inspect & Sanitize `.blend` Scene
#
# > Fetches the target `.blend` file, runs `inspect_scene.py` to audit objects/bounds/materials/paths, and sanitizes legacy Windows paths (`C:\Users\...`), dormant FLIP domain properties, and shaders into `/content/work/flip_fluid_cache`.
#
# <details>
# <summary><b>Scene Diagnostics & Sanitization Pipeline (Click to expand)</b></summary>
#
# - `scripts/inspect_scene.py` inventories every scene object, evaluates world-space bounding boxes (`meshes_only`, `all_objects`), reports active cameras/lights/world strength, checks FLIP Fluids properties vs. dormant `obj["flip_fluid"]` IDProps and Mantaflow modifiers, and flags Windows paths or broken material nodes.
# - `scripts/bake_and_render.py --mode sanitize` rewrites `cache_directory` to fast local storage (`/content/work/flip_fluid_cache`), reactivates dormant FLIP Fluids objects while stripping conflicting Mantaflow modifiers, hides helper bounds (`TYPE_DOMAIN`, `TYPE_INFLOW`, `TYPE_OUTFLOW`) from final renders, and upgrades legacy bare-`Transparent BSDF` water shaders into a physically refractive cyan `Principled BSDF` + `LightPath.IsShadowRay` shader.
#
# </details>

# %%
CACHE_DIR = WORK_DIR / "flip_fluid_cache"
INSPECT_JSON = WORK_DIR / "scene_inspect.json"
SANITIZE_JSON = WORK_DIR / "sanitize_summary.json"


def _normalize_drive_url(url: str) -> str:
    m = re.search(r"drive\.google\.com/file/d/([A-Za-z0-9_-]+)", url)
    if m:
        return f"https://drive.google.com/uc?id={m.group(1)}&export=download"
    return url


def fetch_blend_file(src: str, dest_dir: Path) -> Path:
    src_clean = src.strip()
    if not src_clean:
        raise ValueError("blend_file_url cannot be empty.")
    local_src = Path(src_clean)
    if local_src.is_file():
        print(f"[scene] Using local .blend file: {local_src.resolve()}")
        return local_src.resolve()

    norm_url = _normalize_drive_url(src_clean)
    url_tail = norm_url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    fname = url_tail if url_tail.endswith(".blend") else "scene_input.blend"
    dest = dest_dir / fname
    if dest.is_file() and dest.stat().st_size > 10_000:
        print(f"[scene] Reusing cached .blend ({dest.stat().st_size:,} B): {dest}")
        return dest

    print(f"[scene] Downloading {norm_url} -> {dest}", flush=True)
    tmp = dest.with_suffix(".blend.part")
    req = urllib.request.Request(norm_url, headers={"User-Agent": "Blender-Colab/2.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out_fh:
        shutil.copyfileobj(resp, out_fh, length=1024 * 1024)
    os.replace(tmp, dest)
    print(f"[scene] Downloaded {dest.stat().st_size:,} bytes -> {dest}")
    return dest


RAW_BLEND = fetch_blend_file(blend_file_url, WORK_DIR)
SANITIZED_BLEND = WORK_DIR / f"{RAW_BLEND.stem}_sanitized.blend"

run_streaming(
    [
        BLENDER_BIN,
        "-b",
        str(RAW_BLEND),
        "-P",
        str(SCRIPTS_DIR / "inspect_scene.py"),
        "--",
        "--json-out",
        str(INSPECT_JSON),
        "--quiet",
    ],
    check=True,
)

inspect_data = json.loads(INSPECT_JSON.read_text(encoding="utf-8"))
print("[inspect] Key diagnostics:")
print(json.dumps({
    "source_blend_version": inspect_data.get("blender", {}).get("blend_file_version"),
    "render_engine": inspect_data.get("render_settings", {}).get("engine"),
    "resolution": [
        inspect_data.get("render_settings", {}).get("resolution_x"),
        inspect_data.get("render_settings", {}).get("resolution_y"),
    ],
    "scene_frames": [
        inspect_data.get("render_settings", {}).get("frame_start"),
        inspect_data.get("render_settings", {}).get("frame_end"),
    ],
    "scene_objects": inspect_data.get("object_counts", {}).get("scene_objects"),
    "meshes_only_bbox": inspect_data.get("world_bbox", {}).get("meshes_only"),
    "flip_has_domain": inspect_data.get("flip_fluids_scene", {}).get("has_domain"),
    "dormant_flip_count": len(inspect_data.get("flip_fluids_scene", {}).get("inactive_flip_objects", [])),
    "mantaflow_modifiers": len(inspect_data.get("flip_fluids_scene", {}).get("mantaflow_or_particle_modifiers", [])),
    "missing_items": inspect_data.get("missing_external_paths", {}).get("missing_items", []),
}, indent=2))

run_streaming(
    [
        sys.executable,
        str(SCRIPTS_DIR / "bake_and_render.py"),
        "--mode",
        "sanitize",
        "--blender",
        BLENDER_BIN,
        "--blend",
        str(RAW_BLEND),
        "--sanitized-blend",
        str(SANITIZED_BLEND),
        "--cache-dir",
        str(CACHE_DIR),
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--resolution",
        str(flip_resolution_override),
        "--json-out",
        str(SANITIZE_JSON),
    ],
    check=True,
)

sanitize_data = json.loads(SANITIZE_JSON.read_text(encoding="utf-8"))
print("\n[sanitize] Summary:")
print(json.dumps(sanitize_data.get("sanitize", sanitize_data), indent=2))

# %% [markdown]
# ## 5. Run FLIP Fluids Simulation Bake + GPU Render
#
# > Executes the chosen pipeline (`interleaved`, `sequential`, `render_only`, or `bake_only`) with `use_persistent_data=True`, `--skip-existing` resume safety, and automated MP4/GIF assembly, followed by an inline frame-strip and HTML5 video preview.
#
# <details>
# <summary><b>Pipeline Mode & Persistent-Data Rendering Mechanics (Click to expand)</b></summary>
#
# - **`interleaved` (Default)**: Launches FLIP Fluids CPU baking in one headless Blender worker while a persistent-data Cycles OptiX GPU Blender worker polls `<cache_dir>/bakefiles/finished######.txt` and renders each completed frame immediately.
# - **`sequential`**: Bakes all frames first, then renders in a single persistent-data GPU session.
# - **`render_only` / `bake_only`**: Runs only GPU persistent rendering (against an existing cache) or only simulation baking.
# - **Resume safety (`--skip-existing`)**: Any existing rendered frame PNG (`> 1 KiB`) is skipped automatically on interrupted/resumed sessions, and `num_workers > 1` cleanly shards frames across parallel Colab runtimes (`worker_id`).
#
# </details>

# %%
FRAMES_DIR = WORK_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_JSON = WORK_DIR / "pipeline_run.json"

mode_map = {
    "interleaved": "interleaved",
    "sequential": "sequential",
    "render_only": "render_persistent",
    "bake_only": "bake",
}
if pipeline_mode not in mode_map:
    raise ValueError(f"Unsupported pipeline_mode={pipeline_mode!r}; expected {list(mode_map)}")

pipeline_cmd = [
    sys.executable,
    str(SCRIPTS_DIR / "bake_and_render.py"),
    "--mode",
    mode_map[pipeline_mode],
    "--blender",
    BLENDER_BIN,
    "--blend",
    str(SANITIZED_BLEND),
    "--cache-dir",
    str(CACHE_DIR),
    "--out-dir",
    str(FRAMES_DIR),
    "--start-frame",
    str(start_frame),
    "--end-frame",
    str(end_frame),
    "--resolution",
    str(flip_resolution_override),
    "--engine",
    str(render_engine),
    "--samples",
    str(cycles_samples),
    "--worker-id",
    str(worker_id),
    "--num-workers",
    str(num_workers),
    "--skip-existing",
    "--json-out",
    str(PIPELINE_JSON),
]
if not use_denoising:
    pipeline_cmd.append("--no-denoise")
if not use_adaptive_sampling:
    pipeline_cmd.append("--no-adaptive-sampling")
if DRIVE_OUTPUT_DIR is not None:
    sync_frames_dir = DRIVE_OUTPUT_DIR / f"frames_w{worker_id}"
    sync_frames_dir.mkdir(parents=True, exist_ok=True)
    pipeline_cmd.extend(["--sync-dir", str(sync_frames_dir)])

run_streaming(pipeline_cmd, check=True)

pipeline_summary = json.loads(PIPELINE_JSON.read_text(encoding="utf-8"))
print("\n[pipeline] Execution summary:")
print(json.dumps({
    "mode": pipeline_summary.get("mode"),
    "wall_s": pipeline_summary.get("wall_s"),
    "bake": pipeline_summary.get("bake"),
    "render_count": pipeline_summary.get("render", {}).get("rendered_count"),
    "skipped_count": pipeline_summary.get("render", {}).get("skipped_count"),
    "render_wall_s": pipeline_summary.get("render", {}).get("wall_s"),
    "video": pipeline_summary.get("video"),
}, indent=2))

# %%
import base64
from IPython.display import HTML, display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

png_frames = sorted(FRAMES_DIR.glob("frame_*.png"))
if png_frames:
    n_pick = min(5, len(png_frames))
    if n_pick == 1:
        picked = [png_frames[0]]
    else:
        idxs = sorted({round(i * (len(png_frames) - 1) / (n_pick - 1)) for i in range(n_pick)})
        picked = [png_frames[i] for i in idxs]

    fig, axes = plt.subplots(1, len(picked), figsize=(4.2 * len(picked), 3.0))
    if len(picked) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#12161f")
    for ax, fp in zip(axes, picked):
        ax.imshow(mpimg.imread(str(fp)))
        ax.set_title(fp.stem.replace("frame_", "Frame "), color="white", fontsize=10, pad=6)
        ax.axis("off")
    fig.suptitle(
        f"Rendered Fluid Frames ({len(png_frames)} total | {render_engine} @ {cycles_samples} samples)",
        color="white",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(pad=0.8)
    strip_path = WORK_DIR / "frame_strip_preview.png"
    fig.savefig(str(strip_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
else:
    print("[preview] No PNG frames present (pipeline_mode='bake_only' or 0 frames rendered).")

mp4_path = FRAMES_DIR / "animation.mp4"
gif_path = FRAMES_DIR / "animation_preview.gif"
if mp4_path.is_file() and mp4_path.stat().st_size > 0:
    mp4_bytes = mp4_path.stat().st_size
    if mp4_bytes <= 8 * 1024 * 1024:
        b64 = base64.b64encode(mp4_path.read_bytes()).decode("ascii")
        display(HTML(
            f'<video width="780" controls autoplay loop muted playsinline '
            f'style="border-radius:8px;border:1px solid #334155;">'
            f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
            f"</video>"
            f'<p style="color:#94a3b8;font-size:12px;">MP4: {mp4_path} ({mp4_bytes / (1024*1024):.2f} MiB)</p>'
        ))
    else:
        print(f"[preview] MP4 ready at {mp4_path} ({mp4_bytes / (1024*1024):.2f} MiB)")
elif gif_path.is_file():
    print(f"[preview] GIF preview ready at {gif_path} ({gif_path.stat().st_size / 1024:.1f} KiB)")

# %% [markdown]
# ## 6. Multi-View April Sun Path Render
#
# > Renders four orthographic views (`Top`, `Front`, `Side`, `Isometric`) in a single persistent-data GPU Blender process under April 15 solar noon lighting (`45°N`, elevation `54.48°`, azimuth `180°`) and stitches a `1x4` figure.
#
# <details>
# <summary><b>Solar Geometry & Orthographic Bounding-Box Framing (Click to expand)</b></summary>
#
# - **Solar positioning**: Computes Spencer's solar declination for April 15 (`DOY 105`, declination `9.481°`), yielding `54.481°` solar elevation and `180.0°` azimuth at `12:00` solar noon at `45°N`.
# - **Single-process GPU framing**: Loads the baked `fluid_surface` mesh at the chosen frame, computes the tight world-space bounding box across visible geometry (`culverts array` + `fluid_surface`, excluding hidden inflow/outflow/domain helpers), configures four padded orthographic cameras, renders all four views inside one Cycles OptiX session, and stitches `multiview_april_sun.png` (`< 1.8 MiB`).
#
# </details>

# %%
SUNPATH_DIR = WORK_DIR / "sunpath_views"
SUNPATH_JSON = WORK_DIR / "sunpath_summary.json"

if enable_sunpath_multiview:
    bakefiles_dir = CACHE_DIR / "bakefiles"
    baked_nums = []
    if bakefiles_dir.is_dir():
        for marker in bakefiles_dir.glob("finished*.txt"):
            digits = "".join(ch for ch in marker.stem if ch.isdigit())
            if digits:
                baked_nums.append(int(digits))

    target_sun_frame = min(max(start_frame, 15), end_frame)
    if baked_nums and target_sun_frame not in baked_nums:
        target_sun_frame = max(baked_nums)

    sunpath_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "sunpath_multiview.py"),
        "--blender",
        BLENDER_BIN,
        "--blend",
        str(SANITIZED_BLEND),
        "--cache-dir",
        str(CACHE_DIR),
        "--out-dir",
        str(SUNPATH_DIR),
        "--frame",
        str(target_sun_frame),
        "--month",
        "4",
        "--day",
        "15",
        "--hour",
        "12.0",
        "--latitude",
        "45.0",
        "--engine",
        str(render_engine),
        "--samples",
        str(cycles_samples),
        "--json-out",
        str(SUNPATH_JSON),
    ]
    if not use_denoising:
        sunpath_cmd.append("--no-denoise")
    if not use_adaptive_sampling:
        sunpath_cmd.append("--no-adaptive-sampling")

    run_streaming(sunpath_cmd, check=True)
    sun_meta = json.loads(SUNPATH_JSON.read_text(encoding="utf-8"))
    print(
        f"[sunpath] Frame {sun_meta.get('frame')} | "
        f"Elevation {sun_meta.get('solar', {}).get('used_elevation_deg')} deg | "
        f"Azimuth {sun_meta.get('solar', {}).get('used_azimuth_deg')} deg | "
        f"Wall {sun_meta.get('wall_s')}s"
    )

    combined_png = SUNPATH_DIR / "multiview_april_sun.png"
    if combined_png.is_file():
        from IPython.display import Image as IPyImage, display

        display(IPyImage(filename=str(combined_png)))
else:
    print("[sunpath] Skipped (enable_sunpath_multiview=False).")

# %% [markdown]
# ## 7. Colab Execution Benchmarks (Optional Live Run / Verified Summary)
#
# > Benchmarks Legacy per-frame CLI spawning vs. Persistent CUDA, Persistent OptiX, OptiX + Denoise + Adaptive Sampling, Headless `BLENDER_EEVEE` (`vulkan` & `opengl`), and Sequential vs. Interleaved bake+render when `run_benchmarks = True`.
#
# <details>
# <summary><b>Benchmark Suite Methodology (Click to expand)</b></summary>
#
# `scripts/benchmark_colab.py` executes 6 controlled tests on the active GPU:
# 1. **Legacy Baseline**: Per-frame Blender subprocess launch + `CUDA` + CPU enabled + no persistent data.
# 2. **Persistent CUDA GPU-Only**: Single Blender session + `use_persistent_data=True` + `CUDA` (`cpu_enabled=False`).
# 3. **Persistent OptiX GPU-Only**: Single Blender session + `use_persistent_data=True` + `OPTIX` (`cpu_enabled=False`).
# 4. **Persistent OptiX + Denoise + Adaptive (`0.02`)**: Adds OpenImageDenoise / adaptive sampling.
# 5. **EEVEE Headless Probe**: Tests Blender 5.3 `BLENDER_EEVEE` under `--gpu-backend vulkan` and `--gpu-backend opengl`.
# 6. **Sequential vs. Interleaved Pipeline**: Compares sequential `bake + render` against overlapped CPU FLIP bake || GPU OptiX render.
#
# </details>

# %%
BENCH_DIR = WORK_DIR / "benchmarks"
BENCH_JSON = BENCH_DIR / "colab_benchmark.json"
BENCH_CHART = BENCH_DIR / "colab_benchmark_chart.png"

if run_benchmarks:
    n_span = end_frame - start_frame + 1
    bench_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "benchmark_colab.py"),
        "--blender",
        BLENDER_BIN,
        "--blend",
        str(RAW_BLEND),
        "--out-dir",
        str(BENCH_DIR),
        "--start-frame",
        str(start_frame),
        "--render-frames",
        str(min(5, n_span)),
        "--bake-frames",
        str(min(10, n_span)),
        "--samples",
        str(cycles_samples),
        "--resolution",
        str(flip_resolution_override),
    ]
    run_streaming(bench_cmd, check=True)

# Load live benchmark JSON if generated, else inspect repo outputs if present
display_json = None
for cand in (
    BENCH_JSON,
    SCRIPTS_DIR.parent / "outputs" / "colab_benchmark.json",
    Path.cwd() / "outputs" / "colab_benchmark.json",
):
    if cand.is_file():
        display_json = cand
        break

if display_json is not None:
    bdata = json.loads(display_json.read_text(encoding="utf-8"))
    tel = bdata.get("telemetry", {})
    print(f"[benchmark] Results from: {display_json} ({tel.get('gpu_name', 'GPU')}, {tel.get('blender_version', '')})")
    for cfg in bdata.get("render_configs", []):
        print(
            f"  - {cfg['name']:<46s}: {cfg['wall_s']:6.2f}s "
            f"({cfg['mean_per_frame_s']:5.2f}s/f, {cfg['speedup_vs_baseline']:.3f}x)"
        )
    eevee = bdata.get("eevee_probe", {}).get("backends", {})
    for bk, info in eevee.items():
        print(f"  - Headless BLENDER_EEVEE ({bk}): status={info.get('status')}, wall={info.get('wall_s')}s")
    ov = bdata.get("pipeline_overlap", {})
    if ov:
        print(
            f"  - Bake+Render ({ov.get('frame_count')}f): Sequential={ov['sequential']['wall_s']:.2f}s vs "
            f"Interleaved={ov['interleaved']['wall_s']:.2f}s ({ov.get('speedup')}x, -{ov.get('time_saved_pct')}% wall)"
        )

    chart_cand = display_json.parent / "colab_benchmark_chart.png"
    if not chart_cand.is_file():
        for alt in (
            SCRIPTS_DIR.parent / "assets" / "colab_benchmark_chart.png",
            Path.cwd() / "assets" / "colab_benchmark_chart.png",
        ):
            if alt.is_file():
                chart_cand = alt
                break
    if chart_cand.is_file():
        from IPython.display import Image as IPyImage, display

        display(IPyImage(filename=str(chart_cand)))
else:
    print(
        "[benchmark] run_benchmarks=False and no local colab_benchmark.json found.\n"
        "Set run_benchmarks = True in Section 2 to benchmark this runtime live."
    )

# %% [markdown]
# ## 8. Export & Sync Outputs
#
# > Packages rendered frames, MP4/GIF animations, sun-path views, JSON reports, and benchmark charts into `blender_colab_outputs.zip`, syncing to Google Drive when `save_to_drive = True`.
#
# <details>
# <summary><b>Archive Contents & Browser Download Notes (Click to expand)</b></summary>
#
# - Produces `/content/work/blender_colab_outputs.zip` (excluding raw Blender binaries and multi-hundred-MB FLIP bake caches so the download stays fast).
# - When `save_to_drive = True` and Google Drive is mounted, copies the zip archive, `animation.mp4`, `animation_preview.gif`, and `multiview_april_sun.png` into `MyDrive/BlenderColab_Outputs/<timestamp>/`.
# - Interactive browser download (`google.colab.files.download`) is provided as an opt-in snippet rather than auto-invoked so headless `colab exec` and Papermill pipelines complete deterministically.
#
# </details>

# %%
EXPORT_ZIP = WORK_DIR / "blender_colab_outputs.zip"
export_files: list[tuple[Path, str]] = []

for pattern_root, prefix in ((FRAMES_DIR, "frames"), (SUNPATH_DIR, "sunpath_views")):
    if pattern_root.is_dir():
        for fp in sorted(pattern_root.rglob("*")):
            if fp.is_file():
                export_files.append((fp, f"{prefix}/{fp.relative_to(pattern_root).as_posix()}"))

for extra in (
    INSTALL_JSON,
    INSPECT_JSON,
    SANITIZE_JSON,
    PIPELINE_JSON,
    SUNPATH_JSON,
    WORK_DIR / "frame_strip_preview.png",
    BENCH_JSON,
    BENCH_CHART,
):
    if extra.is_file():
        export_files.append((extra, extra.name))

tmp_zip = EXPORT_ZIP.with_suffix(".zip.tmp")
with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for src_path, arcname in export_files:
        zf.write(src_path, arcname=arcname)
os.replace(tmp_zip, EXPORT_ZIP)

print(
    f"[export] Packaged {len(export_files)} artifacts -> {EXPORT_ZIP} "
    f"({EXPORT_ZIP.stat().st_size / (1024 * 1024):.2f} MiB)"
)

if DRIVE_OUTPUT_DIR is not None:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_run = DRIVE_OUTPUT_DIR / f"run_{stamp}"
    dest_run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPORT_ZIP, dest_run / EXPORT_ZIP.name)
    for key_file in (
        FRAMES_DIR / "animation.mp4",
        FRAMES_DIR / "animation_preview.gif",
        SUNPATH_DIR / "multiview_april_sun.png",
    ):
        if key_file.is_file():
            shutil.copy2(key_file, dest_run / key_file.name)
    print(f"[export] Synced key outputs to Google Drive: {dest_run}")

if IN_COLAB:
    print(
        "\n[tip] To download the archive directly to your browser in an interactive Colab tab, run:\n"
        f"      from google.colab import files; files.download({str(EXPORT_ZIP)!r})"
    )
