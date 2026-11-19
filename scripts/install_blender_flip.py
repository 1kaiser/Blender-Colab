#!/usr/bin/env python3
"""Install Blender (5.3.0 default, 5.2.4 LTS, 4.5.16 LTS) and FLIP Fluids v1.8.9.

Supports three addon source modes:
  - build_from_source (default): builds clean unwatermarked v1.8.9 via scripts/build_flip_fluids.sh
  - custom_url: installs a FLIP Fluids zip from a URL or local path
  - official_demo: downloads rlguy's prebuilt v1.8.9 Demo zip (warns about 3D Demo watermark)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


BLENDER_VERSIONS = {
    "5.3.0": "5.3",
    "5.2.4": "5.2",
    "4.5.16": "4.5",
}

FLIP_TAG = "v1.8.9"
FLIP_VERSION = "1.8.9"
OFFICIAL_DEMO_URL = (
    "https://github.com/rlguy/Blender-FLIP-Fluids/releases/download/"
    "v1.8.9/FLIP_Fluids_addon_1.8.9_demo_release_.2026-11-10.zip"
)
GITHUB_RELEASE_API = (
    "https://api.github.com/repos/rlguy/Blender-FLIP-Fluids/releases/tags/v1.8.9"
)

MINIMAL_RUNTIME_PACKAGES = [
    "libxkbcommon0",
    "libsm6",
    "libxi6",
    "libxxf86vm1",
    "libgl1",
    "libegl1",
    "libvulkan1",
]

SO_TO_APT_PACKAGE = {
    "libxkbcommon.so": "libxkbcommon0",
    "libxkbcommon-x11.so": "libxkbcommon-x11-0",
    "libSM.so": "libsm6",
    "libICE.so": "libice6",
    "libXi.so": "libxi6",
    "libXxf86vm.so": "libxxf86vm1",
    "libXfixes.so": "libxfixes3",
    "libXrender.so": "libxrender1",
    "libX11.so": "libx11-6",
    "libGL.so": "libgl1",
    "libEGL.so": "libegl1",
    "libvulkan.so": "libvulkan1",
    "libGLU.so": "libglu1-mesa",
    "libdbus-1.so": "libdbus-1-3",
    "libfontconfig.so": "libfontconfig1",
}

BUILD_PACKAGES = ["cmake", "g++", "make", "build-essential", "git"]
ALEMBIC_BUILD_PACKAGES = ["libalembic-dev", "libimath-dev"]


def default_work_dir() -> str:
    if os.path.isdir("/content"):
        return "/content/work"
    return os.path.abspath("./work")


def blender_mirror_urls(version: str) -> list[str]:
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid Blender version '{version}'. Expected X.Y.Z format.")
    series = BLENDER_VERSIONS.get(version, f"{parts[0]}.{parts[1]}")
    tar_name = f"blender-{version}-linux-x64.tar.xz"
    return [
        f"https://download.blender.org/release/Blender{series}/{tar_name}",
        f"https://mirrors.dotsrc.org/blender/release/Blender{series}/{tar_name}",
        f"https://ftp.nluug.nl/pub/graphics/blender/release/Blender{series}/{tar_name}",
    ]


def run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        env=env,
        check=check,
        text=True,
        capture_output=capture,
    )


def download_file(url: str, dest: Path, label: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    desc = label or url
    print(f"[download] Fetching {desc} -> {dest}", flush=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Blender-Colab-Installer/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else 0
            downloaded = 0
            next_pct_log = 20
            with tmp_path.open("wb") as out_fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = int(downloaded * 100 / total_bytes)
                        if pct >= next_pct_log:
                            print(
                                f"  ... {pct}% ({downloaded // (1024 * 1024)} / "
                                f"{total_bytes // (1024 * 1024)} MiB)",
                                flush=True,
                            )
                            next_pct_log += 20
        if dest.exists():
            dest.unlink()
        os.replace(tmp_path, dest)
        return dest
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def download_with_mirrors(urls: list[str], dest: Path, label: str) -> Path:
    errors: list[str] = []
    for url in urls:
        try:
            return download_file(url, dest, label=f"{label} ({url})")
        except Exception as exc:
            msg = f"{url}: {exc}"
            print(f"[download] Mirror failed -> {msg}", flush=True)
            errors.append(msg)
    raise RuntimeError(
        f"Failed to download {label} from all mirrors:\n  " + "\n  ".join(errors)
    )


def ensure_blender(
    version: str,
    work_dir: Path,
    install_dir: Path,
    force_reinstall: bool = False,
) -> Path:
    blender_bin = install_dir / "blender"
    if blender_bin.is_file() and not force_reinstall:
        res = run_cmd([str(blender_bin), "--version"], check=False, capture=True)
        if res.returncode == 0:
            print(f"[blender] Reusing existing Blender at {blender_bin}: {res.stdout.strip()}")
            return blender_bin
        print("[blender] Existing binary check failed; re-extracting.", flush=True)

    tar_name = f"blender-{version}-linux-x64.tar.xz"
    tar_path = work_dir / tar_name
    urls = blender_mirror_urls(version)

    def extract_tar(archive: Path) -> bool:
        if install_dir.exists() and force_reinstall:
            shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)
        res = run_cmd(
            ["tar", "-xf", str(archive), "-C", str(install_dir), "--strip-components=1"],
            check=False,
        )
        return res.returncode == 0 and blender_bin.is_file()

    if tar_path.is_file() and tar_path.stat().st_size > 50 * 1024 * 1024:
        print(f"[blender] Found prefetched archive {tar_path} ({tar_path.stat().st_size} bytes)")
        if extract_tar(tar_path):
            return blender_bin
        print("[blender] Existing tarball failed to extract; removing and re-downloading.", flush=True)
        tar_path.unlink(missing_ok=True)
        if install_dir.exists():
            shutil.rmtree(install_dir)

    download_with_mirrors(urls, tar_path, f"Blender {version}")
    if not extract_tar(tar_path):
        raise RuntimeError(f"Failed to extract Blender {version} from {tar_path}")
    return blender_bin


def is_apt_pkg_installed(pkg: str) -> bool:
    res = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}\n", pkg],
        text=True,
        capture_output=True,
    )
    return res.returncode == 0 and "install ok installed" in res.stdout


def install_apt_packages(packages: list[str], skip_apt: bool = False) -> None:
    missing = [p for p in dict.fromkeys(packages) if not is_apt_pkg_installed(p)]
    if not missing:
        print("[deps] All required system packages already installed.")
        return

    if skip_apt:
        print(f"[deps] --skip-apt set; skipping install of missing packages: {missing}")
        return

    if shutil.which("apt-get") is None:
        print(f"[deps] apt-get not available; missing packages: {missing}")
        return

    prefix: list[str] = []
    if os.geteuid() != 0:
        sudo_bin = shutil.which("sudo")
        if sudo_bin and subprocess.run([sudo_bin, "-n", "true"], capture_output=True).returncode == 0:
            prefix = [sudo_bin]
        else:
            print(
                f"[deps] WARNING: Not root and no passwordless sudo; cannot install: {missing}",
                file=sys.stderr,
            )
            return

    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    print(f"[deps] Installing minimal packages: {' '.join(missing)}", flush=True)
    run_cmd(prefix + ["apt-get", "update", "-qq"], env=env)
    run_cmd(
        prefix + ["apt-get", "install", "-y", "-qq", "--no-install-recommends"] + missing,
        env=env,
    )


def find_missing_ldd_libs(binary_path: Path) -> list[str]:
    if shutil.which("ldd") is None:
        return []
    res = subprocess.run(["ldd", str(binary_path)], text=True, capture_output=True)
    missing: list[str] = []
    for line in res.stdout.splitlines():
        if "=> not found" in line:
            lib_name = line.strip().split("=>", 1)[0].strip()
            missing.append(lib_name)
    return missing


def ensure_runtime_and_build_deps(
    blender_bin: Path,
    *,
    need_build_tools: bool,
    with_alembic: bool,
    skip_apt: bool,
) -> None:
    # Explicitly avoid legacy libtcmalloc-minimal4 / LD_PRELOAD overrides.
    missing_libs = find_missing_ldd_libs(blender_bin)
    if missing_libs:
        print(f"[deps] ldd reported missing libs for {blender_bin}: {missing_libs}")
    else:
        print(f"[deps] ldd check on {blender_bin}: no missing shared libs.")

    pkgs = list(MINIMAL_RUNTIME_PACKAGES)
    for lib_name in missing_libs:
        for prefix, pkg in SO_TO_APT_PACKAGE.items():
            if lib_name.startswith(prefix):
                pkgs.append(pkg)

    if need_build_tools:
        pkgs.extend(BUILD_PACKAGES)
        if with_alembic:
            pkgs.extend(ALEMBIC_BUILD_PACKAGES)

    install_apt_packages(pkgs, skip_apt=skip_apt)

    still_missing = find_missing_ldd_libs(blender_bin)
    if still_missing:
        raise RuntimeError(
            f"Blender binary still has unresolved shared libraries after apt install: {still_missing}"
        )


def resolve_official_demo_url() -> str:
    try:
        req = urllib.request.Request(
            GITHUB_RELEASE_API,
            headers={"User-Agent": "Blender-Colab-Installer/2.0"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if name.endswith(".zip") and "demo" in name.lower():
                return asset["browser_download_url"]
    except Exception as exc:
        print(f"[addon] GitHub release API lookup failed ({exc}); using pinned Demo URL.")
    return OFFICIAL_DEMO_URL


def is_valid_cached_flip_zip(zip_path: Path, blender_bin: Path | None = None) -> bool:
    """Verify zip integrity, libffengine.so presence, <=64-char tagline, and Blender validation."""
    if not zip_path.is_file() or zip_path.stat().st_size < 100_000:
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                print(f"[addon] Cached zip CRC error on {bad}: {zip_path}", flush=True)
                return False
            names = set(zf.namelist())
            has_so = any(n.endswith("ffengine/lib/libffengine.so") for n in names)
            if not has_so:
                print(f"[addon] Cached zip missing libffengine.so: {zip_path}", flush=True)
                return False
            has_unwatermarked_marker = any(
                n.endswith("resources/installation_data/unwatermarked_build") for n in names
            )
            if not has_unwatermarked_marker:
                print(
                    f"[addon] Cached zip predates unwatermarked engine patch: {zip_path}",
                    flush=True,
                )
                return False
            manifest_candidates = [
                n for n in names if n == "blender_manifest.toml" or (n.endswith("/blender_manifest.toml") and n.count("/") == 1)
            ]
            if not manifest_candidates:
                print(f"[addon] Cached zip missing blender_manifest.toml: {zip_path}", flush=True)
                return False
            manifest_text = zf.read(manifest_candidates[0]).decode("utf-8", errors="replace")
            for line in manifest_text.splitlines():
                if line.strip().startswith("tagline"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if len(val) > 64:
                        print(
                            f"[addon] Cached zip has invalid >64-char tagline ({len(val)} chars): {zip_path}",
                            flush=True,
                        )
                        return False
    except Exception as exc:
        print(f"[addon] Cached zip unreadable ({exc}): {zip_path}", flush=True)
        return False

    if blender_bin is not None and blender_bin.is_file():
        with tempfile.TemporaryDirectory(prefix="flip_cache_chk_") as tmp_str:
            check_zip, is_ext = prepare_root_manifest_zip(zip_path, Path(tmp_str))
            if is_ext:
                res = run_cmd(
                    [str(blender_bin), "--command", "extension", "validate", str(check_zip)],
                    check=False,
                    capture=True,
                )
                if res.returncode != 0:
                    print(
                        f"[addon] Cached zip failed 'extension validate': {zip_path}\n{res.stdout}\n{res.stderr}",
                        flush=True,
                    )
                    return False
    return True


def obtain_addon_zip(
    args: argparse.Namespace,
    work_dir: Path,
    blender_bin: Path,
) -> tuple[Path, bool]:
    """Return (zip_path, needs_build_tools_used)."""
    addons_dir = work_dir / "addons"
    addons_dir.mkdir(parents=True, exist_ok=True)

    if args.addon_source == "build_from_source":
        candidates = [
            addons_dir / f"flip_fluids_addon_{FLIP_VERSION}_linux_x64_ext.zip",
            addons_dir / f"flip_fluids_addon_{FLIP_VERSION}_linux_x64.zip",
            work_dir / f"flip_fluids_addon_{FLIP_VERSION}_linux_x64_ext.zip",
            work_dir / f"flip_fluids_addon_{FLIP_VERSION}_linux_x64.zip",
        ]
        if not args.force_rebuild:
            for cand in candidates:
                if cand.is_file():
                    if is_valid_cached_flip_zip(cand, blender_bin):
                        print(f"[addon] Reusing verified unwatermarked source build: {cand}")
                        return cand, False
                    print(f"[addon] Removing stale/invalid cached archive: {cand}", flush=True)
                    cand.unlink(missing_ok=True)

        build_script = Path(__file__).resolve().parent / "build_flip_fluids.sh"
        if not build_script.is_file():
            raise FileNotFoundError(f"Missing build script: {build_script}")

        src_dir = work_dir / "src" / "Blender-FLIP-Fluids"
        cmd = [
            "bash",
            str(build_script),
            "--tag",
            FLIP_TAG,
            "--out-dir",
            str(addons_dir),
            "--src-dir",
            str(src_dir),
            "--jobs",
            str(args.jobs),
            "--blender-bin",
            str(blender_bin),
        ]
        cmd.append("--with-alembic" if args.with_alembic else "--without-alembic")
        cmd.append("--no-native" if args.no_native else "--native")
        if args.march_native:
            cmd.append("--march-native")

        run_cmd(cmd)
        ext_zip = addons_dir / f"flip_fluids_addon_{FLIP_VERSION}_linux_x64_ext.zip"
        folder_zip = addons_dir / f"flip_fluids_addon_{FLIP_VERSION}_linux_x64.zip"
        chosen = ext_zip if ext_zip.is_file() else folder_zip
        if not chosen.is_file():
            raise FileNotFoundError(f"Expected built addon zip not found in {addons_dir}")
        return chosen, True

    if args.addon_source == "custom_url":
        target = args.custom_addon_url
        if not target:
            raise ValueError("--custom-addon-url is required when --addon-source custom_url")
        local_path = Path(target)
        if local_path.is_file():
            print(f"[addon] Using local custom addon zip: {local_path}")
            return local_path.resolve(), False
        fname = os.path.basename(target.split("?", 1)[0]) or "custom_flip_fluids_addon.zip"
        dest = addons_dir / fname
        download_file(target, dest, label="Custom FLIP Fluids addon")
        if not zipfile.is_zipfile(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded custom addon is not a valid zip: {target}")
        return dest, False

    if args.addon_source == "official_demo":
        print(
            "\n" + "=" * 78 + "\n"
            "WARNING: --addon-source=official_demo downloads rlguy's prebuilt Demo\n"
            "release. The Demo build bakes a 3D 'FLIP Fluids Demo' text watermark\n"
            "into simulation fluid meshes! Use --addon-source build_from_source for\n"
            "clean, unwatermarked fluid meshes.\n"
            + "=" * 78 + "\n",
            flush=True,
        )
        demo_url = resolve_official_demo_url()
        dest = addons_dir / os.path.basename(demo_url)
        if not (dest.is_file() and dest.stat().st_size > 100_000 and zipfile.is_zipfile(dest)):
            dest.unlink(missing_ok=True)
            download_file(demo_url, dest, label="Official FLIP Fluids Demo v1.8.9")
        return dest, False

    raise ValueError(f"Unsupported --addon-source: {args.addon_source}")


def inspect_zip_manifest(zip_path: Path) -> tuple[bool, str | None]:
    """Return (has_manifest, top_prefix_if_wrapped)."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    if "blender_manifest.toml" in names:
        return True, None
    for n in names:
        if n.endswith("/blender_manifest.toml") and n.count("/") == 1:
            return True, n.split("/", 1)[0] + "/"
    return False, None


def prepare_root_manifest_zip(zip_path: Path, tmp_dir: Path) -> tuple[Path, bool]:
    """Return (installable_zip_path, is_extension_manifest_zip)."""
    has_manifest, prefix = inspect_zip_manifest(zip_path)
    if not has_manifest:
        return zip_path, False
    if prefix is None:
        return zip_path, True

    out_zip = tmp_dir / f"{zip_path.stem}_root_ext.zip"
    print(f"[addon] Repackaging wrapped zip (stripping '{prefix}') -> {out_zip}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zin, zipfile.ZipFile(
        out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zout:
        for info in zin.infolist():
            if info.is_dir():
                continue
            if not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix) :]
            if not rel:
                continue
            new_info = zipfile.ZipInfo(rel, date_time=info.date_time)
            new_info.external_attr = info.external_attr
            new_info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(new_info, zin.read(info.filename))
    return out_zip, True


def run_blender_python(
    blender_bin: Path,
    code: str,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        script_path = Path(fh.name)
    try:
        return run_cmd([str(blender_bin), "-b", "-P", str(script_path)], env=env, check=check, capture=True)
    finally:
        script_path.unlink(missing_ok=True)


def install_extension_into_blender(
    blender_bin: Path,
    addon_zip: Path,
    env: dict[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="flip_ext_") as tmp_str:
        tmp_dir = Path(tmp_str)
        install_zip, is_extension = prepare_root_manifest_zip(addon_zip, tmp_dir)

        if is_extension:
            res = run_cmd(
                [
                    str(blender_bin),
                    "--command",
                    "extension",
                    "install-file",
                    "-r",
                    "user_default",
                    "--enable",
                    str(install_zip),
                ],
                env=env,
                check=False,
                capture=True,
            )
            if res.returncode == 0:
                print(res.stdout.strip())
                print("[install] Blender Extension CLI install-file succeeded.")
                return

            print(
                f"[install] extension install-file returned {res.returncode}; "
                f"falling back to direct repository extraction.\n"
                f"stdout: {res.stdout}\nstderr: {res.stderr}",
                flush=True,
            )
            fallback_code = f"""
import bpy, os, shutil, zipfile, addon_utils
try:
    repo = bpy.context.preferences.extensions.repos.get("user_default")
    ext_root = repo.directory if repo and repo.directory else ""
except Exception:
    ext_root = ""
if not ext_root:
    ext_root = bpy.utils.user_resource("EXTENSIONS", path="user_default")
target_dir = os.path.join(ext_root, "flip_fluids_addon")
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)
os.makedirs(target_dir, exist_ok=True)
with zipfile.ZipFile({str(install_zip)!r}, "r") as zf:
    zf.extractall(target_dir)
try:
    bpy.ops.extensions.repo_refresh_all()
except Exception:
    try:
        bpy.ops.extensions.repo_sync_all()
    except Exception as e:
        print("repo sync note:", e)
mod_name = "bl_ext.user_default.flip_fluids_addon"
addon_utils.enable(mod_name, default_set=True, persistent=True)
bpy.ops.wm.save_userpref()
print("Fallback extracted to:", target_dir)
"""
            fb_res = run_blender_python(blender_bin, fallback_code, env=env, check=True)
            print(fb_res.stdout.strip())
        else:
            legacy_code = f"""
import bpy, addon_utils
bpy.ops.preferences.addon_install(overwrite=True, filepath={str(addon_zip)!r})
addon_utils.enable("flip_fluids_addon", default_set=True, persistent=True)
bpy.ops.wm.save_userpref()
"""
            leg_res = run_blender_python(blender_bin, legacy_code, env=env, check=True)
            print(leg_res.stdout.strip())


def bootstrap_flip_fluids(blender_bin: Path, env: dict[str, str]) -> None:
    bootstrap_code = """
import addon_utils
import bpy
import importlib
import os
import sys

def find_flip_module():
    prefs = bpy.context.preferences
    for key in prefs.addons.keys():
        if key == "flip_fluids_addon" or key.endswith(".flip_fluids_addon"):
            return key
    candidates = []
    for mod in addon_utils.modules():
        name = getattr(mod, "__name__", "")
        if name == "flip_fluids_addon" or name.endswith(".flip_fluids_addon"):
            candidates.append(name)
    for pref in ("bl_ext.user_default.flip_fluids_addon", "flip_fluids_addon"):
        if pref in candidates:
            return pref
    return candidates[0] if candidates else None

mod_name = find_flip_module()
if not mod_name:
    raise RuntimeError("FLIP Fluids module not found in Blender addons/extensions after installation.")

if mod_name not in bpy.context.preferences.addons:
    try:
        bpy.ops.preferences.addon_enable(module=mod_name)
    except Exception:
        addon_utils.enable(mod_name, default_set=True, persistent=True)

mod = addon_utils.enable(mod_name, default_set=True, persistent=True)
if mod is None:
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
addon_dir = os.path.dirname(os.path.realpath(mod.__file__))
inst_dir = os.path.join(addon_dir, "resources", "installation_data")
os.makedirs(inst_dir, exist_ok=True)

res = bpy.ops.flip_fluid_operators.complete_installation()
print("complete_installation() ->", res)

inst_file = os.path.join(inst_dir, "installation_complete")
if not os.path.isfile(inst_file):
    raise RuntimeError(f"Expected installation_complete marker missing at {inst_file}")

bpy.ops.wm.save_userpref()
print("Saved userpref with enabled module:", mod_name)
"""
    res = run_blender_python(blender_bin, bootstrap_code, env=env, check=True)
    print(res.stdout.strip())


def self_check_blender_flip(blender_bin: Path, env: dict[str, str]) -> dict:
    check_code = """
import addon_utils
import bpy
import importlib
import json
import os
import sys

prefs = bpy.context.preferences
enabled_mods = [
    k for k in prefs.addons.keys()
    if k == "flip_fluids_addon" or k.endswith(".flip_fluids_addon")
]
if not enabled_mods:
    raise RuntimeError(
        "FLIP Fluids not auto-enabled in fresh headless Blender session! "
        f"Enabled addons: {list(prefs.addons.keys())}"
    )

mod_name = (
    "bl_ext.user_default.flip_fluids_addon"
    if "bl_ext.user_default.flip_fluids_addon" in enabled_mods
    else enabled_mods[0]
)
mod = importlib.import_module(mod_name)
inst_utils = mod.utils.installation_utils
is_complete = bool(inst_utils.is_installation_complete())
if not is_complete:
    raise RuntimeError("installation_utils.is_installation_complete() returned False!")

ffengine_mod = mod.ffengine.ffengine
ffengine_lib = getattr(ffengine_mod, "ffengine", ffengine_mod)
loaded_lib = ffengine_lib._load_library("ffengine")
lib_path = getattr(loaded_lib, "_name", "")

sim = mod.ffengine.FluidSimulation(10, 10, 10, 0.1)
engine_ver = list(sim.get_version())
del sim

load_post_names = [getattr(fn, "__module__", "") + "." + getattr(fn, "__name__", "") for fn in bpy.app.handlers.load_post]
has_load_post = any("flip_fluids" in name for name in load_post_names)

addon_dir = os.path.dirname(os.path.realpath(mod.__file__))
inst_marker = os.path.join(addon_dir, "resources", "installation_data", "installation_complete")

bl_info = {}
try:
    bl_info = addon_utils.module_bl_info(mod) or {}
except Exception:
    pass
if not bl_info:
    bl_info = getattr(mod, "bl_info", {}) or {}

manifest_data = {}
manifest_path = os.path.join(addon_dir, "blender_manifest.toml")
if os.path.isfile(manifest_path):
    try:
        import tomllib
        with open(manifest_path, "rb") as mf:
            manifest_data = tomllib.load(mf)
    except Exception:
        pass

bl_name = bl_info.get("name") or manifest_data.get("name")
bl_ver = list(bl_info.get("version", ())) if bl_info.get("version") else (
    [int(x) for x in str(manifest_data.get("version", "")).split(".") if x.isdigit()] or None
)
bl_desc = bl_info.get("description") or manifest_data.get("tagline")

summary = {
    "blender_version": list(bpy.app.version),
    "blender_version_string": bpy.app.version_string,
    "blender_binary": bpy.app.binary_path,
    "enabled_module": mod_name,
    "all_enabled_flip_modules": enabled_mods,
    "bl_info_name": bl_name,
    "bl_info_version": bl_ver,
    "bl_info_description": bl_desc,
    "support_license": inst_utils.get_support_license_label(),
    "installation_complete": is_complete,
    "installation_complete_file": inst_marker,
    "installation_complete_file_exists": os.path.isfile(inst_marker),
    "libffengine_path": lib_path,
    "libffengine_version": engine_ver,
    "load_post_registered": has_load_post,
    "compiler_info": inst_utils.get_compiler_info_list(),
    "library_list": inst_utils.get_library_list(),
}
print("__FLIP_SELF_CHECK_JSON_START__")
print(json.dumps(summary, indent=2))
print("__FLIP_SELF_CHECK_JSON_END__")
"""
    res = run_blender_python(blender_bin, check_code, env=env, check=True)
    m = re.search(
        r"__FLIP_SELF_CHECK_JSON_START__\s*(\{.*?\})\s*__FLIP_SELF_CHECK_JSON_END__",
        res.stdout,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError(
            "Self-check did not emit JSON payload.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )
    return json.loads(m.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install Blender (5.3.0 / 5.2.4 / 4.5.16) and FLIP Fluids v1.8.9."
    )
    parser.add_argument(
        "--blender-version",
        default="5.3.0",
        help="Blender version to install: 5.3.0 (default), 5.2.4 LTS, 4.5.16 LTS, or X.Y.Z",
    )
    parser.add_argument(
        "--work-dir",
        default=default_work_dir(),
        help="Working/cache directory (default: /content/work on Colab else ./work)",
    )
    parser.add_argument(
        "--install-dir",
        default="",
        help="Directory to extract Blender into (default: <work-dir>/blender-<version>)",
    )
    parser.add_argument(
        "--addon-source",
        choices=["build_from_source", "custom_url", "official_demo"],
        default="build_from_source",
        help="How to obtain FLIP Fluids v1.8.9 (default: build_from_source)",
    )
    parser.add_argument(
        "--custom-addon-url",
        "--addon-url",
        dest="custom_addon_url",
        default="",
        help="Zip URL or local path when --addon-source custom_url is used",
    )
    parser.add_argument(
        "--with-alembic",
        action="store_true",
        help="Enable Alembic export compilation when building from source",
    )
    parser.add_argument(
        "--without-alembic",
        action="store_true",
        help="Disable Alembic export compilation (default)",
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="Do not patch out force_link_glibc_2.5.h during source build",
    )
    parser.add_argument(
        "--march-native",
        action="store_true",
        help="Compile C++ engine with -march=native",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 4, 12),
        help="Parallel make jobs when building from source",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild FLIP Fluids from source even if a cached zip exists",
    )
    parser.add_argument(
        "--force-reinstall-blender",
        action="store_true",
        help="Re-extract Blender archive even if binary already exists",
    )
    parser.add_argument(
        "--user-config-dir",
        default="",
        help="Optional BLENDER_USER_CONFIG directory override",
    )
    parser.add_argument(
        "--skip-apt",
        action="store_true",
        help="Skip apt-get dependency installation",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create /usr/local/bin/blender symlink pointing to installed binary",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write verification summary JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.without_alembic:
        args.with_alembic = False

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    install_dir = (
        Path(args.install_dir).resolve()
        if args.install_dir
        else work_dir / f"blender-{args.blender_version}"
    )

    blender_bin = ensure_blender(
        args.blender_version,
        work_dir,
        install_dir,
        force_reinstall=args.force_reinstall_blender,
    )

    need_build = args.addon_source == "build_from_source"
    ensure_runtime_and_build_deps(
        blender_bin,
        need_build_tools=need_build,
        with_alembic=args.with_alembic,
        skip_apt=args.skip_apt,
    )

    addon_zip, _ = obtain_addon_zip(args, work_dir, blender_bin)

    env = os.environ.copy()
    if args.user_config_dir:
        cfg_dir = Path(args.user_config_dir).resolve()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        env["BLENDER_USER_CONFIG"] = str(cfg_dir)

    install_extension_into_blender(blender_bin, addon_zip, env)
    bootstrap_flip_fluids(blender_bin, env)
    summary = self_check_blender_flip(blender_bin, env)
    summary["addon_source"] = args.addon_source
    summary["addon_zip"] = str(addon_zip)
    summary["blender_bin_requested"] = str(blender_bin)

    if args.symlink:
        link_path = Path("/usr/local/bin/blender")
        try:
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(blender_bin)
            summary["symlink"] = str(link_path)
        except OSError as exc:
            print(f"[install] Could not symlink {link_path}: {exc}", file=sys.stderr)

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Blender + FLIP Fluids Installation Summary")
    print("=" * 78)
    print(f"  Blender Version      : {summary['blender_version_string']}")
    print(f"  Blender Binary       : {blender_bin}")
    print(f"  Addon Source Mode    : {args.addon_source}")
    print(f"  Addon Zip            : {addon_zip}")
    print(f"  Enabled Module       : {summary['enabled_module']}")
    print(f"  FLIP Fluids Version  : {summary['bl_info_version']} ({summary['bl_info_description']})")
    print(f"  Engine Library       : {summary['libffengine_path']}")
    print(f"  Engine C++ Version   : {summary['libffengine_version']}")
    print(f"  Install Complete     : {summary['installation_complete']}")
    print(f"  load_post Registered : {summary['load_post_registered']}")
    print("=" * 78)
    print(f"BLENDER_BIN={blender_bin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
