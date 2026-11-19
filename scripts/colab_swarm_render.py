#!/usr/bin/env python3
"""Local-to-Colab Distributed Multi-Session Render Orchestrator.

Provisions up to N Google Colab GPU workers (`L4`/`T4`) via `colab-cli`,
uploads the sanitized `.blend` + baked FLIP Fluids cache tarball once per worker,
executes disjoint frame slices (`--worker-id K --num-workers N`) using Blender 5.3
persistent-data OptiX rendering, downloads `.tar.gz` frame bundles (avoiding raw
JSON `colab download` bugs), refreshes expiring proxy tokens (>50 min), guarantees
`colab stop` in `try...finally`, and assembles final `animation.mp4` + preview GIF.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bake_and_render  # noqa: E402


DEFAULT_COLAB_BIN = "/home/kaiser/.miniforge/envs/num_python/bin/colab"


def resolve_colab_bin(explicit: str = "") -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    if os.path.isfile(DEFAULT_COLAB_BIN):
        return DEFAULT_COLAB_BIN
    found = shutil.which("colab")
    if found:
        return found
    raise FileNotFoundError(
        f"Could not find `colab` CLI binary (checked {DEFAULT_COLAB_BIN} and PATH)."
    )


def refresh_colab_proxy_token(session_name: str, verbose: bool = True) -> bool:
    """Refresh Colab runtimeProxyInfo token in ~/.config/colab-cli before 3600s expiry."""
    try:
        from colab_cli.common import State, _apply_proxy_info  # type: ignore
    except ImportError:
        return False

    try:
        st = State()
        s = st.store.get(session_name)
        if s is None:
            return False
        assignments = st.client.list_assignments()
        by_ep = {a.endpoint: a for a in assignments}
        if s.endpoint in by_ep and by_ep[s.endpoint].runtime_proxy_info:
            _apply_proxy_info(s, by_ep[s.endpoint].runtime_proxy_info)
            st.store.add(s)
            if verbose:
                print(
                    f"[swarm:{session_name}] Refreshed proxy token (expires_at={s.token_expires_at})",
                    flush=True,
                )
            return True
    except Exception as exc:
        if verbose:
            print(f"[swarm:{session_name}] Token refresh note: {exc}", flush=True)
    return False


def force_unassign_endpoints(endpoints: list[str], verbose: bool = True) -> None:
    """Fallback server-side cleanup ensuring no Colab endpoints remain assigned."""
    if not endpoints:
        return
    try:
        from colab_cli.common import State  # type: ignore

        st = State()
        active = {a.endpoint for a in st.client.list_assignments()}
        for ep in endpoints:
            if ep in active:
                try:
                    st.client.unassign(ep)
                    if verbose:
                        print(f"[swarm] Force-unassigned leftover endpoint {ep}", flush=True)
                except Exception as exc:
                    if verbose:
                        print(f"[swarm] Failed to unassign {ep}: {exc}", flush=True)
    except Exception:
        pass


def get_session_endpoint(session_name: str) -> Optional[str]:
    try:
        from colab_cli.common import State  # type: ignore

        s = State().store.get(session_name)
        return s.endpoint if s else None
    except Exception:
        return None


def _run_colab_cmd(
    colab_bin: str,
    args: list[str],
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [colab_bin, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def _provision_worker_session(
    colab_bin: str,
    session_name: str,
    gpu_choices: list[str],
) -> tuple[str, str]:
    last_err: Optional[Exception] = None
    for gpu in gpu_choices:
        try:
            print(f"[swarm:{session_name}] Provisioning {gpu} session...", flush=True)
            _run_colab_cmd(colab_bin, ["new", "--gpu", gpu, "-s", session_name], timeout=180)
            ep = get_session_endpoint(session_name) or ""
            print(f"[swarm:{session_name}] Ready ({gpu}, endpoint={ep})", flush=True)
            return gpu, ep
        except Exception as exc:
            last_err = exc
            print(f"[swarm:{session_name}] Provisioning {gpu} failed: {exc}", flush=True)
    raise RuntimeError(f"Could not provision session '{session_name}': {last_err}")


def _build_local_bundles(
    tmp_dir: Path,
    blend_path: Path,
    cache_dir: Path,
    addon_zip: Optional[Path],
) -> tuple[Path, Path]:
    scripts_tar = tmp_dir / "scripts_bundle.tar.gz"
    with tarfile.open(scripts_tar, "w:gz") as tf:
        for py_file in sorted(SCRIPTS_DIR.glob("*")):
            if py_file.is_file() and py_file.suffix in {".py", ".sh"}:
                tf.add(py_file, arcname=f"scripts/{py_file.name}")
        if addon_zip is not None and addon_zip.is_file():
            tf.add(addon_zip, arcname=f"addon/{addon_zip.name}")

    scene_tar = tmp_dir / "scene_cache_bundle.tar.gz"
    with tarfile.open(scene_tar, "w:gz") as tf:
        tf.add(blend_path, arcname=f"scene/{blend_path.name}")
        if cache_dir.is_dir():
            for item in sorted(cache_dir.rglob("*")):
                rel = item.relative_to(cache_dir)
                tf.add(item, arcname=f"flip_fluid_cache/{rel.as_posix()}")

    return scripts_tar, scene_tar


def _write_worker_driver_script(
    path: Path,
    *,
    worker_id: int,
    num_workers: int,
    blend_name: str,
    addon_zip_name: str,
    blender_version: str,
    addon_source: str,
    custom_addon_url: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
    samples: Optional[int],
    adaptive_threshold: float,
    compute_type: str,
    sanitize_on_worker: bool,
) -> None:
    """Generate standalone worker setup + detached launcher script for colab exec -f."""
    import json

    cfg = {
        "worker_id": worker_id,
        "num_workers": num_workers,
        "blend_name": blend_name,
        "addon_zip_name": addon_zip_name,
        "blender_version": blender_version,
        "addon_source": addon_source,
        "custom_addon_url": custom_addon_url,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "samples": samples,
        "adaptive_threshold": adaptive_threshold,
        "compute_type": compute_type,
        "sanitize_on_worker": sanitize_on_worker,
    }
    cfg_json = json.dumps(cfg)

    runner_template = '''import json, subprocess, sys, tarfile
from pathlib import Path

cfg = json.loads(Path("/content/work/worker_cfg.json").read_text(encoding="utf-8"))
wid = int(cfg["worker_id"])
WORK = Path("/content/work")
STATUS = WORK / f"worker_{wid}.status.txt"
frames_dir = WORK / f"frames_worker_{wid}"
frames_dir.mkdir(parents=True, exist_ok=True)
blender_bin = str(WORK / f"blender-{cfg['blender_version']}" / "blender")
blend_file = str(WORK / "scene" / cfg["blend_name"])
cache_dir = str(WORK / "flip_fluid_cache")

cmd = [
    sys.executable,
    str(WORK / "scripts" / "bake_and_render.py"),
    "--mode", "render_persistent",
    "--blender", blender_bin,
    "--blend", blend_file,
    "--cache-dir", cache_dir,
    "--out-dir", str(frames_dir),
    "--worker-id", str(wid),
    "--num-workers", str(cfg["num_workers"]),
    "--compute-type", cfg["compute_type"],
    "--adaptive-threshold", str(cfg["adaptive_threshold"]),
    "--skip-existing",
    "--no-video",
]
if cfg.get("sanitize_on_worker"):
    cmd.append("--sanitize")
if cfg.get("start_frame") is not None:
    cmd.extend(["--start-frame", str(cfg["start_frame"])])
if cfg.get("end_frame") is not None:
    cmd.extend(["--end-frame", str(cfg["end_frame"])])
if cfg.get("samples") is not None:
    cmd.extend(["--samples", str(cfg["samples"])])

rc = subprocess.run(cmd, check=False).returncode
bundle_path = Path(f"/tmp/worker_{wid}_frames.tar.gz")
if rc == 0:
    with tarfile.open(bundle_path, "w:gz") as tf:
        for p in sorted(frames_dir.glob("*.png")):
            tf.add(p, arcname=p.name)
STATUS.write_text(f"DONE rc={rc} bundle={bundle_path}", encoding="utf-8")
sys.exit(rc)
'''

    code = f"""import json, os, subprocess, sys, tarfile
from pathlib import Path

WORK = Path("/content/work")
WORK.mkdir(parents=True, exist_ok=True)
cfg = json.loads({cfg_json!r})
wid = int(cfg["worker_id"])
STATUS_FILE = WORK / f"worker_{{wid}}.status.txt"
LOG_FILE = WORK / f"worker_{{wid}}.log"

def write_status(text: str) -> None:
    tmp = WORK / f"worker_{{wid}}.status.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, STATUS_FILE)

write_status("RUNNING stage=unpack")
with tarfile.open("/content/scripts_bundle.tar.gz", "r:gz") as tf:
    tf.extractall(WORK)
with tarfile.open("/content/scene_cache_bundle.tar.gz", "r:gz") as tf:
    tf.extractall(WORK)

(WORK / "worker_cfg.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

addon_source = cfg["addon_source"]
custom_url = cfg["custom_addon_url"]
addon_zip_name = cfg.get("addon_zip_name") or ""
uploaded_addon = WORK / "addon" / addon_zip_name if addon_zip_name else None
if uploaded_addon and uploaded_addon.is_file():
    addon_source = "custom_url"
    custom_url = str(uploaded_addon)

write_status("RUNNING stage=install_blender")
install_cmd = [
    sys.executable,
    str(WORK / "scripts" / "install_blender_flip.py"),
    "--blender-version", cfg["blender_version"],
    "--work-dir", str(WORK),
    "--addon-source", addon_source,
    "--symlink",
]
if addon_source == "custom_url" and custom_url:
    install_cmd.extend(["--custom-addon-url", custom_url])

subprocess.run(install_cmd, check=True)
runner_py = WORK / f"worker_{{wid}}_detached.py"
runner_py.write_text({runner_template!r}, encoding="utf-8")
write_status("RUNNING stage=render")
log_fh = open(LOG_FILE, "w", encoding="utf-8")
subprocess.Popen(
    [sys.executable, "-u", str(runner_py)],
    stdout=log_fh,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print(f"[worker_{{wid}}] Detached render launched.", flush=True)
"""
    path.write_text(code, encoding="utf-8")


def _run_single_worker(
    worker_id: int,
    *,
    num_workers: int,
    colab_bin: str,
    session_name: str,
    gpu_choices: list[str],
    scripts_tar: Path,
    scene_tar: Path,
    tmp_dir: Path,
    frames_out_dir: Path,
    blend_name: str,
    addon_zip_name: str,
    blender_version: str,
    addon_source: str,
    custom_addon_url: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
    samples: Optional[int],
    adaptive_threshold: float,
    compute_type: str,
    sanitize_on_worker: bool,
    poll_interval_s: float,
    worker_timeout_s: float,
    session_registry: list[str],
    endpoint_registry: list[str],
    reg_lock: threading.Lock,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    gpu_used, endpoint = _provision_worker_session(colab_bin, session_name, gpu_choices)
    with reg_lock:
        session_registry.append(session_name)
        if endpoint:
            endpoint_registry.append(endpoint)

    driver_py = tmp_dir / f"driver_{worker_id}.py"
    keepalive_py = tmp_dir / f"keepalive_{worker_id}.py"
    keepalive_py.write_text("print('keepalive', flush=True)\n", encoding="utf-8")

    _write_worker_driver_script(
        driver_py,
        worker_id=worker_id,
        num_workers=num_workers,
        blend_name=blend_name,
        addon_zip_name=addon_zip_name,
        blender_version=blender_version,
        addon_source=addon_source,
        custom_addon_url=custom_addon_url,
        start_frame=start_frame,
        end_frame=end_frame,
        samples=samples,
        adaptive_threshold=adaptive_threshold,
        compute_type=compute_type,
        sanitize_on_worker=sanitize_on_worker,
    )

    print(f"[swarm:{session_name}] Uploading bundles once per worker...", flush=True)
    _run_colab_cmd(
        colab_bin,
        ["upload", "-s", session_name, str(scripts_tar), "/content/scripts_bundle.tar.gz"],
        timeout=600,
    )
    _run_colab_cmd(
        colab_bin,
        ["upload", "-s", session_name, str(scene_tar), "/content/scene_cache_bundle.tar.gz"],
        timeout=900,
    )

    print(f"[swarm:{session_name}] Installing Blender & launching worker slice {worker_id}/{num_workers}...", flush=True)
    _run_colab_cmd(
        colab_bin,
        ["exec", "-s", session_name, "-f", str(driver_py), "--timeout", "1800"],
        timeout=1860,
    )

    last_token_refresh = time.perf_counter()
    last_keepalive = time.perf_counter()
    local_status = tmp_dir / f"worker_{worker_id}.status.txt"
    status_text = ""

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > worker_timeout_s:
            raise TimeoutError(f"Worker {worker_id} exceeded {worker_timeout_s}s timeout")

        # Refresh Colab proxy token proactively before 60-min JWT expiry (at 50 min intervals)
        if time.perf_counter() - last_token_refresh >= 3000.0:
            refresh_colab_proxy_token(session_name)
            last_token_refresh = time.perf_counter()

        # Keep Jupyter kernel active well below 22.5 min Colab idle reap threshold
        if time.perf_counter() - last_keepalive >= 180.0:
            try:
                _run_colab_cmd(
                    colab_bin,
                    ["exec", "-s", session_name, "-f", str(keepalive_py), "--timeout", "30"],
                    timeout=45,
                    check=False,
                )
            except Exception:
                pass
            last_keepalive = time.perf_counter()

        try:
            _run_colab_cmd(
                colab_bin,
                [
                    "download",
                    "-s",
                    session_name,
                    f"/content/work/worker_{worker_id}.status.txt",
                    str(local_status),
                ],
                timeout=60,
            )
            status_text = local_status.read_text(encoding="utf-8").strip()
            if status_text.startswith("DONE "):
                break
        except Exception:
            pass

        time.sleep(poll_interval_s)

    if "rc=0" not in status_text:
        local_log = tmp_dir / f"worker_{worker_id}.log"
        _run_colab_cmd(
            colab_bin,
            ["download", "-s", session_name, f"/content/work/worker_{worker_id}.log", str(local_log)],
            timeout=60,
            check=False,
        )
        tail = local_log.read_text(encoding="utf-8", errors="ignore")[-1500:] if local_log.is_file() else ""
        raise RuntimeError(f"Worker {worker_id} failed ({status_text}). Log tail:\n{tail}")

    local_bundle = tmp_dir / f"worker_{worker_id}_frames.tar.gz"
    _run_colab_cmd(
        colab_bin,
        [
            "download",
            "-s",
            session_name,
            f"/tmp/worker_{worker_id}_frames.tar.gz",
            str(local_bundle),
        ],
        timeout=600,
    )

    extracted = 0
    with tarfile.open(local_bundle, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith(".png"):
                member.name = os.path.basename(member.name)
                tf.extract(member, path=frames_out_dir)
                extracted += 1

    wall_s = round(time.perf_counter() - t0, 4)
    print(
        f"[swarm:{session_name}] Finished slice {worker_id}/{num_workers}: "
        f"{extracted} frames downloaded in {wall_s:.1f}s",
        flush=True,
    )
    return {
        "worker_id": worker_id,
        "session": session_name,
        "gpu": gpu_used,
        "frames_extracted": extracted,
        "wall_s": wall_s,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed multi-Colab-GPU Blender 5.3 render orchestrator."
    )
    parser.add_argument("--colab-bin", default="", help=f"Path to colab CLI (default: {DEFAULT_COLAB_BIN})")
    parser.add_argument("--blend", required=True, help="Local sanitized .blend file")
    parser.add_argument("--cache-dir", required=True, help="Local baked FLIP Fluids cache directory")
    parser.add_argument("--out-dir", default="./outputs/swarm_render", help="Local output directory")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Colab GPU workers (default: 2)")
    parser.add_argument(
        "--gpu",
        default="L4,T4",
        help="GPU tier or comma-separated fallback priority (e.g. L4,T4 or T4)",
    )
    parser.add_argument("--session-prefix", default="blender53-swarm", help="Colab session prefix")
    parser.add_argument("--blender-version", default="5.3.0", choices=["5.3.0", "5.2.4", "4.5.16"])
    parser.add_argument(
        "--addon-source",
        default="build_from_source",
        choices=["build_from_source", "custom_url", "official_demo"],
    )
    parser.add_argument("--custom-addon-url", default="", help="Custom FLIP Fluids addon zip URL")
    parser.add_argument("--addon-zip", default="", help="Optional local prebuilt FLIP Fluids zip to upload")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--adaptive-threshold", type=float, default=0.02)
    parser.add_argument("--compute-type", default="AUTO", choices=["AUTO", "OPTIX", "CUDA"])
    parser.add_argument("--sanitize-on-worker", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--worker-timeout", type=float, default=5400.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate bundles & plan without creating Colab VMs")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    colab_bin = resolve_colab_bin(args.colab_bin)
    blend_path = Path(args.blend).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    if not blend_path.is_file():
        raise FileNotFoundError(f"Blend file not found: {blend_path}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Baked FLIP Fluids cache directory not found: {cache_dir}")

    addon_zip = Path(args.addon_zip).resolve() if args.addon_zip else None
    gpu_choices = [g.strip() for g in args.gpu.split(",") if g.strip()]
    if not gpu_choices:
        gpu_choices = ["L4", "T4"]

    out_dir = Path(args.out_dir).resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="colab_swarm_"))
    provisioned_sessions: list[str] = []
    provisioned_endpoints: list[str] = []
    reg_lock = threading.Lock()
    run_tag = datetime.datetime.now().strftime("%H%M%S")

    try:
        scripts_tar, scene_tar = _build_local_bundles(tmp_dir, blend_path, cache_dir, addon_zip)
        print(
            f"[swarm] Bundled scripts ({scripts_tar.stat().st_size // 1024} KiB) & "
            f"scene+cache ({scene_tar.stat().st_size // (1024 * 1024)} MiB)",
            flush=True,
        )

        if args.dry_run:
            plan = {
                "dry_run": True,
                "colab_bin": colab_bin,
                "num_workers": args.num_workers,
                "gpu_priority": gpu_choices,
                "sessions": [f"{args.session_prefix}-{run_tag}-w{i}" for i in range(args.num_workers)],
                "scripts_tar_bytes": scripts_tar.stat().st_size,
                "scene_tar_bytes": scene_tar.stat().st_size,
                "out_dir": str(out_dir),
            }
            print(json_dumps_safe(plan), flush=True)
            return 0

        worker_results: list[dict[str, Any]] = []
        # Cap local orchestrator concurrency cleanly (workers execute remotely)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
            futures = []
            for wid in range(args.num_workers):
                sess = f"{args.session_prefix}-{run_tag}-w{wid}"
                futures.append(
                    pool.submit(
                        _run_single_worker,
                        wid,
                        num_workers=args.num_workers,
                        colab_bin=colab_bin,
                        session_name=sess,
                        gpu_choices=gpu_choices,
                        scripts_tar=scripts_tar,
                        scene_tar=scene_tar,
                        tmp_dir=tmp_dir,
                        frames_out_dir=frames_dir,
                        blend_name=blend_path.name,
                        addon_zip_name=addon_zip.name if addon_zip else "",
                        blender_version=args.blender_version,
                        addon_source=args.addon_source,
                        custom_addon_url=args.custom_addon_url,
                        start_frame=args.start_frame,
                        end_frame=args.end_frame,
                        samples=args.samples,
                        adaptive_threshold=args.adaptive_threshold,
                        compute_type=args.compute_type,
                        sanitize_on_worker=args.sanitize_on_worker,
                        poll_interval_s=args.poll_interval,
                        worker_timeout_s=args.worker_timeout,
                        session_registry=provisioned_sessions,
                        endpoint_registry=provisioned_endpoints,
                        reg_lock=reg_lock,
                    )
                )
            for fut in concurrent.futures.as_completed(futures):
                worker_results.append(fut.result())

        video_info = bake_and_render.assemble_video_and_gif(
            frames_dir,
            "frame_",
            out_dir / "animation.mp4",
            out_dir / "animation_preview.gif",
            fps=args.fps,
        )
        summary = {
            "workers": sorted(worker_results, key=lambda w: w["worker_id"]),
            "video": video_info,
        }
        (out_dir / "swarm_summary.json").write_text(json_dumps_safe(summary), encoding="utf-8")
        print(f"[swarm] Complete -> {out_dir / 'animation.mp4'}", flush=True)
        return 0
    finally:
        for sess in provisioned_sessions:
            try:
                refresh_colab_proxy_token(sess, verbose=False)
                print(f"[swarm] Stopping session '{sess}'...", flush=True)
                subprocess.run([colab_bin, "stop", "-s", sess], check=False, timeout=90)
            except Exception as exc:
                print(f"[swarm] Error stopping '{sess}': {exc}", flush=True)
        force_unassign_endpoints(provisioned_endpoints)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def json_dumps_safe(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
