#!/usr/bin/env python3
"""Reproducible Colab GPU Benchmark Harness for Blender 5.3 + FLIP Fluids v1.8.9.

Benchmarks 6 configurations against a baked FLIP Fluids scene (`particles.blend`):
  1. Legacy Baseline             : per-frame Blender CLI spawn + CUDA + CPU enabled + no persistent data
  2. Persistent CUDA GPU-Only    : single Blender process + use_persistent_data + CUDA + CPU off
  3. Persistent OptiX GPU-Only   : single Blender process + use_persistent_data + OPTIX + CPU off
  4. Persistent OptiX + Den + AS : OPTIX + GPU-only + persistent data + Denoiser + Adaptive Sampling (0.02)
  5. EEVEE Headless Probe        : Blender 5.3 BLENDER_EEVEE under --gpu-backend vulkan & opengl
  6. Sequential vs Interleaved   : sequential bake+render vs overlapping CPU FLIP bake + GPU OptiX render

Outputs:
  - <out_dir>/colab_benchmark.json
  - <out_dir>/colab_benchmark_chart.png (< 1.8 MiB)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bake_and_render  # noqa: E402
import gpu_setup  # noqa: E402


MAX_PNG_BYTES = 1_750_000


def collect_system_telemetry(blender_bin: str) -> dict[str, Any]:
    """Collect GPU, CPU, RAM, OS, and Blender version telemetry."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "blender_bin": blender_bin,
    }

    try:
        out = subprocess.check_output([blender_bin, "--version"], text=True, timeout=15)
        info["blender_version"] = out.splitlines()[0].strip() if out else "unknown"
    except Exception as exc:
        info["blender_version"] = f"unknown ({exc})"

    # CPU model & RAM
    if os.path.isfile("/proc/cpuinfo"):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    if os.path.isfile("/proc/meminfo"):
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                info["ram_gib"] = round(kb / (1024.0 * 1024.0), 2)
                break

    # NVIDIA GPU
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            q = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            ).strip()
            if q:
                parts = [p.strip() for p in q.splitlines()[0].split(",")]
                info["gpu_name"] = parts[0] if len(parts) > 0 else ""
                info["gpu_vram_mib"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                info["driver_version"] = parts[2] if len(parts) > 2 else ""
        except Exception as exc:
            info["gpu_query_error"] = str(exc)

    return info


def _run_py_cli(script_path: Path, args_list: list[str]) -> tuple[int, float]:
    cmd = [sys.executable, str(script_path), *args_list]
    print(f"[bench] Running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode), time.perf_counter() - t0


def _ensure_sanitized_bench_blend(
    blender_bin: str,
    src_blend: str,
    sanitized_blend: Path,
    cache_dir: Path,
    start_frame: int,
    end_frame: int,
    resolution: int,
) -> dict[str, Any]:
    san_json = sanitized_blend.with_suffix(".sanitize.json")
    rc, wall_s = _run_py_cli(
        SCRIPTS_DIR / "bake_and_render.py",
        [
            "--mode",
            "sanitize",
            "--blender",
            blender_bin,
            "--blend",
            src_blend,
            "--sanitized-blend",
            str(sanitized_blend),
            "--cache-dir",
            str(cache_dir),
            "--start-frame",
            str(start_frame),
            "--end-frame",
            str(end_frame),
            "--resolution",
            str(resolution),
            "--json-out",
            str(san_json),
        ],
    )
    if rc != 0:
        raise RuntimeError(f"Scene sanitization failed with rc={rc}")
    res = json.loads(san_json.read_text(encoding="utf-8")) if san_json.is_file() else {}
    res["wall_s"] = round(wall_s, 4)
    return res


def _run_legacy_baseline(
    blender_bin: str,
    bench_blend: Path,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    samples: int,
) -> dict[str, Any]:
    """Benchmark 1: per-frame Blender CLI spawn + CUDA + CPU enabled + no persistent data."""
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_setup_script = str((SCRIPTS_DIR / "gpu_setup.py").resolve())
    per_frame_s: list[float] = []
    t_all0 = time.perf_counter()

    for f in range(start_frame, end_frame + 1):
        out_pattern = str(out_dir / "frame_######")
        expected_png = out_dir / f"frame_{f:06d}.png"
        # Note: -E/-o/-noaudio/-f must precede '--' because Blender passes all trailing args to -P
        cmd = [
            blender_bin,
            "-b",
            str(bench_blend),
            "-E",
            "CYCLES",
            "-o",
            out_pattern,
            "-noaudio",
            "-P",
            gpu_setup_script,
            "-f",
            str(f),
            "--",
            "--ensure-flip",
            "--engine",
            "CYCLES",
            "--compute-type",
            "CUDA",
            "--include-cpu",
            "--no-persistent-data",
            "--no-denoise",
            "--no-adaptive-sampling",
            "--no-texture-cache",
            "--samples",
            str(samples),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=False)
        dt = time.perf_counter() - t0
        if proc.returncode != 0 or not expected_png.is_file():
            raise RuntimeError(
                f"Legacy baseline failed on frame {f} (rc={proc.returncode}, file={expected_png})"
            )
        per_frame_s.append(round(dt, 4))
        print(f"[bench:legacy] Frame {f}/{end_frame} in {dt:.2f}s", flush=True)

    wall_s = time.perf_counter() - t_all0
    n_frames = len(per_frame_s)
    return {
        "name": "Legacy Baseline (Per-Frame Spawn + CUDA+CPU)",
        "short_label": "Legacy\n(Spawn+CUDA+CPU)",
        "frames": n_frames,
        "samples": samples,
        "wall_s": round(wall_s, 4),
        "mean_per_frame_s": round(statistics.mean(per_frame_s), 4),
        "median_per_frame_s": round(statistics.median(per_frame_s), 4),
        "per_frame_s": per_frame_s,
        "frames_per_min": round(60.0 * n_frames / max(wall_s, 1e-6), 2),
    }


def _run_persistent_config(
    label: str,
    short_label: str,
    blender_bin: str,
    bench_blend: Path,
    cache_dir: Path,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    samples: int,
    extra_flags: list[str],
) -> dict[str, Any]:
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "run_summary.json"

    rc, wall_s = _run_py_cli(
        SCRIPTS_DIR / "bake_and_render.py",
        [
            "--mode",
            "render_persistent",
            "--blender",
            blender_bin,
            "--blend",
            str(bench_blend),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--start-frame",
            str(start_frame),
            "--end-frame",
            str(end_frame),
            "--samples",
            str(samples),
            "--no-video",
            "--json-out",
            str(json_path),
            *extra_flags,
        ],
    )
    if rc != 0:
        raise RuntimeError(f"Benchmark '{label}' failed with rc={rc}")

    data = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else {}
    frame_times = [f["wall_s"] for f in data.get("render", {}).get("frames", [])]
    n_frames = end_frame - start_frame + 1
    return {
        "name": label,
        "short_label": short_label,
        "frames": n_frames,
        "samples": samples,
        "wall_s": round(wall_s, 4),
        "blender_render_loop_s": data.get("render", {}).get("wall_s"),
        "mean_per_frame_s": round(statistics.mean(frame_times), 4) if frame_times else round(wall_s / n_frames, 4),
        "first_frame_s": frame_times[0] if frame_times else None,
        "steady_frame_mean_s": round(statistics.mean(frame_times[1:]), 4) if len(frame_times) > 1 else None,
        "per_frame_s": frame_times,
        "frames_per_min": round(60.0 * n_frames / max(wall_s, 1e-6), 2),
    }


def _run_seq_vs_interleaved(
    blender_bin: str,
    src_blend: str,
    bench_root: Path,
    start_frame: int,
    end_frame: int,
    resolution: int,
    samples: int,
) -> dict[str, Any]:
    """Benchmark 6: Sequential (bake + render) vs Interleaved (CPU bake || GPU OptiX render)."""
    results: dict[str, Any] = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": end_frame - start_frame + 1,
        "resolution": resolution,
        "samples": samples,
    }

    for mode_name in ("sequential", "interleaved"):
        sub_dir = bench_root / f"pipeline_{mode_name}"
        if sub_dir.is_dir():
            shutil.rmtree(sub_dir, ignore_errors=True)
        sub_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = sub_dir / "cache"
        frames_dir = sub_dir / "frames"
        san_blend = sub_dir / "scene_sanitized.blend"
        json_path = sub_dir / f"{mode_name}_summary.json"

        rc, wall_s = _run_py_cli(
            SCRIPTS_DIR / "bake_and_render.py",
            [
                "--mode",
                mode_name,
                "--sanitize",
                "--reset-bake",
                "--blender",
                blender_bin,
                "--blend",
                src_blend,
                "--sanitized-blend",
                str(san_blend),
                "--cache-dir",
                str(cache_dir),
                "--out-dir",
                str(frames_dir),
                "--start-frame",
                str(start_frame),
                "--end-frame",
                str(end_frame),
                "--resolution",
                str(resolution),
                "--compute-type",
                "OPTIX",
                "--samples",
                str(samples),
                "--adaptive-threshold",
                "0.02",
                "--no-video",
                "--json-out",
                str(json_path),
            ],
        )
        if rc != 0:
            raise RuntimeError(f"Pipeline benchmark '{mode_name}' failed with rc={rc}")
        data = json.loads(json_path.read_text(encoding="utf-8")) if json_path.is_file() else {}
        sanitize_s = data.get("sanitize", {}).get("wall_s", 0.0) or 0.0
        bake_s = data.get("bake", {}).get("wall_s")
        render_s = data.get("render", {}).get("wall_s")
        if mode_name == "sequential":
            # Measure bake+render execution cleanly (excluding one-time sanitize)
            exec_wall = (
                round(float(bake_s) + float(render_s), 4)
                if (bake_s is not None and render_s is not None)
                else round(max(0.0, wall_s - float(sanitize_s)), 4)
            )
            results["sequential"] = {
                "total_with_sanitize_s": round(wall_s, 4),
                "wall_s": exec_wall,
                "bake_s": bake_s,
                "render_s": render_s,
            }
        else:
            inter_s = data.get("interleaved", {}).get("wall_s")
            results["interleaved"] = {
                "total_with_sanitize_s": round(wall_s, 4),
                "wall_s": inter_s if inter_s is not None else round(max(0.0, wall_s - float(sanitize_s)), 4),
                "bake_s": bake_s,
                "render_s": render_s,
            }

    seq_s = float(results["sequential"]["wall_s"])
    int_s = float(results["interleaved"]["wall_s"])
    results["speedup"] = round(seq_s / max(int_s, 1e-6), 3)
    results["time_saved_s"] = round(seq_s - int_s, 3)
    results["time_saved_pct"] = round(100.0 * (seq_s - int_s) / max(seq_s, 1e-6), 2)
    return results


def plot_benchmark_chart(benchmark_data: dict[str, Any], out_png: Path) -> Path:
    """Render publication-grade 2-panel benchmark chart (< 1.8 MiB)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#12161f",
            "axes.facecolor": "#191f2c",
            "axes.edgecolor": "#344058",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "grid.color": "#2a3346",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel 1: Render Configuration Comparison
    render_runs = benchmark_data.get("render_configs", [])
    if render_runs:
        labels = [r["short_label"] for r in render_runs]
        walls = [r["wall_s"] for r in render_runs]
        colors = ["#64748b", "#3b82f6", "#10b981", "#f59e0b"][: len(render_runs)]
        bars = ax1.bar(range(len(labels)), walls, color=colors, width=0.58, edgecolor="#0f172a", linewidth=1.2)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, fontsize=9.5)
        ax1.set_ylabel("Total Wall Time (seconds)", fontsize=10.5, fontweight="bold")
        n_f = render_runs[0].get("frames", "?")
        smp = render_runs[0].get("samples", "?")
        ax1.set_title(
            f"Cycles GPU Render Optimization ({n_f} frames @ {smp} samples)",
            fontsize=11.5,
            fontweight="bold",
            pad=12,
        )
        ax1.grid(axis="y", linestyle="--", alpha=0.55)
        ax1.set_ylim(0, max(walls) * 1.25 if walls else 10)

        baseline_s = walls[0] if walls else 1.0
        for bar, r in zip(bars, render_runs):
            h = bar.get_height()
            sp = baseline_s / max(h, 1e-6)
            pf = r.get("mean_per_frame_s", h / max(r.get("frames", 1), 1))
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max(walls) * 0.025,
                f"{h:.1f}s\n({pf:.2f}s/f · {sp:.2f}x)",
                ha="center",
                va="bottom",
                fontsize=8.8,
                fontweight="bold",
                color="#f8fafc",
            )

    # Annotate EEVEE probe inside Panel 1
    eevee_probe = benchmark_data.get("eevee_probe", {}).get("backends", {})
    if eevee_probe:
        notes = []
        for bk in ("vulkan", "opengl"):
            if bk in eevee_probe:
                binfo = eevee_probe[bk]
                if binfo.get("supported"):
                    notes.append(f"EEVEE {bk}: OK ({binfo.get('render_s') or binfo.get('wall_s')}s)")
                else:
                    notes.append(f"EEVEE {bk}: unavailable")
        if notes:
            ax1.text(
                0.98,
                0.96,
                "Headless EEVEE Probe:\n" + "\n".join(notes),
                transform=ax1.transAxes,
                ha="right",
                va="top",
                fontsize=8.2,
                color="#cbd5e1",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#334155", alpha=0.88),
            )

    # Panel 2: Sequential vs Interleaved Pipeline
    pipe = benchmark_data.get("pipeline_overlap", {})
    seq = pipe.get("sequential", {})
    inter = pipe.get("interleaved", {})
    if seq and inter:
        seq_bake = float(seq.get("bake_s") or 0.0)
        seq_ren = float(seq.get("render_s") or 0.0)
        seq_tot = float(seq.get("wall_s") or (seq_bake + seq_ren))
        int_tot = float(inter.get("wall_s") or 0.0)

        x = [0, 1]
        if seq_bake > 0 and seq_ren > 0:
            ax2.bar(0, seq_bake, width=0.50, color="#6366f1", label="CPU FLIP Bake")
            ax2.bar(0, seq_ren, bottom=seq_bake, width=0.50, color="#06b6d4", label="GPU OptiX Render")
        else:
            ax2.bar(0, seq_tot, width=0.50, color="#6366f1", label="Sequential Total")

        ax2.bar(1, int_tot, width=0.50, color="#22c55e", label="Interleaved (CPU Bake || GPU Render)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["Sequential\n(Bake → Render)", "Interleaved\n(Overlapped)"], fontsize=10)
        ax2.set_ylabel("End-to-End Wall Time (seconds)", fontsize=10.5, fontweight="bold")
        nf_b = pipe.get("frame_count", "?")
        ax2.set_title(
            f"Bake + Render Pipeline Overlap ({nf_b} frames)",
            fontsize=11.5,
            fontweight="bold",
            pad=12,
        )
        ymax = max(seq_tot, int_tot) * 1.28
        ax2.set_ylim(0, ymax if ymax > 0 else 10)
        ax2.grid(axis="y", linestyle="--", alpha=0.55)
        ax2.legend(loc="upper right", fontsize=8.3, facecolor="#0f172a", edgecolor="#334155")

        ax2.text(
            0,
            seq_tot + ymax * 0.02,
            f"{seq_tot:.1f}s\n(bake {seq_bake:.1f}s + ren {seq_ren:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
        )
        sp_p = pipe.get("speedup", 1.0)
        saved_pct = pipe.get("time_saved_pct", 0.0)
        ax2.text(
            1,
            int_tot + ymax * 0.02,
            f"{int_tot:.1f}s\n({sp_p:.2f}x · -{saved_pct:.1f}% wall)",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#86efac",
        )
    else:
        ax2.text(0.5, 0.5, "Pipeline overlap not run", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    tel = benchmark_data.get("telemetry", {})
    gpu_name = tel.get("gpu_name", "GPU")
    bl_ver = tel.get("blender_version", "Blender 5.3.0")
    fig.suptitle(
        f"Blender-Colab Performance Benchmark — {bl_ver} + FLIP Fluids v1.8.9 ({gpu_name})",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93], pad=1.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Enforce < 1.8 MiB artifact budget
    if out_png.stat().st_size > MAX_PNG_BYTES:
        try:
            from PIL import Image  # type: ignore

            with Image.open(out_png) as im:
                im.save(out_png, format="PNG", optimize=True, compress_level=9)
        except Exception:
            pass
    return out_png


def parse_args(argv: list[str]) -> argparse.Namespace:
    wdir = bake_and_render.default_work_dir()
    parser = argparse.ArgumentParser(
        description="Run reproducible Blender 5.3 + FLIP Fluids v1.8.9 benchmarks on Colab GPU."
    )
    parser.add_argument("--blender", default="", help="Path to Blender executable")
    parser.add_argument("--blend", default=str(wdir / "particles.blend"), help="Source .blend scene")
    parser.add_argument("--out-dir", default=str(wdir / "benchmarks"), help="Output directory")
    parser.add_argument("--cache-dir", default="", help="Optional existing baked FLIP cache for configs 1-4")
    parser.add_argument("--start-frame", type=int, default=1, help="Start frame (default: 1)")
    parser.add_argument("--render-frames", type=int, default=5, help="Frames for render configs 1-4 (default: 5)")
    parser.add_argument("--bake-frames", type=int, default=10, help="Frames for seq vs interleaved test (default: 10)")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples per frame (default: 64)")
    parser.add_argument("--resolution", type=int, default=0, help="FLIP resolution override (0 keeps scene default)")
    parser.add_argument("--skip-prebake", action="store_true", help="Reuse --cache-dir without re-baking configs 1-4")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip Legacy per-frame spawn baseline")
    parser.add_argument("--skip-eevee", action="store_true", help="Skip EEVEE Vulkan/OpenGL probe")
    parser.add_argument("--skip-interleaved", action="store_true", help="Skip Sequential vs Interleaved benchmark")
    parser.add_argument("--chart-only", default="", help="Re-plot colab_benchmark_chart.png from existing JSON")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "colab_benchmark.json"
    chart_out = out_dir / "colab_benchmark_chart.png"

    if args.chart_only:
        data = json.loads(Path(args.chart_only).read_text(encoding="utf-8"))
        plot_benchmark_chart(data, chart_out)
        print(f"[bench] Chart saved -> {chart_out} ({chart_out.stat().st_size} B)", flush=True)
        return 0

    blender_bin = bake_and_render.find_blender_executable(args.blender)
    src_blend = str(Path(args.blend).resolve())
    if not os.path.isfile(src_blend):
        raise FileNotFoundError(f"Blend file not found: {src_blend}")

    telemetry = collect_system_telemetry(blender_bin)
    render_end = args.start_frame + args.render_frames - 1
    bake_end = args.start_frame + args.bake_frames - 1

    shared_cache = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else (out_dir / "shared_flip_cache")
    )
    bench_blend = out_dir / "bench_sanitized.blend"
    sanitize_info = _ensure_sanitized_bench_blend(
        blender_bin,
        src_blend,
        bench_blend,
        shared_cache,
        args.start_frame,
        max(render_end, bake_end),
        args.resolution,
    )

    if not args.skip_prebake:
        prebake_json = out_dir / "prebake.json"
        rc, pre_s = _run_py_cli(
            SCRIPTS_DIR / "bake_and_render.py",
            [
                "--mode",
                "bake",
                "--reset-bake",
                "--blender",
                blender_bin,
                "--blend",
                str(bench_blend),
                "--cache-dir",
                str(shared_cache),
                "--start-frame",
                str(args.start_frame),
                "--end-frame",
                str(render_end),
                "--resolution",
                str(args.resolution),
                "--json-out",
                str(prebake_json),
            ],
        )
        if rc != 0:
            raise RuntimeError(f"Pre-bake for render benchmarks failed (rc={rc})")
        print(f"[bench] Shared pre-bake complete in {pre_s:.2f}s", flush=True)

    render_configs: list[dict[str, Any]] = []
    if not args.skip_legacy:
        render_configs.append(
            _run_legacy_baseline(
                blender_bin,
                bench_blend,
                out_dir / "01_legacy_cuda_cpu",
                args.start_frame,
                render_end,
                args.samples,
            )
        )

    render_configs.append(
        _run_persistent_config(
            "Persistent CUDA GPU-Only",
            "Persistent\nCUDA GPU",
            blender_bin,
            bench_blend,
            shared_cache,
            out_dir / "02_persistent_cuda",
            args.start_frame,
            render_end,
            args.samples,
            ["--compute-type", "CUDA", "--no-denoise", "--no-adaptive-sampling"],
        )
    )

    render_configs.append(
        _run_persistent_config(
            "Persistent OptiX GPU-Only",
            "Persistent\nOptiX GPU",
            blender_bin,
            bench_blend,
            shared_cache,
            out_dir / "03_persistent_optix",
            args.start_frame,
            render_end,
            args.samples,
            ["--compute-type", "OPTIX", "--no-denoise", "--no-adaptive-sampling"],
        )
    )

    render_configs.append(
        _run_persistent_config(
            "Persistent OptiX + Denoise + Adaptive (0.02)",
            "OptiX+Den\n+Adaptive",
            blender_bin,
            bench_blend,
            shared_cache,
            out_dir / "04_persistent_optix_den_as",
            args.start_frame,
            render_end,
            args.samples,
            ["--compute-type", "OPTIX", "--adaptive-threshold", "0.02"],
        )
    )

    # Compute speedups relative to first entry (Legacy if run, else Persistent CUDA)
    base_wall = render_configs[0]["wall_s"]
    for cfg in render_configs:
        cfg["speedup_vs_baseline"] = round(base_wall / max(cfg["wall_s"], 1e-6), 3)

    eevee_probe: dict[str, Any] = {}
    if not args.skip_eevee:
        eevee_probe = gpu_setup.probe_eevee_backends(
            blender_bin,
            blend_path=str(bench_blend),
            frame=args.start_frame,
            out_dir=str(out_dir / "05_eevee_probe"),
            samples=min(16, args.samples),
        )

    pipeline_overlap: dict[str, Any] = {}
    if not args.skip_interleaved:
        pipeline_overlap = _run_seq_vs_interleaved(
            blender_bin,
            src_blend,
            out_dir / "06_overlap",
            args.start_frame,
            bake_end,
            args.resolution,
            args.samples,
        )

    benchmark_data = {
        "telemetry": telemetry,
        "sanitize": sanitize_info,
        "render_configs": render_configs,
        "eevee_probe": eevee_probe,
        "pipeline_overlap": pipeline_overlap,
    }
    json_out.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    plot_benchmark_chart(benchmark_data, chart_out)

    print(f"[bench] Wrote JSON -> {json_out}", flush=True)
    print(f"[bench] Wrote Chart -> {chart_out} ({chart_out.stat().st_size} B)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
