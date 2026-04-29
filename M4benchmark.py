"""
OpenCV resize speedup benchmark — Apple Silicon M4 edition.

Workload: 4K (3840x1920) -> 1080p (1920x1080), both RGB and MJPEG inputs.
Adds vs. the original sandbox version:
  - Thread sweep across {1, 2, 4, 6, 8, 10} for OpenCV
  - PyTorch MPS (Metal Performance Shaders) GPU path
  - OpenCV UMat / OpenCL GPU path
  - Core Image via pyobjc GPU path
  - arm64 vs Rosetta detection
  - Power-state reporting

Run on your M4 MacBook (base, 4P + 6E cores). Output: m4_results_extended.json
"""
import os
import sys
import time
import json
import platform
import subprocess
import statistics
import io
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

# ------------------------------------------------------------
# Sanity checks: arm64 only, report power state
# ------------------------------------------------------------
def env_check():
    def sysctl_value(key):
        try:
            result = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True, timeout=3)
            value = result.stdout.strip()
            return value or None
        except Exception:
            return None

    arch = platform.machine()
    if arch != 'arm64':
        print(f"FATAL: Python is running as {arch}, not arm64.")
        print("You're running through Rosetta 2 — performance will be ~half.")
        print("Fix: install an arm64 Python (e.g. via the official installer or 'brew install python').")
        sys.exit(1)

    info = {
        'python_arch': arch,
        'python_version': platform.python_version(),
        'macos': platform.mac_ver()[0],
        'opencv': cv2.__version__,
    }
    try:
        pmset = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, timeout=3)
        info['power'] = pmset.stdout.strip().split('\n')[-1]
    except Exception:
        info['power'] = 'unknown'
    try:
        lpm = subprocess.run(['pmset', '-g'], capture_output=True, text=True, timeout=3)
        info['lowpowermode'] = 'lowpowermode 1' in lpm.stdout.lower()
    except Exception:
        info['lowpowermode'] = 'unknown'
    info['cpu'] = sysctl_value('machdep.cpu.brand_string') or 'Apple Silicon'
    info['hw_model'] = sysctl_value('hw.model') or 'unknown'
    perf_cores = sysctl_value('hw.perflevel0.physicalcpu')
    efficiency_cores = sysctl_value('hw.perflevel1.physicalcpu')
    if perf_cores and efficiency_cores:
        info['cpu_cores'] = f"{perf_cores} performance + {efficiency_cores} efficiency"
    else:
        info['cpu_cores'] = 'unknown'
    info['logical_cpus'] = os.cpu_count()

    print("=" * 60)
    print("Environment")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    print()

    if info['lowpowermode'] is True:
        print("⚠️  LOW POWER MODE IS ON — numbers will be lower than nominal.")
    if 'discharging' in str(info['power']).lower():
        print("⚠️  Running on BATTERY — for cleanest numbers plug into power.")
    print()
    return info


# ------------------------------------------------------------
# Workload
# ------------------------------------------------------------
SRC_W, SRC_H = 3840, 1920
DST_W, DST_H = 1920, 1080
N_FRAMES = 60
N_REPEATS = 5
WARMUP_FRAMES = 8
JPEG_QUALITY = 85
SKIP_OPTIONAL = False

RESULTS = []
QUALITY_RESULTS = []
DIVERSITY_RESULTS = []
THREAD_INTERACTION_RESULTS = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apple Silicon M4 resize benchmark for RGB and MJPEG inputs."
    )
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help="Frames per measured repeat. Default: %(default)s")
    parser.add_argument("--repeats", type=int, default=N_REPEATS,
                        help="Measured repeats per case. Default: %(default)s")
    parser.add_argument("--warmup", type=int, default=WARMUP_FRAMES,
                        help="Warmup frames per case. Default: %(default)s")
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY,
                        help="MJPEG JPEG quality. Default: %(default)s")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip PyTorch MPS, OpenCL/UMat, and Core Image paths.")
    parser.add_argument("--skip-optional", action="store_true",
                        help="Skip Pillow and pyvips paths.")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: 8 frames, 2 repeats, 2 warmup frames.")
    parser.add_argument("--long-run", action="store_true",
                        help="More stable run: 160 frames, 7 repeats, 20 warmup frames.")
    parser.add_argument("--quality-compare", action="store_true",
                        help="Compute MSE/PSNR/SSIM for MJPEG optimized variants against full decode + INTER_LINEAR.")
    parser.add_argument("--quality-samples", type=int, default=5,
                        help="Number of MJPEG frames for quality comparison. Default: %(default)s")
    return parser.parse_args()


def make_test_frames(n=N_FRAMES, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((SRC_H, SRC_W, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    base[..., 0] = 128 + 64 * np.sin(xx / 200) * np.cos(yy / 150)
    base[..., 1] = 128 + 64 * np.cos(xx / 180) * np.sin(yy / 220)
    base[..., 2] = 128 + 64 * np.sin((xx + yy) / 250)
    frames = []
    for i in range(n):
        noisy = base + rng.normal(0, 5, base.shape).astype(np.float32) + i * 0.5
        frames.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return frames


def make_mjpeg_frames(rgb_frames, quality=JPEG_QUALITY):
    encoded = []
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for f in rgb_frames:
        ok, buf = cv2.imencode('.jpg', f, enc_param)
        assert ok
        encoded.append(buf.tobytes())
    return encoded


def make_diverse_test_images(quality=JPEG_QUALITY):
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    images = []

    flat = np.zeros((SRC_H, SRC_W, 3), dtype=np.float32)
    flat[..., 0] = 108 + 10 * (xx / SRC_W)
    flat[..., 1] = 132 + 8 * (yy / SRC_H)
    flat[..., 2] = 156 + 6 * ((xx + yy) / (SRC_W + SRC_H))
    images.append(("flat_low_texture", "Flat/low-texture", np.clip(flat, 0, 255).astype(np.uint8)))

    checker = (((xx.astype(np.int32) // 4) + (yy.astype(np.int32) // 4)) % 2) * 255
    noise = rng.integers(0, 256, size=(SRC_H, SRC_W), dtype=np.uint8)
    high_freq = np.stack([
        checker.astype(np.uint8),
        noise,
        np.bitwise_xor(checker.astype(np.uint8), noise),
    ], axis=2)
    images.append(("high_frequency_texture", "High-frequency texture", high_freq))

    natural = np.zeros((SRC_H, SRC_W, 3), dtype=np.float32)
    natural[..., 0] = 80 + 110 * (xx / SRC_W)
    natural[..., 1] = 60 + 140 * (yy / SRC_H)
    natural[..., 2] = 120 + 50 * np.sin((xx + yy) / 300)
    natural = np.clip(natural, 0, 255).astype(np.uint8)
    cv2.rectangle(natural, (400, 260), (1600, 820), (40, 70, 190), thickness=-1)
    cv2.rectangle(natural, (2100, 900), (3400, 1500), (210, 180, 60), thickness=-1)
    cv2.circle(natural, (2750, 520), 260, (230, 80, 90), thickness=-1)
    images.append(("natural_like", "Natural-like", natural))

    mixed = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
    mixed[:, :SRC_W // 2] = np.clip(flat[:, :SRC_W // 2], 0, 255).astype(np.uint8)
    mixed[:, SRC_W // 2:] = rng.integers(
        0, 256, size=(SRC_H, SRC_W - SRC_W // 2, 3), dtype=np.uint8
    )
    images.append(("mixed_gradient_noise", "Mixed", mixed))

    text_like = np.full((SRC_H, SRC_W, 3), 245, dtype=np.uint8)
    for y in range(120, SRC_H - 80, 48):
        x = 120
        while x < SRC_W - 180:
            block_w = int(rng.integers(18, 120))
            block_h = int(rng.integers(8, 18))
            if rng.random() < 0.82:
                cv2.rectangle(text_like, (x, y), (min(x + block_w, SRC_W - 80), y + block_h), (20, 20, 20), -1)
            x += block_w + int(rng.integers(8, 32))
    images.append(("text_like", "Text-like", text_like))

    encoded = []
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for key, label, image in images:
        ok, buf = cv2.imencode(".jpg", image, enc_param)
        assert ok
        encoded.append({
            "key": key,
            "label": label,
            "image": image,
            "jpeg": buf.tobytes(),
            "jpeg_bytes": int(len(buf)),
        })
    return encoded


def time_it(fn, frames, n_repeats=None, warmup_frames=None):
    if n_repeats is None:
        n_repeats = N_REPEATS
    if warmup_frames is None:
        warmup_frames = WARMUP_FRAMES
    fn(frames[:warmup_frames])
    fps_list = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn(frames)
        elapsed = time.perf_counter() - t0
        fps_list.append(len(frames) / elapsed)
    return fps_list


def record(name, category, fps_list, notes=""):
    row = {
        "name": name,
        "category": category,
        "fps_mean": statistics.mean(fps_list),
        "fps_stdev": statistics.stdev(fps_list) if len(fps_list) > 1 else 0.0,
        "fps_min": min(fps_list),
        "fps_max": max(fps_list),
        "ms_per_frame": 1000.0 / statistics.mean(fps_list),
        "notes": notes,
    }
    RESULTS.append(row)
    print(f"  {name:60s}  {row['fps_mean']:8.1f} fps  ({row['ms_per_frame']:6.2f} ms/frame, ±{row['fps_stdev']:5.1f})")
    return row


def mse_psnr(reference, candidate):
    diff = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return mse, float("inf")
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return mse, float(psnr)


def decode_resize_mjpeg(buf, decode_flag, interpolation):
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, decode_flag)
    return cv2.resize(img, (DST_W, DST_H), interpolation=interpolation)


def consume_mjpeg_variant(frames, decode_flag, interpolation, workers, cv_threads=1):
    cv2.setNumThreads(cv_threads)

    def one(buf):
        return decode_resize_mjpeg(buf, decode_flag, interpolation)

    processed = 0
    if workers == 1:
        for buf in frames:
            _ = one(buf)
            processed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(one, frames):
                processed += 1
    if processed != len(frames):
        raise RuntimeError(f"processed {processed} frames, expected {len(frames)}")


def decode_mjpeg_native(buf, decode_flag):
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, decode_flag)


def consume_mjpeg_decode_only(frames, decode_flag, workers):
    cv2.setNumThreads(1)

    def one(buf):
        return decode_mjpeg_native(buf, decode_flag)

    processed = 0
    if workers == 1:
        for buf in frames:
            _ = one(buf)
            processed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(one, frames):
                processed += 1
    if processed != len(frames):
        raise RuntimeError(f"processed {processed} frames, expected {len(frames)}")


def add_speedups():
    rgb_baseline = next(
        row for row in RESULTS
        if row["name"] == "OpenCV INTER_LINEAR (1 threads)"
    )
    mjpeg_baseline = next(
        row for row in RESULTS
        if row["name"] == "OpenCV imdecode + resize LINEAR (full 4K decode)"
    )

    for row in RESULTS:
        if row["category"].startswith("mjpeg"):
            baseline = mjpeg_baseline
        else:
            baseline = rgb_baseline
        row["baseline"] = baseline["name"]
        row["speedup"] = row["fps_mean"] / baseline["fps_mean"]


def require_ssim():
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "scikit-image is required for SSIM. Install it with: pip install scikit-image"
        ) from exc


def write_markdown_report(output, path="m4_report.md"):
    env = output["env"]
    config = output["config"]
    rows = output["results"]
    quality_rows = output.get("quality", [])

    def table_for(predicate):
        selected = sorted([row for row in rows if predicate(row)], key=lambda r: r["fps_mean"], reverse=True)
        lines = [
            "| Rank | Configuration | FPS | ms/frame | Speedup | Notes |",
            "|---:|---|---:|---:|---:|---|",
        ]
        for rank, row in enumerate(selected, 1):
            lines.append(
                f"| {rank} | {row['name']} | {row['fps_mean']:.1f} | "
                f"{row['ms_per_frame']:.2f} | {row['speedup']:.2f}× | {row['notes']} |"
            )
        return "\n".join(lines)

    best_rgb = max([row for row in rows if not row["category"].startswith("mjpeg")], key=lambda r: r["fps_mean"])
    best_mjpeg = max([row for row in rows if row["category"].startswith("mjpeg")], key=lambda r: r["fps_mean"])
    exact_mjpeg = [
        row for row in rows
        if row["category"].startswith("mjpeg") and row.get("exact_output_size", True)
    ]
    approx_mjpeg = [
        row for row in rows
        if row["category"].startswith("mjpeg") and row.get("exact_output_size") is False
    ]
    best_exact_mjpeg = max(exact_mjpeg, key=lambda r: r["fps_mean"], default=None)
    best_approx_mjpeg = max(approx_mjpeg, key=lambda r: r["fps_mean"], default=None)
    baseline = next(row for row in rows if row["name"] == "OpenCV imdecode + resize LINEAR (full 4K decode)")
    balanced = max(
        [row for row in rows if row.get("mjpeg_variant") == "REDUCED_2 + INTER_LINEAR"],
        key=lambda r: r["fps_mean"],
        default=None,
    )
    max_speed = max(
        [row for row in rows if row["category"].startswith("mjpeg") and row.get("has_final_resize")],
        key=lambda r: r["fps_mean"],
        default=None,
    )

    def quality_table():
        if not quality_rows:
            return "Quality comparison was not requested. Run with `--quality-compare` to generate MSE/PSNR/SSIM."
        lines = [
            "| Variant | Samples | MSE | PSNR | SSIM |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in quality_rows:
            psnr = "inf" if row["psnr_mean"] == float("inf") else f"{row['psnr_mean']:.2f} dB"
            ssim_text = f"{row['ssim_mean']:.4f}" if row.get("ssim_mean") is not None else "n/a"
            lines.append(f"| {row['variant']} | {row['samples']} | {row['mse_mean']:.2f} | {psnr} | {ssim_text} |")
        return "\n".join(lines)

    text = f"""# M4 OpenCV Resize Benchmark Results

## Environment

- CPU: {env.get('cpu') or 'Apple Silicon'}
- macOS: {env.get('macos')}
- Python: {env.get('python_version')} ({env.get('python_arch')})
- OpenCV: {env.get('opencv')}
- Power: {env.get('power')}
- Low Power Mode: {env.get('lowpowermode')}

## Workload

- Source: {config['src_w']}×{config['src_h']} RGB / MJPEG
- Destination: {config['dst_w']}×{config['dst_h']}
- Frames per repeat: {config['n_frames']}
- Repeats: {config['n_repeats']}
- Warmup frames: {config['warmup_frames']}
- JPEG quality: {config['jpeg_quality']}

## Best Results

- RGB winner: {best_rgb['name']} — {best_rgb['fps_mean']:.1f} fps, {best_rgb['speedup']:.2f}× vs OpenCV 1-thread linear.
- MJPEG winner: {best_mjpeg['name']} — {best_mjpeg['fps_mean']:.1f} fps, {best_mjpeg['speedup']:.2f}× vs full decode + resize.
- Best exact-output MJPEG winner: {best_exact_mjpeg['name'] if best_exact_mjpeg else 'not measured'} — {best_exact_mjpeg['fps_mean'] if best_exact_mjpeg else 0:.1f} fps.
- Best approximate-output MJPEG winner: {best_approx_mjpeg['name'] if best_approx_mjpeg else 'not measured'} — {best_approx_mjpeg['fps_mean'] if best_approx_mjpeg else 0:.1f} fps.

## MJPEG Category Summary

- Baseline correctness path: {baseline['name']} — {baseline['fps_mean']:.1f} fps, {baseline['ms_per_frame']:.2f} ms/frame, 1.00×.
- Balanced speed/quality path: {balanced['name'] if balanced else 'not measured'} — {balanced['fps_mean'] if balanced else 0:.1f} fps, {balanced['speedup'] if balanced else 0:.2f}×.
- Maximum-speed exact-output path: {max_speed['name'] if max_speed else 'not measured'} — {max_speed['fps_mean'] if max_speed else 0:.1f} fps, {max_speed['speedup'] if max_speed else 0:.2f}×.
- Maximum-speed approximate-output path: {best_approx_mjpeg['name'] if best_approx_mjpeg else 'not measured'} — {best_approx_mjpeg['fps_mean'] if best_approx_mjpeg else 0:.1f} fps, {best_approx_mjpeg['speedup'] if best_approx_mjpeg else 0:.2f}×.

## RGB Results

{table_for(lambda row: not row['category'].startswith('mjpeg'))}

## MJPEG Results

{table_for(lambda row: row['category'].startswith('mjpeg'))}

## MJPEG Quality Comparison

{quality_table()}

## Recommended Submission Claim

- For decoded RGB, report the fastest quality-acceptable OpenCV path from the RGB table.
- For MJPEG, report the balanced `REDUCED_2 + INTER_LINEAR` path when quality matters, and the `REDUCED_4 + INTER_NEAREST` path when speed is the only judging metric.
- Use the full command `python3 M4benchmark.py --skip-gpu` for stable CPU/MJPEG numbers, plugged into power.
"""
    with open(path, "w") as f:
        f.write(text)
    return path


# ============================================================
# RGB CPU benchmarks — thread sweep
# ============================================================
def bench_rgb_cpu(rgb_frames):
    print("\n=== RGB INPUT, CPU (OpenCV thread sweep) ===")

    n_logical = os.cpu_count() or 10
    sweep = sorted({1, 2, 4, 6, 8, n_logical})

    for n in sweep:
        cv2.setNumThreads(n)

        def cv_linear(frames):
            for f in frames:
                cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
        record(f"OpenCV INTER_LINEAR ({n} threads)", "opencv-cpu",
               time_it(cv_linear, rgb_frames),
               f"setNumThreads({n})")

    cv2.setNumThreads(n_logical)

    dst_pool = [np.empty((DST_H, DST_W, 3), dtype=np.uint8) for _ in rgb_frames]
    def cv_linear_prealloc(frames):
        for i, f in enumerate(frames):
            cv2.resize(f, (DST_W, DST_H), dst=dst_pool[i], interpolation=cv2.INTER_LINEAR)
    record(f"OpenCV INTER_LINEAR preallocated dst ({n_logical} threads)", "opencv-cpu",
           time_it(cv_linear_prealloc, rgb_frames),
           "removes per-frame output allocation")

    def cv_nearest(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_NEAREST)
    record(f"OpenCV INTER_NEAREST ({n_logical} threads)", "opencv-cpu",
           time_it(cv_nearest, rgb_frames), "fastest interp, lowest quality")

    def cv_area(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_AREA)
    record(f"OpenCV INTER_AREA ({n_logical} threads)", "opencv-cpu",
           time_it(cv_area, rgb_frames), "best downsample quality")

    def cv_cubic(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_CUBIC)
    record(f"OpenCV INTER_CUBIC ({n_logical} threads)", "opencv-cpu",
           time_it(cv_cubic, rgb_frames), "higher quality, slower")

    def cv_lanczos(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_LANCZOS4)
    record(f"OpenCV INTER_LANCZOS4 ({n_logical} threads)", "opencv-cpu",
           time_it(cv_lanczos, rgb_frames), "highest quality, slowest")

    cv2.setNumThreads(1)
    def cv_threadpool(frames):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(
                lambda f: cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR),
                frames))
    record("OpenCV LINEAR + ThreadPool(4 P-cores), 1 thread per call", "opencv-cpu",
           time_it(cv_threadpool, rgb_frames), "frame-level parallelism")

    if SKIP_OPTIONAL:
        print("  (optional libraries skipped)")
    else:
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in rgb_frames]
            def pil_bilinear(frames):
                for im in frames:
                    im.resize((DST_W, DST_H), Image.BILINEAR)
            record("Pillow BILINEAR", "pillow", time_it(pil_bilinear, pil_frames),
                   "Pillow on arm64 (NEON)")

            def pil_lanczos(frames):
                for im in frames:
                    im.resize((DST_W, DST_H), Image.LANCZOS)
            record("Pillow LANCZOS", "pillow", time_it(pil_lanczos, pil_frames),
                   "highest-quality Pillow filter")
        except ImportError:
            print("  (Pillow not installed, skipping)")

        try:
            import pyvips
            def make_vips(rgb):
                return pyvips.Image.new_from_memory(rgb.tobytes(), SRC_W, SRC_H, 3, 'uchar')
            vips_frames = [make_vips(f) for f in rgb_frames]
            scale = DST_W / SRC_W
            def vips_resize(frames):
                for im in frames:
                    out = im.resize(scale, kernel='linear')
                    _ = out.write_to_memory()
            record("pyvips resize linear", "pyvips",
                   time_it(vips_resize, vips_frames), "lazy pipeline")
        except (ImportError, OSError) as e:
            print(f"  (pyvips not available: {type(e).__name__}, skipping)")

    cv2.setNumThreads(n_logical)


# ============================================================
# RGB GPU benchmarks
# ============================================================
def bench_rgb_gpu(rgb_frames):
    print("\n=== RGB INPUT, GPU paths ===")

    # ---------- PyTorch MPS ----------
    try:
        import torch
        if not torch.backends.mps.is_available():
            print("  (PyTorch MPS not available, skipping)")
        else:
            device = torch.device('mps')
            mps_frames = rgb_frames[:min(len(rgb_frames), 16)]
            tensors_on_gpu = [
                torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
                for f in mps_frames
            ]

            def mps_resize_pure(frames):
                for t in frames:
                    _ = torch.nn.functional.interpolate(
                        t, size=(DST_H, DST_W), mode='bilinear', align_corners=False)
                torch.mps.synchronize()
            record("PyTorch MPS bilinear (tensors on GPU)", "gpu-mps",
                   time_it(mps_resize_pure, tensors_on_gpu),
                   f"pure GPU compute on {len(mps_frames)} resident tensors")

            def mps_resize_full(frames):
                for f in frames:
                    t = torch.from_numpy(f).to(device).permute(2, 0, 1).unsqueeze(0).float()
                    out = torch.nn.functional.interpolate(
                        t, size=(DST_H, DST_W), mode='bilinear', align_corners=False)
                    _ = out.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                torch.mps.synchronize()
            record("PyTorch MPS bilinear (numpy→GPU→numpy)", "gpu-mps",
                   time_it(mps_resize_full, rgb_frames),
                   "includes host↔device transfer")

            def mps_nearest(frames):
                for t in frames:
                    _ = torch.nn.functional.interpolate(t, size=(DST_H, DST_W), mode='nearest')
                torch.mps.synchronize()
            record("PyTorch MPS nearest (tensors on GPU)", "gpu-mps",
                   time_it(mps_nearest, tensors_on_gpu), "nearest, GPU")
    except ImportError:
        print("  (PyTorch not installed, skipping MPS)")

    # ---------- OpenCV OpenCL via UMat ----------
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            try:
                dev_name = cv2.ocl.Device_getDefault().name()
                print(f"  OpenCL device: {dev_name}")
            except Exception:
                pass

            umats = [cv2.UMat(f) for f in rgb_frames]

            def ocl_resize_uploaded(frames):
                for um in frames:
                    out = cv2.resize(um, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
                    _ = out.get()
            record("OpenCV OpenCL/UMat LINEAR (pre-uploaded)", "gpu-opencl",
                   time_it(ocl_resize_uploaded, umats),
                   "UMat already on GPU; .get() forces sync")

            def ocl_resize_full(frames):
                for f in frames:
                    um = cv2.UMat(f)
                    out = cv2.resize(um, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
                    _ = out.get()
            record("OpenCV OpenCL/UMat LINEAR (numpy→UMat→numpy)", "gpu-opencl",
                   time_it(ocl_resize_full, rgb_frames),
                   "includes upload + download")
        else:
            print("  (OpenCL not available in this OpenCV build, skipping)")
    except Exception as e:
        print(f"  (OpenCL path failed: {e})")

    # ---------- Core Image via pyobjc ----------
    try:
        from Quartz import (CIImage, CIContext, CIFilter, CIVector,
                            CGRectMake, kCIFormatRGBA8)
        from Foundation import NSData
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        ci_context = CIContext.contextWithMTLDevice_(device)

        # Convert frames to RGBA (CI prefers 4-channel)
        rgba_frames = [np.dstack([f, np.full((SRC_H, SRC_W), 255, dtype=np.uint8)])
                       for f in rgb_frames]

        scale = DST_W / SRC_W

        def ci_resize(frames):
            for f in frames:
                data = NSData.dataWithBytes_length_(f.tobytes(), f.nbytes)
                size = CIVector.vectorWithX_Y_(SRC_W, SRC_H)
                ci_img = CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
                    data, SRC_W * 4, (SRC_W, SRC_H), kCIFormatRGBA8, None)
                if ci_img is None:
                    raise RuntimeError("CIImage creation returned None")
                f_filter = CIFilter.filterWithName_("CILanczosScaleTransform")
                f_filter.setValue_forKey_(ci_img, "inputImage")
                f_filter.setValue_forKey_(scale, "inputScale")
                f_filter.setValue_forKey_(1.0, "inputAspectRatio")
                out = f_filter.valueForKey_("outputImage")
                _ = ci_context.createCGImage_fromRect_(out, out.extent())
        record("Core Image CILanczosScaleTransform (Metal)", "gpu-coreimage",
               time_it(ci_resize, rgba_frames),
               "Apple's GPU image pipeline; Lanczos quality")
    except ImportError as e:
        print(f"  (pyobjc/Quartz not installed, skipping Core Image: {e})")
    except Exception as e:
        print(f"  (Core Image path failed: {type(e).__name__}: {e})")


# ============================================================
# MJPEG benchmarks
# ============================================================
def bench_mjpeg(mjpeg_frames, quality_compare=False, quality_samples=5):
    print("\n=== MJPEG INPUT (decode + resize) ===")
    n_logical = os.cpu_count() or 10

    cv2.setNumThreads(n_logical)
    mjpeg_dst_pool = [np.empty((DST_H, DST_W, 3), dtype=np.uint8) for _ in mjpeg_frames]

    def cv_decode_then_resize(frames):
        for buf in frames:
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
    record("OpenCV imdecode + resize LINEAR (full 4K decode)", "mjpeg-opencv",
           time_it(cv_decode_then_resize, mjpeg_frames),
           "baseline: decode at full 4K, then resize")

    def cv_decode_then_resize_prealloc(frames):
        for i, buf in enumerate(frames):
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cv2.resize(img, (DST_W, DST_H), dst=mjpeg_dst_pool[i], interpolation=cv2.INTER_LINEAR)
    record("OpenCV imdecode + resize LINEAR preallocated dst", "mjpeg-opencv",
           time_it(cv_decode_then_resize_prealloc, mjpeg_frames),
           "full decode baseline without resize output allocation")

    print("\n=== MJPEG OPTIMIZED MATRIX (OpenCV threads forced to 1) ===")
    variants = [
        ("REDUCED_2 + INTER_LINEAR", cv2.IMREAD_REDUCED_COLOR_2, cv2.INTER_LINEAR, "balanced speed/quality path"),
        ("REDUCED_2 + INTER_NEAREST", cv2.IMREAD_REDUCED_COLOR_2, cv2.INTER_NEAREST, "faster 1/2 DCT decode with nearest final resize"),
        ("REDUCED_4 + INTER_LINEAR", cv2.IMREAD_REDUCED_COLOR_4, cv2.INTER_LINEAR, "more aggressive 1/4 DCT decode"),
        ("REDUCED_4 + INTER_NEAREST", cv2.IMREAD_REDUCED_COLOR_4, cv2.INTER_NEAREST, "maximum-speed path, lowest quality"),
        ("REDUCED_8 + INTER_LINEAR", cv2.IMREAD_REDUCED_COLOR_8, cv2.INTER_LINEAR, "most aggressive 1/8 DCT decode with final resize"),
        ("REDUCED_8 + INTER_NEAREST", cv2.IMREAD_REDUCED_COLOR_8, cv2.INTER_NEAREST, "maximum DCT shrink, lowest quality"),
    ]
    decode_only_variants = [
        ("REDUCED_2 decode only", cv2.IMREAD_REDUCED_COLOR_2, "approximate output: native 1/2 JPEG decode"),
        ("REDUCED_4 decode only", cv2.IMREAD_REDUCED_COLOR_4, "approximate output: native 1/4 JPEG decode"),
        ("REDUCED_8 decode only", cv2.IMREAD_REDUCED_COLOR_8, "approximate output: native 1/8 JPEG decode"),
    ]
    workers_list = [1, 2, 4, 6, 8, 10, 12, 14, 16]

    print("\n=== MJPEG REDUCED-DECODE-ONLY MATRIX (native output sizes) ===")
    for variant_name, decode_flag, notes in decode_only_variants:
        sample = decode_mjpeg_native(mjpeg_frames[0], decode_flag)
        native_h, native_w = sample.shape[:2]
        for workers in workers_list:
            def run_decode_only(frames, decode_flag=decode_flag, workers=workers):
                consume_mjpeg_decode_only(frames, decode_flag, workers)
            row = record(
                f"OpenCV {variant_name}, ThreadPool({workers})",
                "mjpeg-opencv",
                time_it(run_decode_only, mjpeg_frames),
                f"{notes}; cv2.setNumThreads(1); np.frombuffer; no output list",
            )
            row["mjpeg_variant"] = variant_name
            row["workers"] = workers
            row["output_width"] = native_w
            row["output_height"] = native_h
            row["exact_output_size"] = (native_w == DST_W and native_h == DST_H)
            row["has_final_resize"] = False

    for variant_name, decode_flag, interpolation, notes in variants:
        for workers in workers_list:
            def run_variant(frames, decode_flag=decode_flag, interpolation=interpolation, workers=workers):
                consume_mjpeg_variant(frames, decode_flag, interpolation, workers)
            row = record(
                f"OpenCV {variant_name}, ThreadPool({workers})",
                "mjpeg-opencv",
                time_it(run_variant, mjpeg_frames),
                f"{notes}; cv2.setNumThreads(1); np.frombuffer; no output list",
            )
            row["mjpeg_variant"] = variant_name
            row["workers"] = workers
            row["output_width"] = DST_W
            row["output_height"] = DST_H
            row["exact_output_size"] = True
            row["has_final_resize"] = True

    if quality_compare:
        print("\n=== MJPEG QUALITY COMPARISON vs full decode + INTER_LINEAR ===")
        ssim_fn = require_ssim()
        sample_frames = mjpeg_frames[:max(1, min(quality_samples, len(mjpeg_frames)))]
        baselines = [
            decode_resize_mjpeg(buf, cv2.IMREAD_COLOR, cv2.INTER_LINEAR)
            for buf in sample_frames
        ]
        for variant_name, decode_flag, interpolation, _ in variants:
            mses = []
            psnrs = []
            ssims = []
            for buf, reference in zip(sample_frames, baselines):
                candidate = decode_resize_mjpeg(buf, decode_flag, interpolation)
                mse, psnr = mse_psnr(reference, candidate)
                mses.append(mse)
                psnrs.append(psnr)
                ssims.append(float(ssim_fn(reference, candidate, channel_axis=2)))
            finite_psnrs = [p for p in psnrs if np.isfinite(p)]
            psnr_mean = float(statistics.mean(finite_psnrs)) if finite_psnrs else float("inf")
            quality_row = {
                "variant": variant_name,
                "samples": len(sample_frames),
                "mse_mean": float(statistics.mean(mses)),
                "mse_min": float(min(mses)),
                "mse_max": float(max(mses)),
                "psnr_mean": psnr_mean,
                "psnr_min": float(min(finite_psnrs)) if finite_psnrs else float("inf"),
                "psnr_max": float(max(finite_psnrs)) if finite_psnrs else float("inf"),
                "ssim_mean": float(statistics.mean(ssims)),
                "ssim_min": float(min(ssims)),
                "ssim_max": float(max(ssims)),
            }
            QUALITY_RESULTS.append(quality_row)
            psnr_text = "inf" if quality_row["psnr_mean"] == float("inf") else f"{quality_row['psnr_mean']:.2f} dB"
            print(
                f"  {variant_name:30s}  MSE={quality_row['mse_mean']:8.2f}  "
                f"PSNR={psnr_text}  SSIM={quality_row['ssim_mean']:.4f}"
            )

    cv2.setNumThreads(n_logical)

    if SKIP_OPTIONAL:
        print("  (optional libraries skipped)")
        return

    try:
        from PIL import Image
        def pil_draft(frames):
            for buf in frames:
                im = Image.open(io.BytesIO(buf))
                im.draft('RGB', (DST_W, DST_H))
                im = im.resize((DST_W, DST_H), Image.BILINEAR)
                im.load()
        record("Pillow draft + resize BILINEAR", "mjpeg-pillow",
               time_it(pil_draft, mjpeg_frames),
               "draft() uses libjpeg DCT scaling")
    except ImportError:
        pass

    try:
        import pyvips
        def vips_jpeg_shrink(frames):
            for buf in frames:
                im = pyvips.Image.jpegload_buffer(buf, shrink=2)
                out = im.resize(DST_W / im.width, kernel='linear')
                _ = out.write_to_memory()
        record("pyvips jpegload(shrink=2) + resize", "mjpeg-pyvips",
               time_it(vips_jpeg_shrink, mjpeg_frames),
               "libjpeg-turbo block-shrink inside vips")
    except (ImportError, OSError):
        pass


def bench_mjpeg_diversity(diverse_images):
    print("\n=== MJPEG CONTENT DIVERSITY TEST ===")
    paths = [
        (
            "full_decode_inter_linear",
            "Full decode + INTER_LINEAR",
            cv2.IMREAD_COLOR,
            cv2.INTER_LINEAR,
            1,
            os.cpu_count() or 10,
        ),
        (
            "reduced2_linear_tp10",
            "REDUCED_2 + INTER_LINEAR + ThreadPool(10)",
            cv2.IMREAD_REDUCED_COLOR_2,
            cv2.INTER_LINEAR,
            10,
            1,
        ),
        (
            "reduced8_linear_tp14",
            "REDUCED_8 + INTER_LINEAR + ThreadPool(14)",
            cv2.IMREAD_REDUCED_COLOR_8,
            cv2.INTER_LINEAR,
            14,
            1,
        ),
    ]

    for item in diverse_images:
        frames = [item["jpeg"]] * N_FRAMES
        print(f"  {item['label']} ({item['jpeg_bytes'] / 1024:.1f} KB JPEG)")
        for path_key, path_label, decode_flag, interpolation, workers, cv_threads in paths:
            def run_path(frames, decode_flag=decode_flag, interpolation=interpolation,
                         workers=workers, cv_threads=cv_threads):
                consume_mjpeg_variant(frames, decode_flag, interpolation, workers, cv_threads=cv_threads)

            fps_list = time_it(run_path, frames)
            row = {
                "image_key": item["key"],
                "image_label": item["label"],
                "jpeg_bytes": item["jpeg_bytes"],
                "path_key": path_key,
                "path_label": path_label,
                "workers": workers,
                "cv_threads": cv_threads,
                "fps_mean": float(statistics.mean(fps_list)),
                "fps_stdev": float(statistics.stdev(fps_list)) if len(fps_list) > 1 else 0.0,
                "fps_min": float(min(fps_list)),
                "fps_max": float(max(fps_list)),
                "ms_per_frame": float(1000.0 / statistics.mean(fps_list)),
            }
            DIVERSITY_RESULTS.append(row)
            print(
                f"    {path_label:45s} {row['fps_mean']:8.1f} fps "
                f"({row['ms_per_frame']:5.2f} ms/frame, ±{row['fps_stdev']:5.1f})"
            )


def bench_thread_interaction(mjpeg_frames):
    print("\n=== THREAD INTERACTION SWEEP: REDUCED_2 + INTER_LINEAR ===")
    cv_threads_list = [1, 2, 4]
    workers_list = [2, 4, 6, 8, 10, 12, 14]

    for cv_threads in cv_threads_list:
        for workers in workers_list:
            if cv_threads * workers > 20:
                continue

            def run_variant(frames, cv_threads=cv_threads, workers=workers):
                consume_mjpeg_variant(
                    frames,
                    cv2.IMREAD_REDUCED_COLOR_2,
                    cv2.INTER_LINEAR,
                    workers,
                    cv_threads=cv_threads,
                )

            fps_list = time_it(run_variant, mjpeg_frames)
            row = {
                "cv_threads": cv_threads,
                "workers": workers,
                "thread_product": cv_threads * workers,
                "fps_mean": float(statistics.mean(fps_list)),
                "fps_stdev": float(statistics.stdev(fps_list)) if len(fps_list) > 1 else 0.0,
                "fps_min": float(min(fps_list)),
                "fps_max": float(max(fps_list)),
                "ms_per_frame": float(1000.0 / statistics.mean(fps_list)),
            }
            THREAD_INTERACTION_RESULTS.append(row)
            print(
                f"  cv2.setNumThreads({cv_threads}), ThreadPool({workers}) "
                f"= {row['fps_mean']:8.1f} fps ({row['ms_per_frame']:5.2f} ms/frame)"
            )

    best = max(THREAD_INTERACTION_RESULTS, key=lambda row: row["fps_mean"])
    best_t1 = max(
        [row for row in THREAD_INTERACTION_RESULTS if row["cv_threads"] == 1],
        key=lambda row: row["fps_mean"],
    )
    hybrid_best = max(
        [row for row in THREAD_INTERACTION_RESULTS if row["cv_threads"] > 1],
        key=lambda row: row["fps_mean"],
        default=None,
    )
    print(
        f"  Best overall: T={best['cv_threads']}, W={best['workers']} "
        f"({best['fps_mean']:.1f} fps)"
    )
    if hybrid_best:
        beats = hybrid_best["fps_mean"] > best_t1["fps_mean"]
        print(
            f"  Best hybrid: T={hybrid_best['cv_threads']}, W={hybrid_best['workers']} "
            f"({hybrid_best['fps_mean']:.1f} fps); beats best T=1: {beats}"
        )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.frames = 8
        args.repeats = 2
        args.warmup = 2
    if args.long_run:
        args.frames = max(args.frames, 160)
        args.repeats = max(args.repeats, 7)
        args.warmup = max(args.warmup, 20)

    N_FRAMES = args.frames
    N_REPEATS = args.repeats
    WARMUP_FRAMES = args.warmup
    JPEG_QUALITY = args.quality
    SKIP_OPTIONAL = args.skip_optional

    info = env_check()

    print(f"Source: {SRC_W}x{SRC_H} -> {DST_W}x{DST_H}  ({SRC_W*SRC_H/1e6:.2f}M -> {DST_W*DST_H/1e6:.2f}M pixels)")
    print(f"N_FRAMES={N_FRAMES}, N_REPEATS={N_REPEATS}, WARMUP={WARMUP_FRAMES}, JPEG_QUALITY={JPEG_QUALITY}")
    print()

    print("Generating test frames…")
    rgb_frames = make_test_frames(N_FRAMES)
    mjpeg_frames = make_mjpeg_frames(rgb_frames, quality=JPEG_QUALITY)
    diverse_images = make_diverse_test_images(quality=JPEG_QUALITY)
    avg_kb = sum(len(b) for b in mjpeg_frames) / len(mjpeg_frames) / 1024
    print(f"Average MJPEG buffer: {avg_kb:.1f} KB/frame")

    bench_rgb_cpu(rgb_frames)
    if args.skip_gpu:
        print("\n=== RGB INPUT, GPU paths ===")
        print("  (GPU paths skipped)")
    else:
        bench_rgb_gpu(rgb_frames)
    bench_mjpeg(mjpeg_frames, quality_compare=args.quality_compare, quality_samples=args.quality_samples)
    bench_mjpeg_diversity(diverse_images)
    bench_thread_interaction(mjpeg_frames)

    add_speedups()

    output = {
        'env': info,
        'config': {
            'src_w': SRC_W, 'src_h': SRC_H,
            'dst_w': DST_W, 'dst_h': DST_H,
            'n_frames': N_FRAMES, 'n_repeats': N_REPEATS,
            'warmup_frames': WARMUP_FRAMES,
            'jpeg_quality': JPEG_QUALITY,
        },
        'results': RESULTS,
        'quality': QUALITY_RESULTS,
        'diversity': DIVERSITY_RESULTS,
        'thread_interaction': THREAD_INTERACTION_RESULTS,
    }

    out_path = 'm4_results_extended.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    report_path = write_markdown_report(output)
    print(f"\n✅ Saved {len(RESULTS)} measurements to {out_path}")
    print(f"✅ Wrote summary report to {report_path}")
    print("For final numbers, run on wall power without --quick.")
