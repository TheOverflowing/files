"""
Google Colab Tesla T4 CUDA resize benchmark.

Workload: 3840x1920 RGB -> 1920x1080, plus MJPEG decode + resize.
Output: t4_results.json

The script intentionally avoids building OpenCV with CUDA. It uses OpenCV for
CPU baselines and PyTorch, CuPy, and optional NVIDIA DALI/nvJPEG for GPU paths.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _pip_install(packages: list[str], optional: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    try:
        print("Installing:", " ".join(packages))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:
        msg = f"WARNING: pip install failed for {' '.join(packages)}: {exc}"
        if optional:
            print(msg)
            return False
        raise RuntimeError(msg) from exc


def install_dependencies(skip_dali: bool) -> None:
    """Install missing Colab dependencies. DALI is optional and may fail."""
    required = []
    if not _has_module("cv2"):
        required.append("opencv-python-headless")
    if not _has_module("torch"):
        required.extend(["torch", "torchvision"])
    elif not _has_module("torchvision"):
        required.append("torchvision")
    if not _has_module("cupy"):
        required.append("cupy-cuda12x")
    if not _has_module("skimage"):
        required.append("scikit-image")
    if required:
        _pip_install(required)

    if not skip_dali and not _has_module("nvidia.dali"):
        _pip_install(["nvidia-dali-cuda120"], optional=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tesla T4 CUDA resize benchmark for Google Colab.")
    parser.add_argument("--skip-dali", action="store_true", help="Skip NVIDIA DALI install and benchmarks.")
    parser.add_argument("--quick", action="store_true", help="Use 50 frames x 1 repeat for faster iteration.")
    parser.add_argument("--frames", type=int, default=None, help="Override measured frames per repeat.")
    parser.add_argument("--repeats", type=int, default=None, help="Override measured repeats.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations. Default: 20.")
    parser.add_argument("--mjpeg-frames", type=int, default=None, help="Override MJPEG frames per image type. Default: 80, or 20 with --quick.")
    parser.add_argument("--output", default="t4_results.json", help="JSON output path. Default: t4_results.json.")
    return parser.parse_known_args()[0]


ARGS = parse_args()
install_dependencies(ARGS.skip_dali)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndimage
except Exception as exc:
    cp = None
    cndimage = None
    print(f"WARNING: CuPy unavailable; CuPy benchmarks will be skipped: {exc}")

try:
    if ARGS.skip_dali:
        raise ImportError("--skip-dali was provided")
    import nvidia.dali.fn as dali_fn
    import nvidia.dali.types as dali_types
    from nvidia.dali import pipeline_def
except Exception as exc:
    dali_fn = None
    dali_types = None
    pipeline_def = None
    print(f"WARNING: DALI unavailable; DALI benchmarks will be skipped: {exc}")


SRC_W, SRC_H = 3840, 1920
DST_W, DST_H = 1920, 1080
JPEG_QUALITY = 85
DEFAULT_FRAMES = 200
DEFAULT_REPEATS = 3
QUICK_FRAMES = 50
QUICK_REPEATS = 1
MJPEG_FRAMES_PER_TYPE = 80


def sync_torch() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sync_cupy() -> None:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def mean_fps(frame_count: int, times: list[float]) -> tuple[float, float]:
    fps = [frame_count / t for t in times]
    fps_mean = statistics.mean(fps)
    return fps_mean, 1000.0 / fps_mean


def record_error(errors: list[dict[str, str]], method: str, exc: BaseException) -> None:
    print(f"ERROR: {method} failed: {exc}")
    errors.append(
        {
            "method": method,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }
    )


def benchmark_loop(
    name: str,
    frame_count: int,
    repeats: int,
    warmup: int,
    fn: Callable[[], None],
    sync: Callable[[], None] | None = None,
) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        for _ in range(frame_count):
            fn()
        if sync:
            sync()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    fps, ms = mean_fps(frame_count, times)
    print(f"  {name:54s} {fps:9.2f} fps  {ms:8.3f} ms/frame")
    return fps, ms, times


def benchmark_batches(
    name: str,
    batches_per_repeat: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    fn: Callable[[], None],
    sync: Callable[[], None] | None = None,
) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    total_frames = batches_per_repeat * batch_size
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        for _ in range(batches_per_repeat):
            fn()
        if sync:
            sync()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    fps, ms = mean_fps(total_frames, times)
    print(f"  {name:54s} {fps:9.2f} fps  {ms:8.3f} ms/frame")
    return fps, ms, times


def benchmark_dataset(
    name: str,
    items: list[Any],
    repeats: int,
    warmup_items: int,
    process_batch: Callable[[list[Any]], None],
    sync: Callable[[], None] | None = None,
) -> tuple[float, float, list[float]]:
    """Benchmark a function that processes a list of frames/buffers.

    Unlike benchmark_loop(fn_that_processes_80_frames, frame_count=1), this keeps
    the warmup count in units of frames. That matters for MJPEG because full CPU
    decode is expensive on Colab's small CPU allocation.
    """
    if not items:
        raise ValueError("benchmark_dataset requires at least one item")
    warm = [items[i % len(items)] for i in range(max(1, warmup_items))]
    process_batch(warm)
    if sync:
        sync()
    times = []
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        process_batch(items)
        if sync:
            sync()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    fps, ms = mean_fps(len(items), times)
    print(f"  {name:54s} {fps:9.2f} fps  {ms:8.3f} ms/frame")
    return fps, ms, times


def make_test_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    image = np.zeros((SRC_H, SRC_W, 3), dtype=np.float32)
    image[..., 0] = 128 + 64 * np.sin(xx / 200) * np.cos(yy / 150)
    image[..., 1] = 128 + 64 * np.cos(xx / 180) * np.sin(yy / 220)
    image[..., 2] = 128 + 64 * np.sin((xx + yy) / 250)
    image += rng.normal(0, 5, image.shape).astype(np.float32)
    return np.clip(image, 0, 255).astype(np.uint8)


def make_diverse_test_images() -> list[dict[str, Any]]:
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    images = []

    low = np.zeros((SRC_H, SRC_W, 3), dtype=np.float32)
    low[..., 0] = 108 + 10 * (xx / SRC_W)
    low[..., 1] = 132 + 8 * (yy / SRC_H)
    low[..., 2] = 156 + 6 * ((xx + yy) / (SRC_W + SRC_H))
    images.append(("low_texture", np.clip(low, 0, 255).astype(np.uint8)))

    checker = (((xx.astype(np.int32) // 4) + (yy.astype(np.int32) // 4)) % 2) * 255
    noise = rng.integers(0, 256, size=(SRC_H, SRC_W), dtype=np.uint8)
    high = np.stack([checker.astype(np.uint8), noise, np.bitwise_xor(checker.astype(np.uint8), noise)], axis=2)
    images.append(("high_frequency", high))

    natural = np.zeros((SRC_H, SRC_W, 3), dtype=np.float32)
    natural[..., 0] = 80 + 110 * (xx / SRC_W)
    natural[..., 1] = 60 + 140 * (yy / SRC_H)
    natural[..., 2] = 120 + 50 * np.sin((xx + yy) / 300)
    natural = np.clip(natural, 0, 255).astype(np.uint8)
    cv2.rectangle(natural, (400, 260), (1600, 820), (40, 70, 190), thickness=-1)
    cv2.rectangle(natural, (2100, 900), (3400, 1500), (210, 180, 60), thickness=-1)
    cv2.circle(natural, (2750, 520), 260, (230, 80, 90), thickness=-1)
    images.append(("natural_like", natural))

    mixed = np.zeros((SRC_H, SRC_W, 3), dtype=np.uint8)
    mixed[:, : SRC_W // 2] = np.clip(low[:, : SRC_W // 2], 0, 255).astype(np.uint8)
    mixed[:, SRC_W // 2 :] = rng.integers(0, 256, size=(SRC_H, SRC_W - SRC_W // 2, 3), dtype=np.uint8)
    images.append(("mixed", mixed))

    text = np.full((SRC_H, SRC_W, 3), 245, dtype=np.uint8)
    for y in range(120, SRC_H - 80, 48):
        x = 120
        while x < SRC_W - 180:
            block_w = int(rng.integers(18, 120))
            block_h = int(rng.integers(8, 18))
            if rng.random() < 0.82:
                cv2.rectangle(text, (x, y), (min(x + block_w, SRC_W - 80), y + block_h), (20, 20, 20), -1)
            x += block_w + int(rng.integers(8, 32))
    images.append(("text_like", text))

    encoded = []
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    for key, image in images:
        ok, buf = cv2.imencode(".jpg", image, params)
        if not ok:
            raise RuntimeError(f"JPEG encode failed for {key}")
        encoded.append({"image_type": key, "image": image, "jpeg": buf.tobytes(), "jpeg_bytes": int(len(buf))})
    return encoded


def torch_tensor_from_rgb(image: np.ndarray, batch_size: int = 1) -> torch.Tensor:
    tensor = torch.from_numpy(image).to(device="cuda", non_blocking=False)
    tensor = tensor.permute(2, 0, 1).contiguous().float().div_(255.0)
    if batch_size == 1:
        return tensor.unsqueeze(0).contiguous()
    return tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous()


def torch_resize(tensor: torch.Tensor) -> torch.Tensor:
    return F.interpolate(tensor, size=(DST_H, DST_W), mode="bilinear", align_corners=False)


def torch_output_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    out = tensor[0].detach().clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).contiguous()
    return out.cpu().numpy()


def torch_batch_to_uint8_nhwc(tensor: torch.Tensor) -> np.ndarray:
    out = tensor.detach().clamp(0, 1).mul(255.0).byte().permute(0, 2, 3, 1).contiguous()
    return out.cpu().numpy()


def torch_tensor_from_rgb_batch(batch: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(batch).to(device="cuda", non_blocking=False)
    return tensor.permute(0, 3, 1, 2).contiguous().float().div_(255.0)


def cupy_resize_gpu(image_gpu: "cp.ndarray") -> "cp.ndarray":
    zoom = (DST_H / SRC_H, DST_W / SRC_W, 1.0)
    return cndimage.zoom(image_gpu, zoom=zoom, order=1, mode="nearest", prefilter=False)


def maybe_make_dali_resize_pipeline(batch_size: int, batch: list[np.ndarray]):
    if pipeline_def is None:
        return None

    @pipeline_def(batch_size=batch_size, num_threads=2, device_id=0)
    def resize_pipe():
        images = dali_fn.external_source(source=lambda: batch, device="cpu", batch=True)
        images = images.gpu()
        return dali_fn.resize(images, resize_x=DST_W, resize_y=DST_H, interp_type=dali_types.INTERP_LINEAR)

    pipe = resize_pipe()
    pipe.build()
    return pipe


def maybe_make_dali_jpeg_pipeline(batch_size: int, num_threads: int, batch: list[np.ndarray]):
    if pipeline_def is None:
        return None

    @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    def jpeg_pipe():
        encoded = dali_fn.external_source(source=lambda: batch, device="cpu", batch=True)
        decoded = dali_fn.decoders.image(encoded, device="mixed", output_type=dali_types.RGB)
        return dali_fn.resize(decoded, resize_x=DST_W, resize_y=DST_H, interp_type=dali_types.INTERP_LINEAR)

    pipe = jpeg_pipe()
    pipe.build()
    return pipe


def dali_run_resize(pipe: Any) -> None:
    out = pipe.run()[0]
    out.as_cpu()


def dali_run_jpeg(pipe: Any) -> None:
    out = pipe.run()[0]
    out.as_cpu()


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu": None,
        "vram_gb": None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "cupy_version": getattr(cp, "__version__", None) if cp is not None else None,
        "opencv_version": cv2.__version__,
        "dali_available": pipeline_def is not None,
        "cpu_cores": os.cpu_count(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "note": "Colab T4 commonly exposes only 2 CPU cores, unlike a 10-core M4 Mac.",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["cuda_capability"] = f"{props.major}.{props.minor}"
    return info


def add_speedup(rows: list[dict[str, Any]], baseline_fps: float | None) -> None:
    if not baseline_fps:
        return
    for row in rows:
        row["speedup"] = row["fps"] / baseline_fps


def bench_rgb_resize(image: np.ndarray, frames: int, repeats: int, warmup: int, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    print("\n=== Part 1: Pure RGB resize ===")
    rows: list[dict[str, Any]] = []
    baseline_fps = None

    for threads in [1, 4]:
        method = f"OpenCV resize CPU threads={threads}"
        try:
            cv2.setNumThreads(threads)

            def fn() -> None:
                cv2.resize(image, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)

            fps, ms, times = benchmark_loop(method, frames, repeats, warmup, fn)
            row = {
                "method": method,
                "fps": fps,
                "ms_per_frame": ms,
                "includes_transfer": False,
                "gpu_only": False,
                "times_sec": times,
            }
            rows.append(row)
            if threads == 1:
                baseline_fps = fps
        except Exception as exc:
            record_error(errors, method, exc)

    if torch.cuda.is_available():
        for batch_size in [1, 4, 8, 16, 32]:
            method = f"PyTorch F.interpolate GPU-only batch={batch_size}"
            try:
                x_gpu = torch_tensor_from_rgb(image, batch_size=batch_size)

                @torch.no_grad()
                def fn() -> None:
                    y = torch_resize(x_gpu)
                    _ = y.contiguous()

                batches_per_repeat = max(1, math.ceil(frames / batch_size))
                fps, ms, times = benchmark_batches(method, batches_per_repeat, batch_size, repeats, warmup, fn, sync_torch)
                rows.append(
                    {
                        "method": method,
                        "fps": fps,
                        "ms_per_frame": ms,
                        "includes_transfer": False,
                        "gpu_only": True,
                        "batch_size": batch_size,
                        "times_sec": times,
                    }
                )
                del x_gpu
                torch.cuda.empty_cache()
            except Exception as exc:
                record_error(errors, method, exc)

        for batch_size in [1, 4, 8, 16, 32]:
            method = f"PyTorch F.interpolate end-to-end batch={batch_size}"
            try:
                cpu_batch = np.repeat(image[None, ...], batch_size, axis=0).copy()

                @torch.no_grad()
                def fn() -> None:
                    x_gpu = torch_tensor_from_rgb_batch(cpu_batch)
                    y = torch_resize(x_gpu)
                    _ = torch_batch_to_uint8_nhwc(y)

                batches_per_repeat = max(1, math.ceil(frames / batch_size))
                fps, ms, times = benchmark_batches(method, batches_per_repeat, batch_size, repeats, warmup, fn, sync_torch)
                rows.append(
                    {
                        "method": method,
                        "fps": fps,
                        "ms_per_frame": ms,
                        "includes_transfer": True,
                        "gpu_only": False,
                        "batch_size": batch_size,
                        "times_sec": times,
                    }
                )
                torch.cuda.empty_cache()
            except Exception as exc:
                record_error(errors, method, exc)

    if cp is not None:
        method = "CuPy cupyx.scipy.ndimage.zoom GPU-only"
        try:
            img_gpu = cp.asarray(image)

            def fn() -> None:
                _ = cupy_resize_gpu(img_gpu)

            fps, ms, times = benchmark_loop(method, frames, repeats, warmup, fn, sync_cupy)
            rows.append(
                {
                    "method": method,
                    "fps": fps,
                    "ms_per_frame": ms,
                    "includes_transfer": False,
                    "gpu_only": True,
                    "times_sec": times,
                }
            )
            del img_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            record_error(errors, method, exc)

        method = "CuPy cupyx.scipy.ndimage.zoom end-to-end"
        try:

            def fn() -> None:
                img_gpu = cp.asarray(image)
                out = cupy_resize_gpu(img_gpu)
                _ = cp.asnumpy(out)

            fps, ms, times = benchmark_loop(method, frames, repeats, warmup, fn, sync_cupy)
            rows.append(
                {
                    "method": method,
                    "fps": fps,
                    "ms_per_frame": ms,
                    "includes_transfer": True,
                    "gpu_only": False,
                    "times_sec": times,
                }
            )
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            record_error(errors, method, exc)

    if pipeline_def is not None:
        method = "NVIDIA DALI resize GPU pipeline end-to-end"
        try:
            batch = [image]
            pipe = maybe_make_dali_resize_pipeline(batch_size=1, batch=batch)

            def fn() -> None:
                dali_run_resize(pipe)

            fps, ms, times = benchmark_loop(method, frames, repeats, warmup, fn, sync_torch)
            rows.append(
                {
                    "method": method,
                    "fps": fps,
                    "ms_per_frame": ms,
                    "includes_transfer": True,
                    "gpu_only": False,
                    "batch_size": 1,
                    "times_sec": times,
                }
            )
        except Exception as exc:
            record_error(errors, method, exc)

    add_speedup(rows, baseline_fps)
    return rows


def bench_mjpeg_decode_resize(
    diverse: list[dict[str, Any]], mjpeg_frames: int, repeats: int, warmup: int, errors: list[dict[str, str]]
) -> list[dict[str, Any]]:
    print("\n=== Part 2: MJPEG decode + resize ===")
    rows: list[dict[str, Any]] = []

    for item in diverse:
        image_type = item["image_type"]
        jpeg = item["jpeg"]
        buffers = [jpeg] * mjpeg_frames
        print(f"\n-- {image_type} ({item['jpeg_bytes'] / 1024:.1f} KiB JPEG) --")

        method = "OpenCV imdecode full + resize CPU threads=1"
        try:
            cv2.setNumThreads(1)

            def process_one(buf: bytes) -> None:
                arr = np.frombuffer(buf, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)

            def process_batch(batch: list[bytes]) -> None:
                for buf in batch:
                    process_one(buf)

            fps, ms, times = benchmark_dataset(method, buffers, repeats, warmup, process_batch)
            rows.append({"method": method, "image_type": image_type, "fps": fps, "ms_per_frame": ms, "times_sec": times})
        except Exception as exc:
            record_error(errors, f"{method} {image_type}", exc)

        for workers in [4, 8, 12]:
            method = f"OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool({workers})"
            try:
                cv2.setNumThreads(1)

                def process_one(buf: bytes) -> None:
                    arr = np.frombuffer(buf, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_COLOR_2)
                    cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)

                def process_batch(batch: list[bytes]) -> None:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        list(pool.map(process_one, batch))

                fps, ms, times = benchmark_dataset(method, buffers, repeats, warmup, process_batch)
                rows.append({"method": method, "image_type": image_type, "fps": fps, "ms_per_frame": ms, "times_sec": times})
            except Exception as exc:
                record_error(errors, f"{method} {image_type}", exc)

        if pipeline_def is not None:
            for threads in [2, 4]:
                method = f"NVIDIA DALI nvJPEG mixed decode + GPU resize threads={threads}"
                try:
                    batch_size = 8
                    encoded = [np.frombuffer(jpeg, dtype=np.uint8) for _ in range(batch_size)]
                    pipe = maybe_make_dali_jpeg_pipeline(batch_size=batch_size, num_threads=threads, batch=encoded)
                    batches_per_repeat = max(1, math.ceil(mjpeg_frames / batch_size))

                    def fn() -> None:
                        dali_run_jpeg(pipe)

                    fps, ms, times = benchmark_batches(method, batches_per_repeat, batch_size, repeats, warmup, fn, sync_torch)
                    rows.append(
                        {
                            "method": method,
                            "image_type": image_type,
                            "fps": fps,
                            "ms_per_frame": ms,
                            "batch_size": batch_size,
                            "times_sec": times,
                        }
                    )
                except Exception as exc:
                    record_error(errors, f"{method} {image_type}", exc)
        else:
            print("  DALI unavailable; skipping nvJPEG GPU decode path for this image type.")

    return rows


def bench_batch_scaling(image: np.ndarray, frames: int, repeats: int, warmup: int, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    print("\n=== Part 3: PyTorch GPU batch throughput scaling ===")
    rows: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return rows

    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        method = f"PyTorch F.interpolate preloaded batch={batch_size}"
        try:
            torch.cuda.empty_cache()
            x_gpu = torch_tensor_from_rgb(image, batch_size=batch_size)

            @torch.no_grad()
            def fn() -> None:
                y = torch_resize(x_gpu)
                _ = y.contiguous()

            batches_per_repeat = max(1, math.ceil(frames / batch_size))
            fps, ms, times = benchmark_batches(method, batches_per_repeat, batch_size, repeats, warmup, fn, sync_torch)
            rows.append({"batch_size": batch_size, "fps": fps, "ms_per_frame": ms, "times_sec": times})
            del x_gpu
            torch.cuda.empty_cache()
        except Exception as exc:
            record_error(errors, method, exc)
    return rows


def compute_quality(image: np.ndarray, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    print("\n=== Part 4: Image quality verification ===")
    rows: list[dict[str, Any]] = []
    ref = cv2.resize(image, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)

    def add_quality(method: str, out: np.ndarray, note: str = "") -> None:
        out = np.asarray(out)
        if out.shape != ref.shape:
            raise ValueError(f"{method} output shape {out.shape} != reference {ref.shape}")
        diff = out.astype(np.float64) - ref.astype(np.float64)
        mse = float(np.mean(diff * diff))
        psnr = float("inf") if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
        ssim_val = float(ssim(ref, out, channel_axis=2, data_range=255))
        row = {"method": method, "mse": mse, "psnr": psnr, "ssim": ssim_val, "note": note}
        rows.append(row)
        print(f"  {method:54s} MSE={mse:10.4f} PSNR={psnr:8.3f} SSIM={ssim_val:.6f}")

    if torch.cuda.is_available():
        method = "PyTorch F.interpolate bilinear align_corners=False"
        try:
            with torch.no_grad():
                x_gpu = torch_tensor_from_rgb(image)
                out = torch_output_to_uint8_hwc(torch_resize(x_gpu))
            add_quality(method, out, "PyTorch and OpenCV use slightly different bilinear coordinate/rounding behavior.")
            del x_gpu
            torch.cuda.empty_cache()
        except Exception as exc:
            record_error(errors, method, exc)

    if cp is not None:
        method = "CuPy cupyx.scipy.ndimage.zoom order=1"
        try:
            img_gpu = cp.asarray(image)
            out = cp.asnumpy(cupy_resize_gpu(img_gpu)).astype(np.uint8)
            add_quality(method, out, "SciPy/CuPy zoom order=1 is bilinear-like but not bit-equivalent to OpenCV resize.")
            del img_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as exc:
            record_error(errors, method, exc)

    if pipeline_def is not None:
        method = "NVIDIA DALI resize linear"
        try:
            pipe = maybe_make_dali_resize_pipeline(batch_size=1, batch=[image])
            out = pipe.run()[0].as_cpu().at(0)
            add_quality(method, out, "DALI linear interpolation may differ slightly from OpenCV at pixel centers and borders.")
        except Exception as exc:
            record_error(errors, method, exc)

    return rows


def print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("Summary")
    print("=" * 88)
    env = results["environment"]
    print(f"GPU: {env.get('gpu')}  VRAM: {env.get('vram_gb')} GB  CUDA: {env.get('cuda_version')}")
    print(f"CPU cores visible to Python: {env.get('cpu_cores')}  Python: {env.get('python_version')}")
    print("Note: Colab T4 commonly exposes only 2 CPU cores; CPU results are not expected to match a 10-core M4.")

    def table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
        print("\n" + title)
        if not rows:
            print("  no rows")
            return
        widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
        print("  " + "  ".join(col.ljust(widths[col]) for col in columns))
        print("  " + "  ".join("-" * widths[col] for col in columns))
        for row in rows:
            cells = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    val = f"{val:.3f}"
                cells.append(str(val).ljust(widths[col]))
            print("  " + "  ".join(cells))

    table("RGB resize", results["rgb_resize"], ["method", "fps", "ms_per_frame", "speedup", "includes_transfer"])
    table("MJPEG decode + resize", results["mjpeg_decode_resize"], ["method", "image_type", "fps", "ms_per_frame"])
    table("Batch scaling", results["batch_scaling"], ["batch_size", "fps", "ms_per_frame"])
    table("Quality", results["quality"], ["method", "mse", "psnr", "ssim"])
    if results.get("errors"):
        print("\nErrors / skipped methods")
        for err in results["errors"]:
            print(f"  {err['method']}: {err['error']}")


def main() -> None:
    frames = ARGS.frames if ARGS.frames is not None else (QUICK_FRAMES if ARGS.quick else DEFAULT_FRAMES)
    repeats = ARGS.repeats if ARGS.repeats is not None else (QUICK_REPEATS if ARGS.quick else DEFAULT_REPEATS)
    mjpeg_frames = ARGS.mjpeg_frames if ARGS.mjpeg_frames is not None else (20 if ARGS.quick else MJPEG_FRAMES_PER_TYPE)
    warmup = ARGS.warmup

    print("=" * 88)
    print("T4 CUDA resize benchmark")
    print("=" * 88)
    env = environment_info()
    for key, value in env.items():
        print(f"{key:18s}: {value}")

    if not torch.cuda.is_available():
        print("WARNING: torch.cuda.is_available() is False. GPU benchmarks will be skipped or fail.")

    print(f"\nConfig: frames={frames}, mjpeg_frames={mjpeg_frames}, repeats={repeats}, warmup={warmup}, quick={ARGS.quick}")
    print(f"Workload: {SRC_W}x{SRC_H} RGB -> {DST_W}x{DST_H}, JPEG quality={JPEG_QUALITY}")

    errors: list[dict[str, str]] = []
    image = make_test_image()
    diverse = make_diverse_test_images()

    results = {
        "environment": env,
        "config": {
            "src_w": SRC_W,
            "src_h": SRC_H,
            "dst_w": DST_W,
            "dst_h": DST_H,
            "frames": frames,
            "repeats": repeats,
            "warmup": warmup,
            "jpeg_quality": JPEG_QUALITY,
            "mjpeg_frames_per_type": mjpeg_frames,
        },
        "rgb_resize": bench_rgb_resize(image, frames, repeats, warmup, errors),
        "mjpeg_decode_resize": bench_mjpeg_decode_resize(diverse, mjpeg_frames, repeats, warmup, errors),
        "batch_scaling": bench_batch_scaling(image, frames, repeats, warmup, errors),
        "quality": compute_quality(image, errors),
        "errors": errors,
    }

    with open(ARGS.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print_summary(results)
    print(f"\nWrote {ARGS.output}")


if __name__ == "__main__":
    main()
