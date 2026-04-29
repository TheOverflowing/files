"""
Standalone MJPEG decode + resize benchmark.

Workload:
  Synthetic 3840x1920 images -> JPEG buffers -> decode and resize to 1920x1080.

This intentionally benchmarks in-memory MJPEG-style frames, not VideoCapture
from an AVI/MJPG file. It is meant to isolate JPEG decode, resize, threading,
and optional GPU/DALI pipeline costs.
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


def has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def pip_install(packages: list[str], optional: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    try:
        print("Installing:", " ".join(packages))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:
        if optional:
            print(f"WARNING: optional install failed for {' '.join(packages)}: {exc}")
            return False
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone MJPEG decode + resize benchmark.")
    parser.add_argument("--width", type=int, default=3840, help="Source width. Default: %(default)s")
    parser.add_argument("--height", type=int, default=1920, help="Source height. Default: %(default)s")
    parser.add_argument("--dst-width", type=int, default=1920, help="Output width. Default: %(default)s")
    parser.add_argument("--dst-height", type=int, default=1080, help="Output height. Default: %(default)s")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality. Default: %(default)s")
    parser.add_argument("--frames", type=int, default=80, help="Measured JPEG frames per image type. Default: %(default)s")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats. Default: %(default)s")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup frames per method. Default: %(default)s")
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: 20 frames, 1 repeat, 3 warmup frames.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install missing optional dependencies.")
    parser.add_argument("--skip-pillow", action="store_true", help="Skip Pillow benchmarks.")
    parser.add_argument("--skip-pyvips", action="store_true", help="Skip pyvips benchmarks.")
    parser.add_argument("--skip-dali", action="store_true", help="Skip NVIDIA DALI/nvJPEG benchmarks.")
    parser.add_argument(
        "--dali-batches",
        default="8,16,32,64",
        help="Comma-separated DALI batch sizes to sweep. Default: %(default)s",
    )
    parser.add_argument(
        "--dali-prefetch",
        default="2,3,4",
        help="Comma-separated DALI prefetch_queue_depth values to sweep. Default: %(default)s",
    )
    parser.add_argument("--output", default="mjpeg_results.json", help="JSON output path. Default: %(default)s")
    return parser.parse_known_args()[0]


ARGS = parse_args()
if ARGS.quick:
    ARGS.frames = 20
    ARGS.repeats = 1
    ARGS.warmup = 3
    ARGS.dali_batches = "8,16"
    ARGS.dali_prefetch = "2"


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        items = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers, got {value!r}") from exc
    if not items or any(item <= 0 for item in items):
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return items


DALI_BATCHES = parse_int_list(ARGS.dali_batches, "--dali-batches")
DALI_PREFETCH = parse_int_list(ARGS.dali_prefetch, "--dali-prefetch")

if not ARGS.skip_install:
    required = []
    if not has_module("cv2"):
        required.append("opencv-python-headless")
    if not has_module("numpy"):
        required.append("numpy")
    if required:
        pip_install(required)
    if not ARGS.skip_pillow and not has_module("PIL"):
        pip_install(["Pillow"], optional=True)
    if not ARGS.skip_dali and not has_module("nvidia.dali"):
        pip_install(["nvidia-dali-cuda120"], optional=True)

import cv2
import numpy as np

try:
    if ARGS.skip_pillow:
        raise ImportError("--skip-pillow was provided")
    from PIL import Image
except Exception as exc:
    Image = None
    print(f"WARNING: Pillow unavailable; skipping Pillow benchmarks: {exc}")

try:
    if ARGS.skip_pyvips:
        raise ImportError("--skip-pyvips was provided")
    import pyvips
except Exception as exc:
    pyvips = None
    print(f"WARNING: pyvips unavailable; skipping pyvips benchmarks: {exc}")

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
    print(f"WARNING: DALI unavailable; skipping DALI/nvJPEG benchmarks: {exc}")


SRC_W, SRC_H = ARGS.width, ARGS.height
DST_W, DST_H = ARGS.dst_width, ARGS.dst_height


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_cores": os.cpu_count(),
        "opencv": cv2.__version__,
        "pillow_available": Image is not None,
        "pyvips_available": pyvips is not None,
        "dali_available": pipeline_def is not None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception:
        info["torch"] = None
        info["cuda_available"] = False
    return info


def make_diverse_images() -> list[dict[str, Any]]:
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:SRC_H, 0:SRC_W].astype(np.float32)
    images: list[tuple[str, np.ndarray]] = []

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

    params = [int(cv2.IMWRITE_JPEG_QUALITY), ARGS.quality]
    encoded = []
    for key, image in images:
        ok, buf = cv2.imencode(".jpg", image, params)
        if not ok:
            raise RuntimeError(f"JPEG encode failed for {key}")
        encoded.append({"image_type": key, "jpeg": buf.tobytes(), "jpeg_bytes": int(len(buf))})
    return encoded


def mean_fps(item_count: int, times: list[float]) -> tuple[float, float]:
    fps_values = [item_count / elapsed for elapsed in times]
    fps = statistics.mean(fps_values)
    return fps, 1000.0 / fps


def benchmark_dataset(
    name: str,
    items: list[bytes],
    process_batch: Callable[[list[bytes]], None],
    repeats: int,
    warmup: int,
) -> tuple[float, float, list[float]]:
    if not items:
        raise ValueError("benchmark_dataset requires at least one item")
    warm = [items[i % len(items)] for i in range(max(1, warmup))]
    process_batch(warm)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        process_batch(items)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    fps, ms = mean_fps(len(items), times)
    print(f"  {name:62s} {fps:9.2f} fps  {ms:8.3f} ms/frame")
    return fps, ms, times


def record_error(errors: list[dict[str, str]], method: str, exc: BaseException) -> None:
    print(f"ERROR: {method} failed: {exc}")
    errors.append({"method": method, "error": str(exc), "traceback": traceback.format_exc(limit=4)})


def cv_decode(buf: bytes, flag: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, flag)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return img


def make_dali_jpeg_pipeline(batch_size: int, num_threads: int, prefetch_depth: int, batch: list[np.ndarray]):
    if pipeline_def is None:
        return None

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=0,
        exec_async=True,
        exec_pipelined=True,
        prefetch_queue_depth=prefetch_depth,
    )
    def jpeg_pipe():
        encoded = dali_fn.external_source(source=lambda: batch, device="cpu", batch=True)
        decoded = dali_fn.decoders.image(encoded, device="mixed", output_type=dali_types.RGB)
        return dali_fn.resize(decoded, resize_x=DST_W, resize_y=DST_H, interp_type=dali_types.INTERP_LINEAR)

    pipe = jpeg_pipe()
    pipe.build()
    return pipe


def bench_one_image_type(item: dict[str, Any], errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_type = item["image_type"]
    jpeg = item["jpeg"]
    buffers = [jpeg] * ARGS.frames
    print(f"\n-- {image_type} ({item['jpeg_bytes'] / 1024:.1f} KiB JPEG) --")

    cases: list[tuple[str, Callable[[list[bytes]], None], dict[str, Any]]] = []

    def add_case(name: str, fn: Callable[[list[bytes]], None], **meta: Any) -> None:
        cases.append((name, fn, meta))

    def cv_decode_only_full(frames: list[bytes]) -> None:
        for buf in frames:
            cv_decode(buf, cv2.IMREAD_COLOR)

    add_case("OpenCV imdecode full only", cv_decode_only_full, phase="decode", output_size=f"{SRC_W}x{SRC_H}")

    def cv_decode_full_resize(frames: list[bytes]) -> None:
        for buf in frames:
            img = cv_decode(buf, cv2.IMREAD_COLOR)
            cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)

    add_case("OpenCV imdecode full + resize", cv_decode_full_resize, phase="decode_resize", exact_output_size=True)

    reduced_variants = [
        ("REDUCED_COLOR_2", cv2.IMREAD_REDUCED_COLOR_2, 2),
        ("REDUCED_COLOR_4", cv2.IMREAD_REDUCED_COLOR_4, 4),
        ("REDUCED_COLOR_8", cv2.IMREAD_REDUCED_COLOR_8, 8),
    ]
    interp_variants = [
        ("INTER_LINEAR", cv2.INTER_LINEAR),
        ("INTER_NEAREST", cv2.INTER_NEAREST),
    ]

    def make_decode_only(flag: int) -> Callable[[list[bytes]], None]:
        def process(frames: list[bytes]) -> None:
            for buf in frames:
                cv_decode(buf, flag)

        return process

    def make_decode_resize(flag: int, interpolation: int) -> Callable[[list[bytes]], None]:
        def process(frames: list[bytes]) -> None:
            for buf in frames:
                img = cv_decode(buf, flag)
                cv2.resize(img, (DST_W, DST_H), interpolation=interpolation)

        return process

    def make_threaded_decode_only(flag: int, worker_count: int) -> Callable[[list[bytes]], None]:
        def process(frames: list[bytes]) -> None:
            def one(buf: bytes) -> None:
                cv_decode(buf, flag)

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(one, frames))

        return process

    def make_threaded_decode_resize(flag: int, interpolation: int, worker_count: int) -> Callable[[list[bytes]], None]:
        def process(frames: list[bytes]) -> None:
            def one(buf: bytes) -> None:
                img = cv_decode(buf, flag)
                cv2.resize(img, (DST_W, DST_H), interpolation=interpolation)

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(one, frames))

        return process

    for variant_name, flag, shrink in reduced_variants:
        add_case(
            f"OpenCV imdecode {variant_name} only",
            make_decode_only(flag),
            phase="decode",
            exact_output_size=False,
            output_size=f"{SRC_W // shrink}x{SRC_H // shrink}",
            dct_shrink=shrink,
        )

        for interp_name, interpolation in interp_variants:
            add_case(
                f"OpenCV imdecode {variant_name} + {interp_name}",
                make_decode_resize(flag, interpolation),
                phase="decode_resize",
                exact_output_size=True,
                dct_shrink=shrink,
                interpolation=interp_name,
            )

        for workers in [2, 4, 8, 12, 16]:
            add_case(
                f"OpenCV {variant_name} decode only ThreadPool({workers})",
                make_threaded_decode_only(flag, workers),
                phase="decode",
                exact_output_size=False,
                output_size=f"{SRC_W // shrink}x{SRC_H // shrink}",
                dct_shrink=shrink,
                workers=workers,
            )

            for interp_name, interpolation in interp_variants:
                add_case(
                    f"OpenCV {variant_name} + {interp_name} ThreadPool({workers})",
                    make_threaded_decode_resize(flag, interpolation, workers),
                    phase="decode_resize",
                    exact_output_size=True,
                    dct_shrink=shrink,
                    interpolation=interp_name,
                    workers=workers,
                )

    if Image is not None:
        import io

        def pillow_full_resize(frames: list[bytes]) -> None:
            for buf in frames:
                im = Image.open(io.BytesIO(buf))
                im = im.resize((DST_W, DST_H), Image.BILINEAR)
                im.load()

        add_case("Pillow full decode + resize", pillow_full_resize, phase="decode_resize", exact_output_size=True)

        def pillow_draft_resize(frames: list[bytes]) -> None:
            for buf in frames:
                im = Image.open(io.BytesIO(buf))
                im.draft("RGB", (DST_W, DST_H))
                im = im.resize((DST_W, DST_H), Image.BILINEAR)
                im.load()

        add_case("Pillow draft + resize", pillow_draft_resize, phase="decode_resize", exact_output_size=True)

    if pyvips is not None:
        def vips_shrink_resize(frames: list[bytes]) -> None:
            for buf in frames:
                im = pyvips.Image.jpegload_buffer(buf, shrink=2)
                out = im.resize(DST_W / im.width, kernel="linear")
                out.write_to_memory()

        add_case("pyvips jpegload(shrink=2) + resize", vips_shrink_resize, phase="decode_resize", exact_output_size=True)

    for name, fn, meta in cases:
        try:
            cv2.setNumThreads(1)
            fps, ms, times = benchmark_dataset(name, buffers, fn, ARGS.repeats, ARGS.warmup)
            row = {
                "image_type": image_type,
                "method": name,
                "fps": fps,
                "ms_per_frame": ms,
                "times_sec": times,
                "jpeg_bytes": item["jpeg_bytes"],
            }
            row.update(meta)
            rows.append(row)
        except Exception as exc:
            record_error(errors, f"{name} {image_type}", exc)

    if pipeline_def is not None:
        for batch_size in DALI_BATCHES:
            for prefetch_depth in DALI_PREFETCH:
                for threads in [2, 4]:
                    name = (
                        "NVIDIA DALI nvJPEG mixed decode + GPU resize "
                        f"batch={batch_size} prefetch={prefetch_depth} threads={threads}"
                    )
                    try:
                        encoded = [np.frombuffer(jpeg, dtype=np.uint8) for _ in range(batch_size)]
                        pipe = make_dali_jpeg_pipeline(
                            batch_size=batch_size,
                            num_threads=threads,
                            prefetch_depth=prefetch_depth,
                            batch=encoded,
                        )
                        batches_per_repeat = max(1, math.ceil(ARGS.frames / batch_size))

                        def run_batches(_: list[bytes]) -> None:
                            for _ in range(batches_per_repeat):
                                out = pipe.run()[0]
                                out.as_cpu()

                        measured_items = [jpeg] * (batches_per_repeat * batch_size)
                        fps, ms, times = benchmark_dataset(name, measured_items, run_batches, ARGS.repeats, 1)
                        rows.append(
                            {
                                "image_type": image_type,
                                "method": name,
                                "fps": fps,
                                "ms_per_frame": ms,
                                "times_sec": times,
                                "jpeg_bytes": item["jpeg_bytes"],
                                "phase": "decode_resize",
                                "exact_output_size": True,
                                "batch_size": batch_size,
                                "dali_threads": threads,
                                "prefetch_queue_depth": prefetch_depth,
                            }
                        )
                    except Exception as exc:
                        record_error(errors, f"{name} {image_type}", exc)

    return rows


def add_speedups(rows: list[dict[str, Any]]) -> None:
    baselines: dict[str, float] = {}
    for row in rows:
        if row["method"] == "OpenCV imdecode full + resize":
            baselines[row["image_type"]] = row["fps"]
    for row in rows:
        baseline = baselines.get(row["image_type"])
        if baseline:
            row["speedup_vs_full_cv"] = row["fps"] / baseline


def print_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print("Summary")
    print("=" * 96)
    env = results["environment"]
    print(f"Platform: {env.get('platform')}")
    print(f"CPU cores: {env.get('cpu_cores')}  GPU: {env.get('gpu')}  DALI: {env.get('dali_available')}")
    print(f"Workload: {SRC_W}x{SRC_H} JPEG quality={ARGS.quality} -> {DST_W}x{DST_H}")
    if env.get("dali_available"):
        print(f"DALI sweep: batch={DALI_BATCHES}, prefetch={DALI_PREFETCH}, threads=[2, 4]")

    rows = results["rows"]
    for image_type in sorted({row["image_type"] for row in rows}):
        print(f"\n{image_type}")
        subset = [row for row in rows if row["image_type"] == image_type]
        subset.sort(key=lambda row: row["fps"], reverse=True)
        print("  method                                                         fps       ms/frame  speedup")
        print("  -------------------------------------------------------------  --------  --------  -------")
        for row in subset:
            speedup = row.get("speedup_vs_full_cv")
            speedup_text = f"{speedup:.2f}x" if speedup is not None else ""
            print(f"  {row['method'][:61].ljust(61)}  {row['fps']:8.2f}  {row['ms_per_frame']:8.3f}  {speedup_text:>7s}")

    if results["errors"]:
        print("\nErrors / skipped methods")
        for err in results["errors"]:
            print(f"  {err['method']}: {err['error']}")


def main() -> None:
    print("=" * 96)
    print("Standalone MJPEG benchmark")
    print("=" * 96)
    env = environment_info()
    for key, value in env.items():
        print(f"{key:20s}: {value}")
    print(f"\nConfig: frames={ARGS.frames}, repeats={ARGS.repeats}, warmup={ARGS.warmup}, quick={ARGS.quick}")
    print(f"Workload: {SRC_W}x{SRC_H} JPEG quality={ARGS.quality} -> {DST_W}x{DST_H}")
    if pipeline_def is not None:
        print(f"DALI sweep: batch={DALI_BATCHES}, prefetch={DALI_PREFETCH}, threads=[2, 4]")

    errors: list[dict[str, str]] = []
    diverse = make_diverse_images()
    rows: list[dict[str, Any]] = []
    for item in diverse:
        rows.extend(bench_one_image_type(item, errors))
    add_speedups(rows)

    results = {
        "environment": env,
        "config": {
            "src_w": SRC_W,
            "src_h": SRC_H,
            "dst_w": DST_W,
            "dst_h": DST_H,
            "jpeg_quality": ARGS.quality,
            "frames": ARGS.frames,
            "repeats": ARGS.repeats,
            "warmup": ARGS.warmup,
            "dali_batches": DALI_BATCHES,
            "dali_prefetch": DALI_PREFETCH,
            "dali_threads": [2, 4],
        },
        "rows": rows,
        "errors": errors,
    }
    with open(ARGS.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print_summary(results)
    print(f"\nWrote {ARGS.output}")


if __name__ == "__main__":
    main()
