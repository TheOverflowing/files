"""
OpenCV resize speedup benchmark.

Workload: resize 4K (3840x1920) frames to 1080p (1920x1080).
Measures throughput (fps) across multiple resize backends and configurations.
Inputs covered: raw RGB pixel buffers AND MJPEG-encoded frames.
"""
import os
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import cv2
from PIL import Image
import pyvips

# ----- Config -----
SRC_W, SRC_H = 3840, 1920
DST_W, DST_H = 1920, 1080
N_FRAMES = 20          # frames per measurement
N_REPEATS = 3          # how many times to repeat for variance
WARMUP_FRAMES = 3

RESULTS = []

def make_test_frames(n=N_FRAMES, seed=0):
    """Generate n synthetic 4K BGR frames (uint8)."""
    rng = np.random.default_rng(seed)
    # natural-ish image: low-frequency gradient + noise, in 3 channels
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

def make_mjpeg_frames(rgb_frames, quality=85):
    """Encode each frame as MJPEG (single-frame JPEG) byte buffer."""
    encoded = []
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for f in rgb_frames:
        ok, buf = cv2.imencode('.jpg', f, enc_param)
        assert ok
        encoded.append(buf.tobytes())
    return encoded

def time_it(fn, frames, n_repeats=N_REPEATS):
    """Run fn(frames) n_repeats times. Return list of fps values."""
    # Warmup
    fn(frames[:WARMUP_FRAMES])
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
    print(f"  {name:55s}  {row['fps_mean']:7.1f} fps  ({row['ms_per_frame']:5.2f} ms/frame, ±{row['fps_stdev']:4.1f})")
    return row

# ============================================================
# RGB benchmarks: input is already a decoded numpy array
# ============================================================
def bench_rgb(rgb_frames):
    print("\n=== RGB INPUT (decoded numpy arrays) ===")

    # ----- OpenCV variants -----
    def cv_linear(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
    cv2.setNumThreads(1)
    record("OpenCV INTER_LINEAR (1 thread)", "opencv",
           time_it(cv_linear, rgb_frames),
           "single-threaded baseline")

    cv2.setNumThreads(2)
    record("OpenCV INTER_LINEAR (2 threads)", "opencv",
           time_it(cv_linear, rgb_frames),
           "OpenCV pthreads, all cores")

    def cv_nearest(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_NEAREST)
    cv2.setNumThreads(2)
    record("OpenCV INTER_NEAREST (2 threads)", "opencv",
           time_it(cv_nearest, rgb_frames),
           "fastest interp, lowest quality")

    def cv_area(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_AREA)
    cv2.setNumThreads(2)
    record("OpenCV INTER_AREA (2 threads)", "opencv",
           time_it(cv_area, rgb_frames),
           "best for downsample quality")

    def cv_cubic(frames):
        for f in frames:
            cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_CUBIC)
    cv2.setNumThreads(2)
    record("OpenCV INTER_CUBIC (2 threads)", "opencv",
           time_it(cv_cubic, rgb_frames),
           "higher quality, slower")

    # ----- OpenCV with thread pool over frames -----
    def cv_linear_threadpool(frames):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(
                lambda f: cv2.resize(f, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR),
                frames))
    cv2.setNumThreads(1)  # avoid nested parallelism
    record("OpenCV LINEAR + ThreadPool(2) over frames", "opencv",
           time_it(cv_linear_threadpool, rgb_frames),
           "frame-level parallelism, OpenCV single-thread per call")

    # ----- Pillow -----
    pil_frames = [Image.fromarray(f) for f in rgb_frames]
    def pil_bilinear(frames):
        for im in frames:
            im.resize((DST_W, DST_H), Image.BILINEAR)
    record("Pillow BILINEAR (1 thread)", "pillow",
           time_it(pil_bilinear, pil_frames),
           "PIL converts internally; uses libjpeg-turbo if available")

    def pil_nearest(frames):
        for im in frames:
            im.resize((DST_W, DST_H), Image.NEAREST)
    record("Pillow NEAREST (1 thread)", "pillow", time_it(pil_nearest, pil_frames))

    # ----- pyvips -----
    # vips works on its own image objects; build them from numpy
    def make_vips(rgb):
        return pyvips.Image.new_from_memory(rgb.tobytes(), SRC_W, SRC_H, 3, 'uchar')
    vips_frames = [make_vips(f) for f in rgb_frames]
    scale = DST_W / SRC_W
    def vips_resize(frames):
        for im in frames:
            out = im.resize(scale, kernel='linear')
            _ = out.write_to_memory()  # force computation
    record("pyvips resize linear", "pyvips",
           time_it(vips_resize, vips_frames),
           "lazy pipeline, write_to_memory() forces eval")

    # ----- NumPy strided downsample (nearest) — "optimization" via skipping work -----
    def numpy_stride(frames):
        # When src is exactly 2x dst in both dims, this is equivalent to nearest decimation
        sy = SRC_H // DST_H
        sx = SRC_W // DST_W
        for f in frames:
            _ = f[::sy, ::sx, :].copy()
    record("NumPy strided nearest (2:1 decim)", "numpy",
           time_it(numpy_stride, rgb_frames),
           "valid only when ratio is integer; lowest quality")

# ============================================================
# MJPEG benchmarks: input is a JPEG byte buffer.
# Realistic camera/streaming workload: must decode + resize.
# ============================================================
def bench_mjpeg(mjpeg_frames, rgb_frames):
    print("\n=== MJPEG INPUT (JPEG-encoded buffers, decode+resize) ===")

    # ----- Naive: full decode then full resize -----
    def cv_decode_then_resize(frames):
        for buf in frames:
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
    cv2.setNumThreads(2)
    record("OpenCV imdecode + resize LINEAR", "mjpeg-opencv",
           time_it(cv_decode_then_resize, mjpeg_frames),
           "baseline: decode at full 4K, then resize")

    def cv_decode_reduced(frames):
        # IMREAD_REDUCED_COLOR_2 decodes at 1/2 resolution directly from the
        # JPEG DCT coefficients — no full 4K intermediate buffer.
        for buf in frames:
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_COLOR_2)
            # img is now 1920x960 — resize (or pad) to 1920x1080
            cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
    record("OpenCV imdecode REDUCED_2 + resize", "mjpeg-opencv",
           time_it(cv_decode_reduced, mjpeg_frames),
           "fuses scale-during-decode (DCT-domain 1/2)")

    # ----- Pillow: draft mode does the same (DCT-level downscale at decode) -----
    import io
    def pil_draft_resize(frames):
        for buf in frames:
            im = Image.open(io.BytesIO(buf))
            im.draft('RGB', (DST_W, DST_H))   # snap-decode to nearest power of 2
            im = im.resize((DST_W, DST_H), Image.BILINEAR)
            im.load()
    record("Pillow draft + resize BILINEAR", "mjpeg-pillow",
           time_it(pil_draft_resize, mjpeg_frames),
           "draft() uses libjpeg DCT scaling, big saving")

    def pil_full_decode_resize(frames):
        for buf in frames:
            im = Image.open(io.BytesIO(buf))
            im = im.resize((DST_W, DST_H), Image.BILINEAR)
            im.load()
    record("Pillow full-decode + resize BILINEAR", "mjpeg-pillow",
           time_it(pil_full_decode_resize, mjpeg_frames),
           "no draft mode")

    # ----- pyvips -----
    def vips_jpeg_shrink(frames):
        for buf in frames:
            im = pyvips.Image.jpegload_buffer(buf, shrink=2)
            out = im.resize(DST_W / im.width, kernel='linear')
            _ = out.write_to_memory()
    record("pyvips jpegload(shrink=2) + resize", "mjpeg-pyvips",
           time_it(vips_jpeg_shrink, mjpeg_frames),
           "libjpeg-turbo block-shrink inside vips")

    # ----- ThreadPool over frames (decode is GIL-releasing in OpenCV) -----
    def cv_threadpool_decode_resize(frames):
        def one(buf):
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_REDUCED_COLOR_2)
            return cv2.resize(img, (DST_W, DST_H), interpolation=cv2.INTER_LINEAR)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one, frames))
    cv2.setNumThreads(1)
    record("OpenCV REDUCED_2 + resize, ThreadPool(2)", "mjpeg-opencv",
           time_it(cv_threadpool_decode_resize, mjpeg_frames),
           "frame-level parallelism on top of fused decode")

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print(f"OpenCV {cv2.__version__}, {cv2.getNumberOfCPUs()} CPUs visible to OpenCV")
    print(f"Source: {SRC_W}x{SRC_H} -> {DST_W}x{DST_H}")
    print(f"Pixels per frame: {SRC_W*SRC_H/1e6:.2f}M -> {DST_W*DST_H/1e6:.2f}M")
    print(f"N_FRAMES={N_FRAMES}, N_REPEATS={N_REPEATS}")

    rgb_frames = make_test_frames()
    mjpeg_frames = make_mjpeg_frames(rgb_frames)
    avg_jpeg_kb = sum(len(b) for b in mjpeg_frames) / len(mjpeg_frames) / 1024
    print(f"Average MJPEG buffer: {avg_jpeg_kb:.1f} KB/frame")

    bench_rgb(rgb_frames)
    bench_mjpeg(mjpeg_frames, rgb_frames)

    # Save
    with open('/home/claude/results.json', 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nSaved {len(RESULTS)} rows to results.json")
