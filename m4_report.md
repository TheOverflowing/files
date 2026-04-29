# M4 OpenCV Resize Benchmark Results

## Environment

- CPU: Apple Silicon
- macOS: 26.4.1
- Python: 3.10.9 (arm64)
- OpenCV: 4.13.0
- Power:  -InternalBattery-0 (id=22544483)	80%; AC attached; not charging present: true
- Low Power Mode: False

## Workload

- Source: 3840×1920 RGB / MJPEG
- Destination: 1920×1080
- Frames per repeat: 80
- Repeats: 5
- Warmup frames: 8
- JPEG quality: 85

## Best Results

- RGB winner: OpenCV INTER_NEAREST (10 threads) — 1835.9 fps, 7.16× vs OpenCV 1-thread linear.
- MJPEG winner: OpenCV REDUCED_8 decode only, ThreadPool(16) — 1007.9 fps, 15.92× vs full decode + resize.
- Best exact-output MJPEG winner: OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(12) — 738.4 fps.
- Best approximate-output MJPEG winner: OpenCV REDUCED_8 decode only, ThreadPool(16) — 1007.9 fps.

## MJPEG Category Summary

- Baseline correctness path: OpenCV imdecode + resize LINEAR (full 4K decode) — 63.3 fps, 15.80 ms/frame, 1.00×.
- Balanced speed/quality path: OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(16) — 477.2 fps, 7.54×.
- Maximum-speed exact-output path: OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(12) — 738.4 fps, 11.67×.
- Maximum-speed approximate-output path: OpenCV REDUCED_8 decode only, ThreadPool(16) — 1007.9 fps, 15.92×.

## RGB Results

| Rank | Configuration | FPS | ms/frame | Speedup | Notes |
|---:|---|---:|---:|---:|---|
| 1 | OpenCV INTER_NEAREST (10 threads) | 1835.9 | 0.54 | 7.16× | fastest interp, lowest quality |
| 2 | OpenCV INTER_LINEAR preallocated dst (10 threads) | 1163.0 | 0.86 | 4.54× | removes per-frame output allocation |
| 3 | OpenCV INTER_LINEAR (4 threads) | 1133.1 | 0.88 | 4.42× | setNumThreads(4) |
| 4 | OpenCV INTER_LINEAR (10 threads) | 1130.0 | 0.88 | 4.41× | setNumThreads(10) |
| 5 | OpenCV INTER_LINEAR (2 threads) | 1127.5 | 0.89 | 4.40× | setNumThreads(2) |
| 6 | OpenCV INTER_LINEAR (6 threads) | 1125.2 | 0.89 | 4.39× | setNumThreads(6) |
| 7 | OpenCV INTER_LINEAR (8 threads) | 1109.9 | 0.90 | 4.33× | setNumThreads(8) |
| 8 | OpenCV LINEAR + ThreadPool(4 P-cores), 1 thread per call | 555.3 | 1.80 | 2.17× | frame-level parallelism |
| 9 | OpenCV INTER_CUBIC (10 threads) | 399.8 | 2.50 | 1.56× | higher quality, slower |
| 10 | OpenCV INTER_LINEAR (1 threads) | 256.3 | 3.90 | 1.00× | setNumThreads(1) |
| 11 | OpenCV INTER_AREA (10 threads) | 206.8 | 4.84 | 0.81× | best downsample quality |
| 12 | OpenCV INTER_LANCZOS4 (10 threads) | 183.7 | 5.44 | 0.72× | highest quality, slowest |

## MJPEG Results

| Rank | Configuration | FPS | ms/frame | Speedup | Notes |
|---:|---|---:|---:|---:|---|
| 1 | OpenCV REDUCED_8 decode only, ThreadPool(16) | 1007.9 | 0.99 | 15.92× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 2 | OpenCV REDUCED_8 decode only, ThreadPool(14) | 997.2 | 1.00 | 15.76× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 3 | OpenCV REDUCED_8 decode only, ThreadPool(10) | 963.9 | 1.04 | 15.23× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 4 | OpenCV REDUCED_8 decode only, ThreadPool(12) | 963.4 | 1.04 | 15.22× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 5 | OpenCV REDUCED_8 decode only, ThreadPool(8) | 858.4 | 1.17 | 13.56× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 6 | OpenCV REDUCED_4 decode only, ThreadPool(10) | 837.2 | 1.19 | 13.23× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 7 | OpenCV REDUCED_4 decode only, ThreadPool(12) | 836.5 | 1.20 | 13.22× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 8 | OpenCV REDUCED_4 decode only, ThreadPool(14) | 819.5 | 1.22 | 12.95× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 9 | OpenCV REDUCED_4 decode only, ThreadPool(16) | 816.0 | 1.23 | 12.89× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 10 | OpenCV REDUCED_4 decode only, ThreadPool(8) | 750.2 | 1.33 | 11.85× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 11 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(12) | 738.4 | 1.35 | 11.67× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 12 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(14) | 731.9 | 1.37 | 11.56× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 13 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(12) | 723.0 | 1.38 | 11.42× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 14 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(10) | 716.6 | 1.40 | 11.32× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 15 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(16) | 712.6 | 1.40 | 11.26× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 16 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(14) | 712.0 | 1.40 | 11.25× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 17 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(10) | 710.8 | 1.41 | 11.23× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 18 | OpenCV REDUCED_2 decode only, ThreadPool(14) | 671.6 | 1.49 | 10.61× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 19 | OpenCV REDUCED_8 decode only, ThreadPool(6) | 670.9 | 1.49 | 10.60× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 20 | OpenCV REDUCED_2 decode only, ThreadPool(10) | 655.5 | 1.53 | 10.36× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 21 | OpenCV REDUCED_2 decode only, ThreadPool(12) | 654.9 | 1.53 | 10.35× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 22 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(8) | 646.9 | 1.55 | 10.22× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 23 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(14) | 645.7 | 1.55 | 10.20× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 24 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(8) | 643.9 | 1.55 | 10.17× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 25 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(16) | 642.9 | 1.56 | 10.16× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 26 | OpenCV REDUCED_4 decode only, ThreadPool(6) | 626.8 | 1.60 | 9.90× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 27 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(10) | 620.9 | 1.61 | 9.81× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 28 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(12) | 620.0 | 1.61 | 9.80× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 29 | OpenCV REDUCED_2 decode only, ThreadPool(8) | 619.7 | 1.61 | 9.79× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 30 | OpenCV REDUCED_2 decode only, ThreadPool(16) | 597.1 | 1.67 | 9.43× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 31 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(16) | 583.5 | 1.71 | 9.22× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 32 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(10) | 582.1 | 1.72 | 9.20× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 33 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(14) | 579.9 | 1.72 | 9.16× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 34 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(16) | 576.0 | 1.74 | 9.10× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 35 | OpenCV REDUCED_8 decode only, ThreadPool(4) | 571.4 | 1.75 | 9.03× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 36 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(8) | 565.8 | 1.77 | 8.94× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 37 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(12) | 552.9 | 1.81 | 8.74× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 38 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(6) | 531.8 | 1.88 | 8.40× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 39 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(8) | 530.1 | 1.89 | 8.38× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 40 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(6) | 519.3 | 1.93 | 8.21× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 41 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(12) | 518.0 | 1.93 | 8.18× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 42 | OpenCV REDUCED_2 decode only, ThreadPool(6) | 515.9 | 1.94 | 8.15× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 43 | OpenCV REDUCED_4 decode only, ThreadPool(4) | 514.0 | 1.95 | 8.12× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 44 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(14) | 513.1 | 1.95 | 8.11× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 45 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(10) | 504.4 | 1.98 | 7.97× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 46 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(16) | 499.4 | 2.00 | 7.89× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 47 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(16) | 477.2 | 2.10 | 7.54× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 48 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(6) | 476.5 | 2.10 | 7.53× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 49 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(10) | 466.8 | 2.14 | 7.37× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 50 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(6) | 462.5 | 2.16 | 7.31× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 51 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(12) | 459.5 | 2.18 | 7.26× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 52 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(8) | 457.4 | 2.19 | 7.23× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 53 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(14) | 443.8 | 2.25 | 7.01× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 54 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(4) | 432.3 | 2.31 | 6.83× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 55 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(8) | 427.1 | 2.34 | 6.75× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 56 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(4) | 426.7 | 2.34 | 6.74× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 57 | OpenCV REDUCED_2 decode only, ThreadPool(4) | 421.8 | 2.37 | 6.66× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 58 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(6) | 403.0 | 2.48 | 6.37× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 59 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(4) | 392.1 | 2.55 | 6.20× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 60 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(4) | 369.0 | 2.71 | 5.83× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 61 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(6) | 367.9 | 2.72 | 5.81× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 62 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(4) | 337.0 | 2.97 | 5.32× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 63 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(4) | 301.9 | 3.31 | 4.77× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 64 | OpenCV REDUCED_8 decode only, ThreadPool(2) | 288.5 | 3.47 | 4.56× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 65 | OpenCV REDUCED_4 decode only, ThreadPool(2) | 261.5 | 3.82 | 4.13× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 66 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(2) | 240.2 | 4.16 | 3.79× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 67 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(2) | 227.7 | 4.39 | 3.60× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 68 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(2) | 221.2 | 4.52 | 3.50× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 69 | OpenCV REDUCED_2 decode only, ThreadPool(2) | 218.7 | 4.57 | 3.46× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 70 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(2) | 206.3 | 4.85 | 3.26× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 71 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(2) | 186.1 | 5.37 | 2.94× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 72 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(2) | 164.4 | 6.08 | 2.60× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 73 | OpenCV REDUCED_8 decode only, ThreadPool(1) | 144.5 | 6.92 | 2.28× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 74 | OpenCV REDUCED_4 decode only, ThreadPool(1) | 131.4 | 7.61 | 2.08× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 75 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(1) | 123.2 | 8.12 | 1.95× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 76 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(1) | 120.6 | 8.29 | 1.91× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 77 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(1) | 109.9 | 9.10 | 1.74× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 78 | OpenCV REDUCED_2 decode only, ThreadPool(1) | 109.8 | 9.11 | 1.74× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 79 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(1) | 103.7 | 9.64 | 1.64× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 80 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(1) | 95.8 | 10.44 | 1.51× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 81 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(1) | 86.6 | 11.55 | 1.37× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 82 | OpenCV imdecode + resize LINEAR preallocated dst | 63.8 | 15.67 | 1.01× | full decode baseline without resize output allocation |
| 83 | OpenCV imdecode + resize LINEAR (full 4K decode) | 63.3 | 15.80 | 1.00× | baseline: decode at full 4K, then resize |

## MJPEG Quality Comparison

| Variant | Samples | MSE | PSNR | SSIM |
|---|---:|---:|---:|---:|
| REDUCED_2 + INTER_LINEAR | 5 | 1.09 | 47.76 dB | 0.9836 |
| REDUCED_2 + INTER_NEAREST | 5 | 2.66 | 43.87 dB | 0.9609 |
| REDUCED_4 + INTER_LINEAR | 5 | 3.14 | 43.16 dB | 0.9514 |
| REDUCED_4 + INTER_NEAREST | 5 | 3.30 | 42.95 dB | 0.9501 |
| REDUCED_8 + INTER_LINEAR | 5 | 3.78 | 42.36 dB | 0.9423 |
| REDUCED_8 + INTER_NEAREST | 5 | 4.08 | 42.02 dB | 0.9389 |

## Recommended Submission Claim

- For decoded RGB, report the fastest quality-acceptable OpenCV path from the RGB table.
- For MJPEG, report the balanced `REDUCED_2 + INTER_LINEAR` path when quality matters, and the `REDUCED_4 + INTER_NEAREST` path when speed is the only judging metric.
- Use the full command `python3 M4benchmark.py --skip-gpu` for stable CPU/MJPEG numbers, plugged into power.
