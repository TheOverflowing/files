# OpenCV 图像缩放加速实验报告：Apple Silicon M4

## 1. 实验目标

本实验针对 `3840×1920 @ 60–200 fps` RGB / MJPEG 图像缩放场景，在 Apple Silicon M4 MacBook Air 上测试 OpenCV 缩放与 MJPEG 解码优化路径。目标指标是吞吐量 fps、单帧耗时 ms/frame，以及相对基准路径的 speedup。

本轮新增三项改进：

1. 在 resize 矩阵中加入 `IMREAD_REDUCED_COLOR_8 + INTER_LINEAR` 与 `IMREAD_REDUCED_COLOR_8 + INTER_NEAREST`。
2. 增加 reduced-decode-only 路径：`REDUCED_2/4/8` 只解码、不做最终 `cv2.resize`，并报告原生输出尺寸。
3. 增加可选 long-run 模式：`--long-run` 会提高 frames、repeats 和 warmup，用于更稳定的最终测量。

## 2. 结论摘要

当前 `opencv-python` 4.13.0 没有 Metal / MPS 后端，因此本报告不声称“OpenCV with Metal Support”。本次可复现的最高收益来自 **JPEG DCT 层面缩减解码 + 帧级并行**。

| 类别 | 推荐方案 | 输出尺寸 | FPS | ms/frame | 加速比 |
|---|---|---:|---:|---:|---:|
| Baseline correctness path | 完整 JPEG 解码 + `INTER_LINEAR` | 1920×1080 | 65.3 | 15.33 | 1.00× |
| Balanced speed/quality exact-output path | `REDUCED_2 + INTER_LINEAR`, ThreadPool(10) | 1920×1080 | 479.7 | 2.08 | 7.35× |
| Maximum-speed exact-output path | `REDUCED_8 + INTER_LINEAR`, ThreadPool(14) | 1920×1080 | 731.2 | 1.37 | 11.21× |
| Maximum-speed approximate-output path | `REDUCED_8 decode only`, ThreadPool(14) | 480×240 | 986.6 | 1.01 | 15.12× |

最强 exact-output 提交主张：

> 对 `3840×1920 MJPEG → 1920×1080`，使用 `REDUCED_8 + INTER_LINEAR + ThreadPool(14)` 达到 **731.2 fps**，相比完整解码 + `INTER_LINEAR` 的 **65.3 fps**，实现 **11.21× 加速**。

如果允许 approximate-output，即只要 reduced decoder 的原生输出，不要求 `1920×1080`，则 `REDUCED_8 decode only + ThreadPool(14)` 达到 **986.6 fps / 15.12×**，但输出尺寸为 **480×240**，不能直接替代 1080p 输出。

## 3. 实验环境

| 项目 | 值 |
|---|---|
| 机器 | Apple Silicon M4 MacBook Air |
| Python 架构 | arm64 |
| Python 版本 | 3.10.9 |
| macOS | 26.4.1 |
| OpenCV | 4.13.0 |
| 逻辑 CPU 数 | 10 |
| 电源状态 |  -InternalBattery-0 (id=22020195)	80%; AC attached; not charging present: true |
| 低电量模式 | False |
| 输入尺寸 | 3840×1920 |
| 目标输出尺寸 | 1920×1080 |
| JPEG 质量 | 85 |
| 测试帧数 | 80 frames × 5 repeats |
| 预热帧数 | 8 |

## 4. 方法说明

### 4.1 Baseline correctness path

完整解码 4K JPEG，再做 `INTER_LINEAR` resize：

```python
arr = np.frombuffer(buf, dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
out = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
```

该路径输出 `1920×1080`，作为质量参考和 speedup 基准。

### 4.2 Exact-output-size paths

Exact-output-size paths 都会执行最终 `cv2.resize(..., (1920, 1080))`，因此输出尺寸严格为 `1920×1080`。矩阵包括：

- `REDUCED_2 + INTER_LINEAR / INTER_NEAREST`
- `REDUCED_4 + INTER_LINEAR / INTER_NEAREST`
- `REDUCED_8 + INTER_LINEAR / INTER_NEAREST`
- worker counts: `[1, 2, 4, 6, 8, 10, 12, 14, 16]`

计时要求：固定 `cv2.setNumThreads(1)`，使用 `np.frombuffer`，不把所有输出帧保存到 list，只消费 iterator 并计数。

### 4.3 Approximate-output-size paths

Approximate-output-size paths 只做 reduced JPEG decode，不做最终 resize，因此输出是 decoder 原生尺寸：

- `REDUCED_2 decode only`: `1920×960`
- `REDUCED_4 decode only`: `960×480`
- `REDUCED_8 decode only`: `480×240`

这些路径可以作为“解码阶段理论上限”或下游接受低分辨率输入时的方案，但不能直接等价于 `1920×1080` 输出。

### 4.4 Quality comparison

质量对照只针对 exact-output-size paths，因为它们和 baseline 同为 `1920×1080`。Reduced JPEG decoding 不是 bit-exact，因此以完整解码 + `INTER_LINEAR` 为参考计算 MSE 和 PSNR。

## 5. RGB 结果

RGB 基准为 `OpenCV INTER_LINEAR (1 threads)`，吞吐量 **261.2 fps**。

| 类别 | 推荐结果 | FPS | ms/frame | 加速比 |
|---|---|---:|---:|---:|
| RGB 最高速度 | OpenCV INTER_NEAREST (10 threads) | 2430.2 | 0.41 | 9.30× |
| RGB 双线性质量路径 | OpenCV INTER_LINEAR (6 threads) | 1161.0 | 0.86 | 4.45× |

## 6. MJPEG 结果
### 6.1 MJPEG exact-output-size 结果

这些路径最终输出 `1920×1080`，可直接和 baseline 比较。

| 排名 | 配置 | 输出尺寸 | FPS | ms/frame | 加速比 | 备注 |
|---:|---|---:|---:|---:|---:|---|
| 1 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(14) | 1920×1080 | 731.2 | 1.37 | 11.21× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 2 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(12) | 1920×1080 | 726.2 | 1.38 | 11.13× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 3 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(16) | 1920×1080 | 726.0 | 1.38 | 11.13× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 4 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(10) | 1920×1080 | 723.5 | 1.38 | 11.09× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 5 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(12) | 1920×1080 | 721.6 | 1.39 | 11.06× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 6 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(10) | 1920×1080 | 713.2 | 1.40 | 10.93× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 7 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(16) | 1920×1080 | 708.2 | 1.41 | 10.85× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 8 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(14) | 1920×1080 | 703.6 | 1.42 | 10.78× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 9 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(14) | 1920×1080 | 644.8 | 1.55 | 9.88× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 10 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(12) | 1920×1080 | 643.9 | 1.55 | 9.87× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 11 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(16) | 1920×1080 | 640.2 | 1.56 | 9.81× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 12 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(8) | 1920×1080 | 636.8 | 1.57 | 9.76× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 13 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(10) | 1920×1080 | 636.4 | 1.57 | 9.75× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 14 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(8) | 1920×1080 | 634.2 | 1.58 | 9.72× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 15 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(16) | 1920×1080 | 617.5 | 1.62 | 9.46× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 16 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(14) | 1920×1080 | 615.1 | 1.63 | 9.43× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 17 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(10) | 1920×1080 | 613.2 | 1.63 | 9.40× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 18 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(12) | 1920×1080 | 612.6 | 1.63 | 9.39× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 19 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(8) | 1920×1080 | 563.6 | 1.77 | 8.64× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 20 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(8) | 1920×1080 | 535.3 | 1.87 | 8.20× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 21 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(6) | 1920×1080 | 533.8 | 1.87 | 8.18× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 22 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(6) | 1920×1080 | 531.9 | 1.88 | 8.15× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 23 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(12) | 1920×1080 | 518.9 | 1.93 | 7.95× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 24 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(16) | 1920×1080 | 510.3 | 1.96 | 7.82× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 25 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(14) | 1920×1080 | 510.1 | 1.96 | 7.82× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 26 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(10) | 1920×1080 | 508.4 | 1.97 | 7.79× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 27 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(6) | 1920×1080 | 480.3 | 2.08 | 7.36× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 28 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(10) | 1920×1080 | 479.7 | 2.08 | 7.35× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 29 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(12) | 1920×1080 | 478.7 | 2.09 | 7.34× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 30 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(16) | 1920×1080 | 474.4 | 2.11 | 7.27× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 31 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(14) | 1920×1080 | 470.8 | 2.12 | 7.22× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 32 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(6) | 1920×1080 | 452.9 | 2.21 | 6.94× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 33 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(4) | 1920×1080 | 434.4 | 2.30 | 6.66× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 34 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(8) | 1920×1080 | 432.9 | 2.31 | 6.63× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 35 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(8) | 1920×1080 | 430.1 | 2.33 | 6.59× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 36 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(4) | 1920×1080 | 428.5 | 2.33 | 6.57× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 37 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(4) | 1920×1080 | 394.8 | 2.53 | 6.05× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 38 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(6) | 1920×1080 | 379.1 | 2.64 | 5.81× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 39 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(6) | 1920×1080 | 363.8 | 2.75 | 5.58× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 40 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(4) | 1920×1080 | 363.7 | 2.75 | 5.57× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 41 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(4) | 1920×1080 | 328.6 | 3.04 | 5.04× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 42 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(4) | 1920×1080 | 291.6 | 3.43 | 4.47× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 43 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(2) | 1920×1080 | 238.8 | 4.19 | 3.66× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 44 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(2) | 1920×1080 | 237.1 | 4.22 | 3.63× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 45 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(2) | 1920×1080 | 219.9 | 4.55 | 3.37× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 46 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(2) | 1920×1080 | 209.8 | 4.77 | 3.22× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 47 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(2) | 1920×1080 | 184.1 | 5.43 | 2.82× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 48 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(2) | 1920×1080 | 164.6 | 6.07 | 2.52× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 49 | OpenCV REDUCED_8 + INTER_NEAREST, ThreadPool(1) | 1920×1080 | 122.0 | 8.19 | 1.87× | maximum DCT shrink, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 50 | OpenCV REDUCED_8 + INTER_LINEAR, ThreadPool(1) | 1920×1080 | 121.8 | 8.21 | 1.87× | most aggressive 1/8 DCT decode with final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 51 | OpenCV REDUCED_4 + INTER_NEAREST, ThreadPool(1) | 1920×1080 | 112.8 | 8.86 | 1.73× | maximum-speed path, lowest quality; cv2.setNumThreads(1); np.frombuffer; no output list |
| 52 | OpenCV REDUCED_4 + INTER_LINEAR, ThreadPool(1) | 1920×1080 | 105.2 | 9.50 | 1.61× | more aggressive 1/4 DCT decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 53 | OpenCV REDUCED_2 + INTER_NEAREST, ThreadPool(1) | 1920×1080 | 95.3 | 10.49 | 1.46× | faster 1/2 DCT decode with nearest final resize; cv2.setNumThreads(1); np.frombuffer; no output list |
| 54 | OpenCV REDUCED_2 + INTER_LINEAR, ThreadPool(1) | 1920×1080 | 85.5 | 11.70 | 1.31× | balanced speed/quality path; cv2.setNumThreads(1); np.frombuffer; no output list |
| 55 | OpenCV imdecode + resize LINEAR (full 4K decode) | 1920×1080 | 65.3 | 15.33 | 1.00× | baseline: decode at full 4K, then resize |
| 56 | OpenCV imdecode + resize LINEAR preallocated dst | 1920×1080 | 64.4 | 15.53 | 0.99× | full decode baseline without resize output allocation |

### 6.2. MJPEG approximate-output-size / decode-only 结果

这些路径不做最终 `cv2.resize`，输出尺寸不是 `1920×1080`。

| 排名 | 配置 | 输出尺寸 | FPS | ms/frame | 加速比 | 备注 |
|---:|---|---:|---:|---:|---:|---|
| 1 | OpenCV REDUCED_8 decode only, ThreadPool(14) | 480×240 | 986.6 | 1.01 | 15.12× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 2 | OpenCV REDUCED_8 decode only, ThreadPool(10) | 480×240 | 982.0 | 1.02 | 15.05× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 3 | OpenCV REDUCED_8 decode only, ThreadPool(16) | 480×240 | 974.1 | 1.03 | 14.93× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 4 | OpenCV REDUCED_8 decode only, ThreadPool(12) | 480×240 | 970.9 | 1.03 | 14.88× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 5 | OpenCV REDUCED_8 decode only, ThreadPool(8) | 480×240 | 852.3 | 1.17 | 13.06× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 6 | OpenCV REDUCED_4 decode only, ThreadPool(12) | 960×480 | 843.8 | 1.19 | 12.93× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 7 | OpenCV REDUCED_4 decode only, ThreadPool(14) | 960×480 | 840.5 | 1.19 | 12.88× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 8 | OpenCV REDUCED_4 decode only, ThreadPool(10) | 960×480 | 839.2 | 1.19 | 12.86× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 9 | OpenCV REDUCED_4 decode only, ThreadPool(16) | 960×480 | 838.3 | 1.19 | 12.85× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 10 | OpenCV REDUCED_4 decode only, ThreadPool(8) | 960×480 | 749.2 | 1.33 | 11.48× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 11 | OpenCV REDUCED_8 decode only, ThreadPool(6) | 480×240 | 719.2 | 1.39 | 11.02× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 12 | OpenCV REDUCED_2 decode only, ThreadPool(14) | 1920×960 | 680.9 | 1.47 | 10.44× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 13 | OpenCV REDUCED_2 decode only, ThreadPool(10) | 1920×960 | 680.4 | 1.47 | 10.43× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 14 | OpenCV REDUCED_2 decode only, ThreadPool(12) | 1920×960 | 677.6 | 1.48 | 10.38× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 15 | OpenCV REDUCED_2 decode only, ThreadPool(16) | 1920×960 | 669.6 | 1.49 | 10.26× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 16 | OpenCV REDUCED_4 decode only, ThreadPool(6) | 960×480 | 629.4 | 1.59 | 9.65× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 17 | OpenCV REDUCED_2 decode only, ThreadPool(8) | 1920×960 | 609.1 | 1.64 | 9.34× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 18 | OpenCV REDUCED_8 decode only, ThreadPool(4) | 480×240 | 564.2 | 1.77 | 8.65× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 19 | OpenCV REDUCED_2 decode only, ThreadPool(6) | 1920×960 | 516.9 | 1.93 | 7.92× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 20 | OpenCV REDUCED_4 decode only, ThreadPool(4) | 960×480 | 509.7 | 1.96 | 7.81× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 21 | OpenCV REDUCED_2 decode only, ThreadPool(4) | 1920×960 | 414.9 | 2.41 | 6.36× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 22 | OpenCV REDUCED_8 decode only, ThreadPool(2) | 480×240 | 290.2 | 3.45 | 4.45× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 23 | OpenCV REDUCED_4 decode only, ThreadPool(2) | 960×480 | 263.5 | 3.80 | 4.04× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 24 | OpenCV REDUCED_2 decode only, ThreadPool(2) | 1920×960 | 218.5 | 4.58 | 3.35× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 25 | OpenCV REDUCED_8 decode only, ThreadPool(1) | 480×240 | 146.7 | 6.82 | 2.25× | approximate output: native 1/8 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 26 | OpenCV REDUCED_4 decode only, ThreadPool(1) | 960×480 | 132.6 | 7.54 | 2.03× | approximate output: native 1/4 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |
| 27 | OpenCV REDUCED_2 decode only, ThreadPool(1) | 1920×960 | 109.1 | 9.17 | 1.67× | approximate output: native 1/2 JPEG decode; cv2.setNumThreads(1); np.frombuffer; no output list |

### 6.3. 质量对照结果

| Variant | Samples | MSE | PSNR |
|---|---:|---:|---:|
| REDUCED_2 + INTER_LINEAR | 5 | 1.09 | 47.76 dB |
| REDUCED_2 + INTER_NEAREST | 5 | 2.66 | 43.87 dB |
| REDUCED_4 + INTER_LINEAR | 5 | 3.14 | 43.16 dB |
| REDUCED_4 + INTER_NEAREST | 5 | 3.30 | 42.95 dB |
| REDUCED_8 + INTER_LINEAR | 5 | 3.78 | 42.36 dB |
| REDUCED_8 + INTER_NEAREST | 5 | 4.08 | 42.02 dB |

质量结论：

- `REDUCED_2 + INTER_LINEAR` 质量最好，MSE **1.09**，PSNR **47.76 dB**。
- `REDUCED_8 + INTER_LINEAR` 是当前最快的 exact-output path，但 PSNR 降到 **42.36 dB**。
- `REDUCED_8 + INTER_NEAREST` 的质量最低，PSNR **42.02 dB**，但本次 exact-output 速度不如 `REDUCED_8 + INTER_LINEAR`。

### 6.4 MJEPG策略划分

#### 6.4.1 质量平衡方案

```python
cv2.setNumThreads(1)
img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
out = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
```

使用 `ThreadPoolExecutor(max_workers=10)`：**479.7 fps / 7.35× / PSNR 47.76 dB**。

#### 6.4.2 最高 exact-output 速度方案

```python
cv2.setNumThreads(1)
img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_8)
out = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
```

使用 `ThreadPoolExecutor(max_workers=14)`：**731.2 fps / 11.21×**。

#### 6.4.3 最高 approximate-output 速度方案

```python
cv2.setNumThreads(1)
img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_8)
```

使用 `ThreadPoolExecutor(max_workers=14)`：**986.6 fps / 15.12×**，但输出为 **480×240**。

## 7. GPU / VideoToolbox 说明

本机 OpenCV wheel 没有 Metal/MPS 后端。OpenCL/UMat 对照实验显示速度低于 CPU OpenCV resize，因此不作为最终方案。

已尝试 Swift `VTDecompressionSession` + `kCMVideoCodecType_JPEG`。系统查询显示 `VTIsHardwareDecodeSupported(kCMVideoCodecType_JPEG) = true`，但当前裸 JPEG sample buffer 创建 session 返回 `OSStatus -12911`。该方向仍可能通过 QuickTime/MJPEG sample description、AVFoundation 容器路径或真实摄像头 MJPEG 流继续探索。

## 8. 复现命令

常规稳定运行：

```bash
python3 M4benchmark.py --frames 80 --repeats 5 --warmup 8 --skip-gpu --skip-optional --quality-compare --quality-samples 5
```

更稳定但更慢的 long-run：

```bash
python3 M4benchmark.py --long-run --skip-gpu --skip-optional --quality-compare --quality-samples 5
```

输出文件：

- `m4_results.json`：完整原始结果和 quality metrics。
- `m4_report.md`：自动生成摘要。

