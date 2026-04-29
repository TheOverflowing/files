# 基于 OpenCV 的图像缩放加速实验报告

## 摘要

本实验围绕 3840×1920 图像向 1920×1080 图像的实时缩放任务，评估 Apple Silicon M4 平台上 OpenCV 在 RGB 图像缩放与 MJPEG 解码缩放中的加速效果，并补充测试 Google Colab Tesla T4 GPU 在同类任务上的表现。实验分别测试了已解码 RGB 图像的多种插值方式、OpenCV 内部多线程、MJPEG 的完整解码路径、JPEG DCT 层面的缩减解码路径、帧级并行处理方式，以及 T4 上的 PyTorch/CuPy/DALI GPU 路径。

实验结果表明，对于已解码 RGB 图像，OpenCV `INTER_LINEAR` 在 4 线程下达到 1133.1 fps，相对于单线程 `INTER_LINEAR` 的 256.3 fps 实现 4.42 倍加速；若采用画质较低的 `INTER_NEAREST`，最高可达到 1835.9 fps，也就是 7.16 倍的加速比。

对于 MJPEG 输入，完整 4K JPEG 解码后再缩放的基准吞吐量为 63.3 fps；采用 `IMREAD_REDUCED_COLOR_2` 结合 `INTER_LINEAR` 与 10 个线程的帧级并行后，吞吐量达到 466.8 fps，加速比为 7.37 倍，且 PSNR 为 47.76 dB、SSIM 为 0.9836，属于较好的速度与画质折中方案。若优先追求最高输出速度，`IMREAD_REDUCED_COLOR_8 + INTER_LINEAR + ThreadPool(12)` 可达到 723.0 fps，加速比为 11.42 倍，但画质相对下降。

T4 补充实验显示，GPU 在“纯 resize”上具有很高吞吐量：预加载到显存后的 PyTorch `F.interpolate` 最高达到 2290.2 fps，相对 Colab CPU 单线程 resize 为 19.65 倍。但在包含 CPU/GPU 数据传输的端到端 RGB resize 中，PyTorch、CuPy 与 DALI 均未超过 CPU resize baseline。对于 MJPEG，DALI/nvJPEG 在绝对吞吐量上优于 Colab CPU 路径，但相对完整解码 baseline 的加速比主要集中在 4.50× 到 6.30×，低于 M4 上 DCT 缩减解码与多核并行达到的 11× 级别加速。这说明 T4 的 MJPEG 结果瓶颈不在 resize kernel，而在 JPEG 解码、CPU/GPU pipeline 与 Colab 仅 2 个可见 CPU 核心的限制。

## 1. 实验目的

本实验的目标是验证在 CPU 环境下，OpenCV 是否能够有效提升 4K 图像向 1080p 图像缩放的处理速度，并分析不同加速方法在速度、输出尺寸和图像质量之间的权衡关系。实验重点关注以下问题：

1. 对于已解码 RGB 图像，OpenCV 的多线程缩放能否显著提高吞吐量。
2. 对于 MJPEG 输入，是否可以避免完整 4K 解码带来的额外计算开销。
3. JPEG 缩减解码、最终 resize 与帧级并行组合后，能否满足实时或高帧率图像处理需求。
4. 不同方法在速度提升的同时会对图像质量产生何种影响。

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| 设备 | Apple Silicon M4 MacBook Air |
| CPU 架构 | arm64 |
| 操作系统 | macOS 26.4.1 |
| Python 版本 | 3.10.9 |
| OpenCV 版本 | 4.13.0 |
| 逻辑 CPU 数 | 10 |
| 电源状态 | 接入电源，低电量模式关闭 |
| 输入图像尺寸 | 3840×1920 |
| 输出图像尺寸 | 1920×1080 |
| JPEG 质量 | 85 |
| 测试规模 | 80 frames × 5 repeats |
| 预热帧数 | 8 |

T4 补充实验环境如下：

| 项目 | 配置 |
|---|---|
| 设备 | Google Colab Tesla T4 |
| GPU 显存 | 14.56 GB |
| CUDA 版本 | 12.8 |
| GPU 计算能力 | 7.5 |
| Python 版本 | 3.12.13 |
| PyTorch 版本 | 2.10.0+cu128 |
| CuPy 版本 | 14.0.1 |
| OpenCV 版本 | 4.13.0 |
| DALI | 可用 |
| Python 可见 CPU 核心数 | 2 |
| 测试规模 | RGB 200 frames × 3 repeats；MJPEG 80 frames × 3 repeats |
| 预热帧数 | RGB 20；MJPEG 20 |

除主测试所用的合成 MJPEG 序列外，本实验还生成了 5 类 3840×1920 的合成测试图像以考察内容复杂度对 JPEG 解码吞吐量的影响：低纹理平滑图像、高频纹理图像、类自然图像、平滑与噪声混合图像，以及模拟文本排版的白底黑块图像。所有图像均使用 OpenCV 以 JPEG quality 85 编码。

## 3. 实验方法与原理

### 3.1 基准方法

实验以完整解码后再缩放作为 MJPEG 处理的基准路径。该方法首先将 MJPEG 字节流完整解码为 3840×1920 的 BGR 图像，再使用双线性插值缩放到 1920×1080。其核心流程如下：

```python
arr = np.frombuffer(buf, dtype=np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
out = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
```

该路径的优点是实现直接、输出尺寸精确、图像质量稳定，因此适合作为正确性和加速比的参考基准。其缺点是完整解码会产生 3840×1920 的中间图像，而后续缩放又会丢弃大量像素，因此存在明显的冗余计算与内存访问开销。

### 3.2 OpenCV 多线程缩放原理

OpenCV 的 `cv2.resize` 在部分平台和构建配置下可以使用多线程执行。图像缩放本质上是对输出图像中每个像素独立计算其在源图像中的对应位置，并根据插值方式生成像素值。因此，不同行或不同图像块之间依赖较少，适合并行处理。

在本实验中，通过 `cv2.setNumThreads(N)` 控制 OpenCV 内部线程数。对于已解码 RGB 图像，实验比较了 `INTER_LINEAR`、`INTER_NEAREST`、`INTER_CUBIC`、`INTER_AREA` 和 `INTER_LANCZOS4` 等插值方式。不同插值方式的计算量不同：

- `INTER_NEAREST` 只选择最近的源像素，计算量最低，但图像边缘容易出现锯齿或块状失真。
- `INTER_LINEAR` 使用相邻 2×2 像素进行双线性加权，是速度与画质较均衡的常用方法。
- `INTER_CUBIC` 使用更大的邻域进行三次插值，画质可能更平滑，但计算量更高。
- `INTER_AREA` 更适合高质量下采样，但在实时场景下速度较慢。
- `INTER_LANCZOS4` 使用更复杂的滤波核，质量较高但计算成本最大。

### 3.3 JPEG DCT 缩减解码原理

MJPEG 的每一帧通常是独立 JPEG 图像。JPEG 编码会把图像划分为 8×8 像素块，并将每个块转换到离散余弦变换（DCT）频域。图像的低频信息主要表示整体亮度和颜色变化，高频信息主要表示边缘和细节。

在解码阶段，libjpeg-turbo 等 JPEG 解码器可以直接输出 1/2、1/4 或 1/8 分辨率的图像。这种方式不需要先完整恢复原始 4K 图像，而是在 DCT 解码过程中只保留与低分辨率输出相关的信息。因此，它可以显著减少 IDCT 计算量、内存写入量和后续 resize 的输入规模。

OpenCV 通过以下参数暴露该能力：

| 参数 | 原生解码输出尺寸 |
|---|---|
| `cv2.IMREAD_REDUCED_COLOR_2` | 1920×960 |
| `cv2.IMREAD_REDUCED_COLOR_4` | 960×480 |
| `cv2.IMREAD_REDUCED_COLOR_8` | 480×240 |

由于实验目标输出为 1920×1080，缩减解码得到的原生尺寸不一定与目标尺寸完全一致。因此，exact-output-size 路径会在缩减解码后继续执行一次 `cv2.resize`，使最终输出严格保持 1920×1080。

### 3.4 帧级并行方法

MJPEG 帧之间相互独立，因此可以把每一帧的“解码 + 缩放”作为一个任务提交到线程池中并行执行。本实验使用 `ThreadPoolExecutor` 测试不同 worker 数量下的吞吐量。在帧级并行路径中，每个任务内部设置 `cv2.setNumThreads(1)`，目的是避免 OpenCV 内部多线程与外部线程池形成嵌套并行，造成线程过多和调度开销增加。

帧级并行的基本流程如下：

```python
cv2.setNumThreads(1)
img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8),
                   cv2.IMREAD_REDUCED_COLOR_2)
out = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
```

该方法的核心思想是：在单帧内部减少解码和缩放工作量，同时在多帧之间利用 CPU 多核心并行处理，从而提高整体吞吐量。

在 Python 实现中使用线程池而不是多进程还有一个重要原因：`cv2.imdecode` 和 `cv2.resize` 的主要计算发生在 OpenCV 的 C/C++ 实现中，这些调用在执行耗时计算时会释放 Python GIL。因此，多线程可以并行推进多个解码和缩放任务，同时避免多进程带来的大图像数据序列化、进程间通信和内存复制开销。

### 3.5 图像质量评价方法

质量对照仅针对最终输出为 1920×1080 的路径。实验以完整解码后使用 `INTER_LINEAR` 缩放的结果作为参考图像，计算不同缩减解码路径的 MSE、PSNR 和 SSIM。MSE 越低表示像素均方误差越小，PSNR 越高表示图像越接近参考结果；SSIM 从亮度、对比度和结构相似性角度评价图像，数值越接近 1 表示结构越接近参考图像。

### 3.6 T4 GPU 补充实验方法

T4 实验分为四部分：第一部分测试已解码 RGB 图像的 resize，包括 OpenCV CPU、PyTorch GPU-only、PyTorch end-to-end、CuPy end-to-end 和 DALI resize pipeline；第二部分测试 MJPEG decode + resize，包括 OpenCV 完整解码、OpenCV `IMREAD_REDUCED_COLOR_2` 缩减解码与 DALI/nvJPEG mixed decode + GPU resize；第三部分单独测试预加载 GPU tensor 的 batch resize 上限；第四部分比较 PyTorch、CuPy 和 DALI resize 输出与 OpenCV CPU reference 的 MSE、PSNR 和 SSIM。

需要强调的是，T4 的 DALI/nvJPEG 路径并不等价于 OpenCV 的 `IMREAD_REDUCED_COLOR_8`。前者主要优化完整 JPEG decode + GPU resize pipeline，后者是在 JPEG DCT 解码阶段减少输出分辨率，属于算法级少算。因此，二者的 speedup ratio 不能直接视为同一种优化方法的横向替代。

## 4. 实验结果

### 4.1 RGB 图像缩放结果

RGB 图像部分以 `OpenCV INTER_LINEAR (1 thread)` 为基准，其吞吐量为 256.3 fps。

| 方法 | FPS | ms/frame | 加速比 | 说明 |
|---|---:|---:|---:|---|
| `INTER_NEAREST`，10 线程 | 1835.9 | 0.54 | 7.16× | 速度最高，画质较低 |
| `INTER_LINEAR`，4 线程 | 1133.1 | 0.88 | 4.42× | 推荐的速度与画质平衡方案 |
| `INTER_LINEAR`，10 线程 | 1130.0 | 0.88 | 4.41× | 与 4 线程结果接近 |
| `INTER_CUBIC`，10 线程 | 399.8 | 2.50 | 1.56× | 质量较高但速度较慢 |
| `INTER_LINEAR`，1 线程 | 256.3 | 3.90 | 1.00× | RGB 基准 |
| `INTER_AREA`，10 线程 | 206.8 | 4.84 | 0.81× | 高质量下采样，实时性能较低 |
| `INTER_LANCZOS4`，10 线程 | 183.7 | 5.44 | 0.72× | 计算成本最高 |

结果说明，已解码 RGB 图像的缩放能够明显受益于 OpenCV 内部多线程。双线性插值在 4 线程时达到 1133.1 fps，继续增加线程数后收益趋于饱和，说明此时性能可能受到内存带宽、调度开销或实现内部并行粒度的限制。

### 4.2 MJPEG 精确输出尺寸结果

MJPEG 部分以完整 4K 解码后再 `INTER_LINEAR` 缩放为基准，其吞吐量为 63.3 fps。

| 方法 | 输出尺寸 | FPS | ms/frame | 加速比 | 说明 |
|---|---:|---:|---:|---:|---|
| 完整 JPEG 解码 + `INTER_LINEAR` | 1920×1080 | 63.3 | 15.80 | 1.00× | 基准路径 |
| `REDUCED_2 + INTER_LINEAR`，ThreadPool(10) | 1920×1080 | 466.8 | 2.14 | 7.37× | 推荐的质量平衡方案 |
| `REDUCED_4 + INTER_LINEAR`，ThreadPool(10) | 1920×1080 | 582.1 | 1.72 | 9.19× | 更高速度，质量略降 |
| `REDUCED_8 + INTER_LINEAR`，ThreadPool(12) | 1920×1080 | 723.0 | 1.38 | 11.42× | 最高双线性精确输出速度 |
| `REDUCED_8 + INTER_NEAREST`，ThreadPool(12) | 1920×1080 | 738.4 | 1.35 | 11.66× | 最高精确输出速度，但画质最低 |

结果表明，MJPEG 处理的主要瓶颈在于完整 JPEG 解码。通过 DCT 层面的缩减解码，可以避免生成完整 4K 中间图像，从而显著减少计算和内存访问。在此基础上叠加帧级并行后，吞吐量从 63.3 fps 提升至最高 738.4 fps。

### 4.3 MJPEG 近似输出尺寸结果

部分路径只执行缩减解码，不再 resize 到 1920×1080。此类方法速度更高，但输出尺寸不是目标尺寸，因此不能直接替代 1080p 输出。

| 方法 | 输出尺寸 | FPS | ms/frame | 加速比 |
|---|---:|---:|---:|---:|
| `REDUCED_2 decode only`，ThreadPool(14) | 1920×960 | 671.6 | 1.49 | 10.61× |
| `REDUCED_4 decode only`，ThreadPool(10) | 960×480 | 837.2 | 1.19 | 13.23× |
| `REDUCED_8 decode only`，ThreadPool(16) | 480×240 | 1007.9 | 0.99 | 15.93× |

该结果可作为解码阶段理论上限的参考。如果后续算法可以直接接受较低分辨率输入，例如预览、快速检测或多级处理流水线，则 decode-only 路径具有实际价值；若必须输出 1920×1080，则仍需采用 exact-output-size 路径。

### 4.4 图像质量结果

| 方法 | 样本数 | MSE | PSNR | SSIM |
|---|---:|---:|---:|---:|
| `REDUCED_2 + INTER_LINEAR` | 5 | 1.09 | 47.76 dB | 0.9836 |
| `REDUCED_2 + INTER_NEAREST` | 5 | 2.66 | 43.87 dB | 0.9609 |
| `REDUCED_4 + INTER_LINEAR` | 5 | 3.14 | 43.16 dB | 0.9514 |
| `REDUCED_4 + INTER_NEAREST` | 5 | 3.30 | 42.95 dB | 0.9501 |
| `REDUCED_8 + INTER_LINEAR` | 5 | 3.78 | 42.36 dB | 0.9423 |
| `REDUCED_8 + INTER_NEAREST` | 5 | 4.08 | 42.02 dB | 0.9389 |

质量结果显示，`REDUCED_2 + INTER_LINEAR` 的 PSNR 和 SSIM 均最高，说明其输出最接近完整解码基准；`REDUCED_8 + INTER_LINEAR` 虽然速度较高，但由于解码阶段保留的信息更少，PSNR 降至 42.36 dB，SSIM 降至 0.9423。因此，实际应用中应根据质量要求选择合适的缩减倍率。

### 4.5 图像内容对解码性能的影响

为分析图像内容复杂度对 MJPEG 解码吞吐量的影响，实验额外生成 5 类 3840×1920 合成图像，并分别以 JPEG quality 85 编码。每类图像均重复构成 80 帧输入序列，测试完整解码基准、`REDUCED_2 + INTER_LINEAR + ThreadPool(10)` 和 `REDUCED_8 + INTER_LINEAR + ThreadPool(14)` 三条关键路径。

| 图像类型 | JPEG 大小 | 完整解码 + `INTER_LINEAR` | `REDUCED_2 + INTER_LINEAR`，TP(10) | `REDUCED_8 + INTER_LINEAR`，TP(14) |
|---|---:|---:|---:|---:|
| 低纹理平滑图像 | 113.8 KB | 127.4 fps | 789.4 fps | 1485.9 fps |
| 高频纹理图像 | 5596.7 KB | 24.0 fps | 158.7 fps | 179.9 fps |
| 类自然图像 | 172.4 KB | 107.9 fps | 707.4 fps | 1360.4 fps |
| 平滑与噪声混合图像 | 2778.5 KB | 40.9 fps | 281.5 fps | 355.3 fps |
| 模拟文本图像 | 392.4 KB | 110.0 fps | 632.2 fps | 1250.6 fps |

结果显示，图像内容对 JPEG 解码吞吐量影响显著。低纹理图像的 JPEG 文件仅 113.8 KB，完整解码可达 127.4 fps；高频纹理图像的 JPEG 文件达到 5596.7 KB，完整解码仅 24.0 fps。其原因在于低纹理图像的 DCT 系数主要集中在低频部分，熵编码后数据量较小，解码和内存访问压力较低；高频噪声或细密棋盘格会产生大量非零高频 DCT 系数，压缩率下降，熵解码和 IDCT 相关工作量随之增加。

缩减解码对所有图像类型均有效，但收益幅度也受内容影响。低纹理与类自然图像在 `REDUCED_8 + INTER_LINEAR` 下分别达到 1485.9 fps 和 1360.4 fps，而高频纹理图像仅为 179.9 fps。说明 DCT 缩减解码虽然减少了输出像素规模，但仍无法完全消除高复杂度 JPEG 比特流本身带来的熵解码负担。

### 4.6 线程交互分析

为验证 OpenCV 内部线程数与帧级线程池之间的相互影响，实验固定使用 `REDUCED_2 + INTER_LINEAR` 路径，系统测试 `cv2.setNumThreads(T)` 与 `ThreadPool(W)` 的组合。为避免在 10 个逻辑核心上形成过度超订阅，跳过了 `T × W > 20` 的组合。

| OpenCV 内部线程 T | ThreadPool workers W | T×W | FPS | ms/frame |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 148.6 | 6.73 |
| 1 | 4 | 4 | 277.6 | 3.60 |
| 1 | 6 | 6 | 338.7 | 2.95 |
| 1 | 8 | 8 | 403.1 | 2.48 |
| 1 | 10 | 10 | 419.5 | 2.38 |
| 1 | 12 | 12 | 398.8 | 2.51 |
| 1 | 14 | 14 | 422.2 | 2.37 |
| 2 | 2 | 4 | 184.9 | 5.41 |
| 2 | 4 | 8 | 291.7 | 3.43 |
| 2 | 6 | 12 | 349.7 | 2.86 |
| 2 | 8 | 16 | 393.6 | 2.54 |
| 2 | 10 | 20 | 420.0 | 2.38 |
| 4 | 2 | 8 | 172.8 | 5.79 |
| 4 | 4 | 16 | 270.6 | 3.69 |

本次扫描的最优组合为 `T=1, W=14`，吞吐量为 422.2 fps。最佳混合组合为 `T=2, W=10`，吞吐量为 420.0 fps，未超过最佳 `T=1` 配置。因此，主实验中采用“每帧内部单线程、帧间并行”的策略仍然成立。

从机制上看，MJPEG 帧级任务已经包含解码、缩减解码和最终 resize。若每个任务内部再使用多个 OpenCV 线程，会增加线程调度、缓存竞争和核心争用。Apple Silicon M4 同时包含性能核心和能效核心，线程池数量较高时，任务可能跨不同类型核心调度；此时让单帧任务保持较小的内部并行粒度，更有利于多帧任务在核心之间均匀排布。`T=2, W=10` 与 `T=1, W=14` 的结果非常接近，说明轻度混合并行并非不可用，但在本次测试中没有稳定优势。

### 4.7 Tesla T4 GPU 补充结果

T4 上的 RGB resize 结果显示，纯 GPU resize kernel 很快，但端到端路径受到 CPU/GPU 数据传输和 pipeline 开销限制。以 Colab CPU 上的 OpenCV 单线程 resize 116.6 fps 为基准，PyTorch 预加载 GPU tensor 的最高吞吐量为 2290.2 fps，即 19.65×；但包含每帧 CPU 到 GPU 传输和结果拷回的 PyTorch end-to-end 路径最高只有 112.1 fps，未超过 CPU baseline。

| T4 RGB 方法 | FPS | ms/frame | 加速比 | 是否包含传输 |
|---|---:|---:|---:|---|
| OpenCV resize CPU threads=1 | 116.6 | 8.58 | 1.00× | 否 |
| OpenCV resize CPU threads=4 | 130.5 | 7.67 | 1.12× | 否 |
| PyTorch `F.interpolate` GPU-only batch=4 | 2290.2 | 0.44 | 19.65× | 否 |
| PyTorch `F.interpolate` end-to-end batch=4 | 112.1 | 8.92 | 0.96× | 是 |
| CuPy zoom end-to-end | 90.6 | 11.03 | 0.78× | 是 |
| DALI resize GPU pipeline end-to-end | 98.3 | 10.17 | 0.84× | 是 |

T4 上的 MJPEG 结果不如 M4 DCT 路径理想。DALI/nvJPEG 在所有图像类型上都明显快于 Colab CPU 完整解码 baseline，但相对加速比主要在 4.50× 到 6.30×，低于 M4 上 `REDUCED_8 + INTER_LINEAR + ThreadPool(12)` 的 11.42×。同时，Colab 仅暴露 2 个 CPU 核心，OpenCV 缩减解码 + ThreadPool 的收益明显低于 M4 多核心环境。

| 图像类型 | JPEG 大小 | T4 完整解码 + resize baseline | T4 最佳 OpenCV REDUCED_2 | T4 最佳 DALI/nvJPEG | DALI 加速比 |
|---|---:|---:|---:|---:|---:|
| low_texture | 113.8 KB | 40.2 fps | 91.6 fps | 233.7 fps | 5.82× |
| high_frequency | 5596.7 KB | 9.6 fps | 14.6 fps | 43.2 fps | 4.50× |
| natural_like | 172.4 KB | 32.8 fps | 77.4 fps | 206.6 fps | 6.30× |
| mixed | 2778.5 KB | 16.2 fps | 25.4 fps | 74.1 fps | 4.58× |
| text_like | 392.4 KB | 33.8 fps | 77.6 fps | 167.7 fps | 4.95× |

T4 上的 PyTorch batch scaling 进一步说明 resize 本身不是瓶颈。预加载 GPU tensor 后，batch size 从 1 到 64 的吞吐量都在 2035 fps 以上，说明 T4 对该尺寸的 bilinear resize 有充足算力；MJPEG 路径的性能主要受 JPEG 解码、DALI/nvJPEG pipeline、CPU 参与部分和结果回传影响。

| Batch size | PyTorch GPU-only FPS | ms/frame |
|---:|---:|---:|
| 1 | 2314.5 | 0.43 |
| 4 | 2286.0 | 0.44 |
| 16 | 2197.5 | 0.46 |
| 64 | 2035.6 | 0.49 |

质量验证方面，PyTorch `F.interpolate` 与 OpenCV reference 最接近，PSNR 为 52.33 dB、SSIM 为 0.997；DALI resize 的 PSNR 为 43.54 dB、SSIM 为 0.960，仍处于可用范围，但与 OpenCV 的像素级结果存在更明显差异。

## 5. 结果分析

对于已解码 RGB 输入，OpenCV 内部多线程可以显著降低单帧缩放耗时。`INTER_LINEAR` 是较合理的默认选择，因为它在约 0.88 ms/frame 的速度下仍保持较好的视觉质量。`INTER_NEAREST` 虽然速度最高，但会牺牲插值质量，适合缩略图、快速预览或后续模型对细节不敏感的场景。

对于 MJPEG 输入，实验结果说明加速的关键不是单纯优化 resize，而是减少解码阶段的工作量。完整解码会先恢复全部 3840×1920 像素，再缩放到 1920×1080；而 DCT 缩减解码可以在 JPEG 解码阶段直接生成低分辨率图像，大幅降低后续处理的数据规模。结合帧级并行后，多帧任务能够更充分地利用 M4 的多核心计算能力。

从速度与质量的综合角度看，`REDUCED_2 + INTER_LINEAR + ThreadPool(10)` 是较均衡的方案。它实现 7.37 倍加速，同时 PSNR 达到 47.76 dB、SSIM 达到 0.9836，图像误差较小。若任务更关注极限吞吐量，`REDUCED_8 + INTER_LINEAR + ThreadPool(12)` 可作为最高双线性精确输出速度方案，但需要接受更明显的质量损失。

新增的图像内容多样性实验说明，MJPEG 吞吐量并不是固定常数，而是与图像内容强相关。低纹理图像和类自然图像压缩后文件较小，DCT 系数更集中，解码速度明显更高；高频纹理图像和混合噪声图像保留了大量高频信息，JPEG 文件体积显著增大，熵解码与 IDCT 相关开销更高。因此，在真实应用中，仅使用单一合成图像估计摄像头吞吐量可能过于乐观或过于保守，应使用接近实际场景的素材重新测量。

线程交互实验进一步表明，帧级并行是本任务更有效的并行层次。`T=1, W=14` 的组合达到 422.2 fps，是本次 sweep 中的最优结果；`T=2, W=10` 虽然接近，但没有超过该结果。说明在 M4 的 10 个逻辑核心上，适度增加帧级 worker 可以充分利用多核心，而在单帧内部再启用较多 OpenCV 线程容易引入核心争用和缓存压力。

T4 补充实验进一步说明，GPU 加速并不自动带来端到端加速。RGB 部分中，预加载 GPU tensor 的 resize 速度达到 2000 fps 以上，但一旦把 CPU/GPU 传输和输出回传计入时间，端到端吞吐量反而低于 CPU baseline。MJPEG 部分也类似：DALI/nvJPEG 的绝对速度明显高于 Colab CPU 路径，但它优化的是完整 decode + resize pipeline，并没有像 OpenCV `IMREAD_REDUCED_COLOR_8` 那样在 DCT 阶段直接减少解码工作量。因此，T4 上的 DALI 加速比只有 4.50× 到 6.30×，低于 M4 上 DCT 缩减解码配合多核心帧级并行得到的 11× 级别加速。

此外，Colab T4 环境仅向 Python 暴露 2 个 CPU 核心，这使得 OpenCV DCT 缩减解码 + ThreadPool 无法像 M4 上一样充分并行。对 MJPEG 这类仍包含 CPU Huffman 解码、调度和内存搬运的任务，GPU 的 resize 算力并不是唯一决定因素。为了进一步探索 T4 的上限，新增的独立 `mjpeg_benchmark.py` 已加入 `REDUCED_COLOR_2/4/8` 的 decode-only、`INTER_LINEAR`、`INTER_NEAREST` 和 ThreadPool sweep，并加入 DALI `batch_size` 与 `prefetch_queue_depth` sweep，用于测试 batched nvJPEG decoding 和 pipeline overlap 是否能改善 speedup ratio。

## 6. 结论

本实验验证了 OpenCV 在 Apple Silicon M4 平台上对 4K 到 1080p 图像缩放任务具有明显加速效果。对于已解码 RGB 图像，使用 OpenCV 多线程 `INTER_LINEAR` 可实现 4.42 倍加速，适合作为通用实时缩放方案。对于 MJPEG 输入，最佳优化思路是使用 JPEG DCT 缩减解码减少完整解码开销，并结合帧级并行提高整体吞吐量。

T4 补充实验表明，GPU 在纯 resize kernel 上优势明显，但在包含解码、数据传输和 pipeline 同步的端到端任务中，speedup ratio 不一定更高。对于本实验的 MJPEG 输入，T4 DALI/nvJPEG 的最佳加速比为 6.30×，没有超过 M4 DCT 路径的 11.42×。因此，如果评分更重视 speedup ratio，而不是绝对 fps 或 GPU 使用率，M4 上的 DCT 缩减解码路径反而是更有利的主结果；T4 结果应作为 GPU pipeline 的补充分析，说明瓶颈已经从 resize 转移到 JPEG decode 与数据搬运。

综合实验数据，推荐方案如下：

| 使用场景 | 推荐方法 | 性能 |
|---|---|---|
| RGB 已解码图像，兼顾速度与质量 | `cv2.resize(..., INTER_LINEAR)`，4 线程 | 1133.1 fps，4.42× |
| MJPEG 输入，兼顾速度与质量 | `REDUCED_2 + INTER_LINEAR + ThreadPool(10)` | 466.8 fps，7.37×，PSNR 47.76 dB，SSIM 0.9836 |
| MJPEG 输入，追求最高双线性 1080p 输出速度 | `REDUCED_8 + INTER_LINEAR + ThreadPool(12)` | 723.0 fps，11.42× |
| 只需要低分辨率近似输出 | `REDUCED_8 decode only + ThreadPool(16)` | 1007.9 fps，15.93×，输出 480×240 |
| T4 GPU 补充，纯 resize 上限 | PyTorch GPU-only batch=4 | 2290.2 fps，19.65×，不含传输 |
| T4 GPU 补充，MJPEG 端到端 | DALI/nvJPEG mixed decode + GPU resize | 最高 6.30×，低于 M4 DCT 路径 |

因此，在必须输出 1920×1080 且需要较好画质的实时场景中，建议采用 `IMREAD_REDUCED_COLOR_2` 缩减解码、`INTER_LINEAR` 最终缩放和帧级线程池并行的组合方法。

## 7. 局限性与后续工作

本实验已经通过 5 类合成图像部分覆盖了图像内容复杂度差异，并通过线程交互 sweep 部分验证了内部线程与帧级并行之间的关系。因此，早期单一图像和单一线程组合带来的代表性限制已得到一定缓解。

不过，真实摄像头画面的纹理复杂度、运动情况、曝光噪声、JPEG 编码器实现和码率控制仍可能影响解码速度与质量指标。M4 实验使用的 OpenCV Python wheel 未提供 Metal 或 MPS 后端，因此 M4 主结论仍以 CPU/OpenCV 为主。T4 实验使用的是 Colab 虚拟化环境，其 CPU 核心数、后台负载和 DALI/nvJPEG 版本也会影响结果；特别是 T4 的 `hw_decoder_load` 等较新 GPU 架构参数不一定适用。因此，T4 结果更适合用于说明 GPU pipeline 的瓶颈，而不是替代 M4 上的 DCT 缩减解码结论。

此外，本实验主要使用吞吐量、单帧耗时、MSE、PSNR 和 SSIM 评价结果。若应用场景对主观视觉质量更敏感，可以进一步引入人工视觉对照；若应用场景面向机器视觉模型，也应测试不同缩减解码策略对模型准确率的影响。

## 8. 复现方式

常规复现实验可运行：

```bash
python3 M4benchmark.py --frames 80 --repeats 5 --warmup 8 --skip-gpu --skip-optional --quality-compare --quality-samples 5
```

若需要更稳定但耗时更长的测量，可运行：

```bash
python3 M4benchmark.py --long-run --skip-gpu --skip-optional --quality-compare --quality-samples 5
```

扩展实验的原始结果保存在 `m4_results_extended.json` 中，自动生成的结果摘要保存在 `m4_report.md` 中。

T4 CUDA 补充实验可运行：

```bash
python3 T4_cuda_benchmark.py --output t4_results.json > output.md
```

单独 MJPEG sweep 可运行：

```bash
python3 mjpeg_benchmark.py --output mjpeg_results.json > mjpeg_output.md
```

若需要重点搜索更高 speedup ratio，可使用新增的 DCT 与 DALI sweep：

```bash
python3 mjpeg_benchmark.py \
  --dali-batches 8,16,32,64 \
  --dali-prefetch 2,3,4 \
  --output mjpeg_dct_dali_results.json > mjpeg_dct_dali_output.md
```

其中 `mjpeg_benchmark.py` 会测试 `REDUCED_COLOR_2/4/8`、decode-only、`INTER_LINEAR`、`INTER_NEAREST`、ThreadPool，以及 DALI batch size 和 `prefetch_queue_depth` 组合。
