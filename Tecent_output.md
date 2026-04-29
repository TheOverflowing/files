========================================================================================
T4 CUDA resize benchmark
========================================================================================
/home/ubuntu/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
gpu               : A10
vram_gb           : 24GB
cuda_version      : 13.0
torch_version     : 2.11.0+cu130
cupy_version      : 14.0.1
opencv_version    : 4.13.0
dali_available    : True
cpu_cores         : 28
python_version    : 3.10.12
platform          : Linux-5.15.0-126-generic-x86_64-with-glibc2.35
note              : Tencent cloud
WARNING: torch.cuda.is_available() is False. GPU benchmarks will be skipped or fail.

Config: frames=200, mjpeg_frames=80, repeats=3, warmup=20, quick=False
Workload: 3840x1920 RGB -> 1920x1080, JPEG quality=85

=== Part 1: Pure RGB resize ===
  OpenCV resize CPU threads=1                               218.84 fps     4.570 ms/frame
  OpenCV resize CPU threads=4                               469.07 fps     2.132 ms/frame
ERROR: CuPy cupyx.scipy.ndimage.zoom GPU-only failed: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
ERROR: CuPy cupyx.scipy.ndimage.zoom end-to-end failed: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
  NVIDIA DALI resize GPU pipeline end-to-end                129.17 fps     7.741 ms/frame

=== Part 2: MJPEG decode + resize ===

-- low_texture (113.8 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                70.81 fps    14.123 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)      397.63 fps     2.515 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)      648.08 fps     1.543 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)     957.78 fps     1.044 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    676.06 fps     1.479 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    848.79 fps     1.178 ms/frame

-- high_frequency (5596.7 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                15.06 fps    66.395 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       57.55 fps    17.376 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)      108.16 fps     9.246 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)     151.57 fps     6.598 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2     96.79 fps    10.332 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    192.28 fps     5.201 ms/frame

-- natural_like (172.4 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                60.00 fps    16.666 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)      339.58 fps     2.945 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)      539.08 fps     1.855 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)     741.14 fps     1.349 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    540.70 fps     1.849 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    817.84 fps     1.223 ms/frame

-- mixed (2778.5 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                24.85 fps    40.237 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       81.23 fps    12.311 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)      176.45 fps     5.667 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)     267.02 fps     3.745 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    173.31 fps     5.770 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    280.74 fps     3.562 ms/frame

-- text_like (392.4 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                62.22 fps    16.071 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)      278.62 fps     3.589 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)      548.42 fps     1.823 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)     828.26 fps     1.207 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    468.46 fps     2.135 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    630.92 fps     1.585 ms/frame

=== Part 3: PyTorch GPU batch throughput scaling ===

=== Part 4: Image quality verification ===
ERROR: CuPy cupyx.scipy.ndimage.zoom order=1 failed: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
  NVIDIA DALI resize linear                              MSE=    2.8801 PSNR=  43.537 SSIM=0.959753

========================================================================================
Summary
========================================================================================
GPU: None  VRAM: None GB  CUDA: 13.0
CPU cores visible to Python: 28  Python: 3.10.12
Note: Colab T4 commonly exposes only 2 CPU cores; CPU results are not expected to match a 10-core M4.

RGB resize
  method                                      fps                 ms_per_frame        speedup             includes_transfer
  ------------------------------------------  ------------------  ------------------  ------------------  -----------------
  OpenCV resize CPU threads=1                 218.837             4.570               1.000               False
  OpenCV resize CPU threads=4                 469.075             2.132               2.143               False
  NVIDIA DALI resize GPU pipeline end-to-end  129.174             7.741               0.590               True

MJPEG decode + resize
  method                                                  image_type      fps                 ms_per_frame
  ------------------------------------------------------  --------------  ------------------  ------------------
  OpenCV imdecode full + resize CPU threads=1             low_texture     70.806              14.123
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    low_texture     397.630             2.515
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    low_texture     648.081             1.543
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   low_texture     957.776             1.044
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  low_texture     676.058             1.479
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  low_texture     848.785             1.178
  OpenCV imdecode full + resize CPU threads=1             high_frequency  15.061              66.395
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    high_frequency  57.551              17.376
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    high_frequency  108.161             9.246
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   high_frequency  151.568             6.598
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  high_frequency  96.787              10.332
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  high_frequency  192.281             5.201
  OpenCV imdecode full + resize CPU threads=1             natural_like    60.001              16.666
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    natural_like    339.580             2.945
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    natural_like    539.082             1.855
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   natural_like    741.145             1.349
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  natural_like    540.695             1.849
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  natural_like    817.840             1.223
  OpenCV imdecode full + resize CPU threads=1             mixed           24.853              40.237
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    mixed           81.227              12.311
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    mixed           176.449             5.667
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   mixed           267.018             3.745
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  mixed           173.313             5.770
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  mixed           280.738             3.562
  OpenCV imdecode full + resize CPU threads=1             text_like       62.223              16.071
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    text_like       278.621             3.589
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    text_like       548.420             1.823
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   text_like       828.256             1.207
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  text_like       468.459             2.135
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  text_like       630.916             1.585

Batch scaling
  no rows

Quality
  method                     mse                 psnr                ssim
  -------------------------  ------------------  ------------------  ------------------
  NVIDIA DALI resize linear  2.880               43.537              0.960

Errors / skipped methods
  CuPy cupyx.scipy.ndimage.zoom GPU-only: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
  CuPy cupyx.scipy.ndimage.zoom end-to-end: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
  CuPy cupyx.scipy.ndimage.zoom order=1: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid

Wrote t4_results.json