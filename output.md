========================================================================================
T4 CUDA resize benchmark
========================================================================================
gpu               : Tesla T4
vram_gb           : 14.56
cuda_version      : 12.8
torch_version     : 2.10.0+cu128
cupy_version      : 14.0.1
opencv_version    : 4.13.0
dali_available    : True
cpu_cores         : 2
python_version    : 3.12.13
platform          : Linux-6.6.113+-x86_64-with-glibc2.35
note              : Colab T4 commonly exposes only 2 CPU cores, unlike a 10-core M4 Mac.
cuda_capability   : 7.5

Config: frames=200, mjpeg_frames=80, repeats=3, warmup=20, quick=False
Workload: 3840x1920 RGB -> 1920x1080, JPEG quality=85

=== Part 1: Pure RGB resize ===
  OpenCV resize CPU threads=1                               116.56 fps     8.580 ms/frame
  OpenCV resize CPU threads=4                               130.46 fps     7.665 ms/frame
  PyTorch F.interpolate GPU-only batch=1                   2215.22 fps     0.451 ms/frame
  PyTorch F.interpolate GPU-only batch=4                   2290.17 fps     0.437 ms/frame
  PyTorch F.interpolate GPU-only batch=8                   2244.82 fps     0.445 ms/frame
  PyTorch F.interpolate GPU-only batch=16                  2195.71 fps     0.455 ms/frame
  PyTorch F.interpolate GPU-only batch=32                  2122.50 fps     0.471 ms/frame
  PyTorch F.interpolate end-to-end batch=1                  110.71 fps     9.033 ms/frame
  PyTorch F.interpolate end-to-end batch=4                  112.10 fps     8.921 ms/frame
  PyTorch F.interpolate end-to-end batch=8                   83.87 fps    11.924 ms/frame
  PyTorch F.interpolate end-to-end batch=16                  82.48 fps    12.124 ms/frame
  PyTorch F.interpolate end-to-end batch=32                  84.80 fps    11.792 ms/frame
  CuPy cupyx.scipy.ndimage.zoom GPU-only                    259.54 fps     3.853 ms/frame
  CuPy cupyx.scipy.ndimage.zoom end-to-end                   90.64 fps    11.033 ms/frame
  NVIDIA DALI resize GPU pipeline end-to-end                 98.35 fps    10.168 ms/frame

=== Part 2: MJPEG decode + resize ===

-- low_texture (113.8 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                40.19 fps    24.879 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       91.62 fps    10.915 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)       59.92 fps    16.690 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)      89.82 fps    11.133 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    230.60 fps     4.336 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    233.74 fps     4.278 ms/frame

-- high_frequency (5596.7 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                 9.61 fps   104.049 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       13.58 fps    73.633 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)       14.60 fps    68.511 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)      14.46 fps    69.156 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2     43.22 fps    23.136 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4     42.95 fps    23.282 ms/frame

-- natural_like (172.4 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                32.82 fps    30.473 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       76.02 fps    13.154 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)       77.40 fps    12.920 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)      66.31 fps    15.081 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    206.17 fps     4.850 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    206.60 fps     4.840 ms/frame

-- mixed (2778.5 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                16.17 fps    61.856 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       24.11 fps    41.475 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)       24.02 fps    41.638 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)      25.36 fps    39.435 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2     74.11 fps    13.494 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4     70.47 fps    14.191 ms/frame

-- text_like (392.4 KiB JPEG) --
  OpenCV imdecode full + resize CPU threads=1                33.84 fps    29.550 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)       77.64 fps    12.879 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)       56.59 fps    17.673 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)      76.25 fps    13.115 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2    167.68 fps     5.964 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4    167.11 fps     5.984 ms/frame

=== Part 3: PyTorch GPU batch throughput scaling ===
  PyTorch F.interpolate preloaded batch=1                  2314.52 fps     0.432 ms/frame
  PyTorch F.interpolate preloaded batch=2                  2307.83 fps     0.433 ms/frame
  PyTorch F.interpolate preloaded batch=4                  2286.00 fps     0.437 ms/frame
  PyTorch F.interpolate preloaded batch=8                  2246.01 fps     0.445 ms/frame
  PyTorch F.interpolate preloaded batch=16                 2197.54 fps     0.455 ms/frame
  PyTorch F.interpolate preloaded batch=32                 2133.71 fps     0.469 ms/frame
  PyTorch F.interpolate preloaded batch=64                 2035.55 fps     0.491 ms/frame

=== Part 4: Image quality verification ===
  PyTorch F.interpolate bilinear align_corners=False     MSE=    0.3799 PSNR=  52.334 SSIM=0.996935
  CuPy cupyx.scipy.ndimage.zoom order=1                  MSE=    4.0632 PSNR=  42.042 SSIM=0.950613
  NVIDIA DALI resize linear                              MSE=    2.8801 PSNR=  43.537 SSIM=0.959753

========================================================================================
Summary
========================================================================================
GPU: Tesla T4  VRAM: 14.56 GB  CUDA: 12.8
CPU cores visible to Python: 2  Python: 3.12.13
Note: Colab T4 commonly exposes only 2 CPU cores; CPU results are not expected to match a 10-core M4.

RGB resize
  method                                      fps                 ms_per_frame         speedup             includes_transfer
  ------------------------------------------  ------------------  -------------------  ------------------  -----------------
  OpenCV resize CPU threads=1                 116.557             8.580                1.000               False            
  OpenCV resize CPU threads=4                 130.458             7.665                1.119               False            
  PyTorch F.interpolate GPU-only batch=1      2215.222            0.451                19.006              False            
  PyTorch F.interpolate GPU-only batch=4      2290.171            0.437                19.649              False            
  PyTorch F.interpolate GPU-only batch=8      2244.819            0.445                19.259              False            
  PyTorch F.interpolate GPU-only batch=16     2195.706            0.455                18.838              False            
  PyTorch F.interpolate GPU-only batch=32     2122.501            0.471                18.210              False            
  PyTorch F.interpolate end-to-end batch=1    110.705             9.033                0.950               True             
  PyTorch F.interpolate end-to-end batch=4    112.098             8.921                0.962               True             
  PyTorch F.interpolate end-to-end batch=8    83.866              11.924               0.720               True             
  PyTorch F.interpolate end-to-end batch=16   82.478              12.124               0.708               True             
  PyTorch F.interpolate end-to-end batch=32   84.801              11.792               0.728               True             
  CuPy cupyx.scipy.ndimage.zoom GPU-only      259.543             3.853                2.227               False            
  CuPy cupyx.scipy.ndimage.zoom end-to-end    90.640              11.033               0.778               True             
  NVIDIA DALI resize GPU pipeline end-to-end  98.349              10.168               0.844               True             

MJPEG decode + resize
  method                                                  image_type      fps                 ms_per_frame      
  ------------------------------------------------------  --------------  ------------------  ------------------
  OpenCV imdecode full + resize CPU threads=1             low_texture     40.194              24.879            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    low_texture     91.617              10.915            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    low_texture     59.916              16.690            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   low_texture     89.823              11.133            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  low_texture     230.601             4.336             
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  low_texture     233.741             4.278             
  OpenCV imdecode full + resize CPU threads=1             high_frequency  9.611               104.049           
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    high_frequency  13.581              73.633            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    high_frequency  14.596              68.511            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   high_frequency  14.460              69.156            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  high_frequency  43.222              23.136            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  high_frequency  42.953              23.282            
  OpenCV imdecode full + resize CPU threads=1             natural_like    32.815              30.473            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    natural_like    76.023              13.154            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    natural_like    77.401              12.920            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   natural_like    66.309              15.081            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  natural_like    206.167             4.850             
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  natural_like    206.596             4.840             
  OpenCV imdecode full + resize CPU threads=1             mixed           16.167              61.856            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    mixed           24.111              41.475            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    mixed           24.016              41.638            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   mixed           25.358              39.435            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  mixed           74.109              13.494            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  mixed           70.468              14.191            
  OpenCV imdecode full + resize CPU threads=1             text_like       33.840              29.550            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(4)    text_like       77.644              12.879            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(8)    text_like       56.585              17.673            
  OpenCV IMREAD_REDUCED_COLOR_2 + resize ThreadPool(12)   text_like       76.250              13.115            
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2  text_like       167.678             5.964             
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4  text_like       167.106             5.984             

Batch scaling
  batch_size  fps                 ms_per_frame       
  ----------  ------------------  -------------------
  1           2314.515            0.432              
  2           2307.828            0.433              
  4           2286.000            0.437              
  8           2246.009            0.445              
  16          2197.540            0.455              
  32          2133.705            0.469              
  64          2035.552            0.491              

Quality
  method                                              mse                  psnr                ssim              
  --------------------------------------------------  -------------------  ------------------  ------------------
  PyTorch F.interpolate bilinear align_corners=False  0.380                52.334              0.997             
  CuPy cupyx.scipy.ndimage.zoom order=1               4.063                42.042              0.951             
  NVIDIA DALI resize linear                           2.880                43.537              0.960             

Wrote t4_results.json
