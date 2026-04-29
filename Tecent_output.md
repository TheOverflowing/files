ubuntu@VM-0-13-ubuntu:~/temp/test1/files$ python3 Tecent_benchmark.py
/home/ubuntu/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
========================================================================================
Tencent Cloud CUDA resize benchmark
========================================================================================
gpu               : None
vram_gb           : None
nvidia_driver_version: None
nvidia_smi_cuda_version: None
torch_cuda_runtime: 13.0
torch_cuda_available: False
torch_cuda_error  : torch.cuda.is_available() is False
cupy_cuda_available: True
cupy_cuda_error   : None
torch_version     : 2.11.0+cu130
cupy_version      : 14.0.1
opencv_version    : 4.13.0
dali_available    : True
cpu_cores         : 28
python_version    : 3.10.12
platform          : Linux-5.15.0-126-generic-x86_64-with-glibc2.35
note              : Tencent Cloud GPU server: expected A10-class GPU, 24 GB VRAM, 28 CPU cores, 116 GB RAM, 30+ TFLOPS SP.
nvidia_smi_error  : Command '['nvidia-smi', '--query-gpu=name,memory.total,driver_version,cuda_version', '--format=csv,noheader,nounits']' returned non-zero exit status 2.
WARNING: PyTorch CUDA benchmarks will be skipped: torch.cuda.is_available() is False

Config: frames=200, mjpeg_frames=80, repeats=3, warmup=20, quick=False
Workload: 3840x1920 RGB -> 1920x1080, JPEG quality=85

=== Part 1: Pure RGB resize ===
  OpenCV resize CPU threads=1                               222.15 fps     4.501 ms/frame
  OpenCV resize CPU threads=4                               438.36 fps     2.281 ms/frame
  CuPy cupyx.scipy.ndimage.zoom GPU-only                    522.10 fps     1.915 ms/frame
  CuPy cupyx.scipy.ndimage.zoom end-to-end                  246.26 fps     4.061 ms/frame
  NVIDIA DALI resize GPU pipeline end-to-end                356.04 fps     2.809 ms/frame

=== Part 2: MJPEG DCT reduced decode + ThreadPool sweep ===

-- low_texture (113.8 KiB JPEG) --
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=1    103.62 fps     9.650 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=2    195.48 fps     5.116 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=4    286.82 fps     3.487 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=8    522.56 fps     1.914 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=12    654.53 fps     1.528 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=16    790.02 fps     1.266 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=20    852.37 fps     1.173 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=24    830.75 fps     1.204 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=28    842.48 fps     1.187 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=32    826.31 fps     1.210 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=40    778.64 fps     1.284 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=48    798.99 fps     1.252 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=4    328.46 fps     3.045 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=8    578.44 fps     1.729 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=12    738.21 fps     1.355 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=16    787.01 fps     1.271 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=20    852.70 fps     1.173 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=24    839.39 fps     1.191 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=4    281.28 fps     3.555 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=6    420.13 fps     2.380 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=8    509.42 fps     1.963 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=12    738.82 fps     1.354 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=1     70.59 fps    14.166 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=2    132.39 fps     7.554 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=4    196.13 fps     5.099 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=8    353.42 fps     2.830 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=12    535.59 fps     1.867 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=16    573.18 fps     1.745 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=20    588.52 fps     1.699 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=24    609.68 fps     1.640 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=28    601.25 fps     1.663 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=32    612.90 fps     1.632 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=40    588.24 fps     1.700 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=48    616.36 fps     1.622 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=4    232.69 fps     4.298 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=8    396.46 fps     2.522 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=12    518.86 fps     1.927 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=16    596.15 fps     1.677 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=20    631.12 fps     1.584 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=24    622.21 fps     1.607 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=4    276.13 fps     3.621 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=6    323.01 fps     3.096 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=8    471.97 fps     2.119 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=12    541.55 fps     1.847 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=1     83.98 fps    11.908 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=2    162.47 fps     6.155 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=4    251.72 fps     3.973 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=8    424.06 fps     2.358 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=12    582.46 fps     1.717 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=16    678.17 fps     1.475 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=20    691.05 fps     1.447 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=24    683.48 fps     1.463 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=28    667.80 fps     1.497 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=32    691.64 fps     1.446 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=40    690.97 fps     1.447 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=48    661.58 fps     1.512 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=4    299.25 fps     3.342 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=8    458.27 fps     2.182 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=12    629.76 fps     1.588 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=16    695.45 fps     1.438 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=20    663.54 fps     1.507 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=24    674.35 fps     1.483 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=4    291.73 fps     3.428 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=6    366.54 fps     2.728 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=8    501.98 fps     1.992 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=12    634.48 fps     1.576 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=1    214.97 fps     4.652 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=2    421.56 fps     2.372 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=4    662.91 fps     1.509 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=8   1211.97 fps     0.825 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=12   1660.45 fps     0.602 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=16   1994.02 fps     0.502 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=20   2255.47 fps     0.443 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=24   2516.33 fps     0.397 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=28   2545.86 fps     0.393 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=32   2457.16 fps     0.407 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=40   2330.48 fps     0.429 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=48   2303.47 fps     0.434 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=4    713.44 fps     1.402 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=8   1211.93 fps     0.825 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=12   1582.67 fps     0.632 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=16   2111.55 fps     0.474 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=20   2204.51 fps     0.454 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=24   2451.85 fps     0.408 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=4    816.24 fps     1.225 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=6    994.84 fps     1.005 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=8   1284.51 fps     0.779 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=12   1835.79 fps     0.545 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=1    125.14 fps     7.991 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=2    234.75 fps     4.260 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=4    325.87 fps     3.069 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=8    782.79 fps     1.277 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1001.23 fps     0.999 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1173.90 fps     0.852 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=20   1356.93 fps     0.737 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=24   1474.70 fps     0.678 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=28   1464.74 fps     0.683 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=32   1437.48 fps     0.696 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=40   1380.03 fps     0.725 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=48   1352.82 fps     0.739 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=4    407.69 fps     2.453 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=8    759.42 fps     1.317 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=12   1033.38 fps     0.968 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1205.54 fps     0.830 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=20   1329.63 fps     0.752 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=24   1458.35 fps     0.686 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=4    483.80 fps     2.067 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=6    685.42 fps     1.459 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=8    828.21 fps     1.207 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1122.26 fps     0.891 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=1    146.12 fps     6.844 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=2    288.79 fps     3.463 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=4    457.87 fps     2.184 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=8    828.92 fps     1.206 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1167.28 fps     0.857 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1401.18 fps     0.714 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1604.75 fps     0.623 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=24   1638.97 fps     0.610 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=28   1644.32 fps     0.608 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=32   1556.66 fps     0.642 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1510.08 fps     0.662 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=48   1545.81 fps     0.647 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=4    501.02 fps     1.996 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=8    921.14 fps     1.086 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1275.77 fps     0.784 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1420.52 fps     0.704 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=20   1724.89 fps     0.580 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=24   1687.28 fps     0.593 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=4    530.28 fps     1.886 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=6    795.40 fps     1.257 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=8    975.82 fps     1.025 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1216.94 fps     0.822 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=1    294.33 fps     3.398 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=2    567.77 fps     1.761 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=4    822.70 fps     1.216 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=8   1569.37 fps     0.637 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=12   2478.79 fps     0.403 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=16   2708.25 fps     0.369 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=20   3083.31 fps     0.324 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=24   3427.07 fps     0.292 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=28   3621.51 fps     0.276 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=32   3368.66 fps     0.297 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=40   3027.41 fps     0.330 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=48   3200.76 fps     0.312 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=4   1157.24 fps     0.864 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=8   1833.90 fps     0.545 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=12   2352.43 fps     0.425 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=16   2945.48 fps     0.340 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=20   3309.21 fps     0.302 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=24   3480.32 fps     0.287 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=4    814.58 fps     1.228 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=6   1189.43 fps     0.841 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=8   1680.63 fps     0.595 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=12   2091.92 fps     0.478 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=1    177.88 fps     5.622 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=2    349.13 fps     2.864 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=4    532.09 fps     1.879 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=8   1028.76 fps     0.972 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1375.30 fps     0.727 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1693.59 fps     0.590 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=20   2002.45 fps     0.499 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=24   2208.30 fps     0.453 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=28   2254.89 fps     0.443 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=32   2139.90 fps     0.467 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=40   2015.63 fps     0.496 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=48   2054.43 fps     0.487 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=4    751.40 fps     1.331 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=8   1112.52 fps     0.899 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=12   1504.11 fps     0.665 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1749.59 fps     0.572 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=20   1893.59 fps     0.528 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=24   2202.77 fps     0.454 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=4    737.57 fps     1.356 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=6   1019.06 fps     0.981 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=8   1184.84 fps     0.844 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1556.96 fps     0.642 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=1    180.40 fps     5.543 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=2    354.46 fps     2.821 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=4    484.98 fps     2.062 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=8   1103.08 fps     0.907 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1343.85 fps     0.744 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1728.00 fps     0.579 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1909.50 fps     0.524 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=24   2130.11 fps     0.469 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=28   2223.14 fps     0.450 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=32   2145.66 fps     0.466 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1932.47 fps     0.517 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=48   2049.76 fps     0.488 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=4    539.85 fps     1.852 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=8   1159.73 fps     0.862 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1332.28 fps     0.751 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1653.94 fps     0.605 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=20   2015.35 fps     0.496 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=24   2164.83 fps     0.462 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=4    602.38 fps     1.660 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=6    998.32 fps     1.002 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=8   1318.84 fps     0.758 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1551.56 fps     0.645 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=1    398.96 fps     2.507 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=2    783.83 fps     1.276 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=4   1432.93 fps     0.698 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=8   1979.81 fps     0.505 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=12   2876.44 fps     0.348 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=16   3813.33 fps     0.262 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20   4222.63 fps     0.237 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24   4501.02 fps     0.222 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28   4588.38 fps     0.218 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32   4181.22 fps     0.239 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40   4225.93 fps     0.237 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48   4075.60 fps     0.245 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=4   1402.25 fps     0.713 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=8   2279.81 fps     0.439 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=12   3516.95 fps     0.284 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=16   3884.42 fps     0.257 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=20   4133.91 fps     0.242 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24   3349.12 fps     0.299 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=4   1418.26 fps     0.705 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=6   2005.03 fps     0.499 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=8   2466.86 fps     0.405 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=12   3332.78 fps     0.300 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=1    231.43 fps     4.321 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=2    434.93 fps     2.299 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=4    616.55 fps     1.622 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=8   1042.61 fps     0.959 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1681.09 fps     0.595 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=16   2324.16 fps     0.430 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=20   2547.37 fps     0.393 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=24   2756.72 fps     0.363 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=28   2883.95 fps     0.347 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=32   2628.70 fps     0.380 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=40   2669.64 fps     0.375 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=48   2552.95 fps     0.392 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=4    854.82 fps     1.170 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=8   1441.80 fps     0.694 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=12   2022.47 fps     0.494 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=16   2381.92 fps     0.420 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=20   2578.28 fps     0.388 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=24   2943.75 fps     0.340 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=4    758.40 fps     1.319 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=6   1267.04 fps     0.789 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=8   1328.00 fps     0.753 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1975.87 fps     0.506 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=1    209.04 fps     4.784 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=2    421.85 fps     2.370 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=4    836.84 fps     1.195 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=8   1048.06 fps     0.954 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1655.99 fps     0.604 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1854.07 fps     0.539 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=20   2232.26 fps     0.448 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=24   2508.92 fps     0.399 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=28   2514.43 fps     0.398 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=32   2347.65 fps     0.426 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=40   2383.56 fps     0.420 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=48   2288.37 fps     0.437 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=4    805.56 fps     1.241 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=8   1247.33 fps     0.802 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1785.06 fps     0.560 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=16   2037.61 fps     0.491 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=20   2216.18 fps     0.451 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=24   2573.13 fps     0.389 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=4    819.40 fps     1.220 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=6   1183.18 fps     0.845 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=8   1350.36 fps     0.741 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1745.48 fps     0.573 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=False   1311.58 fps     0.762 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=True   1250.55 fps     0.800 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=False   1176.46 fps     0.850 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=True   1309.28 fps     0.764 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=False    840.08 fps     1.190 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=True   1001.80 fps     0.998 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=False   1083.51 fps     0.923 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=True    945.57 fps     1.058 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=False   1196.18 fps     0.836 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=True   1083.79 fps     0.923 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=False   1195.14 fps     0.837 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=True   1093.69 fps     0.914 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=False    967.98 fps     1.033 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=True    833.42 fps     1.200 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=False    941.98 fps     1.062 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=True    944.80 fps     1.058 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=False    945.53 fps     1.058 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=True    945.11 fps     1.058 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=False    932.80 fps     1.072 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=True    941.06 fps     1.063 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=False    939.42 fps     1.064 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=True    876.62 fps     1.141 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=False    940.47 fps     1.063 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=True    936.78 fps     1.067 ms/frame

-- high_frequency (5596.7 KiB JPEG) --
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=1     16.11 fps    62.068 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=2     32.05 fps    31.201 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=4     55.01 fps    18.179 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=8     99.50 fps    10.050 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=12    147.62 fps     6.774 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=16    181.76 fps     5.502 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=20    217.45 fps     4.599 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=24    242.33 fps     4.127 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=28    275.64 fps     3.628 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=32    279.05 fps     3.584 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=40    289.82 fps     3.450 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=48    273.79 fps     3.652 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=4     54.67 fps    18.292 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=8     99.76 fps    10.024 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=12    148.25 fps     6.746 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=16    178.05 fps     5.616 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=20    215.24 fps     4.646 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=24    241.57 fps     4.140 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=4     55.30 fps    18.082 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=6     78.43 fps    12.750 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=8    105.69 fps     9.462 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=12    143.58 fps     6.965 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=1     15.01 fps    66.624 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=2     29.98 fps    33.356 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=4     51.58 fps    19.386 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=8    100.10 fps     9.990 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=12    125.44 fps     7.972 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=16    163.33 fps     6.123 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=20    198.19 fps     5.046 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=24    213.28 fps     4.689 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=28    243.66 fps     4.104 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=32    249.62 fps     4.006 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=40    261.34 fps     3.826 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=48    251.42 fps     3.977 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=4     53.18 fps    18.803 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=8     95.99 fps    10.418 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=12    140.59 fps     7.113 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=16    168.42 fps     5.937 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=20    207.07 fps     4.829 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=24    218.68 fps     4.573 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=4     53.10 fps    18.833 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=6     77.18 fps    12.957 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=8    106.53 fps     9.387 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=12    145.44 fps     6.876 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=1     15.49 fps    64.572 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=2     30.93 fps    32.330 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=4     49.25 fps    20.304 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=8     98.62 fps    10.139 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=12    142.45 fps     7.020 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=16    172.64 fps     5.793 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=20    206.98 fps     4.831 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=24    226.55 fps     4.414 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=28    253.60 fps     3.943 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=32    260.58 fps     3.838 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=40    268.16 fps     3.729 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=48    270.81 fps     3.693 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=4     46.52 fps    21.496 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=8     96.52 fps    10.360 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=12    147.89 fps     6.762 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=16    175.57 fps     5.696 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=20    213.67 fps     4.680 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=24    236.12 fps     4.235 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=4     52.07 fps    19.206 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=6     80.82 fps    12.373 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=8    100.63 fps     9.937 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=12    143.72 fps     6.958 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=1     17.70 fps    56.488 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=2     35.26 fps    28.364 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=4     60.39 fps    16.559 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=8    100.48 fps     9.952 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=12    171.98 fps     5.814 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=16    202.10 fps     4.948 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=20    249.47 fps     4.009 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=24    267.18 fps     3.743 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=28    331.99 fps     3.012 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=32    331.06 fps     3.021 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=40    331.04 fps     3.021 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=48    322.72 fps     3.099 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=4     60.36 fps    16.567 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=8    109.21 fps     9.157 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=12    163.09 fps     6.132 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=16    202.07 fps     4.949 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=20    251.82 fps     3.971 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=24    275.38 fps     3.631 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=4     56.65 fps    17.654 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=6     93.45 fps    10.701 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=8    120.05 fps     8.330 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=12    162.63 fps     6.149 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=1     16.73 fps    59.775 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=2     33.40 fps    29.944 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=4     46.74 fps    21.396 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=8     99.57 fps    10.043 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=12    160.15 fps     6.244 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=16    185.67 fps     5.386 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=20    230.36 fps     4.341 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=24    251.15 fps     3.982 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=28    302.79 fps     3.303 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=32    300.25 fps     3.331 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=40    298.85 fps     3.346 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=48    294.16 fps     3.400 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=4     55.36 fps    18.064 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=8    105.53 fps     9.476 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=12    151.02 fps     6.622 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=16    195.03 fps     5.127 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=20    232.46 fps     4.302 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=24    257.63 fps     3.882 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=4     59.68 fps    16.756 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=6     83.30 fps    12.005 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=8    107.95 fps     9.263 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=12    156.61 fps     6.385 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=1     17.07 fps    58.572 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=2     33.74 fps    29.638 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=4     57.60 fps    17.360 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=8    114.63 fps     8.724 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=12    161.22 fps     6.203 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=16    190.18 fps     5.258 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=20    236.17 fps     4.234 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=24    247.18 fps     4.046 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=28    300.90 fps     3.323 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=32    310.03 fps     3.226 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=40    312.21 fps     3.203 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=48    306.22 fps     3.266 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=4     59.72 fps    16.745 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=8    108.97 fps     9.177 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=12    151.21 fps     6.614 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=16    195.21 fps     5.123 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=20    238.32 fps     4.196 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=24    265.79 fps     3.762 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=4     60.37 fps    16.565 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=6     86.85 fps    11.514 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=8    108.40 fps     9.225 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=12    160.90 fps     6.215 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=1     18.62 fps    53.715 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=2     35.69 fps    28.019 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=4     57.17 fps    17.491 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=8    119.03 fps     8.401 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=12    175.31 fps     5.704 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=16    211.83 fps     4.721 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=20    266.60 fps     3.751 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=24    285.10 fps     3.508 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=28    344.06 fps     2.907 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=32    340.53 fps     2.937 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=40    340.48 fps     2.937 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=48    340.46 fps     2.937 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=4     64.17 fps    15.583 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=8    133.28 fps     7.503 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=12    174.79 fps     5.721 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=16    218.77 fps     4.571 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=20    263.95 fps     3.789 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=24    286.74 fps     3.487 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=4     63.86 fps    15.659 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=6     91.23 fps    10.962 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=8    117.82 fps     8.487 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=12    176.66 fps     5.661 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=1     17.85 fps    56.007 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=2     35.00 fps    28.575 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=4     62.04 fps    16.118 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=8    115.74 fps     8.640 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=12    158.76 fps     6.299 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=16    206.96 fps     4.832 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=20    249.24 fps     4.012 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=24    266.83 fps     3.748 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=28    330.47 fps     3.026 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=32    319.63 fps     3.129 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=40    316.81 fps     3.156 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=48    315.99 fps     3.165 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=4     55.41 fps    18.046 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=8    130.97 fps     7.635 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=12    170.95 fps     5.850 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=16    202.30 fps     4.943 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=20    251.76 fps     3.972 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=24    276.75 fps     3.613 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=4     67.55 fps    14.805 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=6     93.64 fps    10.679 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=8    124.06 fps     8.060 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=12    171.94 fps     5.816 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=1     17.84 fps    56.061 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=2     35.73 fps    27.990 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=4     60.31 fps    16.581 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=8    111.60 fps     8.960 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=12    175.80 fps     5.688 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=16    199.77 fps     5.006 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=20    249.01 fps     4.016 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=24    272.15 fps     3.674 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=28    328.10 fps     3.048 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=32    319.13 fps     3.133 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=40    323.28 fps     3.093 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=48    317.19 fps     3.153 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=4     57.73 fps    17.322 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=8    115.73 fps     8.641 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=12    167.47 fps     5.971 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=16    208.27 fps     4.801 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=20    255.44 fps     3.915 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=24    270.68 fps     3.694 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=4     68.65 fps    14.568 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=6     88.77 fps    11.265 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=8    121.94 fps     8.201 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=12    175.70 fps     5.692 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=1     21.25 fps    47.051 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=2     42.32 fps    23.629 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=4     74.09 fps    13.497 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=8    150.02 fps     6.666 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=12    208.21 fps     4.803 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=16    272.46 fps     3.670 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20    338.29 fps     2.956 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24    357.83 fps     2.795 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28    445.91 fps     2.243 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32    422.78 fps     2.365 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40    428.05 fps     2.336 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48    424.27 fps     2.357 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=4     74.50 fps    13.422 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=8    144.36 fps     6.927 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=12    202.00 fps     4.951 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=16    271.90 fps     3.678 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=20    337.06 fps     2.967 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24    353.96 fps     2.825 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=4     73.51 fps    13.604 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=6    109.40 fps     9.141 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=8    141.99 fps     7.043 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=12    200.17 fps     4.996 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=1     20.50 fps    48.774 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=2     40.81 fps    24.501 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=4     73.81 fps    13.549 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=8    137.39 fps     7.278 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=12    197.46 fps     5.064 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=16    258.05 fps     3.875 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=20    319.25 fps     3.132 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=24    342.12 fps     2.923 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=28    423.89 fps     2.359 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=32    424.25 fps     2.357 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=40    403.51 fps     2.478 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=48    394.48 fps     2.535 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=4     75.16 fps    13.305 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=8    144.74 fps     6.909 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=12    210.59 fps     4.749 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=16    264.16 fps     3.786 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=20    323.05 fps     3.095 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=24    346.39 fps     2.887 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=4     73.13 fps    13.675 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=6    111.36 fps     8.980 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=8    143.18 fps     6.984 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=12    205.79 fps     4.859 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=1     20.32 fps    49.205 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=2     40.63 fps    24.612 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=4     64.35 fps    15.540 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=8    140.96 fps     7.094 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=12    182.25 fps     5.487 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=16    254.83 fps     3.924 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=20    313.39 fps     3.191 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=24    323.06 fps     3.095 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=28    416.55 fps     2.401 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=32    409.54 fps     2.442 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=40    407.66 fps     2.453 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=48    389.89 fps     2.565 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=4     74.23 fps    13.471 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=8    137.55 fps     7.270 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=12    197.09 fps     5.074 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=16    257.47 fps     3.884 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=20    317.98 fps     3.145 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=24    336.04 fps     2.976 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=4     71.07 fps    14.071 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=6    114.23 fps     8.754 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=8    148.46 fps     6.736 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=12    206.63 fps     4.840 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=False    156.65 fps     6.384 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=True    156.71 fps     6.381 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=False    156.35 fps     6.396 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=True    155.99 fps     6.411 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=False    156.50 fps     6.390 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=True    157.19 fps     6.362 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=False    157.12 fps     6.365 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=True    157.02 fps     6.369 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=False    156.94 fps     6.372 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=True    157.09 fps     6.366 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=False    157.30 fps     6.357 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=True    157.48 fps     6.350 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=False    192.74 fps     5.188 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=True    191.18 fps     5.231 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=False    192.57 fps     5.193 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=True    190.87 fps     5.239 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=False    189.02 fps     5.290 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=True    192.40 fps     5.198 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=False    190.88 fps     5.239 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=True    191.43 fps     5.224 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=False    192.88 fps     5.184 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=True    192.88 fps     5.185 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=False    186.64 fps     5.358 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=True    193.20 fps     5.176 ms/frame

-- natural_like (172.4 KiB JPEG) --
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=1     83.74 fps    11.941 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=2    163.55 fps     6.114 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=4    245.97 fps     4.066 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=8    427.76 fps     2.338 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=12    580.23 fps     1.723 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=16    728.59 fps     1.373 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=20    781.84 fps     1.279 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=24    785.85 fps     1.273 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=28    790.63 fps     1.265 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=32    837.78 fps     1.194 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=40    813.46 fps     1.229 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=48    818.13 fps     1.222 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=4    287.22 fps     3.482 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=8    478.72 fps     2.089 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=12    623.32 fps     1.604 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=16    696.57 fps     1.436 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=20    749.12 fps     1.335 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=24    805.60 fps     1.241 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=4    278.64 fps     3.589 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=6    414.86 fps     2.410 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=8    456.87 fps     2.189 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=12    576.32 fps     1.735 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=1     60.62 fps    16.495 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=2    118.61 fps     8.431 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=4    198.83 fps     5.029 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=8    373.30 fps     2.679 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=12    437.31 fps     2.287 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=16    505.95 fps     1.976 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=20    537.97 fps     1.859 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=24    578.50 fps     1.729 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=28    561.91 fps     1.780 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=32    594.31 fps     1.683 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=40    580.57 fps     1.722 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=48    554.60 fps     1.803 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=4    183.66 fps     5.445 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=8    332.48 fps     3.008 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=12    478.42 fps     2.090 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=16    580.28 fps     1.723 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=20    575.71 fps     1.737 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=24    594.70 fps     1.682 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=4    218.83 fps     4.570 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=6    316.56 fps     3.159 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=8    370.20 fps     2.701 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=12    520.13 fps     1.923 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=1     70.24 fps    14.237 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=2    137.43 fps     7.277 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=4    225.95 fps     4.426 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=8    367.51 fps     2.721 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=12    511.25 fps     1.956 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=16    593.23 fps     1.686 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=20    634.74 fps     1.575 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=24    638.50 fps     1.566 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=28    620.74 fps     1.611 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=32    646.73 fps     1.546 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=40    648.33 fps     1.542 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=48    638.52 fps     1.566 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=4    229.92 fps     4.349 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=8    395.07 fps     2.531 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=12    582.17 fps     1.718 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=16    609.85 fps     1.640 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=20    637.79 fps     1.568 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=24    640.28 fps     1.562 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=4    242.12 fps     4.130 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=6    359.26 fps     2.783 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=8    428.33 fps     2.335 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=12    573.76 fps     1.743 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=1    147.84 fps     6.764 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=2    282.87 fps     3.535 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=4    480.14 fps     2.083 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=8    873.07 fps     1.145 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=12   1318.11 fps     0.759 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=16   1494.84 fps     0.669 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=20   1728.71 fps     0.578 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=24   1955.49 fps     0.511 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=28   2113.99 fps     0.473 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=32   1887.90 fps     0.530 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=40   1991.43 fps     0.502 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=48   1778.74 fps     0.562 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=4    540.73 fps     1.849 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=8    964.46 fps     1.037 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=12   1376.26 fps     0.727 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=16   1703.84 fps     0.587 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=20   1755.58 fps     0.570 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=24   1950.94 fps     0.513 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=4    533.51 fps     1.874 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=6    682.22 fps     1.466 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=8    962.99 fps     1.038 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=12   1392.43 fps     0.718 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=1     97.41 fps    10.265 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=2    194.34 fps     5.146 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=4    285.41 fps     3.504 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=8    578.97 fps     1.727 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=12    804.03 fps     1.244 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1050.31 fps     0.952 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=20   1116.97 fps     0.895 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=24   1305.09 fps     0.766 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=28   1298.61 fps     0.770 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=32   1274.34 fps     0.785 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=40   1253.63 fps     0.798 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=48   1264.69 fps     0.791 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=4    350.89 fps     2.850 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=8    644.91 fps     1.551 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=12    909.13 fps     1.100 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1118.17 fps     0.894 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=20   1183.06 fps     0.845 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=24   1282.79 fps     0.780 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=4    358.34 fps     2.791 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=6    543.07 fps     1.841 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=8    654.69 fps     1.527 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=12    885.32 fps     1.130 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=1    106.36 fps     9.402 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=2    220.91 fps     4.527 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=4    402.42 fps     2.485 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=8    679.80 fps     1.471 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=12    923.10 fps     1.083 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1103.45 fps     0.906 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1218.68 fps     0.821 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=24   1357.64 fps     0.737 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=28   1413.48 fps     0.707 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=32   1389.27 fps     0.720 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1312.76 fps     0.762 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=48   1363.73 fps     0.733 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=4    405.27 fps     2.468 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=8    783.96 fps     1.276 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=12    886.45 fps     1.128 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1206.19 fps     0.829 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=20   1308.86 fps     0.764 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=24   1433.62 fps     0.698 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=4    408.97 fps     2.445 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=6    593.81 fps     1.684 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=8    751.61 fps     1.330 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=12    999.11 fps     1.001 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=1    197.17 fps     5.072 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=2    386.81 fps     2.585 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=4    637.52 fps     1.569 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=8   1222.72 fps     0.818 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=12   1756.36 fps     0.569 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=16   2034.41 fps     0.492 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=20   2353.41 fps     0.425 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=24   2594.25 fps     0.385 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=28   2823.04 fps     0.354 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=32   2671.86 fps     0.374 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=40   2552.04 fps     0.392 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=48   2493.12 fps     0.401 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=4    650.97 fps     1.536 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=8   1073.15 fps     0.932 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=12   1623.62 fps     0.616 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=16   2145.07 fps     0.466 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=20   2319.46 fps     0.431 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=24   2607.00 fps     0.384 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=4    677.23 fps     1.477 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=6    914.75 fps     1.093 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=8   1230.19 fps     0.813 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=12   1600.82 fps     0.625 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=1    135.49 fps     7.381 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=2    256.00 fps     3.906 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=4    456.21 fps     2.192 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=8   1052.64 fps     0.950 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1198.20 fps     0.835 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1323.71 fps     0.755 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=20   1553.20 fps     0.644 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=24   1826.92 fps     0.547 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=28   1913.08 fps     0.523 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=32   1787.81 fps     0.559 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=40   1772.67 fps     0.564 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=48   1749.11 fps     0.572 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=4    530.46 fps     1.885 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=8    870.03 fps     1.149 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=12   1293.24 fps     0.773 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1497.45 fps     0.668 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=20   1569.99 fps     0.637 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=24   1805.16 fps     0.554 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=4    587.14 fps     1.703 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=6    735.00 fps     1.361 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=8   1100.44 fps     0.909 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1186.12 fps     0.843 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=1    137.37 fps     7.280 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=2    263.93 fps     3.789 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=4    418.94 fps     2.387 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=8    834.06 fps     1.199 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1225.92 fps     0.816 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1475.59 fps     0.678 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1518.73 fps     0.658 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=24   1837.15 fps     0.544 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=28   1872.21 fps     0.534 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=32   1809.35 fps     0.553 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1829.67 fps     0.547 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=48   1684.66 fps     0.594 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=4    438.56 fps     2.280 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=8    868.31 fps     1.152 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1249.03 fps     0.801 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1487.62 fps     0.672 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=20   1607.60 fps     0.622 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=24   1820.01 fps     0.549 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=4    564.66 fps     1.771 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=6    676.02 fps     1.479 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=8    856.35 fps     1.168 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1276.76 fps     0.783 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=1    251.13 fps     3.982 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=2    507.36 fps     1.971 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=4    996.29 fps     1.004 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=8   1708.34 fps     0.585 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=12   2369.57 fps     0.422 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=16   2662.52 fps     0.376 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20   3144.26 fps     0.318 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24   3441.26 fps     0.291 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28   3640.21 fps     0.275 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32   3201.12 fps     0.312 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40   3200.26 fps     0.312 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48   3259.35 fps     0.307 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=4    816.55 fps     1.225 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=8   1595.58 fps     0.627 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=12   2096.16 fps     0.477 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=16   2537.91 fps     0.394 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=20   3065.10 fps     0.326 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24   3398.31 fps     0.294 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=4    866.66 fps     1.154 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=6   1305.93 fps     0.766 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=8   1642.75 fps     0.609 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=12   2095.63 fps     0.477 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=1    175.74 fps     5.690 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=2    330.74 fps     3.024 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=4    573.74 fps     1.743 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=8   1132.86 fps     0.883 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1549.95 fps     0.645 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1906.82 fps     0.524 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=20   2055.51 fps     0.486 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=24   2370.42 fps     0.422 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=28   2543.11 fps     0.393 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=32   2428.74 fps     0.412 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=40   2213.33 fps     0.452 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=48   2217.20 fps     0.451 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=4    613.41 fps     1.630 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=8   1190.13 fps     0.840 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=12   1592.27 fps     0.628 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1781.32 fps     0.561 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=20   2113.73 fps     0.473 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=24   2392.85 fps     0.418 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=4    711.67 fps     1.405 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=6   1037.97 fps     0.963 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=8   1226.65 fps     0.815 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1705.17 fps     0.586 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=1    162.59 fps     6.150 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=2    293.53 fps     3.407 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=4    498.73 fps     2.005 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=8    999.74 fps     1.000 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1295.31 fps     0.772 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1451.01 fps     0.689 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1912.52 fps     0.523 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=24   2130.26 fps     0.469 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=28   2211.25 fps     0.452 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=32   2086.36 fps     0.479 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=40   2015.70 fps     0.496 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=48   2053.26 fps     0.487 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=4    546.96 fps     1.828 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=8    925.98 fps     1.080 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1317.19 fps     0.759 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1757.25 fps     0.569 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=20   2036.42 fps     0.491 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=24   2134.92 fps     0.468 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=4    648.78 fps     1.541 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=6    840.63 fps     1.190 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=8   1206.29 fps     0.829 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1410.10 fps     0.709 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=False    819.84 fps     1.220 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=True    723.16 fps     1.383 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=False   1025.62 fps     0.975 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=True   1052.26 fps     0.950 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=False   1053.25 fps     0.949 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=True   1051.29 fps     0.951 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=False    724.75 fps     1.380 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=True    862.66 fps     1.159 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=False    861.99 fps     1.160 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=True    867.40 fps     1.153 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=False    863.63 fps     1.158 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=True    861.65 fps     1.161 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=False    837.96 fps     1.193 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=True    833.95 fps     1.199 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=False    839.21 fps     1.192 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=True    836.35 fps     1.196 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=False    835.82 fps     1.196 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=True    836.57 fps     1.195 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=False    907.33 fps     1.102 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=True    770.59 fps     1.298 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=False    795.08 fps     1.258 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=True    791.30 fps     1.264 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=False    678.04 fps     1.475 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=True    785.12 fps     1.274 ms/frame

-- mixed (2778.5 KiB JPEG) --
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=1     28.06 fps    35.636 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=2     55.38 fps    18.058 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=4     96.92 fps    10.318 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=8    164.44 fps     6.081 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=12    250.42 fps     3.993 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=16    305.31 fps     3.275 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=20    366.80 fps     2.726 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=24    377.60 fps     2.648 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=28    434.52 fps     2.301 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=32    444.49 fps     2.250 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=40    446.04 fps     2.242 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=48    443.30 fps     2.256 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=4     84.99 fps    11.766 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=8    183.38 fps     5.453 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=12    261.38 fps     3.826 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=16    303.54 fps     3.294 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=20    368.80 fps     2.711 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=24    388.47 fps     2.574 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=4     95.55 fps    10.465 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=6    137.45 fps     7.275 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=8    162.82 fps     6.142 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=12    249.08 fps     4.015 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=1     24.89 fps    40.184 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=2     48.06 fps    20.809 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=4     78.86 fps    12.681 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=8    141.76 fps     7.054 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=12    216.11 fps     4.627 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=16    288.21 fps     3.470 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=20    309.39 fps     3.232 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=24    327.29 fps     3.055 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=28    356.70 fps     2.803 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=32    373.87 fps     2.675 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=40    386.31 fps     2.589 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=48    372.10 fps     2.687 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=4     90.04 fps    11.106 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=8    170.64 fps     5.860 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=12    236.57 fps     4.227 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=16    273.74 fps     3.653 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=20    307.89 fps     3.248 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=24    358.30 fps     2.791 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=4     83.03 fps    12.044 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=6    147.46 fps     6.781 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=8    177.84 fps     5.623 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=12    241.17 fps     4.147 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=1     26.36 fps    37.941 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=2     52.36 fps    19.098 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=4     73.10 fps    13.680 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=8    162.52 fps     6.153 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=12    230.54 fps     4.338 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=16    275.76 fps     3.626 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=20    325.87 fps     3.069 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=24    351.32 fps     2.846 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=28    371.36 fps     2.693 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=32    407.44 fps     2.454 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=40    407.55 fps     2.454 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=48    389.55 fps     2.567 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=4     95.62 fps    10.458 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=8    169.89 fps     5.886 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=12    234.39 fps     4.266 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=16    300.45 fps     3.328 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=20    339.38 fps     2.947 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=24    364.60 fps     2.743 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=4     92.94 fps    10.760 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=6    128.12 fps     7.805 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=8    170.29 fps     5.872 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=12    242.17 fps     4.129 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=1     33.18 fps    30.135 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=2     65.81 fps    15.195 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=4    112.90 fps     8.857 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=8    204.33 fps     4.894 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=12    286.11 fps     3.495 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=16    369.29 fps     2.708 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=20    464.14 fps     2.155 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=24    489.27 fps     2.044 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=28    598.86 fps     1.670 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=32    562.67 fps     1.777 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=40    563.66 fps     1.774 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=48    549.50 fps     1.820 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=4     97.38 fps    10.269 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=8    244.43 fps     4.091 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=12    313.52 fps     3.190 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=16    382.41 fps     2.615 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=20    458.35 fps     2.182 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=24    506.12 fps     1.976 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=4    101.87 fps     9.816 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=6    164.74 fps     6.070 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=8    213.84 fps     4.676 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=12    300.04 fps     3.333 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=1     29.82 fps    33.538 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=2     58.84 fps    16.995 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=4     88.79 fps    11.262 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=8    191.43 fps     5.224 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=12    257.58 fps     3.882 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=16    341.75 fps     2.926 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=20    394.32 fps     2.536 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=24    442.28 fps     2.261 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=28    515.04 fps     1.942 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=32    469.17 fps     2.131 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=40    468.89 fps     2.133 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=48    469.71 fps     2.129 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=4    119.52 fps     8.367 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=8    203.02 fps     4.926 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=12    262.68 fps     3.807 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=16    355.54 fps     2.813 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=20    409.42 fps     2.442 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=24    432.10 fps     2.314 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=4    105.30 fps     9.497 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=6    170.32 fps     5.871 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=8    205.70 fps     4.861 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=12    280.73 fps     3.562 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=1     30.92 fps    32.345 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=2     54.47 fps    18.358 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=4    101.20 fps     9.881 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=8    197.39 fps     5.066 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=12    273.10 fps     3.662 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=16    343.46 fps     2.912 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=20    415.56 fps     2.406 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=24    449.12 fps     2.227 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=28    516.14 fps     1.937 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=32    504.86 fps     1.981 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=40    497.16 fps     2.011 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=48    499.14 fps     2.003 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=4     99.16 fps    10.085 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=8    187.71 fps     5.327 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=12    280.86 fps     3.561 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=16    348.27 fps     2.871 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=20    420.77 fps     2.377 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=24    446.14 fps     2.241 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=4    105.95 fps     9.438 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=6    154.08 fps     6.490 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=8    200.66 fps     4.983 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=12    295.86 fps     3.380 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=1     35.43 fps    28.228 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=2     70.25 fps    14.234 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=4    124.04 fps     8.062 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=8    238.84 fps     4.187 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=12    303.03 fps     3.300 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=16    415.65 fps     2.406 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=20    492.26 fps     2.031 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=24    527.27 fps     1.897 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=28    639.38 fps     1.564 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=32    608.57 fps     1.643 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=40    602.81 fps     1.659 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=48    596.82 fps     1.676 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=4    123.68 fps     8.085 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=8    249.84 fps     4.003 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=12    343.80 fps     2.909 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=16    422.94 fps     2.364 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=20    493.69 fps     2.026 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=24    526.01 fps     1.901 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=4    128.05 fps     7.809 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=6    196.92 fps     5.078 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=8    234.30 fps     4.268 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=12    294.40 fps     3.397 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=1     32.79 fps    30.494 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=2     65.45 fps    15.279 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=4    106.49 fps     9.391 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=8    197.73 fps     5.057 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=12    293.15 fps     3.411 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=16    355.95 fps     2.809 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=20    443.91 fps     2.253 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=24    489.48 fps     2.043 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=28    580.03 fps     1.724 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=32    545.82 fps     1.832 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=40    547.39 fps     1.827 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=48    553.54 fps     1.807 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=4     98.65 fps    10.136 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=8    241.36 fps     4.143 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=12    337.93 fps     2.959 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=16    366.77 fps     2.727 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=20    451.07 fps     2.217 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=24    492.92 fps     2.029 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=4    118.04 fps     8.472 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=6    168.75 fps     5.926 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=8    232.06 fps     4.309 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=12    317.47 fps     3.150 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=1     32.82 fps    30.468 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=2     65.05 fps    15.373 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=4     96.57 fps    10.355 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=8    218.09 fps     4.585 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=12    289.91 fps     3.449 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=16    356.05 fps     2.809 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=20    441.02 fps     2.267 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=24    501.08 fps     1.996 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=28    580.43 fps     1.723 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=32    550.33 fps     1.817 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=40    560.40 fps     1.784 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=48    538.72 fps     1.856 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=4    110.47 fps     9.052 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=8    204.11 fps     4.899 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=12    300.33 fps     3.330 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=16    378.99 fps     2.639 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=20    445.03 fps     2.247 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=24    501.43 fps     1.994 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=4    102.21 fps     9.784 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=6    169.38 fps     5.904 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=8    220.97 fps     4.525 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=12    310.95 fps     3.216 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=1     41.31 fps    24.206 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=2     81.89 fps    12.212 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=4    128.71 fps     7.770 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=8    265.43 fps     3.767 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=12    392.85 fps     2.546 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=16    510.10 fps     1.960 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20    630.57 fps     1.586 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24    668.42 fps     1.496 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28    831.43 fps     1.203 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32    799.05 fps     1.251 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40    760.99 fps     1.314 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48    758.81 fps     1.318 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=4    135.09 fps     7.402 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=8    261.02 fps     3.831 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=12    392.08 fps     2.551 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=16    511.95 fps     1.953 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=20    627.34 fps     1.594 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24    644.55 fps     1.551 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=4    145.11 fps     6.891 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=6    209.97 fps     4.763 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=8    284.73 fps     3.512 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=12    393.24 fps     2.543 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=1     38.55 fps    25.942 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=2     77.09 fps    12.972 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=4    131.97 fps     7.578 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=8    241.29 fps     4.144 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=12    375.55 fps     2.663 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=16    457.74 fps     2.185 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=20    568.51 fps     1.759 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=24    610.66 fps     1.638 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=28    751.51 fps     1.331 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=32    703.26 fps     1.422 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=40    689.58 fps     1.450 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=48    693.38 fps     1.442 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=4    141.10 fps     7.087 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=8    257.13 fps     3.889 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=12    382.29 fps     2.616 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=16    482.55 fps     2.072 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=20    582.49 fps     1.717 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=24    625.68 fps     1.598 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=4    148.70 fps     6.725 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=6    212.49 fps     4.706 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=8    281.06 fps     3.558 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=12    366.33 fps     2.730 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=1     37.89 fps    26.394 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=2     74.87 fps    13.357 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=4    117.48 fps     8.512 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=8    258.43 fps     3.869 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=12    344.06 fps     2.906 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=16    454.39 fps     2.201 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=20    556.90 fps     1.796 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=24    591.39 fps     1.691 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=28    729.23 fps     1.371 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=32    696.49 fps     1.436 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=40    685.17 fps     1.459 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=48    663.65 fps     1.507 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=4    130.82 fps     7.644 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=8    249.07 fps     4.015 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=12    342.85 fps     2.917 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=16    470.34 fps     2.126 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=20    559.30 fps     1.788 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=24    586.39 fps     1.705 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=4    140.94 fps     7.095 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=6    206.77 fps     4.836 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=8    285.82 fps     3.499 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=12    388.69 fps     2.573 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=False    283.52 fps     3.527 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=True    276.62 fps     3.615 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=False    344.76 fps     2.901 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=True    281.53 fps     3.552 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=False    279.57 fps     3.577 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=True    280.88 fps     3.560 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=False    273.76 fps     3.653 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=True    273.95 fps     3.650 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=False    272.99 fps     3.663 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=True    273.09 fps     3.662 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=False    274.14 fps     3.648 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=True    343.91 fps     2.908 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=False    274.62 fps     3.641 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=True    282.48 fps     3.540 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=False    306.45 fps     3.263 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=True    309.77 fps     3.228 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=False    304.76 fps     3.281 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=True    310.08 fps     3.225 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=False    325.39 fps     3.073 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=True    274.93 fps     3.637 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=False    303.93 fps     3.290 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=True    298.09 fps     3.355 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=False    274.54 fps     3.643 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=True    346.96 fps     2.882 ms/frame

-- text_like (392.4 KiB JPEG) --
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=1     88.02 fps    11.361 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=2    170.01 fps     5.882 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=4    275.87 fps     3.625 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=8    502.08 fps     1.992 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=12    674.09 fps     1.483 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=16    777.25 fps     1.287 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=20    777.68 fps     1.286 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=24    817.53 fps     1.223 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=28    828.10 fps     1.208 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=32    840.49 fps     1.190 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=40    841.28 fps     1.189 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=1 workers=48    851.47 fps     1.174 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=4    242.40 fps     4.125 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=8    488.33 fps     2.048 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=12    635.66 fps     1.573 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=16    757.39 fps     1.320 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=20    762.18 fps     1.312 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=2 workers=24    812.89 fps     1.230 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=4    277.57 fps     3.603 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=6    375.36 fps     2.664 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=8    483.13 fps     2.070 ms/frame
  OpenCV IMREAD_COLOR decode only cv2_threads=4 workers=12    637.78 fps     1.568 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=1     62.95 fps    15.886 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=2    123.08 fps     8.125 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=4    161.47 fps     6.193 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=8    341.25 fps     2.930 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=12    466.52 fps     2.144 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=16    558.19 fps     1.791 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=20    557.31 fps     1.794 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=24    597.89 fps     1.673 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=28    594.25 fps     1.683 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=32    611.23 fps     1.636 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=40    612.62 fps     1.632 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=1 workers=48    623.00 fps     1.605 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=4    201.91 fps     4.953 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=8    382.31 fps     2.616 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=12    499.08 fps     2.004 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=16    571.37 fps     1.750 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=20    577.34 fps     1.732 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=2 workers=24    619.94 fps     1.613 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=4    224.54 fps     4.453 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=6    307.53 fps     3.252 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=8    405.71 fps     2.465 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_LINEAR cv2_threads=4 workers=12    510.26 fps     1.960 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=1     73.45 fps    13.615 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=2    136.63 fps     7.319 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=4    194.21 fps     5.149 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=8    438.18 fps     2.282 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=12    518.18 fps     1.930 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=16    632.43 fps     1.581 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=20    674.22 fps     1.483 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=24    649.07 fps     1.541 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=28    668.45 fps     1.496 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=32    690.23 fps     1.449 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=40    687.83 fps     1.454 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=1 workers=48    675.20 fps     1.481 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=4    222.04 fps     4.504 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=8    436.89 fps     2.289 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=12    542.04 fps     1.845 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=16    655.54 fps     1.525 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=20    669.07 fps     1.495 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=2 workers=24    682.26 fps     1.466 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=4    241.68 fps     4.138 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=6    336.75 fps     2.970 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=8    445.09 fps     2.247 ms/frame
  OpenCV IMREAD_COLOR decode + resize INTER_NEAREST cv2_threads=4 workers=12    581.50 fps     1.720 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=1    160.36 fps     6.236 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=2    302.38 fps     3.307 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=4    461.02 fps     2.169 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=8    876.13 fps     1.141 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=12   1164.87 fps     0.858 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=16   1480.48 fps     0.675 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=20   1802.64 fps     0.555 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=24   2036.71 fps     0.491 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=28   2180.25 fps     0.459 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=32   2071.59 fps     0.483 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=40   1952.08 fps     0.512 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=1 workers=48   2030.57 fps     0.492 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=4    510.18 fps     1.960 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=8    952.30 fps     1.050 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=12   1321.10 fps     0.757 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=16   1632.12 fps     0.613 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=20   1780.65 fps     0.562 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=2 workers=24   2003.92 fps     0.499 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=4    551.73 fps     1.812 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=6    699.92 fps     1.429 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=8   1053.86 fps     0.949 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode only cv2_threads=4 workers=12   1287.95 fps     0.776 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=1    103.24 fps     9.686 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=2    201.86 fps     4.954 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=4    324.19 fps     3.085 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=8    584.87 fps     1.710 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=12    740.93 fps     1.350 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1005.03 fps     0.995 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=20   1198.88 fps     0.834 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=24   1300.44 fps     0.769 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=28   1346.81 fps     0.742 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=32   1285.13 fps     0.778 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=40   1054.25 fps     0.949 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=1 workers=48   1276.51 fps     0.783 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=4    341.63 fps     2.927 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=8    749.42 fps     1.334 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=12    922.39 fps     1.084 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1093.09 fps     0.915 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=20   1144.81 fps     0.874 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=2 workers=24   1296.64 fps     0.771 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=4    408.93 fps     2.445 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=6    526.74 fps     1.898 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=8    656.68 fps     1.523 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_LINEAR cv2_threads=4 workers=12    958.94 fps     1.043 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=1    115.40 fps     8.665 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=2    225.14 fps     4.442 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=4    372.25 fps     2.686 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=8    656.51 fps     1.523 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=12    896.01 fps     1.116 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1195.90 fps     0.836 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1244.85 fps     0.803 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=24   1467.97 fps     0.681 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=28   1543.25 fps     0.648 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=32   1434.48 fps     0.697 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1394.69 fps     0.717 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=1 workers=48   1340.57 fps     0.746 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=4    408.47 fps     2.448 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=8    659.79 fps     1.516 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1021.32 fps     0.979 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1152.51 fps     0.868 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=20   1351.20 fps     0.740 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=2 workers=24   1489.90 fps     0.671 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=4    470.41 fps     2.126 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=6    676.19 fps     1.479 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=8    814.38 fps     1.228 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_2 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1104.28 fps     0.906 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=1    203.08 fps     4.924 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=2    403.45 fps     2.479 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=4    800.77 fps     1.249 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=8   1310.57 fps     0.763 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=12   1777.64 fps     0.563 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=16   2188.31 fps     0.457 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=20   2442.74 fps     0.409 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=24   2615.15 fps     0.382 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=28   2796.81 fps     0.358 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=32   2569.18 fps     0.389 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=40   2455.45 fps     0.407 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=48   2458.69 fps     0.407 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=4    718.38 fps     1.392 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=8   1187.79 fps     0.842 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=12   1577.54 fps     0.634 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=16   2156.71 fps     0.464 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=20   2311.42 fps     0.433 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=24   2649.12 fps     0.377 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=4    634.51 fps     1.576 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=6    838.30 fps     1.193 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=8   1187.12 fps     0.842 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=4 workers=12   1570.84 fps     0.637 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=1    139.10 fps     7.189 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=2    275.62 fps     3.628 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=4    419.77 fps     2.382 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=8    830.45 fps     1.204 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1028.04 fps     0.973 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1323.38 fps     0.756 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=20   1520.62 fps     0.658 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=24   1870.45 fps     0.535 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=28   1883.27 fps     0.531 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=32   1825.94 fps     0.548 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=40   1738.14 fps     0.575 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=1 workers=48   1669.49 fps     0.599 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=4    574.52 fps     1.741 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=8    945.70 fps     1.057 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=12   1191.31 fps     0.839 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1320.38 fps     0.757 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=20   1547.95 fps     0.646 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=2 workers=24   1785.33 fps     0.560 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=4    521.03 fps     1.919 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=6    714.85 fps     1.399 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=8    875.51 fps     1.142 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1282.48 fps     0.780 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=1    136.92 fps     7.303 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=2    279.37 fps     3.579 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=4    553.71 fps     1.806 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=8    856.82 fps     1.167 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1142.07 fps     0.876 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1436.90 fps     0.696 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1576.25 fps     0.634 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=24   1826.71 fps     0.547 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=28   1892.29 fps     0.528 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=32   1769.55 fps     0.565 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1771.38 fps     0.565 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=1 workers=48   1716.20 fps     0.583 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=4    536.75 fps     1.863 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=8    858.81 fps     1.164 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1184.55 fps     0.844 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1514.19 fps     0.660 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=20   1703.18 fps     0.587 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=2 workers=24   1857.20 fps     0.538 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=4    500.05 fps     2.000 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=6    729.24 fps     1.371 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=8    924.88 fps     1.081 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_4 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1281.35 fps     0.780 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=1    259.40 fps     3.855 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=2    507.41 fps     1.971 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=4    801.60 fps     1.248 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=8   1478.96 fps     0.676 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=12   2072.02 fps     0.483 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=16   2524.86 fps     0.396 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20   3002.57 fps     0.333 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24   3387.96 fps     0.295 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28   3550.39 fps     0.282 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32   3261.18 fps     0.307 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40   3148.04 fps     0.318 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48   2970.94 fps     0.337 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=4    879.29 fps     1.137 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=8   1497.30 fps     0.668 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=12   2117.96 fps     0.472 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=16   2458.56 fps     0.407 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=20   2367.75 fps     0.422 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24   3183.82 fps     0.314 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=4    806.45 fps     1.240 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=6   1214.34 fps     0.823 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=8   1393.55 fps     0.718 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=12   2123.45 fps     0.471 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=1    179.99 fps     5.556 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=2    329.32 fps     3.037 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=4    578.54 fps     1.728 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=8    995.87 fps     1.004 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=12   1604.38 fps     0.623 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=16   1917.65 fps     0.521 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=20   1995.17 fps     0.501 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=24   2344.98 fps     0.426 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=28   2468.90 fps     0.405 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=32   2321.06 fps     0.431 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=40   2221.87 fps     0.450 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=1 workers=48   2253.38 fps     0.444 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=4    713.49 fps     1.402 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=8   1052.79 fps     0.950 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=12   1567.32 fps     0.638 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=16   1852.40 fps     0.540 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=20   2045.83 fps     0.489 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=2 workers=24   2383.97 fps     0.419 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=4    627.52 fps     1.594 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=6   1079.43 fps     0.926 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=8   1165.77 fps     0.858 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_LINEAR cv2_threads=4 workers=12   1696.46 fps     0.589 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=1    163.88 fps     6.102 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=2    314.71 fps     3.178 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=4    456.37 fps     2.191 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=8   1015.48 fps     0.985 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=12   1430.25 fps     0.699 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=16   1459.14 fps     0.685 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=20   1782.59 fps     0.561 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=24   2149.12 fps     0.465 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=28   2209.14 fps     0.453 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=32   2037.82 fps     0.491 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=40   1930.94 fps     0.518 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=1 workers=48   1988.10 fps     0.503 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=4    570.97 fps     1.751 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=8    955.54 fps     1.047 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=12   1445.15 fps     0.692 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=16   1671.39 fps     0.598 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=20   1956.04 fps     0.511 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=2 workers=24   2184.52 fps     0.458 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=4    633.18 fps     1.579 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=6    851.21 fps     1.175 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=8   1054.87 fps     0.948 ms/frame
  OpenCV IMREAD_REDUCED_COLOR_8 decode + resize INTER_NEAREST cv2_threads=4 workers=12   1361.58 fps     0.734 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=False    718.75 fps     1.391 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=2 copy_back=True    673.11 fps     1.486 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=False    644.14 fps     1.552 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=3 copy_back=True    755.50 fps     1.324 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=False    630.06 fps     1.587 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=8 prefetch=4 copy_back=True    686.73 fps     1.456 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=False    746.19 fps     1.340 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=2 copy_back=True    746.99 fps     1.339 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=False    698.30 fps     1.432 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=3 copy_back=True    744.77 fps     1.343 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=False    697.97 fps     1.433 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=16 prefetch=4 copy_back=True    744.43 fps     1.343 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=False    729.34 fps     1.371 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=2 copy_back=True    729.26 fps     1.371 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=False    729.58 fps     1.371 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=3 copy_back=True    727.45 fps     1.375 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=False    729.26 fps     1.371 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=32 prefetch=4 copy_back=True    729.63 fps     1.371 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=False    749.85 fps     1.334 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=2 copy_back=True    708.54 fps     1.411 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=False    748.84 fps     1.335 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=3 copy_back=True    815.05 fps     1.227 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=False    691.71 fps     1.446 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize batch=64 prefetch=4 copy_back=True    813.00 fps     1.230 ms/frame

=== Part 3: PyTorch GPU batch throughput scaling ===

=== Part 4: Image quality verification ===
  CuPy cupyx.scipy.ndimage.zoom order=1                  MSE=    4.0632 PSNR=  42.042 SSIM=0.950613
  NVIDIA DALI resize linear                              MSE=    2.8801 PSNR=  43.537 SSIM=0.959753

========================================================================================
Summary
========================================================================================
GPU: None  VRAM: None GB  Driver: None  CUDA driver: None  PyTorch CUDA runtime: 13.0
CPU cores visible to Python: 28  Python: 3.10.12
Note: Tencent Cloud GPU server: expected A10-class GPU, 24 GB VRAM, 28 CPU cores, 116 GB RAM, 30+ TFLOPS SP.
PyTorch CUDA skipped: torch.cuda.is_available() is False

RGB resize
  method                                      fps                 ms_per_frame        speedup             includes_transfer
  ------------------------------------------  ------------------  ------------------  ------------------  -----------------
  OpenCV resize CPU threads=1                 222.151             4.501               1.000               False
  OpenCV resize CPU threads=4                 438.364             2.281               1.973               False
  CuPy cupyx.scipy.ndimage.zoom GPU-only      522.100             1.915               2.350               False
  CuPy cupyx.scipy.ndimage.zoom end-to-end    246.263             4.061               1.109               True
  NVIDIA DALI resize GPU pipeline end-to-end  356.037             2.809               1.603               True

MJPEG decode + resize top 30
  method                                                              image_type    fps                 ms_per_frame         speedup_ratio       output_shape
  ------------------------------------------------------------------  ------------  ------------------  -------------------  ------------------  -------------
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28  low_texture   4588.383            0.218                64.998              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24  low_texture   4501.023            0.222                63.760              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40  low_texture   4225.929            0.237                59.863              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20  low_texture   4222.631            0.237                59.817              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32  low_texture   4181.215            0.239                59.230              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=20  low_texture   4133.909            0.242                58.560              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48  low_texture   4075.605            0.245                57.734              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=16  low_texture   3884.423            0.257                55.026              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=16  low_texture   3813.335            0.262                54.019              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28  natural_like  3640.207            0.275                60.046              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=28  low_texture   3621.514            0.276                51.301              [480, 960, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=28  text_like     3550.388            0.282                56.403              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=12  low_texture   3516.951            0.284                49.820              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=24  low_texture   3480.321            0.287                49.301              [480, 960, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24  natural_like  3441.259            0.291                56.764              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=24  low_texture   3427.073            0.292                48.547              [480, 960, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24  natural_like  3398.310            0.294                56.056              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=24  text_like     3387.961            0.295                53.823              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=32  low_texture   3368.665            0.297                47.720              [480, 960, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24  low_texture   3349.116            0.299                47.443              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=4 workers=12  low_texture   3332.775            0.300                47.211              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=2 workers=20  low_texture   3309.209            0.302                46.877              [480, 960, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32  text_like     3261.183            0.307                51.809              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=48  natural_like  3259.351            0.307                53.763              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=32  natural_like  3201.122            0.312                52.803              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_4 decode only cv2_threads=1 workers=48  low_texture   3200.759            0.312                45.341              [480, 960, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40  natural_like  3200.264            0.312                52.789              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=2 workers=24  text_like     3183.821            0.314                50.580              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=40  text_like     3148.041            0.318                50.011              [240, 480, 3]
  OpenCV IMREAD_REDUCED_COLOR_8 decode only cv2_threads=1 workers=20  natural_like  3144.264            0.318                51.865              [240, 480, 3]

Batch scaling
  no rows

Quality
  method                                 mse                 psnr                ssim
  -------------------------------------  ------------------  ------------------  ------------------
  CuPy cupyx.scipy.ndimage.zoom order=1  4.063               42.042              0.951
  NVIDIA DALI resize linear              2.880               43.537              0.960

Wrote tencent_results.json
Wrote tencent_results.md