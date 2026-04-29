WARNING: pyvips unavailable; skipping pyvips benchmarks: No module named 'pyvips'
================================================================================================
Standalone MJPEG benchmark
================================================================================================
python              : 3.12.13
platform            : Linux-6.6.113+-x86_64-with-glibc2.35
cpu_cores           : 2
opencv              : 4.13.0
pillow_available    : True
pyvips_available    : False
dali_available      : True
torch               : 2.10.0+cu128
cuda_available      : True
cuda                : 12.8
gpu                 : Tesla T4
vram_gb             : 14.56

Config: frames=80, repeats=3, warmup=8, quick=False
Workload: 3840x1920 JPEG quality=85 -> 1920x1080

-- low_texture (113.8 KiB JPEG) --
  OpenCV imdecode full only                                          52.82 fps    18.931 ms/frame
  OpenCV imdecode full + resize                                      39.73 fps    25.170 ms/frame
  OpenCV imdecode REDUCED_COLOR_2 only                              131.93 fps     7.580 ms/frame
  OpenCV imdecode REDUCED_COLOR_2 + INTER_LINEAR                     76.56 fps    13.061 ms/frame
  OpenCV imdecode REDUCED_COLOR_2 + INTER_NEAREST                    79.26 fps    12.616 ms/frame
  OpenCV REDUCED_COLOR_2 decode only ThreadPool(2)                  161.52 fps     6.191 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_LINEAR ThreadPool(2)                94.90 fps    10.538 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_NEAREST ThreadPool(2)              113.86 fps     8.782 ms/frame
  OpenCV REDUCED_COLOR_2 decode only ThreadPool(4)                  157.90 fps     6.333 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_LINEAR ThreadPool(4)                71.19 fps    14.046 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_NEAREST ThreadPool(4)              109.31 fps     9.149 ms/frame
  OpenCV REDUCED_COLOR_2 decode only ThreadPool(8)                  155.18 fps     6.444 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_LINEAR ThreadPool(8)                92.65 fps    10.793 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_NEAREST ThreadPool(8)              114.00 fps     8.772 ms/frame
  OpenCV REDUCED_COLOR_2 decode only ThreadPool(12)                 155.34 fps     6.438 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_LINEAR ThreadPool(12)               65.87 fps    15.181 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_NEAREST ThreadPool(12)             115.73 fps     8.641 ms/frame
  OpenCV REDUCED_COLOR_2 decode only ThreadPool(16)                 154.75 fps     6.462 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_LINEAR ThreadPool(16)               92.19 fps    10.847 ms/frame
  OpenCV REDUCED_COLOR_2 + INTER_NEAREST ThreadPool(16)             114.62 fps     8.725 ms/frame
  OpenCV imdecode REDUCED_COLOR_4 only                              160.55 fps     6.228 ms/frame
  OpenCV imdecode REDUCED_COLOR_4 + INTER_LINEAR                     93.58 fps    10.686 ms/frame
  OpenCV imdecode REDUCED_COLOR_4 + INTER_NEAREST                   125.22 fps     7.986 ms/frame
  OpenCV REDUCED_COLOR_4 decode only ThreadPool(2)                  215.58 fps     4.639 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_LINEAR ThreadPool(2)               132.55 fps     7.545 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_NEAREST ThreadPool(2)              140.67 fps     7.109 ms/frame
  OpenCV REDUCED_COLOR_4 decode only ThreadPool(4)                  211.35 fps     4.731 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_LINEAR ThreadPool(4)               107.01 fps     9.345 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_NEAREST ThreadPool(4)              103.91 fps     9.624 ms/frame
  OpenCV REDUCED_COLOR_4 decode only ThreadPool(8)                  211.63 fps     4.725 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_LINEAR ThreadPool(8)               134.09 fps     7.457 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_NEAREST ThreadPool(8)              144.60 fps     6.916 ms/frame
  OpenCV REDUCED_COLOR_4 decode only ThreadPool(12)                 213.61 fps     4.681 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_LINEAR ThreadPool(12)              134.92 fps     7.412 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_NEAREST ThreadPool(12)              96.43 fps    10.370 ms/frame
  OpenCV REDUCED_COLOR_4 decode only ThreadPool(16)                 109.50 fps     9.132 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_LINEAR ThreadPool(16)              130.68 fps     7.652 ms/frame
  OpenCV REDUCED_COLOR_4 + INTER_NEAREST ThreadPool(16)             143.93 fps     6.948 ms/frame
  OpenCV imdecode REDUCED_COLOR_8 only                              274.49 fps     3.643 ms/frame
  OpenCV imdecode REDUCED_COLOR_8 + INTER_LINEAR                    161.21 fps     6.203 ms/frame
  OpenCV imdecode REDUCED_COLOR_8 + INTER_NEAREST                   156.74 fps     6.380 ms/frame
  OpenCV REDUCED_COLOR_8 decode only ThreadPool(2)                  286.84 fps     3.486 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_LINEAR ThreadPool(2)               177.81 fps     5.624 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_NEAREST ThreadPool(2)              108.88 fps     9.185 ms/frame
  OpenCV REDUCED_COLOR_8 decode only ThreadPool(4)                  197.94 fps     5.052 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_LINEAR ThreadPool(4)               183.41 fps     5.452 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_NEAREST ThreadPool(4)              171.73 fps     5.823 ms/frame
  OpenCV REDUCED_COLOR_8 decode only ThreadPool(8)                  279.79 fps     3.574 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_LINEAR ThreadPool(8)               180.18 fps     5.550 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_NEAREST ThreadPool(8)              164.29 fps     6.087 ms/frame
  OpenCV REDUCED_COLOR_8 decode only ThreadPool(12)                 283.54 fps     3.527 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_LINEAR ThreadPool(12)              180.91 fps     5.528 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_NEAREST ThreadPool(12)             156.34 fps     6.396 ms/frame
  OpenCV REDUCED_COLOR_8 decode only ThreadPool(16)                 162.02 fps     6.172 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_LINEAR ThreadPool(16)              144.48 fps     6.922 ms/frame
  OpenCV REDUCED_COLOR_8 + INTER_NEAREST ThreadPool(16)             169.57 fps     5.897 ms/frame
  Pillow full decode + resize                                        11.36 fps    87.999 ms/frame
  Pillow draft + resize                                              11.50 fps    86.982 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=2            236.40 fps     4.230 ms/frame
  NVIDIA DALI nvJPEG mixed decode + GPU resize threads=4            130.77 fps     7.647 ms/frame
