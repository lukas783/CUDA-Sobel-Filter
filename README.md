Sobel Edge Detector

Date: 10/16/2017
 
Compiling: Requires a Nvidia CUDA capable graphics card and the Nvidia GPU Computing Toolkit.
      Linux: nvcc -Wno-deprecated-gpu-targets -O3 -o edge sobelFilter.cu lodepng.cpp -std=c++11 -Xcompiler -fopenmp
 
Usage:Linux: >> edge [filename.png]
 
Description: This program is meant to run a sobel filter using three different methods. Each sobel
      filter function runs in a different way than the others, one is a basic sobel filter running
      through just the cpu on a single thread, another runs through openmp to parallelize the 
      single threaded cpu function, and the last one runs through a NVIDIA gpu to parallelize the 
      function onto the many cores available on the gpu. The resulting output from the functions 
      should be the same no matter which way it is ran, the only difference will be the time taken
      to process each method. In most cases, the CPU will run the slowest, then openmp, and finally
      the GPU method will run the fastest (due to having more cores to handle many small computations).
      The result will be an image showing an edge map produced by each method, the edge maps should all
      be the same.
