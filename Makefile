################ MAKEFILE TEMPLATE ################

# Author : Lucas Carpenter

# Usage : make target1

# What compiler are we using? (gcc, g++, nvcc, etc)
LINK = nvcc

# Name of our binary executable
OUT_FILE = edge

# Any weird flags ( -O2/-O3/-Wno-deprecated-gpu-targets/-fopenmp/etc)
FLAGS = -Wno-deprecated-gpu-targets -O2 -Xcompiler -fopenmp -std=c++11

all: edge

edge: sobelFilter.cu lodepng.cpp
	$(LINK) -o $(OUT_FILE) $(FLAGS) $^

clean: 
	rm -f *.o *~ core
