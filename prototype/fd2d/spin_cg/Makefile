
all: fdtd2d

fdtd2d: fdtd2d.cu
	nvcc -arch=sm_61 -rdc=true --std=c++11 -Xcompiler -fopenmp -lcuda -lcudart -lgomp -o fdtd2d fdtd2d.cu

clean:
	rm -rf fdtd2d
