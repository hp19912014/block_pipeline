
all: fdtd2d

fdtd2d: fdtd2d.cu
	nvcc -Xcompiler -fopenmp -lcuda -lcudart -lgomp -o fdtd2d fdtd2d.cu

clean:
	rm -rf fdtd2d
