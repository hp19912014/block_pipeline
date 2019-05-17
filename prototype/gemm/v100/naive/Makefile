
all: gemm

gemm: gemm.cu
	nvcc -Xcompiler -fopenmp -lcuda -lcudart -lgomp -o gemm gemm.cu

clean:
	rm -rf gemm
