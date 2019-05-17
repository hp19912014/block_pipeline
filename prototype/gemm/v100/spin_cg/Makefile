
all: gemm

gemm: gemm.cu
	nvcc -arch=sm_61 -rdc=true -Xcompiler  -lcuda -lcudart -lgomp -o gemm gemm.cu

clean:
	rm -rf gemm
