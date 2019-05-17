
all: 3dconv

3dconv: 3DConvolution.cu
	nvcc -Xcompiler -fopenmp -lcuda -lcudart    -o 3dconv 3DConvolution.cu

clean:
	rm -rf 3dconv
