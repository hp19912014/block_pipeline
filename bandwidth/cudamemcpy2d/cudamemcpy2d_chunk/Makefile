
all: 2dconv

2dconv: 2DConvolution.cu
	nvcc -Xcompiler -fopenmp -lcuda -lcudart -lgomp  -o 2dconv 2DConvolution.cu

clean:
	rm -rf 2dconv
