
all: 2dconv

2dconv: 2DConvolution.cu
	nvcc -arch=sm_61 -rdc=true -Xcompiler  -lcuda -lcudart    -o 2dconv 2DConvolution.cu

clean:
	rm -rf 2dconv
