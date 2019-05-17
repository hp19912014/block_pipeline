
all: 3dconv

3dconv: 3DConvolution.cu
	nvcc -arch=sm_61 -rdc=true --std=c++11 -Xcompiler  -lcuda -lcudart    -o 3dconv 3DConvolution.cu

clean:
	rm -rf 3dconv
