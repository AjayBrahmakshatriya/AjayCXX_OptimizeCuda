-include Makefile.inc
CUDA_DIR?=/usr/local/cuda

# assume GNU make
$(shell mkdir -p build)

HEADERS=$(wildcard include/*)

CUXX=$(CUDA_DIR)/bin/nvcc
CUXX_FLAGS=-rdc=true --use_fast_math -Xptxas "-dlcm=ca --maxrregcount=64" -std=c++11 -gencode arch=compute_70,code=sm_70
CUXX_FLAGS+=-I include/ -O3 

EXECUTABLES=build/load_matrix build/simple build/spvdv build/spmv
 
all: $(EXECUTABLES)


build/load_matrix: src/load_matrix.cu $(HEADERS)
	$(CUXX) $(CUXX_FLAGS) $< -o $@

build/simple: src/simple.cu $(HEADERS)
	$(CUXX) $(CUXX_FLAGS) $< -o $@

build/spvdv: src/spvdv.cu $(HEADERS)
	$(CUXX) $(CUXX_FLAGS) $< -o $@

build/spmv: src/spmv.cu $(HEADERS)
	$(CUXX) $(CUXX_FLAGS) $< -o $@
clean:
	- rm -rf build/
