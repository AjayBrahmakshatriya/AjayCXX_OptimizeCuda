-include Makefile.inc
CUDA_DIR?=/usr/local/cuda

# assume GNU make
$(shell mkdir -p build)

HEADERS=$(wildcard include/*)

CUXX=$(CUDA_DIR)/bin/nvcc
CUXX_FLAGS=-rdc=true --use_fast_math -Xptxas "-dlcm=ca --maxrregcount=64" -std=c++11 -gencode arch=compute_80,code=sm_80
CUXX_FLAGS+=-I include/

all: build/load_graph


build/load_graph: src/load_graph.cu $(HEADERS)
	$(CUXX) $(CUXX_FLAGS) $< -o $@

