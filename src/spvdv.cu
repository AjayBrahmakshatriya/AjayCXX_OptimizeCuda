#include <iostream>
#include "utils.h"

#define VEC_SIZE (100000)


void __global__ SpVDV (float * d_vector, float * d_sp_values, uint32_t * d_sp_idx, float * d_output, int nnz, int vec_size) {
	
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_id >= nnz) return;

	float val = d_sp_values[thread_id];
	uint32_t idx = d_sp_idx[thread_id];

	// c[i] = a[i] * b[i]
	d_output[idx] = d_vector[idx] * val;

	// d_output[d_sp_idx[thread_id]] = d_vector[d_sp_idx[thread_id]] * d_sp_values[thread_id]
}
	



int main (int argc, char* argv[]) {

	float * h_vector = new float [VEC_SIZE];
	
	for (int i = 0; i < VEC_SIZE; i++)
		h_vector[i] = i;

	// Allocate GPU memory
	float * d_vector;
	cudaMalloc((void**) &d_vector, sizeof(*d_vector) * VEC_SIZE);
	cudaMemcpy(d_vector, h_vector, sizeof(*d_vector) * VEC_SIZE, cudaMemcpyHostToDevice);

	float * h_sp_values = new float [VEC_SIZE];	
	uint32_t * h_sp_idx = new uint32_t [VEC_SIZE];

	int nnz = VEC_SIZE / 20;
	for (int i = 0; i < nnz; i++) {
		h_sp_values[i] = i + 1;
		h_sp_idx[i] = i * 20;
	}

	float * d_sp_values;
	uint32_t * d_sp_idx;
	
	cudaMalloc((void**) &d_sp_values, sizeof(*d_sp_values) * nnz);
	cudaMalloc((void**) &d_sp_idx, sizeof(*d_sp_idx) * nnz);

	cudaMemcpy(d_sp_values, h_sp_values, sizeof(*d_sp_values) * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sp_idx, h_sp_idx, sizeof(*d_sp_idx) * nnz, cudaMemcpyHostToDevice);

	float * h_output = new float [VEC_SIZE];
	float * d_output;		
	cudaMalloc((void**) &d_output, sizeof(*d_output) * VEC_SIZE);

	int block_size = 512;		
	int num_blocks = (nnz + 512 - 1) / 512;

	float ttr = run_and_time([&] (int rid) {
		SpVDV<<<num_blocks, block_size>>> (d_vector, d_sp_values, d_sp_idx, d_output, nnz, VEC_SIZE);
		cudaCheckLastError();
	}, 1, 0);

	std::cout << "Execution time = " << ttr << "(ms)" << std::endl;

	return 0;
}
