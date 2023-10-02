#include <iostream>
#include "utils.h"

#define VEC_SIZE (10000000)


void __global__ vector_scale(float * d_vector, float scalar, size_t vec_size) {

	// threadIdx.x - id of the thread within the block
	// blockDim.x - size of the block
	// blockIdx.x - id of the block in the grid

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (thread_id >= vec_size) return;

	/// a[i] = d * a[i]

	d_vector[thread_id] = d_vector[thread_id] * scalar;

	return;
}

int main (int argc, char* argv[]) {

	float * h_vector = new float [VEC_SIZE];
	
	for (int i = 0; i < VEC_SIZE; i++)
		h_vector[i] = i;

	// Allocate GPU memory
	float * d_vector;
	cudaMalloc((void**) &d_vector, sizeof(*d_vector) * VEC_SIZE);
	cudaMemcpy(d_vector, h_vector, sizeof(*d_vector) * VEC_SIZE, cudaMemcpyHostToDevice);
	
	int block_size = 512;		
	int num_blocks = (VEC_SIZE + 512 - 1) / 512;

	float ttr = run_and_time([&] (int rid) {
		vector_scale<<<num_blocks, block_size>>> (d_vector, 4.2, VEC_SIZE);
	}, 1000, 0);
	
	std::cout << "Execution time = " << ttr << "(ms)" << std::endl;

	cudaMemcpy(h_vector, d_vector, sizeof(*h_vector) * VEC_SIZE, cudaMemcpyDeviceToHost);	

	return 0;
}
