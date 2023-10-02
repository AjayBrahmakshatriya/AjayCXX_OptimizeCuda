#include "matrix.h"
#include "utils.h"
#include <iostream>


void __global__ SpMV (matrix_t A, float * d_B, float * d_C, uint32_t num_rows, uint32_t num_blocks) {

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id >= num_rows) return;
	int rid = thread_id;	

	// rid = i
	// col = j

	// C[i] += A[i][j] * B[j]

	// printf("rid = %d\n", rid);

	d_C[rid] = 0.0;

	for (int cidx = A.d_row_ptrs[rid]; cidx < A.d_row_ptrs[rid + 1]; cidx++) {
		uint32_t col = A.d_cols[cidx];
		float val = A.d_values[cidx];

		d_C[rid] += val * d_B[col];
	}

}

void __global__ SpMV_co_warp (matrix_t A, float * d_B, float * d_C, uint32_t num_rows, uint32_t num_blocks) {

	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	int rid = thread_id / 32;

	if (rid >= num_rows) return;

	int lane_id = thread_id % 32;

	d_C[rid] = 0.0;

	// LaneID 1: 0, 32, 64, 96th non zero the row
	// LaneID 2: 1, 33, 65, 97th non zero the row

	for (int cidx = A.d_row_ptrs[rid] + lane_id; cidx < A.d_row_ptrs[rid + 1]; cidx+=32) {
		
		uint32_t col = A.d_cols[cidx];
		float val = A.d_values[cidx];

		d_C[rid] += val * d_B[col];
	}

}


int main(int argc, char* argv[]) {
	assert(argc >= 2 && "Usage: executable <matrix name>");
	matrix_t A = load_matrix(argv[1]);

		
	std::cout << A.num_rows << " " << A.num_values << std::endl;


	// C[i] += A[i][j] * B[j]

	float * h_C = new float[A.num_rows];
	float * h_C_ref = new float[A.num_rows];
	float * h_B = new float[A.num_rows];

	float * d_C = (float*) cuda_allocate(A.num_rows * sizeof (float));
	float * d_B = (float*) cuda_allocate(A.num_rows * sizeof (float));

	for (int i = 0; i < A.num_rows; i++) {
		int deg = A.h_row_ptrs[i + 1] - A.h_row_ptrs[i];
		h_B[i] = 1.0 / (float) deg;
		h_C[i] = 0;
		h_C_ref[i] = 0;
	}
	
	for (int i = 0; i < A.num_rows; i++) {
		for (int cidx = A.h_row_ptrs[i]; cidx < A.h_row_ptrs[i+1]; cidx++) {
			int col = A.h_cols[cidx];
			float val = A.h_values[cidx];
			
			h_C_ref[i] += val * h_B[col];
		}
	}



	cudaMemcpy(d_B, h_B, sizeof(float) * A.num_rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(float) * A.num_rows, cudaMemcpyHostToDevice);
	

	// Runnign simple row mapping version - 1 thread = 1 row			
	int block_size = 512;		
	int num_blocks = (A.num_rows + block_size - 1) / block_size;
	
	std::cout << block_size << " " << num_blocks << std::endl;	
	float ttr = 0;

	ttr = run_and_time([&] (int rid) {
		SpMV<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows, num_blocks);
		cudaCheckLastError();
	}, 100, 10);
	std::cout << "Execution time [Basic] = " << ttr << "(ms)" << std::endl;

	// Running Co-warp version - 32 threads in warp simulataneusly process a row
	block_size = 512;	
	num_blocks = (A.num_rows * 32 + block_size - 1) / block_size;

	cudaMemcpy(h_C, d_C, sizeof(float) * A.num_rows, cudaMemcpyDeviceToHost);

	ttr = run_and_time([&] (int rid) {
		SpMV_co_warp<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows, num_blocks);
		cudaCheckLastError();
	}, 100, 10);
	std::cout << "Execution time [Co-Warp] = " << ttr << "(ms)" << std::endl;
	
	for (int i = 0; i < A.num_rows; i++) {
		if (std::abs(h_C[i] - h_C_ref[i]) > 0.001 * h_C_ref[i])
			printf("mismatch at %d: %f, %f\n", i, h_C[i], h_C_ref[i]);
	}

	
	return 0;
}
