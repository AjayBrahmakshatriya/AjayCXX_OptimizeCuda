#include "matrix.h"
#include "utils.h"
#include <iostream>

#define CTA_SIZE (512)
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

	if (lane_id == 0)		
		d_C[rid] = 0.0;

	// LaneID 1: 0, 32, 64, 96th non zero the row
	// LaneID 2: 1, 33, 65, 97th non zero the row

	for (int cidx = A.d_row_ptrs[rid] + lane_id; cidx < A.d_row_ptrs[rid + 1]; cidx+=32) {
		
		uint32_t col = A.d_cols[cidx];
		float val = A.d_values[cidx];

		//d_C[rid] += val * d_B[col];
		atomicAdd(&d_C[rid], val * d_B[col]);
	}

}

void __global__ SpMV_WM (matrix_t A, float * d_B, float * d_C, uint32_t num_rows) {	
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int lane_id = thread_id  % 32;
	int chunk_id = threadIdx.x / 32;
	int chunk_off = chunk_id * 32;

	int32_t __shared__ shared_deg[CTA_SIZE];
	uint32_t __shared__ shared_row[CTA_SIZE];
	uint32_t __shared__ shared_rptr[CTA_SIZE];	

	if (thread_id < num_rows) {
		shared_row[threadIdx.x] = thread_id;
		shared_rptr[threadIdx.x] = A.d_row_ptrs[thread_id];
		shared_deg[threadIdx.x] = A.d_row_ptrs[thread_id + 1] - A.d_row_ptrs[thread_id];

		d_C[thread_id] = 0;
	} else {
		shared_row[threadIdx.x] = 0;
		shared_rptr[threadIdx.x] = 0;
		shared_deg[threadIdx.x] = 0;
	}
	__syncthreads();
	// Prefix sum	
	int32_t deg = shared_deg[threadIdx.x];	

	for (uint32_t d = 1; d < 32; d *=2) {
		int32_t tmp = __shfl_up_sync((unsigned)-1, deg, d);
		if (lane_id >= d) deg += tmp;
	}


	int32_t tot_deg = __shfl_sync((unsigned)-1, deg, 31);
	
	if (lane_id == 31) deg = 0;
	shared_deg[chunk_off + (lane_id + 1) % 32] = deg; // __shfl_sync((unsigned)-1, deg, lane_id);

	__syncthreads();

	int warp_end = thread_id - lane_id + 32;
	if (warp_end > num_rows) warp_end = num_rows;
	int len = warp_end - (thread_id - lane_id);

	for (int nnzid = lane_id; nnzid < tot_deg; nnzid += 32) {
		int ridx = binary_search_upperbound(shared_deg + chunk_off, len, nnzid) - 1;
		
		uint32_t rid = shared_row[chunk_off + ridx];
		uint32_t rptr = shared_rptr[chunk_off + ridx] + nnzid - shared_deg[chunk_off + ridx];

		uint32_t col = A.d_cols[rptr];
		float val = A.d_values[rptr];

		atomicAdd(&d_C[rid], val * d_B[col]);	
	}
	
}

void __global__ SpMV_EO_init(matrix_t A, float * d_B, float *d_C, uint32_t num_rows) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < num_rows) 
		d_C[thread_id] = 0;
}

void __global__ SpMV_EO (matrix_t A, float * d_B, float * d_C, uint32_t num_rows) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

	for (int nnzid = thread_id; nnzid < A.num_values; nnzid += num_threads) {
		int rid = A.d_rows[nnzid];
		int cid = A.d_cols[nnzid];
		float val = A.d_values[nnzid];		

		atomicAdd(&d_C[rid], val * d_B[cid]);	
	}
}

void __global__ SpMV_EO_Blocked(matrix_t A, float * d_B, float * d_C, uint32_t block_start, uint32_t block_end) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;

	for (int nnzid = thread_id + block_start; nnzid < block_end; nnzid += num_threads) {
		int rid = A.d_rows_blocked[nnzid];
		int cid = A.d_cols_blocked[nnzid];
		float val = A.d_values_blocked[nnzid];		

		atomicAdd(&d_C[rid], val * d_B[cid]);	
	}
}

int main(int argc, char* argv[]) {
	assert(argc >= 2 && "Usage: executable <matrix name>");
	matrix_t A = load_matrix(argv[1], 800000);
	

		
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
	}, 100, 10);
	cudaCheckLastError();

	std::cout << "Execution time [Basic] = " << ttr << "(ms)" << std::endl;

	// Running Co-warp version - 32 threads in warp simulataneusly process a row
	block_size = 512;	
	num_blocks = (A.num_rows * 32 + block_size - 1) / block_size;


	ttr = run_and_time([&] (int rid) {
		SpMV_co_warp<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows, num_blocks);
	}, 100, 10);
	cudaCheckLastError();

	std::cout << "Execution time [Co-Warp] = " << ttr << "(ms)" << std::endl;


	// Running WM version - Threads is a warp collaboratively process non zeros
	// in a load balanced way
	block_size = CTA_SIZE;	
	num_blocks = (A.num_rows + block_size - 1) / block_size;

	ttr = run_and_time([&] (int rid) {
		SpMV_WM<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows);
	}, 100, 10);
	cudaCheckLastError();
	std::cout << "Execution time [WM] = " << ttr << "(ms)" << std::endl;

	// Running EO version - All threads collaboratively process all non-zeros in COO format
	ttr = run_and_time([&] (int rid) {
		block_size = CTA_SIZE;	
		num_blocks = (A.num_rows + block_size - 1) / block_size;
		SpMV_EO_init<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows);
		
		block_size = 512;
		num_blocks = 80;

		SpMV_EO<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows);

	}, 100, 10);
	cudaCheckLastError();
	std::cout << "Execution time [EO] = " << ttr << "(ms)" << std::endl;


	// Running EO Blocked version - All threads collaboratively process all non-zeros in COO format
	// COO is partitioned into blocks for better locality
	ttr = run_and_time([&] (int rid) {
		block_size = CTA_SIZE;	
		num_blocks = (A.num_rows + block_size - 1) / block_size;
		SpMV_EO_init<<<num_blocks, block_size>>> (A, d_B, d_C, A.num_rows);
		
		block_size = 512;
		num_blocks = 160;
		for (int i = 0; i < A.num_blocks; i++) {
			uint32_t block_start = i == 0?0: A.h_bin_offsets[i-1];
			uint32_t block_end = A.h_bin_offsets[i];
			SpMV_EO_Blocked<<<num_blocks, block_size>>> (A, d_B, d_C, block_start, block_end);
		}

	}, 100, 10);
	cudaCheckLastError();
	std::cout << "Execution time [EO Blocked] = " << ttr << "(ms)" << std::endl;
	cudaMemcpy(h_C, d_C, sizeof(*h_C) * A.num_rows, cudaMemcpyDeviceToHost);

	for (int i = 0; i < A.num_rows; i++) {
		if (std::abs(h_C[i] - h_C_ref[i]) > 0.001 * h_C_ref[i])
			printf("mismatch at %d: %f, %f\n", i, h_C[i], h_C_ref[i]);
	}

	
	return 0;
}
