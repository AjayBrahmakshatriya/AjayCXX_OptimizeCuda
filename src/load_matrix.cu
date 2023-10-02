#include "matrix.h"
#include "utils.h"
#include <iostream>

int main(int argc, char* argv[]) {
	assert(argc >= 2 && "Usage: executable <matrix name>");
	matrix_t g = load_matrix(argv[1]);
	
	std::cout << g.num_rows << " " << g.num_values << std::endl;

	uint32_t* vec;
	cudaMalloc((void**)&vec, sizeof(*vec) * g.num_rows);

	std::cout << run_and_time([&] (int run) { 
		cudaMemcpy(vec, g.d_row_ptrs, sizeof(*vec) * g.num_rows, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}, 1000, 0) << "ms" << std::endl;
	
	return 0;
}
