#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

struct matrix_t {
	// Dimensions of the matrix
	uint32_t num_rows;
	uint32_t num_values;

	// Store the offsets into the edges array for each vertex
	uint32_t* h_row_ptrs; // num_rows + 1
	uint32_t* d_row_ptrs; // num_rows + 1

	//  Array to store the row of each non zero
	uint32_t* h_rows;
	uint32_t* d_rows;
	uint32_t* h_rows_blocked;
	uint32_t* d_rows_blocked;

	// Array to store the column of each array
	uint32_t* h_cols; // nnz
	uint32_t* d_cols; // nnz
	uint32_t* h_cols_blocked;
	uint32_t* d_cols_blocked;

	// Array to store the value of each edge
	uint32_t* h_values; // nnz
	uint32_t* d_values; // nnz
	uint32_t* h_values_blocked;
	uint32_t* d_values_blocked;


	// Array to store bucket offsets
	uint32_t* h_bin_offsets;
	uint32_t num_blocks;
};

static matrix_t load_matrix(const char* matrix_name, int block_size = -1) {
	FILE* f = fopen(matrix_name, "rb");
	assert(f != NULL && "Failed to open matrix file");


	matrix_t g;

	int res = 0;

	// Read the dimensions
	res = fread(&g.num_rows, sizeof(g.num_rows), 1, f);
	res = fread(&g.num_values, sizeof(g.num_values), 1, f);
	
	// Allocate all the host arrays
	g.h_row_ptrs = (uint32_t*)malloc((g.num_rows + 1) * sizeof(*g.h_row_ptrs));
	g.h_cols = (uint32_t*)malloc(g.num_values * sizeof(*g.h_cols));
	g.h_rows = (uint32_t*)malloc(g.num_values * sizeof(*g.h_rows));
	g.h_values = (uint32_t*)malloc(g.num_values * sizeof(*g.h_values));



	assert(g.h_row_ptrs && g.h_cols && g.h_values && "Failed to allocate host side arrays for matrix");
	
	// Read the row_ptrs
	res = fread(g.h_row_ptrs, sizeof(*g.h_row_ptrs), g.num_rows + 1, f);
	assert(res == (g.num_rows + 1) && "Failed to load row_ptrs");

	// Read the edges	
	res = fread(g.h_cols, sizeof(*g.h_cols), g.num_values, f);
	assert(res == g.num_values && "Failed to load edges");

	// Read the values
	res = fread(g.h_values, sizeof(*g.h_values), g.num_values, f);
	assert(res == g.num_values && "Failed to load values");


	fclose(f);

	for (int rid = 0; rid < g.num_rows; rid++) {
		for (int cidx = g.h_row_ptrs[rid]; cidx < g.h_row_ptrs[rid+1]; cidx++) {
			g.h_rows[cidx] = rid;		
		}
	}

	
	if (block_size != -1) {

		int num_blocks = (g.num_rows + block_size - 1) / block_size;
		g.num_blocks = num_blocks;
		g.h_bin_offsets = new uint32_t[num_blocks + 1];

		uint32_t* h_cols = (uint32_t*)malloc(g.num_values * sizeof(*g.h_cols));
		uint32_t* h_rows = (uint32_t*)malloc(g.num_values * sizeof(*g.h_rows));
		uint32_t* h_values = (uint32_t*)malloc(g.num_values * sizeof(*g.h_values));

		for (int block_id = 0; block_id < num_blocks + 1; block_id++) {
			g.h_bin_offsets[block_id] = 0;
		}
		for (uint32_t nnzid = 0; nnzid < g.num_values; nnzid++) {
			uint32_t colid = g.h_cols[nnzid];

			uint32_t block_id = colid / block_size;
			g.h_bin_offsets[block_id + 1]++;
		}
		uint32_t sum = 0; 
		for (int block_id = 0; block_id < num_blocks + 1; block_id++) {
			sum += g.h_bin_offsets[block_id];
			g.h_bin_offsets[block_id] = sum;	
		}

		for (uint32_t nnzid = 0; nnzid < g.num_values; nnzid++) {
			uint32_t colid = g.h_cols[nnzid];

			uint32_t block_id = colid / block_size;
			uint32_t offset = g.h_bin_offsets[block_id];	

			h_rows[offset] = g.h_rows[nnzid];
			h_cols[offset] = g.h_cols[nnzid];
			h_values[offset] = g.h_values[nnzid];
			
			offset++;
			g.h_bin_offsets[block_id] = offset;
		}
		g.h_rows_blocked = h_rows;
		g.h_cols_blocked = h_cols;
		g.h_values_blocked = h_values;

		cudaMalloc((void**) &g.d_cols_blocked, sizeof(*g.d_cols) * g.num_values);
		cudaMalloc((void**) &g.d_rows_blocked, sizeof(*g.d_rows) * g.num_values);
		cudaMalloc((void**) &g.d_values_blocked, sizeof(*g.d_values) * g.num_values);
	
		assert(g.d_cols_blocked && g.d_values_blocked && g.d_rows_blocked && 
			"Failed to allocate device arrays for blocked matrix");

		cudaMemcpy(g.d_cols_blocked, g.h_cols_blocked, sizeof(*g.h_cols) * g.num_values, cudaMemcpyHostToDevice);
		cudaMemcpy(g.d_rows_blocked, g.h_rows_blocked, sizeof(*g.h_rows) * g.num_values, cudaMemcpyHostToDevice);
		cudaMemcpy(g.d_values_blocked, g.h_values_blocked, sizeof(*g.h_values) * g.num_values, cudaMemcpyHostToDevice);
		
	}

 
	// Now allocate and copy over the values to the device memory
	cudaMalloc((void**) &g.d_row_ptrs, sizeof(*g.d_row_ptrs) * (g.num_rows + 1));
	cudaMalloc((void**) &g.d_cols, sizeof(*g.d_cols) * g.num_values);
	cudaMalloc((void**) &g.d_rows, sizeof(*g.d_rows) * g.num_values);
	cudaMalloc((void**) &g.d_values, sizeof(*g.d_values) * g.num_values);

	assert(g.d_row_ptrs && g.d_cols && g.d_values && g.d_rows && 
		"Failed to allocate device arrays for matrix");
	
	cudaMemcpy(g.d_row_ptrs, g.h_row_ptrs, sizeof(*g.h_row_ptrs) * (g.num_rows + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_cols, g.h_cols, sizeof(*g.h_cols) * g.num_values, cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_rows, g.h_rows, sizeof(*g.h_rows) * g.num_values, cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_values, g.h_values, sizeof(*g.h_values) * g.num_values, cudaMemcpyHostToDevice);

	// return the loaded matrix object
	return g;
}


#endif
