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

	// Array to store the destination of each array
	uint32_t* h_cols; // nnz
	uint32_t* d_cols; // nnz

	// Array to store the value of each edge
	uint32_t* h_values; // nnz
	uint32_t* d_values; // nnz
};

static matrix_t load_matrix(const char* matrix_name) {
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

 
	// Now allocate and copy over the values to the device memory
	cudaMalloc((void**) &g.d_row_ptrs, sizeof(*g.d_row_ptrs) * (g.num_rows + 1));
	cudaMalloc((void**) &g.d_cols, sizeof(*g.d_cols) * g.num_values);
	cudaMalloc((void**) &g.d_values, sizeof(*g.d_values) * g.num_values);

	assert(g.d_row_ptrs && g.d_cols && g.d_values && "Failed to allocate device arrays for matrix");
	
	cudaMemcpy(g.d_row_ptrs, g.h_row_ptrs, sizeof(*g.h_row_ptrs) * (g.num_rows + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_cols, g.h_cols, sizeof(*g.h_cols) * g.num_values, cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_values, g.h_values, sizeof(*g.h_values) * g.num_values, cudaMemcpyHostToDevice);

	// return the loaded matrix object
	return g;
}


#endif
