#ifndef GRAPH_H
#define GRAPH_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

struct graph_t {
	// Dimensions of the graph
	uint32_t num_vertices;
	uint32_t num_edges;

	// Store the offsets into the edges array for each vertex
	uint32_t* h_row_ptrs;
	uint32_t* d_row_ptrs;

	// Array to store the destination of each array
	uint32_t* h_edges;	
	uint32_t* d_edges;

	// Array to store the value of each edge
	uint32_t* h_values;
	uint32_t* d_values;
};

graph_t load_graph(const char* graph_name) {
	FILE* f = fopen(graph_name, "r");
	assert(f != NULL && "Failed to open graph file");


	graph_t g;

	// Read the dimensions
	fread(&g.num_vertices, sizeof(g.num_vertices), 1, f);
	fread(&g.num_edges, sizeof(g.num_edges), 1, f);
	
	// Allocate all the host arrays
	g.h_row_ptrs = (uint32_t*)malloc((g.num_vertices + 1) * sizeof(*g.h_row_ptrs));
	g.h_edges = (uint32_t*)malloc(g.num_edges * sizeof(*g.h_edges));
	g.h_values = (uint32_t*)malloc(g.num_edges * sizeof(*g.h_values));

	assert(g.h_row_ptrs && g.h_edges && g.h_values && "Failed to allocate host side arrays for graph");
	
	// Read the row_ptrs
	int res = fread(g.h_row_ptrs, sizeof(*g.h_row_ptrs), g.num_vertices + 1, f);
	assert(res == (g.num_vertices + 1) && "Failed to load row_ptrs");

	// Read the edges	
	res = fread(g.h_edges, sizeof(*g.h_edges), g.num_edges, f);
	assert(res == g.num_edges && "Failed to load edges");

	// Read the values
	res = fread(g.h_values, sizeof(*g.h_values), g.num_edges, f);
	assert(res == g.num_edges && "Failed to load values");


	fclose(f);

 
	// Now allocate and copy over the values to the device memory
	cudaMalloc((void**) &g.d_row_ptrs, sizeof(*g.d_row_ptrs) * (g.num_vertices + 1));
	cudaMalloc((void**) &g.d_edges, sizeof(*g.d_edges) * g.num_edges);
	cudaMalloc((void**) &g.d_values, sizeof(*g.d_values) * g.num_edges);

	assert(g.d_row_ptrs && g.d_edges && g.d_values && "Failed to allocate device arrays for graph");
	
	cudaMemcpy(g.d_row_ptrs, g.h_row_ptrs, sizeof(*g.h_row_ptrs) * (g.num_vertices + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_edges, g.h_edges, sizeof(*g.h_edges) * g.num_edges, cudaMemcpyHostToDevice);
	cudaMemcpy(g.d_values, g.h_values, sizeof(*g.h_values) * g.num_edges, cudaMemcpyHostToDevice);

	// return the loaded graph object
	return g;
}


#ifdef __cplusplus
}
#endif

#endif
