#ifndef UTILS_H
#define UTILS_H

#include <functional>
#include <sys/time.h>

static long long get_time_in_us(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

static float run_and_time(std::function<void(int)> f, int runs, int warm_up_runs = 10) {
	// Run warm up runs
	for (int i = 0; i < warm_up_runs; i++) {
		f(-1);	
	}	

	// Actual timed runs
	long long start_time = get_time_in_us();
	for (int i = 0; i < runs; i++) {
		f(i);
	}
	// It is the f's job to synchronize for CUDA
	long long stop_time = get_time_in_us();
	
	return ((float)(stop_time - start_time)) / 1000.0;	
}


static int32_t __device__ binary_search_upperbound(int32_t *array, int32_t len, int32_t key){
        int32_t s = 0;
        while(len>0){
                int32_t half = len>>1;
                int32_t mid = s + half;
                if(array[mid] > key){
                        len = half;
                }else{
                        s = mid+1;
                        len = len-half-1;
                }
        }
        return s;
}

static void cudaCheckLastError(void) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                printf("Error: %s\n", cudaGetErrorString(err));
                exit(-1);
        }
}

void* cuda_allocate(size_t size) {
	void* ptr = NULL;
	cudaMalloc(&ptr, size);
	if (ptr == NULL) printf("Cannot allocate memory\n");
	cudaCheckLastError();
	return ptr;
}

#endif



