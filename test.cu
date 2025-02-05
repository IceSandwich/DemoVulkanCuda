#include <cuda_runtime.h>

static constexpr size_t LOOP_N = 10000;
static constexpr size_t STREAMS_N = 4;

__global__ void __kernel__test1() {
	double sum = 0.0;
	for (int i = 0; i < LOOP_N; ++i) {
		sum += tan(0.2) * tan(0.1);
	}
}
__global__ void __kernel__test2() {
	double sum = 0.0;
	for (int i = 0; i < LOOP_N; ++i) {
		sum += tan(0.3) * tan(0.1);
	}
}

__global__ void __kernel__test3() {
	double sum = 0.0;
	for (int i = 0; i < LOOP_N; ++i) {
		sum += tan(0.4) * tan(0.1);
	}
}

__global__ void __kernel__test4() {
	double sum = 0.0;
	for (int i = 0; i < LOOP_N; ++i) {
		sum += tan(0.5) * tan(0.1);
	}
}

#include <array>
#include <iostream>
void runtest() {
	std::array<cudaStream_t, STREAMS_N> streams;
	cudaEvent_t start, stop;

	for (auto &item : streams) {
		cudaStreamCreate(&item);
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block{ 1 };
	dim3 grid{ 1 };
	cudaEventRecord(start);
	for (int i = 0; i < STREAMS_N; ++i) {
		__kernel__test1 << <grid, block, 0, streams[i] >> > ();
		__kernel__test2 << <grid, block, 0, streams[i] >> > ();
		__kernel__test3 << <grid, block, 0, streams[i] >> > ();
		__kernel__test4 << <grid, block, 0, streams[i] >> > ();
	}
	cudaEventRecord(stop);

	float elapsed_time_ms;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);

	std::cout << "kernel cost: " << elapsed_time_ms << "ms" << std::endl;

	for (auto &item : streams) {
		cudaStreamDestroy(item);
	}
}
