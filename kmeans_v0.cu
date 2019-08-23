#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdlib.h> 
#include <time.h>
#include <stdbool.h>

#define GRID_SIZE 8
#define BLOCK_SIZE 1024

#define min(a, b) (a < b ? a : b)
#define abs(a) (a > 0 ? a : -1 * a)

__global__ void kMeansStep1(float *d_data, float* d_clusters, float* d_assignments, float *d_distances, int n_clusters){

	int start_data = blockDim.x * blockIdx.x + threadIdx.x * TILE_WIDTH;
	int end_data = start_data + TILE_WIDTH;

	int start_cluster = blockDim.y * blockIdx.y + threadIdx.y * TILE_WIDTH;
	int end_cluster =  start_cluster + TILE_WIDTH;

	for (int data=start_data; data<end_data; data++){
		for (int cluster=start_cluster, cluster<end_cluster; cluster++){
			float distance = 0.0
			for (int j=0; j<d; j++){
				distance += (d_clustes[cluster * d + j] - d_data[data * d + j]) * (d_clustes[cluster * d + j] - d_data[data * d + j]);
			}
			d_distances[data * n_clusters + cluster] = distance;
		}
	}
}

__global__ void kMeansStep2(float *assignments, float *d_clusters, int n_data, int n_clusters){

	int data = blockIdx.x;

	int start = threadIdx.x;
	while (start < n_clusters){

	}
	
	//reduction


	__syncthreads();

	if (tid == 0){
		for int(j = 0; j<d; j++)
			atomicAdd(&d_clusters[cluster][j], d_data[data][j]);
	}

}

__global void checkConverged(float *d_prev_clusters, float *d_new_clusters, bool *converged){

	int start_dim = blockDim.x * blockIdx.x + threadIdx.x * TILE_WIDTH;
	int end_dim = start_dim + TILE_WIDTH;

	int start_cluster = blockDim.y * blockIdx.y + threadIdx.y * TILE_WIDTH;
	int end_cluster = start_cluster + TILE_WIDTH;

	for (int data=start_dim; data<end_dim; data++){
		for (int cluster=start_cluster; cluster<end_cluster; cluster++){
			if (abs(d_prev_clusters - d_new_clusters) > 0.1)
				atomicAnd(not_converged, 1);
		}
	}
}


__global__ void salam(){

}

int main(){

	int n;  // number of data points
	int k;  //number of centorids
	int d;  //dimension of each point

	int n_data = 10;
	int n_clusters = 5;
	int d = 4;

	int size_data = sizeof(float) * n_data * d;
	int size_clusters = sizeof(float) * n_clusters * d;
	int size_distances = sizeof(float) * n_data * n_clusters;

	int h_converged = true;
	float *h_data = (float *)malloc(n * sizeof(float));
	float *h_clusters = (float *)malloc(k * sizeof(float));
	int *h_assignments =  (int*) malloc(n * sizeof(int));

	float *d_data, *d_clusters, *d_assignments, *d_distances, *d_converged;

	for (int i=0; i<n*d; i++)
		h_data[i] = (float *) malloc(d * sizeof(float));	
	

	for (int i=0; i<n_clusters*d; i++)
		h_clusters[i] = (float *) malloc(d * sizeof(float));


	cudaMalloc((void **)&d_data, size_data;
	cudaMalloc((void **)&d_clusters, size_clusters);
	cudaMalloc((void **)&d_prev_clusters, size_clusters)
	cudaMalloc((void **)&d_assignments, n*sizeof(int));
	cudaMalloc((void **)&d_assignments, size_distances);
	cudaMalloc((void *)&converged, size_distances);

	cudaMemcpy(d_data, h_data, size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(d_clusters, h_data, size_data, cudaMemcpyHostToDevice);

	float *d1 = d_clusters;
	float *d2 = d_prev_clusters;

	while(1){
		kMeansStep1 <<<GRID_SIZE, BLOCK_SIZE>>> (d_data, d1, d_assignments, n_data, n_clusters);
		cudaThreadSynchronize();
		cudaMemset(d2, 0, size_clusters);
		kMeansStep2 <<<GRID_SIZE, BLOCK_SIZE>>> (d_assignments, d2);
		checkConverged <<<GRID_SIZE, BLOCK_SIZE>>> (d1, d2, &converged);

		d1 = d1 == d_in ? d_out : d_in;
        d2 = d2 == d_in ? d_out : d_in;
	}
	return 0;
}