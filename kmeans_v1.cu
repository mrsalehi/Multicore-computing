#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdlib.h> 
#include <time.h>
#include <cfloat>


#define min(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)
#define abs(a) (a > 0 ? a : -1 * a)

#define BLOCK_SIZE 1024
#define MAX_GRID 1024


__global__ void kMeansStep2(int *d_counts, float *d_new_clusters, float *d_prev_clusters, int *converged, int n_clusters, int d, int n_threads){
	
	int cluster =  blockIdx.x * blockDim.x + threadIdx.x;
	while(cluster < n_clusters){
		// printf("cluster number %d, count %d\n", cluster, d_counts[cluster]);
		int count = max(1, d_counts[cluster]);
		for (int j=0; j<d; j++){
			d_new_clusters[cluster * d + j] /= count;
			if (abs(d_new_clusters[cluster * d + j] - d_prev_clusters[cluster * d + j]) > 0.01) 
				atomicAnd(&converged[0], 0);
		}
		cluster += n_threads;
	}
}


__global__ void kMeansStep1(float *d_data, float *d_prev_clusters, float *d_new_clusters, int *d_counts, int n_data, int n_clusters, int d, 
	int n_threads){

	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < n_data){
		
		float best_distance = FLT_MAX;
		int best_cluster = -1;

		for (int cluster=0; cluster<n_clusters; cluster++){
			float distance = 0.0;
			for (int j=0; j<d; j++)
				distance += (d_prev_clusters[cluster * d + j] - d_data[tid * d + j]) * (d_prev_clusters[cluster * d + j] - d_data[tid * d + j]);

			if (distance < best_distance){
				best_distance = distance;
				best_cluster = cluster;
			}
		}
		// printf("data point number %d assigned to cluster %d\n", tid, best_cluster);
		for (int j=0; j<d; j++)
			atomicAdd(&d_new_clusters[best_cluster * d + j], d_data[tid * d + j]);
  		
  		atomicAdd(&d_counts[best_cluster], 1);

  		tid += n_threads;
	}

}



int main(){
	srand((unsigned int)time(NULL));

	int n_data = 100000;
	int n_clusters = 10;
	int d = 100;

	int size_data = sizeof(float) * n_data * d;
	int size_clusters = sizeof(float) * n_clusters * d;

	int *h_converged = (int *)malloc(1 * sizeof(int));
	float *h_data = (float *)malloc(size_data);
	float *h_clusters = (float *)malloc(size_clusters);

	// printf("data:\n");
	for (int i=0; i<n_data*d; i++){
		h_data[i] = ((float)rand()/(float)(RAND_MAX)) * 100.0;
		// printf("%f ", h_data[i]);
		// if ((i+1) % d == 0)
		// 	printf("\n");
	}

	printf("\ninitial clusters:\n");
	for (int i=0; i<n_clusters*d; i++){
		h_clusters[i] = ((float)rand()/(float)(RAND_MAX)) * 100.0;
		printf("%f ", h_clusters[i]);
		if ((i+1) % d == 0)
			printf("\n");
	}

	float *d_data, *d_new_clusters, *d_prev_clusters;
	int *d_converged, *d_counts;

	cudaMalloc((void **)&d_data, size_data);
	cudaMalloc((void **)&d_new_clusters, size_clusters);
	cudaMalloc((void **)&d_prev_clusters, size_clusters);
	cudaMalloc((void **)&d_counts, n_clusters * sizeof(int));
	cudaMalloc((void **)&d_converged, sizeof(int));

	cudaMemcpy(d_data, h_data, size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(d_prev_clusters, h_clusters, size_clusters, cudaMemcpyHostToDevice);

	float *d1 = d_prev_clusters;
	float *d2 = d_new_clusters;

	int n_data_blocks = min( (int)(n_data / BLOCK_SIZE) + 1, MAX_GRID);
	int n_clusters_blocks = min( (int)(n_clusters / BLOCK_SIZE) + 1, MAX_GRID);

	int iteration = 1;
	clock_t start_time = clock();
	while(1){
		// printf("\nITERATION NUMBER %d STARTED!!!!\n", iteration);
		cudaMemset(d2, 0.0, size_clusters);
		cudaMemset(d_counts, 0, n_clusters * sizeof(int));

		kMeansStep1 <<<n_data_blocks, BLOCK_SIZE>>> (d_data, d1, d2, d_counts, n_data, n_clusters, d, n_data_blocks*BLOCK_SIZE);
		cudaThreadSynchronize();

		h_converged[0] = 1;
		cudaMemcpy(d_converged, h_converged, sizeof(int), cudaMemcpyHostToDevice);
		
		kMeansStep2 <<<n_clusters_blocks, BLOCK_SIZE>>> (d_counts, d2, d1, d_converged, n_clusters, d, n_clusters_blocks*BLOCK_SIZE);
		cudaThreadSynchronize();

		// cudaMemcpy(h_clusters, d1, size_clusters, cudaMemcpyDeviceToHost);
		// printf("\niteration %d prev cluster:\n", iteration);
		// for(int i=0; i<n_clusters*d; i++){
		// 	printf("%f ", h_clusters[i]);
		// 	if ((i+1) % d == 0)
		// 		printf("\n");
		// }

		// cudaMemcpy(h_clusters, d2, size_clusters, cudaMemcpyDeviceToHost);
		// printf("\niteration %d new cluster:\n", iteration);
		// for(int i=0; i<n_clusters*d; i++){
		// 	printf("%f ", h_clusters[i]);
		// 	if ((i+1) % d == 0)
		// 		printf("\n");
		// }

		cudaMemcpy(h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

		if (h_converged[0] == 1){
			cudaMemcpy(h_clusters, d2, size_clusters, cudaMemcpyDeviceToHost);
			break;
		}

		d1 = d1 == d_prev_clusters ? d_new_clusters : d_prev_clusters;
        d2 = d2 == d_prev_clusters ? d_new_clusters : d_prev_clusters;
		iteration += 1;
	}
	clock_t end_time = clock();

	printf("\nFinished!!\n");
	printf("Final clusters:\n");
	for (int i=0; i<n_clusters*d; i++){
		printf("%f ", h_clusters[i]);
		if ((i+1) % d == 0)
			printf("\n");
	}
	double total_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("total time: %f\n", total_time);

	return 0;
}
