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
#define MAX_BLOCKS 50000


__global__ void kMeansStep2(int *d_counts, float *d_new_clusters, float *d_prev_clusters, int *converged, int n_clusters, int d){
	
	int cluster =  threadIdx.x;
	int dim = threadIdx.y;

	if (dim == 0){
		printf("cluster number %d, count %d\n", cluster, d_counts[cluster]);
	}

	int count = max(1, d_counts[cluster]);

	d_new_clusters[cluster * d + dim] /= (float)count;
	if (abs(d_new_clusters[cluster * d + dim] - d_prev_clusters[cluster * d + dim]) > 0.01)
		atomicAnd(&converged[0], 0);
}


__global__ void kMeansStep1(float *d_data, float *d_prev_clusters, float *d_new_clusters, int *d_counts, int n_data, int n_clusters, int d){

	int data = blockIdx.x;
	int cluster = threadIdx.x;
	int dim = threadIdx.y;

	// if (blockIdx.x > 5000)
	// 	printf("data number %d\n", blockIdx.x);
	while(data < n_data){
		extern __shared__ float s[];
		float *shared_dist = s;
		float *shared_data = (float*)&shared_dist[n_clusters];
		float *shared_prev_clusters = (float*)&shared_data[d];

		shared_dist[cluster] = 0.0;
		shared_prev_clusters[cluster * d + dim] = d_prev_clusters[cluster * d + dim];
		if (cluster == 0)
			shared_data[dim] = d_data[data * d + dim];
		__syncthreads();

		float tmp_dist = shared_prev_clusters[cluster * d + dim] - shared_data[dim];
		float dist_data_cluster_dim =  tmp_dist * tmp_dist;
		atomicAdd(&shared_dist[cluster], dist_data_cluster_dim);
		__syncthreads();

		__shared__ int best_cluster;

		if (cluster == 0 && dim == 0){
			float best_distance = FLT_MAX;
			best_cluster = -1;

			for (int j=0; j<n_clusters; j++)
				if (shared_dist[j] < best_distance){
					best_distance = shared_dist[j];
					best_cluster = j;
				}
			printf("data point number %d assigned to cluster %d\n", data, best_cluster);
			atomicAdd(&d_counts[best_cluster], 1);
		}
		__syncthreads();
		
		if (cluster == 0){
			atomicAdd(&d_new_clusters[best_cluster * d + dim], shared_data[dim]);
			// printf("%f is added to new clusters %d , %d\n", shared_data[dim], best_cluster, dim);
		}
		data += MAX_BLOCKS;
		__syncthreads();
	}

}


int main(){
	srand((unsigned int)time(NULL));

	int n_data = 30;
	int n_clusters = 4;
	int d = 2;

	int size_data = sizeof(float) * n_data * d;
	int size_clusters = sizeof(float) * n_clusters * d;

	int *h_converged = (int *)malloc(1 * sizeof(int));
	float *h_data = (float *)malloc(size_data);
	float *h_clusters = (float *)malloc(size_clusters);
	
	int data_x[30] = {25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46};
	int data_y[30] = {79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7};


	for (int i=0; i<n_data*d; i++){
		// h_data[i] = ((float)rand()/(float)(RAND_MAX)) * 100.0;
		if (i % 2 == 0)
			h_data[i] = data_x[i / 2];
		else
			h_data[i] = data_y[i / 2];
		
		printf("%f ", h_data[i]);
		if ((i+1) % d == 0)
			printf("\n");
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

	dim3 bd(n_clusters, d);
	int n_data_blocks = min(n_data, MAX_BLOCKS);
	int sharedMemSize1 = (n_clusters + d + n_clusters * d) * sizeof(float);

	int iteration = 1;
	clock_t start_time = clock();
	
	while(1){

		cudaMemset(d2, 0.0, size_clusters);
		cudaMemset(d_counts, 0, n_clusters * sizeof(int));

		kMeansStep1 <<<n_data_blocks, bd, sharedMemSize1>>> (d_data, d1, d2, d_counts, n_data, n_clusters, d);
		cudaThreadSynchronize();

		h_converged[0] = 1;
		cudaMemcpy(d_converged, h_converged, sizeof(int), cudaMemcpyHostToDevice);
		
		kMeansStep2 <<<1, bd>>> (d_counts, d2, d1, d_converged, n_clusters, d);
		cudaThreadSynchronize();

		cudaMemcpy(h_clusters, d1, size_clusters, cudaMemcpyDeviceToHost);
		printf("\niteration %d prev cluster:\n", iteration);
		for(int i=0; i<n_clusters*d; i++){
			printf("%f ", h_clusters[i]);
			if ((i+1) % d == 0)
				printf("\n");
		}

		cudaMemcpy(h_clusters, d2, size_clusters, cudaMemcpyDeviceToHost);
		printf("\niteration %d new cluster:\n", iteration);
		for(int i=0; i<n_clusters*d; i++){
			printf("%f ", h_clusters[i]);
			if ((i+1) % d == 0)
				printf("\n");
		}

		cudaMemcpy(h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

		if (h_converged[0] == 1){
			cudaMemcpy(h_clusters, d2, size_clusters, cudaMemcpyDeviceToHost);
			break;
		}

		d1 = d1 == d_prev_clusters ? d_new_clusters : d_prev_clusters;
        d2 = d2 == d_prev_clusters ? d_new_clusters : d_prev_clusters;
		iteration += 1;
		if (iteration > 10)
			break;
	}
	clock_t end_time = clock();

	printf("\nFinished!!\n");
	printf("Final clusters:\n");
	for (int i=0; i<n_clusters*d; i++){
		printf("%f ", h_clusters[i]);
		if ((i+1) % d == 0)
			printf("\n");
	}
	printf("number of iterations is %d \n", iteration);
	double total_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("total time: %f\n", total_time);

	return 0;
}
