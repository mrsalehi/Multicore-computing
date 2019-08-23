#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <time.h>


#define GRID_SIZE 8
#define BLOCK_SIZE 32
#define min(a, b) (a < b ? a : b)


__global__ void mergeSort(int *d1, int *d2, int width, int n){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int n_threads = gridDim.x * blockDim.x;

	int start = tid * width;
	int middle = min(start + (width >> 1), n);
	int end = min(start + width, n);


	// each thread may sort more than one tile (or zero tiles if start >= n)
	while(start < n){
		int a = start;
	    int b = middle;

	    //printf("thread id %d start %d end %d middle %d width %d\n", tid, start, end, middle, width);

	    // merge
	    for (int k = start; k < end; k++) {
	        if (a < middle && (b >= end || d1[a] < d1[b])) {
	            d2[k] = d1[a];
	            a += 1;
	        } else {
	            d2[k] = d1[b];
	            b += 1;
	        }
	    }

	    start += n_threads * width;
	    middle = min(start + (width >> 1), n);
		end = min(start + width, n);
	}

}

int main(){
	
	int n;
	n = 100000;
	int size = n * sizeof(int);

	int *h_in = (int *)malloc(size);
	int *h_out = (int *)malloc(size);

	int *d_in, *d_out;

	for (int i=0; i<n; i++){
		h_in[i] = 100000 - i;
	}

	cudaMalloc((void **)&d_in, size);
	cudaMalloc((void **)&d_out, size);

	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

	int *d1 = d_in;
	int *d2 = d_out;

	clock_t start_time = clock();
	for (int width = 2; width < (n << 1); width <<= 1){
		mergeSort <<<GRID_SIZE, BLOCK_SIZE>>>(d1, d2, width, n);
		cudaThreadSynchronize();

		d1 = d1 == d_in ? d_out : d_in;
        d2 = d2 == d_in ? d_out : d_in;

	}
	clock_t end_time = clock();

	cudaMemcpy(h_out, d1, size, cudaMemcpyDeviceToHost);

	for(int i=0; i<n; i++)
		printf("%d\n", h_out[i]);

	printf("\n");

	double total_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("total time: %f\n", total_time);

	return 0;
}