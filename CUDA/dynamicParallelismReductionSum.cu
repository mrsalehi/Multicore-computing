#include <stdio.h>


__global__ void gpusum_dyn_par_1 (int *input, int *output) {
    
    int tid = threadIdx.x;

    int size = 2*blockDim.x; 

   
    int *ptr_input = input + blockIdx.x*size;
    int *ptr_output = &output[blockIdx.x];

   
    if (size == 2 && tid == 0) {
        ptr_output[0] = ptr_input[0]  +  ptr_input[1];
        return;
    }

    int s = size/2;

    if(s > 1 && tid < s) {
        ptr_input[tid] += ptr_input[tid + s];
    }
    
     __syncthreads();

    
    if(tid==0) {
        gpusum_dyn_par_1 <<<1, s/2>>>(ptr_input,ptr_output);
        cudaDeviceSynchronize();
    }

   

}


int main(){

    int n = 1024;
    int n_per_block = 32;
    int threads_per_block = n_per_block / 2;
    int num_blocks = n / n_per_block;

    int * v = (int *) malloc(n  * sizeof(int));
    for (int i = 0; i < n ; i++){
        v[i] = i;
    }
    int * v_out = (int * ) malloc( num_blocks * sizeof(int)   );

    int  * d_v;
    int  * d_v_out;

    cudaMalloc( (void**)&d_v , n*sizeof(int)   );
    cudaMalloc( (void**)&d_v_out , num_blocks*sizeof(int)   );
    cudaMemcpy( (void*) d_v , (void*) v , n*sizeof(int) , cudaMemcpyHostToDevice );


    gpusum_dyn_par_1<<< num_blocks ,  threads_per_block >>>(d_v, d_v_out);

    cudaMemcpy( (void*) v_out , (void*) d_v_out , num_blocks*sizeof(int) , cudaMemcpyDeviceToHost );

    int total_gpu = 0;
    for (int i = 0; i < num_blocks ; i++){
        total_gpu = total_gpu + v_out[i];
    }
    printf("%d\n", total_gpu );

}