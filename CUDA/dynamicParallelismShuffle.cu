#include <stdio.h>

__inline__ __device__
int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}


__inline__ __device__
int blockReduceSum(int val) {

  static __shared__ int shared[32]; 
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);    
  if (lane==0) shared[wid]=val; 
  __syncthreads();              

  
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); 
 
  return val;
  
}


__global__ void deviceReduceKernel(int *in, int* out, int N) {

  int sum= 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}


int main(){
    
    int n = 1024;
    int threads = 64;
    int blocks = n/threads;
    

    int * in = (int *) malloc(n  * sizeof(int));
    for (int i = 0; i < n ; i++){
        in[i] = i;
    }
    int * out = (int * ) malloc( blocks * sizeof(int)  );

    int * d_in;
    int * d_out;

    cudaMalloc( (void**)&d_in , n*sizeof(int)   );
    cudaMalloc( (void**)&d_out , blocks*sizeof(int)   );
    cudaMemcpy( (void*) d_in , (void*) in , n*sizeof(int) , cudaMemcpyHostToDevice );


     deviceReduceKernel<<<blocks, threads>>>(d_in, d_out, n);
     deviceReduceKernel<<<1, threads>>>(d_out, d_out, blocks);

    cudaMemcpy( (void*) out , (void*) d_out , blocks*sizeof(int) , cudaMemcpyDeviceToHost );

    
    printf("%d\n", out[0] );

}