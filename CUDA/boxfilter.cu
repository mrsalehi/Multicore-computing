#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

using namespace std;
#define TILE_SIZEX 32
#define TILE_SIZEY 32
#define sizeImage 512
#define W 3
#define ITERATIONS ( 1 )

//////////////////////////////////////////
__global__ void boxFilterKernel(int* input, int* output)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int x,y;
    if ( row==0 || col==0 || row==sizeImage-1 || col==sizeImage-1)
        output[row*sizeImage+col] = 0;
    
    else
    {
    	int sum = 0;
        for ( x=0 ; x<W ; x++ )
            for ( y=0 ; y<W ; y++ )
            {
                
                sum += input[(row+x-1)*sizeImage+(col+y-1)];
                	
            }
   
    	output[row*sizeImage+col] = sum / (W*W);

    }
}

/********************************************************

Main function

*********************************************************/
int main(void)
{
    int *in,*out;
    int *d_in, *d_out;
    int size = sizeImage*sizeImage*sizeof(int);
    int i,j;

    in = (int*)malloc( size );
    out = (int*)malloc( size );
    out = (int*)malloc( size ); 

    cudaMalloc( (void**)&d_in, size );
    cudaMalloc( (void**)&d_out, size );
    
    for( i=0 ; i<sizeImage ; i++ )
    	for( j=0 ; j<sizeImage ; j++ )
    		in[ i*sizeImage + j ] = (rand() % 256);
           

   
    cudaMemcpy( d_in, in, size, cudaMemcpyHostToDevice );
    
	

    dim3 dimBlock(TILE_SIZEX , TILE_SIZEY);
    dim3 dimGrid((int)ceil((float) sizeImage / (float)TILE_SIZEX),
                (int)ceil((float) sizeImage / (float)TILE_SIZEY));
    
    
   
    boxFilterKernel <<< dimGrid, dimBlock >>> (d_in, d_out);
   
    
    cudaMemcpy( out, d_out, size, cudaMemcpyDeviceToHost );

    

    
    
	return 0;
}
