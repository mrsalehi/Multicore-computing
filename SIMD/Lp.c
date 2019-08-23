#include <pmmintrin.h>
#include <stdio.h>
#include <iostream>

using namespace std;
#define n 8

void lp(float *x, float *y) {
for (int i = 0; i < n/2; i++)
  y[i] = (x[2*i] + x[2*i+1])/2;
}


void lp_vec(float *x, float *y) {
__m128 half, v1, v2, avg;
half = _mm_set1_ps(0.5);
for(int i = 0; i < n/8; i++) {
  v1 = _mm_load_ps(x+i*8);
  v2 = _mm_load_ps(x+4+i*8);
  avg = _mm_hadd_ps(v1, v2);
  avg = _mm_mul_ps(avg, half);
  _mm_store_ps(y+i*4, avg);

 }
}



int main(int argc, char** argv)                                                                                                                                                                                  
{
  float x[n],y[n/2];
  int i;
  for(i=0 ; i<n ; i++)
  {
  	x[i] = 1;
  } 
  lp_vec(x,y);

  for(i=0; i<n/2 ; i++)
  	cout << y[i] << endl;                                                                                                                                                                                                     
                                                                                                                                                                                                
 return 0;
}