#include <emmintrin.h>
#include <stdio.h>
#include <iostream>
using namespace std;
#define n 12

void addindex(float *x) {
for (int i = 0; i < n; i++)
	x[i] = x[i] + i;
}

void addindex_vec1(float *x) {
__m128 index, x_vec;
 for (int i = 0; i < n; i+=4) {
 x_vec = _mm_load_ps(x+i); // load 4 floats
 index = _mm_set_ps(i+3, i+2, i+1, i); // create vector with indexes
 x_vec = _mm_add_ps(x_vec, index); // add the two
 _mm_store_ps(x+i, x_vec); // store back
}
}                                                                                                                                                                                                               
 
void addindex_vec2(float *x) {
__m128 x_vec, ind, incr;
ind = _mm_set_ps(3, 2, 1, 0);
incr = _mm_set1_ps(4);
for (int i = 0; i < n; i+=4) {
	x_vec = _mm_load_ps(x+i);
	x_vec = _mm_add_ps(x_vec, ind);
	ind = _mm_add_ps(ind, incr);
	_mm_store_ps(x+i, x_vec);
}
}


int main(int argc, char** argv)                                                                                                                                                                                  
{
  float a[n];
  int i;
  for(i=0 ; i<n ; i++)
  {
  	a[i] = 1;
  } 
  addindex_vec2(a);

  for(i=0; i<n ; i++)
  	cout << a[i] << endl;                                                                                                                                                                                                     
                                                                                                                                                                                                
 return 0;
}