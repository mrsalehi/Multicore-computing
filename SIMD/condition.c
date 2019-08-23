#include <pmmintrin.h>
#include <stdio.h>
#include <iostream>
using namespace std;
#define n 8

void fcond(float *x) {
int i;
for(i = 0; i < n; i++) {
if(x[i] > 0.5)
x[i] += 1.;
else x[i] -= 1.;
}
}



void fcond_vec(float *a) {
int i;
__m128 vt, vr, vtp1, vtm1, vmask, ones, thresholds;
ones
 = _mm_set1_ps(1.);
thresholds = _mm_set1_ps(0.5);
for(i = 0; i < n; i+=4) {
vt = _mm_load_ps(a+i);
vmask = _mm_cmpgt_ps(vt, thresholds);
vtp1 = _mm_add_ps(vt, ones);
vtm1 = _mm_sub_ps(vt, ones);
vr = _mm_or_ps(_mm_and_ps(vmask, vtp1), 
  _mm_andnot_ps(vmask, vtm1));
_mm_store_ps(a+i, vr);
}
}




int main(int argc, char** argv)                                                                                                                                                                                  
{
  float x[n];
  int i;
  for(i=0 ; i<n ; i++)
  {
  	x[i] = 0.1;
  } 
  fcond_vec(x);

  for(i=0; i<n/2 ; i++)
  	cout << x[i] << endl;                                                                                                                                                                                                     
                                                                                                                                                                                                
 return 0;
}