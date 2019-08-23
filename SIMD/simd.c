#include <stdio.h>
#include <stdlib.h>
#include <pmmintrin.h>

int n=3, m=5, k=3;

void matmul(float **A, float **B, float **C){

	__m128 a_vec, b_vec, c_vec;
	
	for (int iii=0; iii<k; iii+=4){

		for (int i=0; i<m; i++){
			c_vec = _mm_set1_ps(0.0);

			for (int ii=0; ii<n; ii++){
				a_vec = _mm_set1_ps(A[i][ii]);
				b_vec = _mm_load_ps(B[ii]+iii);
				c_vec += _mm_mul_ps(a_vec, b_vec);
			}
			_mm_store_ps(C[i]+iii, c_vec);
		}
	}
}

int main(){

	//scanf("%d", &m);
	//scanf("%d", &n);
	//scanf("%d", &k);

	float **A = malloc(m * sizeof(float*));
	for (int i=0; i<m; i++){
		A[i] = malloc(n * sizeof(float));
		for (int j =0; j<n; j++){
			A[i][j] = n * i + j;
			printf("%f ", A[i][j]);

		}
		printf("\n");
	}

	printf("\n");

	float **B = malloc(n * sizeof(float*));
	for (int i=0; i<n; i++){
		B[i] = malloc(k * sizeof(float));
		for (int j =0; j<k; j++){
			B[i][j] = k * i + j;
			printf("%f ", B[i][j]);
		}
		printf("\n");
	}

	float **C = malloc(m * sizeof(float*));
	for (int i=0; i<m; i++){
		C[i] = malloc(k * sizeof(float));
		for (int j=0; j<k; j++)
			C[i][j] = 0.0;
	}

	printf("\n");

	matmul(A, B, C);

	for (int i=0; i<m; i++){
		for (int j=0; j<k; j++)
			printf("%f ", C[i][j]);
		
		printf("\n");
	}

	return 0;

}