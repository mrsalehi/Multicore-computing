#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 16

int MAX = -999999;

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void bucketSort(int *data) 
{ 
    int n_buckets = 4;
    int size_bucket = MAX / n_buckets + 1;
    
    int **buckets = (int **) malloc(n_buckets * sizeof(int*));
    for (int i=0; i<n_buckets; i++){
    	buckets[i] = (int *) malloc(size_bucket * sizeof(int));
    }

    int *bucket_sizes = (int *) malloc(n_buckets * sizeof(int));
    memset(bucket_sizes, 0, sizeof(bucket_sizes));

    for (int i=0; i<SIZE; i++){
       int bucket_idx = data[i] / size_bucket;
       buckets[bucket_idx][bucket_sizes[bucket_idx]] = data[i];
       bucket_sizes[bucket_idx] += 1;
    }
  	
  	#pragma omp parallel
  	{	
  		#pragma omp for schedule (dynamic)
    	for (int i=0; i<n_buckets; i++)
			qsort(buckets[i], bucket_sizes[i], sizeof(int), cmpfunc);
  	}
    int idx = 0;
    for (int i = 0; i<n_buckets; i++)
        for (int j = 0; j<bucket_sizes[i]; j++){
          data[idx] = buckets[i][j];
          idx += 1;
        }
}


int main(){
	int *data;
  	data = (int *) malloc(sizeof(int)*SIZE);

  	for (int i=0; i<SIZE; i++){
  		scanf("%d", &data[i]);
  		if (data[i] > MAX)
  			MAX = data[i];
  	}

	double start = omp_get_wtime() ;
	bucketSort (data) ;
	double end = omp_get_wtime();
	
	double time = end - start;
	printf ("Runtime = %g\n" , time);

	return 0;
}