//#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

#define SIZE 16

void segmentedScan(int *input, int *flags, int* output){
	int segments = 0;
	int *starts = (int *) malloc(sizeof(int)*SIZE);
	int *ends = (int *) malloc(sizeof(int)*SIZE);

	starts[0] = 0;
	int iterator = 1;
	while(iterator < SIZE){
		if (flags[iterator]){
			ends[segments] = iterator-1;
			segments += 1;
			starts[segments] = iterator;
		}
		iterator += 1;
	}
	ends[segments] = SIZE - 1;
	#pragma omp parallel
	{
	#pragma omp for schedule(dynamic)
	for (int i=0; i<=segments; i++){
		//int id = omp_get_thread_num();
		//printf("thread id: %d iteration %d\n", id, i);
		int sum = input[starts[i]];
		output[starts[i]] = sum;
		for (int j=starts[i]+1; j<=ends[i]; j++){
			sum += input[j];
			output[j] = sum;
		}
	}
	}
}

int main(){
	int *input = (int *) malloc(sizeof(int)*SIZE); 
	int *output = (int *) malloc(sizeof(int)*SIZE); 
	int *flags = (int *) malloc(sizeof(int)*SIZE);

	
	for (int i = 0; i<SIZE; i++)
		//input[i] = i;
		scanf("%d", &input[i]);


	// getting the flags
	for (int i=0; i<SIZE; i++)
		//flags[i] = rand() % 2;
		scanf("%d", &flags[i]);

	
	double start, end;
	
	start = omp_get_wtime();
	segmentedScan(input ,flags ,output);
	end = omp_get_wtime();
	
	double time = (end - start) / 10;
	printf("Runtime = %f\n", time);


	return 0;
}