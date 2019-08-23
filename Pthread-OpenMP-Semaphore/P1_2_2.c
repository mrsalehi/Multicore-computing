#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>

#define COUNT 1000

int N_THREADS;
char **MSG;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;
pthread_attr_t attr;
int counter = 0;


void *Thread_work(void* id) { 

	int t_id = * (int *)id;
	
	for (int i = 1; i<=COUNT; i++) {
		printf("Iteration %d, thread id %d\n", i, t_id);
		sprintf(MSG[(t_id+1)%N_THREADS], "Hello to %d from %d", (t_id+1)%N_THREADS, t_id);

		if (strlen(MSG[t_id]) > 0){
    		printf("%d: %s\n", *(int *)id, MSG[t_id]);
		}else{
    		printf("%d: No message from %d\n", t_id, (t_id+N_THREADS-1)%N_THREADS);
		}

		pthread_mutex_lock(&count_mutex);
		counter += 1;
		if (counter == N_THREADS * i){
			for (int i=0; i<N_THREADS; i++){
				free(MSG[i]);
        		MSG[i] = (char *) malloc(100 * sizeof(char));
        		strcpy(MSG[i], "");
    		}	
    		printf("Iteration %d finished \n\n", i);
			pthread_cond_broadcast(&count_threshold_cv);
		}else{
			pthread_cond_wait(&count_threshold_cv, &count_mutex);
		}
		pthread_mutex_unlock(&count_mutex);
	}
	pthread_exit(NULL);
}


int main(void) {
	scanf("%d", &N_THREADS);
    
    pthread_t threads[N_THREADS];
    long t_ids[N_THREADS];

    pthread_attr_init (&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void *status;
    pthread_mutex_init(&count_mutex, NULL);
  	pthread_cond_init (&count_threshold_cv, NULL);

	MSG = malloc(N_THREADS * sizeof(char *));
    for (int i=0; i<N_THREADS; i++){
        MSG[i] = (char *) malloc(100 * sizeof(char));
    }

    int rc;

    for (long id=0; id<N_THREADS; id++){
        t_ids[id] = id;
        rc = pthread_create(&threads[id], &attr, Thread_work, (void *) &t_ids[id]);
        
        if (rc){
            printf("Error; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    for (long id=0; id<N_THREADS; id++){
        
        rc = pthread_join(threads[id], &status);
        
        if (rc){
            printf("Error; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

	pthread_exit(NULL);
	return 0; 
}