#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <unistd.h>

#define COUNT 1000

int N_THREADS;
char **MSG;
pthread_mutex_t counter_lock;
pthread_attr_t attr;
int counter[2];

void *Thread_work(void* id) { 
	int t_id = * (int *)id;
	
	for (int i = 0; i<COUNT; i++) {

		printf("Iteration %d, thread id %d\n", i, t_id);

		sprintf(MSG[(t_id+1)%N_THREADS], "Hello to %d from %d", (t_id+1)%N_THREADS, t_id);

		if (strlen(MSG[t_id]) > 0){
    		printf("%d: %s\n", *(int *)id, MSG[t_id]);
		}else{
    		printf("%d: No message from %d\n", t_id, (t_id+N_THREADS-1)%N_THREADS);
		}

		pthread_mutex_lock(&counter_lock);
		counter[i%2] += 1;
		pthread_mutex_unlock(&counter_lock);

		while (counter[i%2] <= N_THREADS && counter[i%2] > 0){
		}
	}
	pthread_exit(NULL);
}


void *watch_count(){
	for (int i = 0; i<COUNT; i++){
		while (counter[i%2] < N_THREADS){
		}
		
		for (int i=0; i<N_THREADS; i++){
			free(MSG[i]);
        	MSG[i] = (char *) malloc(100 * sizeof(char));
        	strcpy(MSG[i], "");
    	}
    	printf("\n");
    	pthread_mutex_lock(&counter_lock);
    	counter[(i+1)%2] = 0;
    	counter[i%2] = 0;
    	pthread_mutex_unlock(&counter_lock);

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
    pthread_mutex_init(&counter_lock, NULL);

	MSG = malloc(N_THREADS * sizeof(char *));
    
    for (int i=0; i<N_THREADS; i++){
        MSG[i] = (char *) malloc(100 * sizeof(char));
    }

    int rc;

    pthread_t t_count; 
    pthread_create(&t_count, &attr, watch_count, NULL);

    if (rc){
        printf("Error; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }


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
    rc = pthread_join(t_count, &status);

	pthread_exit(NULL);
	return 0; 
}
