#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <unistd.h>

#define COUNT 1000

int N_THREADS;
char **MSG;
sem_t s[COUNT+3];
sem_t count_mutex;
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
        
        sem_wait(&count_mutex);
        counter += 1;
        sem_post(&count_mutex);

        if(counter == N_THREADS * i){
            for (int j=0; j<N_THREADS; j++){
                free(MSG[j]);
                MSG[j] = (char *) malloc(100 * sizeof(char));
                strcpy(MSG[j], "");
            }   
            printf("Iteration %d finished \n\n", i);    
            sem_init(&s[i+1], 0, 0);
            sem_post(&s[i]);    
        }
        sem_wait(&s[i]);
        sem_post(&s[i]);
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
    
    sem_init(&s[1], 0, 0);
    sem_init(&count_mutex, 0, 1);      

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

    for (int i=0; i<COUNT+3; i++)
        sem_destroy(&s[i]);
	
    pthread_exit(NULL);
	return 0; 
}
