#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>


sem_t MUTEX;

int N_THREADS;
char **MSG;
pthread_attr_t attr;

void *send_message(void *id){
    
	sem_wait(&MUTEX);
	
    int t_id = * (int *)id;
    sprintf(MSG[(t_id+1)%N_THREADS], "Hello to %d from %d", (t_id+1)%N_THREADS, t_id);
    
    if (strlen(MSG[t_id]) > 0){
        printf("%d: %s\n", *(int *)id, MSG[t_id]);
    }else{
        printf("%d: No message from %d\n", t_id, (t_id+N_THREADS-1)%N_THREADS);
    }
    sem_post(&MUTEX);
    
    pthread_exit(NULL);
}


int main(){
	scanf("%d", &N_THREADS);
    
    pthread_t threads[N_THREADS];
    long t_ids[N_THREADS];

    pthread_attr_init (&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void *status;

	MSG = malloc(N_THREADS * sizeof(char *));
    for (int i=0; i<N_THREADS; i++){
        MSG[i] = (char *) malloc(100 * sizeof(char));
    }

    sem_init(&MUTEX, 0, 1);

    int rc;
    for (int id=0; id<N_THREADS; id++){
        t_ids[id] = id;
        rc = pthread_create(&threads[id], NULL, send_message, (void *) &t_ids[id]);
        
        if (rc){
            printf("Error; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    sem_destroy(&MUTEX);
	pthread_exit(NULL);
    return 0;
}