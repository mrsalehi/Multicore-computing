#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

int N_THREADS;
char **MSG;
pthread_attr_t attr;

void *send_message(void *id){
    int t_id = * (long *)id;
    sprintf(MSG[(t_id+1)%N_THREADS], "Hello to %d from %d", (t_id+1)%N_THREADS, t_id);
    
    if (strlen(MSG[t_id]) > 0){
        printf("%d: %s\n", *(int *)id, MSG[t_id]);
    }else{
        printf("%d: No message from %d\n", t_id, (t_id+N_THREADS-1)%N_THREADS);
    }
    pthread_exit(NULL);
}


int main(int argc, char *argv[]){
    scanf("%d", &N_THREADS);
    
    pthread_t threads[N_THREADS];
    long t_ids[N_THREADS];

    pthread_attr_init (&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    void *status;

    
    MSG = malloc(N_THREADS * sizeof(char *));
    for (long i=0; i<N_THREADS; i++){
        MSG[i] = (char *) malloc(100);
    }
    
    int rc;
    for (long id=0; id<N_THREADS; id++){
        t_ids[id] = id;
        rc = pthread_create(&threads[id], &attr, send_message, (void *) &t_ids[id]);
        
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

