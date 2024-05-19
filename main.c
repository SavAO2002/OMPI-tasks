#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>
#include <setjmp.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  N 8

int RANK_TO_BE_KILLED = -1;

double maxeps = 0.1e-7;
int itmax = 100;

jmp_buf jbuf;
MPI_Comm main_comm;

double eps;
double A[N][N];
double A_checkpoint[N][N];

double tmp_mas[N];

void relax();
void init();
void verify();

int w_rank;
int w_size;
int w_start;
int w_end;

int first_row, last_row;

void copy_from_to(double from_matrix[N][N], double to_matrix[N][N]) {
    for (int i = 1; i < N - 2; i++) {
        for (int j = 1; j <= N - 2; j++) {
            to_matrix[i][j] = from_matrix[i][j];
        }
    }
}

void redistribute() {
    double one_unit = N / w_size;

    first_row = one_unit * w_rank;
    last_row = first_row + one_unit;

    if (w_rank == w_size - 1) {
        last_row = N;
    }
    fprintf(stderr, "rank %d, from %d to %d\n", w_rank, first_row, last_row);
    fflush(stdout);
}

static void verbose_errhandler(MPI_Comm *comm, int *err, ...) {
    RANK_TO_BE_KILLED = -1; // don't kill anyone

    int len;
    char errstr[MPI_MAX_ERROR_STRING];

    MPI_Error_string(*err, errstr, &len);
    errstr[len] = 0;

    printf("Error in handler: %s\n",errstr);

    int erro = MPIX_Comm_shrink(*comm, &main_comm);
    //fprintf(stderr, "hello0: %d\n", erro);
	//fflush(stdout);
    MPI_Comm_rank(main_comm, &w_rank);
    MPI_Comm_size(main_comm, &w_size);
    MPI_Barrier(main_comm);
    redistribute();

    copy_from_to(A_checkpoint, A);
    
    MPI_Barrier(main_comm);
	
	//fprintf(stderr, "hello2: \n");
    longjmp(jbuf, 0);
}

int main(int an, char **as){
        main_comm = MPI_COMM_WORLD;
        int code;
        
        if (code = MPI_Init(&an, &as)) {
            printf("error on start\n");
            MPI_Abort(main_comm, code);
            return code;
        };
        
        MPI_Errhandler errh;
        MPI_Comm_create_errhandler(verbose_errhandler, &errh);
        MPI_Comm_set_errhandler(main_comm, errh);
        
        MPI_Comm_rank(main_comm, &w_rank);
        MPI_Comm_size(main_comm, &w_size);
        
        MPI_Barrier(main_comm);
        
        if (!w_rank) {
            printf("running on %d processes\n", w_size);
        }
        
        MPI_Barrier(main_comm);

        redistribute();

        double time = MPI_Wtime();

        init();
        copy_from_to(A, A_checkpoint);
        for(int it = 1; it <= itmax; it++) {
                setjmp(jbuf);
                eps = 0.;
                relax();
                if (w_rank == 0) {
                        //printf( "it=%4i   eps=%f\n", it,eps);
                        if (eps < maxeps) break;
                }
                copy_from_to(A, A_checkpoint);
        }
        verify();

        double all_time = MPI_Wtime() - time;

        if (w_rank == 0) {
                printf("size = %d\ntime = %lf\n", w_size, all_time);
        }
    
        MPI_Finalize();
        return 0;
}

void init() {
        for (int i = 0; i <= N - 1; i++) {
                for(int j = 0; j <= N - 1; j++) {
                        if (j == 0 || i == 0 || j == N - 1 || i == N - 1) A[i][j] = 0;
                        else A[i][j] = (1. + i + j) ;
                }
        }
}

void relax() {
        MPI_Status status;  
		MPI_Barrier(main_comm);
        for(int i = 1; i < N - 2; i++) {
                for(int j = 1; j < N - 2; j++) {
                    A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
                }
        }
        if (first_row != 0) {
                MPI_Send(A[first_row], N, MPI_DOUBLE, w_rank - 1, 0, main_comm); 
        }
        if (last_row != N) {
                int m = 0;
                MPI_Recv(tmp_mas, N, MPI_DOUBLE, w_rank + 1, 0, main_comm, &status);
                for (int j = 0; j < N - 1; j++) {
                        A[last_row][j] = tmp_mas[m++];
                }
        }
        if (last_row != N) {
                MPI_Send(A[last_row - 1], N, MPI_DOUBLE, w_rank + 1, 1, main_comm);
        }
        if (first_row != 0) {
        		int m = 0;
                MPI_Recv(tmp_mas, N, MPI_DOUBLE, w_rank - 1, 1, main_comm, &status);
                for (int j = 0; j < N; j++) {
                        A[first_row - 1][j] = tmp_mas[m++];
                }
        }
        MPI_Barrier(main_comm); 
        double local_eps = 0;

        for(int i = first_row; i < last_row; i++) {
                for(int j = 1; j < N - 2; j++) {
                    double e = A[i][j];
                    A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
                    local_eps = Max(local_eps, fabs(e - A[i][j]));
                }
        }
        if (first_row != 0) {
                MPI_Send(A[first_row], N, MPI_DOUBLE, w_rank - 1, 0, main_comm); 
        }
        if (last_row != N) {
        		int m = 0;
                MPI_Recv(tmp_mas, N, MPI_DOUBLE, w_rank + 1, 0, main_comm, &status);
                for (int j = 0; j < N; j++) {
                        A[last_row][j] = tmp_mas[m++];
                }
        }
        
        if (w_rank == RANK_TO_BE_KILLED) {
            fprintf(stderr, "killed1 %d\n", w_rank);
            fflush(stdout);
            raise(SIGKILL);
        }

        if (last_row != N) {
                MPI_Send(A[last_row - 1], N, MPI_DOUBLE, w_rank + 1, 1, main_comm);
        }
        if (first_row != 0) {
        		int m = 0;
                MPI_Recv(tmp_mas, N, MPI_DOUBLE, w_rank - 1, 1, main_comm, &status);
                for (int j = 0; j < N; j++) {
                        A[first_row - 1][j] = tmp_mas[m++];
                }
        }
        MPI_Barrier(main_comm);
        MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, main_comm);
}

void verify() {
        double local = 0.;
        double global = 0.;
        for(int i = first_row; i < last_row; i++) {
                for(int j = 0; j <= N - 1; j++) {
                        local = local + A[i][j] * (i + 1) * (j + 1) / (N * N);
                }
        }
        MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, main_comm);

        if (w_rank == 0) {
                printf("  S = %f\n", global);
        }
}
