#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                      \
    if( e != MPI_SUCCESS ) {                          \
        printf("Failed: MPI error %s:%d '%d'\n",        \
            __FILE__,__LINE__, e);   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)
  
#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",             \
            __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",             \
            __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)


// Type definition for real numbers
typedef float real;

// Simple Jacobi Stencil: a_new[i,j] = 0.25 * (a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1])
// Also computes L2 norm of the difference (for convergence checking)
__global__ void jacobi_kernel(int nx, int ny, real* a, real* a_new, real* l2_norm) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 because row 0 is halo
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 because col 0 is boundary

    if (i < ny - 1 && j < nx - 1) {
        real val = 0.25f * (a[(i - 1) * nx + j] + a[(i + 1) * nx + j] +
                            a[i * nx + j - 1] + a[i * nx + j + 1]);
        a_new[i * nx + j] = val;
        // Compute squared difference for L2 norm
        real diff = val - a[i * nx + j];
        atomicAdd(l2_norm, diff * diff);
    }
}

int main(int argc, char *argv[]) {
    // Initialize MPI (used for rank/size and broadcasting NCCL unique ID)
    MPICHECK(MPI_Init(&argc, &argv));
    
    int rank, size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (size < 2) {
        if (rank == 0) printf("This demo requires at least 2 MPI ranks.\n");
        MPI_Finalize();
        return 0;
    }

    // setup device
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    CUDACHECK(cudaSetDevice(rank % dev_count));

    // 2. GET AND BROADCAST NCCL UNIQUE ID
    ncclUniqueId id;
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    // 3. INITIALIZE NCCL COMMUNICATOR
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    // 4. CREATE CUDA STREAM (NCCL is stream-aware)
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // 5. ALLOCATE AND INITIALIZE GPU DATA
    // Problem Parameters: nx is the number of columns, ny is the number of rows.
    // Here, we split the 1024 x 1024 grid into two sets of rows.
    int nx = 1024;
    int global_ny = 1024;
    int chunk_ny = global_ny / size;

    int local_ny = chunk_ny + 2;       // each rank needs chunk_ny + 2 rows (1 top halo, 1 bottom halo)
    size_t bytes = nx * local_ny * sizeof(real);    // total num bytes of a single rank's grid

    int num_cols = nx;
    int num_local_rows = local_ny;

    // Each rank allocates this memory
    real* d_a;
    real* d_a_new;
    real* d_l2;  // Device memory for L2 norm (local contribution)
    real* h_l2;  // Host memory for L2 norm (local)
    real l2_global;  // Global L2 norm after allreduce
    
    CUDACHECK(cudaMalloc(&d_a, bytes));
    CUDACHECK(cudaMemset(d_a, 0, bytes));
    CUDACHECK(cudaMalloc(&d_a_new, bytes));
    CUDACHECK(cudaMemset(d_a_new, 0, bytes));
    CUDACHECK(cudaMalloc(&d_l2, sizeof(real)));
    h_l2 = (real*)malloc(sizeof(real));

    // Initialize boundary conditions (the "heat source")
    // If we are Rank 0, we set the top edge (row 0) to 1.0 (hot).
    // This "heat" will diffuse down through all ranks via the halo exchange.
    if (rank == 0) {
        real* h_temp = (real*)malloc(nx * sizeof(real));
        for(int i = 0; i < nx; i++) h_temp[i] = 1.0f;
        CUDACHECK(cudaMemcpy(d_a, h_temp, nx * sizeof(real), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_a_new, h_temp, nx * sizeof(real), cudaMemcpyHostToDevice));
        free(h_temp);
    }

    // Main iteration loop
    for (int i = 0; i < 100; i++) {
        // Reset L2 norm for this iteration
        CUDACHECK(cudaMemsetAsync(d_l2, 0, sizeof(real), stream));
        
        // Halo Exchange: My top neighbor is rank - 1, bottom neighbor is rank + 1.
        // Note: NCCL doesn't support MPI_PROC_NULL, so we conditionally skip communication
        int top = (rank > 0) ? rank - 1 : -1;
        int bottom = (rank < size - 1) ? rank + 1 : -1;

        // Exchange halos using ncclGroup (equivalent to MPI_Sendrecv)
        // First exchange: Send my top interior row (index 1) to top neighbor's bottom halo
        //                 Receive from bottom neighbor's top interior row into my bottom halo
        if (top >= 0 || bottom >= 0) {
            ncclGroupStart();
            if (top >= 0) {
                // Send row 1 to top neighbor (will be received into their row chunk_ny + 1)
                NCCLCHECK(ncclSend(d_a + (1 * num_cols), num_cols, ncclFloat, top, comm, stream));
            }
            if (bottom >= 0) {
                // Receive from bottom neighbor's row 1 into my row chunk_ny + 1
                NCCLCHECK(ncclRecv(d_a + ((chunk_ny + 1) * num_cols), num_cols, ncclFloat, bottom, comm, stream));
            }
            ncclGroupEnd();
        }

        // Second exchange: Send my bottom interior row (index chunk_ny) to bottom neighbor's top halo
        //                  Receive from top neighbor's bottom interior row into my top halo
        if (top >= 0 || bottom >= 0) {
            ncclGroupStart();
            if (bottom >= 0) {
                // Send row chunk_ny to bottom neighbor (will be received into their row 0)
                NCCLCHECK(ncclSend(d_a + (chunk_ny * num_cols), num_cols, ncclFloat, bottom, comm, stream));
            }
            if (top >= 0) {
                // Receive from top neighbor's row chunk_ny into my row 0
                NCCLCHECK(ncclRecv(d_a + (0 * num_cols), num_cols, ncclFloat, top, comm, stream));
            }
            ncclGroupEnd();
        }

        // Now that our "halo" rows are filled, we can do some compute:
        dim3 threads(32, 32);
        dim3 blocks((nx + 31) / 32, (local_ny + 31) / 32);
        jacobi_kernel<<<blocks, threads, 0, stream>>>(nx, local_ny, d_a, d_a_new, d_l2);

        // Use ncclAllreduce to sum L2 norms across all ranks (operates on device memory)
        NCCLCHECK(ncclAllReduce(d_l2, d_l2, 1, ncclFloat, ncclSum, comm, stream));
        
        // Wait for allreduce to complete, then copy result to host
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaMemcpy(&l2_global, d_l2, sizeof(real), cudaMemcpyDeviceToHost));
        
        // Compute square root for actual L2 norm
        l2_global = sqrtf(l2_global);

        // swap pointers to new grid
        real* tmp = d_a;
        d_a = d_a_new;
        d_a_new = tmp;

        // Print convergence info
        if (rank == 0 && (i % 10 == 0 || i < 5)) {
            printf("Iteration %d: L2 norm = %e\n", i, l2_global);
        }
    }

    printf("Rank %d finished successfully.\n", rank);

    // 8. CLEANUP
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaFree(d_a));
    CUDACHECK(cudaFree(d_a_new));
    CUDACHECK(cudaFree(d_l2));
    free(h_l2);
    MPICHECK(MPI_Finalize());
    return 0;
}