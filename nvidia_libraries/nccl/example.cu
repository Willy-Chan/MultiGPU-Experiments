#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

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


/**
 * CUDA Kernel: Increments each element by a value
 */
__global__ void add_value(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += value;
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

    // 1. DEVICE SELECTION (Crucial for multi-GPU nodes)
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    CUDACHECK(cudaSetDevice(rank % deviceCount));

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
    int n = 1024;
    size_t bytes = n * sizeof(float);
    float *d_data;
    CUDACHECK(cudaMalloc(&d_data, bytes));
    float *h_init = (float*)malloc(bytes);
    for(int i=0; i<n; i++) h_init[i] = (float)rank;
    CUDACHECK(cudaMemcpy(d_data, h_init, bytes, cudaMemcpyHostToDevice));
    free(h_init);

    // 6. COMPUTE ON GPU
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_value<<<blocks, threads, 0, stream>>>(d_data, 1.0f, n);
    
    // 7. NCCL COMMUNICATION (stream-aware, device-to-device)
    // We pass 'd_data' (a GPU pointer) directly to NCCL.
    // NCCL operations are asynchronous and respect CUDA streams.
    if (rank == 1) {
        printf("Rank 1: Sending GPU buffer to Rank 0 via NCCL...\n");
        NCCLCHECK(ncclSend(d_data, n, ncclFloat, 0, comm, stream));
    } 
    else if (rank == 0) {
        float *d_recv;
        CUDACHECK(cudaMalloc(&d_recv, bytes));
        
        NCCLCHECK(ncclRecv(d_recv, n, ncclFloat, 1, comm, stream));
        
        // Wait for NCCL operation to complete
        CUDACHECK(cudaStreamSynchronize(stream));
        
        // Verification
        float first_val;
        CUDACHECK(cudaMemcpy(&first_val, d_recv, sizeof(float), cudaMemcpyDeviceToHost));
        printf("Rank 0: Received value %f from Rank 1's GPU via NCCL\n", first_val);
        
        CUDACHECK(cudaFree(d_recv));
    }

    // 8. CLEANUP
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    CUDACHECK(cudaFree(d_data));
    MPICHECK(MPI_Finalize());
    return 0;
}