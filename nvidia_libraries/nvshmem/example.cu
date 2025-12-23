#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

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


/**
 * CUDA Kernel: Increments each element by a value
 */
__global__ void add_value(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += value;
}

int main(int argc, char *argv[]) {
    // Initialize MPI first
    MPICHECK(MPI_Init(&argc, &argv));
    
    int rank, size;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (size < 2) {
        if (rank == 0) printf("This demo requires at least 2 MPI ranks.\n");
        MPI_Finalize();
        return 0;
    }

    // 1. DEVICE SELECTION
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    CUDACHECK(cudaSetDevice(rank % deviceCount));

    // 2. INITIALIZE NVSHMEM with MPI
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Get nvshmem rank and size
    int nvshmem_rank = nvshmem_my_pe();
    int nvshmem_size = nvshmem_n_pes();
    
    if (rank == 0) {
        printf("Initialized NVSHMEM with %d PEs\n", nvshmem_size);
    }

    // 3. CREATE CUDA STREAM
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // 4. ALLOCATE SYMMETRIC MEMORY (PGAS model)
    // nvshmem_malloc allocates symmetric memory that is accessible from all PEs
    int n = 1024;
    size_t bytes = n * sizeof(float);
    float *d_data = (float*)nvshmem_malloc(bytes);
    
    if (d_data == NULL) {
        printf("Rank %d: Failed to allocate symmetric memory\n", rank);
        exit(EXIT_FAILURE);
    }

    // 5. INITIALIZE DATA
    float *h_init = (float*)malloc(bytes);
    for(int i = 0; i < n; i++) h_init[i] = (float)rank;
    CUDACHECK(cudaMemcpy(d_data, h_init, bytes, cudaMemcpyHostToDevice));
    free(h_init);

    // 6. COMPUTE ON GPU
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_value<<<blocks, threads, 0, stream>>>(d_data, 1.0f, n);
    CUDACHECK(cudaStreamSynchronize(stream));

    // 7. NVSHMEM ONE-SIDED COMMUNICATION (put/get)
    // NVSHMEM uses a PGAS model: we can directly access remote memory
    // Each PE has symmetric memory allocated. We use nvshmem_ptr to get remote addresses.
    
    // Allocate a receive buffer on rank 0
    float *d_recv = NULL;
    if (rank == 0) {
        CUDACHECK(cudaMalloc(&d_recv, bytes));
    }
    
    // Synchronize before communication
    nvshmem_barrier_all();
    
    if (rank == 1) {
        printf("Rank 1: Sending GPU buffer to Rank 0 via NVSHMEM put...\n");
        // CORRECT USAGE: Pass the symmetric pointer and the target PE ID.
        // NVSHMEM handles the address calculation for Peer 0 automatically.

        // THIS IS USELESS: RANK0 JUST GETS
        nvshmem_float_put(d_data, d_data, n, 0); 
        nvshmem_quiet();  
    } 
    else if (rank == 0) {
        // CORRECT USAGE: Rank 0 gets data from Rank 1's 'd_data' buffer
        // and puts it into its local 'd_recv' (which is just standard cudaMalloc'd memory)
        nvshmem_float_get(d_recv, d_data, n, 1);
        nvshmem_quiet(); 
        
        // Verification
        float first_val;
        CUDACHECK(cudaMemcpy(&first_val, d_recv, sizeof(float), cudaMemcpyDeviceToHost));
        printf("Rank 0: Received value %f from Rank 1's GPU via NVSHMEM\n", first_val);
    }
    
    // Synchronize all PEs before cleanup
    nvshmem_barrier_all();
    
    if (d_recv != NULL) {
        CUDACHECK(cudaFree(d_recv));
    }

    // 8. CLEANUP
    nvshmem_barrier_all();  // Ensure all operations complete before cleanup
    nvshmem_free(d_data);
    CUDACHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
    MPICHECK(MPI_Finalize());
    return 0;
}