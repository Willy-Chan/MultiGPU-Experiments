#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

// This script initializes 0-vector on rank 0, 1-vector on rank 1.
// Both ranks launch a "compute" kernel that just adds 1 to the vectors.
// Then rank 0 sends its new 1-vector to rank 1.

/**
 * CUDA Kernel: Increments each element by a value
 */
__global__ void add_value(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += value;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("This demo requires at least 2 MPI ranks.\n");
        MPI_Finalize();
        return 0;
    }

    // 1. DEVICE SELECTION (Crucial for multi-GPU nodes)
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % deviceCount); 

    int n = 1024;
    size_t bytes = n * sizeof(float);

    // 2. ALLOCATE GPU MEMORY
    float *d_data;
    cudaMalloc(&d_data, bytes);

    // 3. INITIALIZE DATA (Directly on GPU using cudaMemset or a kernel)
    // For simplicity, let's just fill it with the rank value
    float *h_init = (float*)malloc(bytes);
    for(int i=0; i<n; i++) h_init[i] = (float)rank;
    cudaMemcpy(d_data, h_init, bytes, cudaMemcpyHostToDevice);
    free(h_init);

    // 4. COMPUTE
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_value<<<blocks, threads>>>(d_data, 1.0f, n);
    
    // 5. CUDA-AWARE COMMUNICATION
    // We pass 'd_data' (a GPU pointer) directly to MPI.
    if (rank == 1) {
        printf("Rank 1: Sending GPU buffer to Rank 0...\n");
        MPI_Send(d_data, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 0) {
        float *d_recv;
        cudaMalloc(&d_recv, bytes);
        
        MPI_Recv(d_recv, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Verification
        float first_val;
        cudaMemcpy(&first_val, d_recv, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Rank 0: Received value %f from Rank 1's GPU\n", first_val);
        
        cudaFree(d_recv);
    }

    cudaFree(d_data);
    MPI_Finalize();
    return 0;
}