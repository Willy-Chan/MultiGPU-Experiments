#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// READ THIS: https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/
// SOLUTION CODE: https://github.com/NVIDIA/multi-gpu-programming-models/tree/master/mpi

// Run with: modal run run.py --cuda-file jacobi_solver.cu

// Type definition for real numbers
typedef float real;

// Simple Jacobi Stencil: a_new[i,j] = 0.25 * (a[i-1,j] + a[i+1,j] + a[i,j-1] + a[i,j+1])
__global__ void jacobi_kernel(int nx, int ny, real* a, real* a_new) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 because row 0 is halo
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 because col 0 is boundary

    if (i < ny - 1 && j < nx - 1) {
        a_new[i * nx + j] = 0.25f * (a[(i - 1) * nx + j] + a[(i + 1) * nx + j] +
                                     a[i * nx + j - 1] + a[i * nx + j + 1]);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // setup device
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(rank % dev_count);

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
    cudaMalloc(&d_a, bytes);
    cudaMemset(d_a, 0, bytes);
    cudaMalloc(&d_a_new, bytes);

    // Main iteration loop
    for (int i = 0; i < 100; i++) {
        // Halo Exchange: My top neighbor is rank - 1, bottom neighbor is rank + 1.
        int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int bottom = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        // Send my top row (index 1) to neighbor's bottom halo (index chunk_ny + 1)
        MPI_Sendrecv(d_a + (1 * num_cols), num_cols, MPI_FLOAT, top, 0,
                     d_a + ((chunk_ny + 1) * num_cols), num_cols, MPI_FLOAT, bottom, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // "bottom" does a send to destination "top". "top" does a receive from source "bottom".
        // Send my bottom row (index chunk_ny) to neighbor's top halo (index 0)
        MPI_Sendrecv(d_a + (chunk_ny * num_cols), num_cols, MPI_FLOAT, bottom, 1,
                     d_a +  (0 * num_cols), num_cols, MPI_FLOAT, top, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // "top" does a send to destination "bottom". "bottom" does a receive from source "top".

        // Now that our "halo" rows are filled, we can do some compute:
        dim3 threads(32, 32);
        dim3 blocks((nx + 31) / 32, (local_ny + 31) / 32);
        jacobi_kernel<<<blocks, threads>>>(nx, local_ny, d_a, d_a_new);

        // swap pointers to new grid
        real* tmp = d_a;
        d_a = d_a_new;
        d_a_new = tmp;

        if (rank == 0 && i % 10 == 0) printf("Iteration %d\n", i);
    }

    printf("Rank %d finished successfully.\n", rank);

    cudaFree(d_a);
    cudaFree(d_a_new);
    MPI_Finalize();
    return 0;
}