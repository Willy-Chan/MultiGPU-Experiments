// In the prior example, we don't overlap compute and communication at all.
// Looking at NSight though, we can see that in the time it takes us to move data, we can get away with more kernel launches!

#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

typedef float real;

// Standard 5-point stencil kernel
__global__ void jacobi_kernel(int nx, real* a, real* a_new, real* l2_norm, int start_row, int end_row) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + start_row;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < end_row && j < nx - 1) {
        real val = 0.25f * (a[(i-1)*nx + j] + a[(i+1)*nx + j] + a[i*nx + j-1] + a[i*nx + j+1]);
        a_new[i*nx + j] = val;
        real diff = val - a[i*nx + j];
        atomicAdd(l2_norm, diff * diff);
    }
}

int jacobiSolver(real* d_a, real* d_a_new, real* d_l2, real* h_l2, int chunk_ny, int nx, int rank, int size, 
                 cudaStream_t s_bound, cudaStream_t s_int) {

    real l2_global = 1.0f;
    int iter = 0;
    const int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int bottom = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Boundary rows for this rank
    const int row_top = 1;
    const int row_bot = chunk_ny;

    MPI_Request reqs[4];

    // for iteration:
    //      compute boundaries
    //      start async communication to halo boundaries
    //      compute interior
    //      sync and update L2 and grid

    while (l2_global > 1e-6 && iter < 1000) {
        cudaMemsetAsync(d_l2, 0, sizeof(real), s_int);

        // --- STEP 1: COMPUTE BOUNDARIES (Stream: s_bound) ---
        dim3 threads(32, 1);
        dim3 blocks_b((nx + 31) / 32, 1);
        jacobi_kernel<<<blocks_b, threads, 0, s_bound>>>(nx, d_a, d_a_new, d_l2, row_top, row_top + 1);
        jacobi_kernel<<<blocks_b, threads, 0, s_bound>>>(nx, d_a, d_a_new, d_l2, row_bot, row_bot + 1);

        // We MUST ensure the GPU is done with boundaries before MPI reads from them
        cudaStreamSynchronize(s_bound);

        // --- STEP 2: START ASYNC COMMUNICATION (CPU + Network) ---
        // Receive into halos
        MPI_Irecv(d_a_new, nx, MPI_FLOAT, top, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(d_a_new + (chunk_ny + 1) * nx, nx, MPI_FLOAT, bottom, 1, MPI_COMM_WORLD, &reqs[1]);
        // Send from computed boundaries
        MPI_Isend(d_a_new + nx, nx, MPI_FLOAT, top, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(d_a_new + chunk_ny * nx, nx, MPI_FLOAT, bottom, 0, MPI_COMM_WORLD, &reqs[3]);

        // --- STEP 3: COMPUTE INTERIOR (Stream: s_int) ---
        // This runs ON THE GPU while the NETWORK is moving the boundary data!
        if (chunk_ny > 2) {
            dim3 blocks_i((nx + 31) / 32, (chunk_ny - 2 + 31) / 32);
            jacobi_kernel<<<blocks_i, threads, 0, s_int>>>(nx, d_a, d_a_new, d_l2, 2, chunk_ny);
        }

        // --- STEP 4: SYNCHRONIZE ---
        // We do irecv and isend with 4 elements in the reqs array: so here we just wait for all 4 of them
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE); // Wait for network
        cudaStreamSynchronize(s_int);              // Wait for interior compute

        // L2 Norm Reduction
        // L2 Norm is the "L2 distance" between a_new and a: when it stops changing we have "converged".
        // Each rank computes its own d_l2, and so we do an allreduce to sum those up and then take a sqrt at the end.
        cudaMemcpyAsync(h_l2, d_l2, sizeof(real), cudaMemcpyDeviceToHost, s_int);
        cudaStreamSynchronize(s_int);
        MPI_Allreduce(h_l2, &l2_global, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        l2_global = sqrtf(l2_global);

        if (rank == 0 && iter % 100 == 0) printf("Iter %d, Norm: %e\n", iter, l2_global);

        std::swap(d_a, d_a_new);
        iter++;
    }
    return iter;
}

int main(int argc, char* argv[]) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 2. Global Problem Size
    int nx = 1024;
    int global_ny = 1024;
    
    // 3. Domain Decomposition
    // Each rank gets a "chunk" of rows. 
    // Example: 1024 rows / 2 GPUs = 512 rows per GPU.
    int chunk_ny = global_ny / size;
    
    // 4. Device Selection (One GPU per MPI Rank)
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(rank % dev_count);

    // 5. Memory Allocation
    // We allocate chunk_ny + 2 rows to account for the top and bottom halos.
    real *d_a, *d_a_new, *d_l2, *h_l2;
    size_t bytes = nx * (chunk_ny + 2) * sizeof(real);
    
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_a_new, bytes);
    cudaMalloc(&d_l2, sizeof(real));
    h_l2 = (real*)malloc(sizeof(real));

    // Initialize the entire local grid to 0.0
    cudaMemset(d_a, 0, bytes);
    cudaMemset(d_a_new, 0, bytes);

    // 6. INITIALIZE BOUNDARY CONDITIONS (The "Heat Source")
    // If we are Rank 0, we set the top edge (row 0) to 1.0 (hot).
    // This "heat" will diffuse down through all ranks via the halo exchange.
    if (rank == 0) {
        real* h_temp = (real*)malloc(nx * sizeof(real));
        for(int i=0; i<nx; i++) h_temp[i] = 1.0f;
        cudaMemcpy(d_a, h_temp, nx * sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_new, h_temp, nx * sizeof(real), cudaMemcpyHostToDevice);
        free(h_temp);
    }

    // 7. Create Multiple Streams for Overlap
    // s_bound: High priority for the boundary rows
    // s_int: Normal priority for the bulk interior
    cudaStream_t s_bound, s_int;
    cudaStreamCreateWithPriority(&s_bound, cudaStreamNonBlocking, -1); // -1 is high priority
    cudaStreamCreateWithPriority(&s_int, cudaStreamNonBlocking, 0);

    // 8. Run the Solver
    if (rank == 0) printf("Starting Jacobi Solver on %d GPUs...\n", size);
    
    int total_iters = jacobiSolver(d_a, d_a_new, d_l2, h_l2, 
                                   chunk_ny, nx, rank, size, 
                                   s_bound, s_int);

    if (rank == 0) printf("Converged after %d iterations.\n", total_iters);

    // 9. Cleanup
    cudaFree(d_a); cudaFree(d_a_new); cudaFree(d_l2);
    free(h_l2);
    cudaStreamDestroy(s_bound);
    cudaStreamDestroy(s_int);
    
    MPI_Finalize();
    return 0;
}