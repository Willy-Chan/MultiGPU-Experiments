
#define TILE_SIZE 32

// CUDA kernel for tiled matrix multiplication
__global__ void matmulTiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;      // each block corresponds a single tile of elements
    int col = bx * TILE_SIZE + tx;      // each thread in a block still corresponds to one element

    float sum = 0.0f;
    for (int t = 0; t < (K - 1) / TILE_SIZE + 1; t++) {
        if ()
    }
}

void main() {

    int N = 1024;

    // allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    float* h_C_GPU = (float*)malloc(size);


    // allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    // LAUNCH THE KERNEL
    dim3 blockDim(16, 16);    // 16x16 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    // launching 64 x 64 = 4096 blocks in my grid, with 16 x 16 = 256 threads in each block.
    // 4096 * 256 = 1024 * 1024 = 1048576 total elements of C.


    // Ensure kernel is finished
    cudaDeviceSynchronize();
    // copy result matrix C from device to host
    cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost);
}