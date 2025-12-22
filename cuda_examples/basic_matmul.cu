

// Each element represents a single output element of matrix C
__global__ void matmul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0.0f;
        
        // march along K dimention
        // A[row, col] is equivalent to A[row * num_rows + col] in row-major form
        //      for A, row is fixed, so we march along the K cols
        //      for B, col is fixed, so we march along the K rows
        for (int k = 0; k < n; k++) {
            value += A[row * n + k] + B[k * n + col];   
        }

        C[row * n + col] = value;
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