// AUTOMATICALLY GENERATED CODE FOR CUSADI

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__constant__ int nnz_in[] = {19};
__constant__ int nnz_out[] = {9};
__constant__ int n_w = 23;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
}
}


__global__ void evaluate_kernel (
        const double *inputs[],
        double *work,
        double *outputs[],
        const int batch_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int env_idx = idx * n_w;
    if (idx < batch_size) {
        work[env_idx + 0] = 1.0000000000000000;
        work[env_idx + 1] = inputs[0][idx * nnz_in[0] + 4];
        work[env_idx + 2] = 2.*(work[env_idx + 1]);
        work[env_idx + 3] = work[env_idx + 2] * work[env_idx + 1];
        work[env_idx + 4] = inputs[0][idx * nnz_in[0] + 5];
        work[env_idx + 5] = 2.*(work[env_idx + 4]);
        work[env_idx + 4] = work[env_idx + 5] * work[env_idx + 4];
        work[env_idx + 6] = work[env_idx + 3] + work[env_idx + 4];
        work[env_idx + 6] = work[env_idx + 0] - work[env_idx + 6];
        work[env_idx + 7] = inputs[0][idx * nnz_in[0] + 8];
        work[env_idx + 8] = cos(work[env_idx + 7]);
        work[env_idx + 9] = work[env_idx + 6] * work[env_idx + 8];
        work[env_idx + 10] = inputs[0][idx * nnz_in[0] + 3];
        work[env_idx + 11] = work[env_idx + 5] * work[env_idx + 10];
        work[env_idx + 12] = inputs[0][idx * nnz_in[0] + 6];
        work[env_idx + 13] = work[env_idx + 2] * work[env_idx + 12];
        work[env_idx + 14] = work[env_idx + 11] + work[env_idx + 13];
        work[env_idx + 15] = inputs[0][idx * nnz_in[0] + 7];
        work[env_idx + 16] = cos(work[env_idx + 15]);
        work[env_idx + 17] = work[env_idx + 14] * work[env_idx + 16];
        work[env_idx + 2] = work[env_idx + 2] * work[env_idx + 10];
        work[env_idx + 18] = work[env_idx + 5] * work[env_idx + 12];
        work[env_idx + 19] = work[env_idx + 2] - work[env_idx + 18];
        work[env_idx + 15] = sin(work[env_idx + 15]);
        work[env_idx + 20] = work[env_idx + 19] * work[env_idx + 15];
        work[env_idx + 17] = work[env_idx + 17] - work[env_idx + 20];
        work[env_idx + 7] = sin(work[env_idx + 7]);
        work[env_idx + 20] = work[env_idx + 17] * work[env_idx + 7];
        work[env_idx + 9] = work[env_idx + 9] - work[env_idx + 20];
        work[env_idx + 20] = inputs[0][idx * nnz_in[0] + 9];
        work[env_idx + 21] = cos(work[env_idx + 20]);
        work[env_idx + 22] = work[env_idx + 9] * work[env_idx + 21];
        work[env_idx + 6] = work[env_idx + 6] * work[env_idx + 7];
        work[env_idx + 17] = work[env_idx + 17] * work[env_idx + 8];
        work[env_idx + 6] = work[env_idx + 6] + work[env_idx + 17];
        work[env_idx + 20] = sin(work[env_idx + 20]);
        work[env_idx + 17] = work[env_idx + 6] * work[env_idx + 20];
        work[env_idx + 22] = work[env_idx + 22] - work[env_idx + 17];
        outputs[0][idx * nnz_out[0] + 0] = work[env_idx + 22];
        work[env_idx + 19] = work[env_idx + 19] * work[env_idx + 16];
        work[env_idx + 14] = work[env_idx + 14] * work[env_idx + 15];
        work[env_idx + 19] = work[env_idx + 19] + work[env_idx + 14];
        outputs[0][idx * nnz_out[0] + 1] = work[env_idx + 19];
        work[env_idx + 9] = work[env_idx + 9] * work[env_idx + 20];
        work[env_idx + 6] = work[env_idx + 6] * work[env_idx + 21];
        work[env_idx + 9] = work[env_idx + 9] + work[env_idx + 6];
        outputs[0][idx * nnz_out[0] + 2] = work[env_idx + 9];
        work[env_idx + 2] = work[env_idx + 2] + work[env_idx + 18];
        work[env_idx + 18] = work[env_idx + 2] * work[env_idx + 8];
        work[env_idx + 5] = work[env_idx + 5] * work[env_idx + 1];
        work[env_idx + 1] = 2.*(work[env_idx + 10]);
        work[env_idx + 12] = work[env_idx + 1] * work[env_idx + 12];
        work[env_idx + 9] = work[env_idx + 5] - work[env_idx + 12];
        work[env_idx + 6] = work[env_idx + 9] * work[env_idx + 16];
        work[env_idx + 1] = work[env_idx + 1] * work[env_idx + 10];
        work[env_idx + 4] = work[env_idx + 1] + work[env_idx + 4];
        work[env_idx + 4] = work[env_idx + 0] - work[env_idx + 4];
        work[env_idx + 10] = work[env_idx + 4] * work[env_idx + 15];
        work[env_idx + 6] = work[env_idx + 6] - work[env_idx + 10];
        work[env_idx + 10] = work[env_idx + 6] * work[env_idx + 7];
        work[env_idx + 18] = work[env_idx + 18] - work[env_idx + 10];
        work[env_idx + 10] = work[env_idx + 18] * work[env_idx + 21];
        work[env_idx + 2] = work[env_idx + 2] * work[env_idx + 7];
        work[env_idx + 6] = work[env_idx + 6] * work[env_idx + 8];
        work[env_idx + 2] = work[env_idx + 2] + work[env_idx + 6];
        work[env_idx + 6] = work[env_idx + 2] * work[env_idx + 20];
        work[env_idx + 10] = work[env_idx + 10] - work[env_idx + 6];
        outputs[0][idx * nnz_out[0] + 3] = work[env_idx + 10];
        work[env_idx + 4] = work[env_idx + 4] * work[env_idx + 16];
        work[env_idx + 9] = work[env_idx + 9] * work[env_idx + 15];
        work[env_idx + 4] = work[env_idx + 4] + work[env_idx + 9];
        outputs[0][idx * nnz_out[0] + 4] = work[env_idx + 4];
        work[env_idx + 18] = work[env_idx + 18] * work[env_idx + 20];
        work[env_idx + 2] = work[env_idx + 2] * work[env_idx + 21];
        work[env_idx + 18] = work[env_idx + 18] + work[env_idx + 2];
        outputs[0][idx * nnz_out[0] + 5] = work[env_idx + 18];
        work[env_idx + 11] = work[env_idx + 11] - work[env_idx + 13];
        work[env_idx + 13] = work[env_idx + 11] * work[env_idx + 8];
        work[env_idx + 1] = work[env_idx + 1] + work[env_idx + 3];
        work[env_idx + 0] = work[env_idx + 0] - work[env_idx + 1];
        work[env_idx + 1] = work[env_idx + 0] * work[env_idx + 16];
        work[env_idx + 5] = work[env_idx + 5] + work[env_idx + 12];
        work[env_idx + 12] = work[env_idx + 5] * work[env_idx + 15];
        work[env_idx + 1] = work[env_idx + 1] - work[env_idx + 12];
        work[env_idx + 12] = work[env_idx + 1] * work[env_idx + 7];
        work[env_idx + 13] = work[env_idx + 13] - work[env_idx + 12];
        work[env_idx + 12] = work[env_idx + 13] * work[env_idx + 21];
        work[env_idx + 11] = work[env_idx + 11] * work[env_idx + 7];
        work[env_idx + 1] = work[env_idx + 1] * work[env_idx + 8];
        work[env_idx + 11] = work[env_idx + 11] + work[env_idx + 1];
        work[env_idx + 1] = work[env_idx + 11] * work[env_idx + 20];
        work[env_idx + 12] = work[env_idx + 12] - work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 6] = work[env_idx + 12];
        work[env_idx + 5] = work[env_idx + 5] * work[env_idx + 16];
        work[env_idx + 0] = work[env_idx + 0] * work[env_idx + 15];
        work[env_idx + 5] = work[env_idx + 5] + work[env_idx + 0];
        outputs[0][idx * nnz_out[0] + 7] = work[env_idx + 5];
        work[env_idx + 13] = work[env_idx + 13] * work[env_idx + 20];
        work[env_idx + 11] = work[env_idx + 11] * work[env_idx + 21];
        work[env_idx + 13] = work[env_idx + 13] + work[env_idx + 11];
        outputs[0][idx * nnz_out[0] + 8] = work[env_idx + 13];
    }
}


extern "C" {

float evaluate(const double *inputs[],
            double *work,
            double *outputs[],
            const int batch_size) {
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                            work,
                                            outputs,
                                            batch_size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    return milliseconds/1000;
}

}