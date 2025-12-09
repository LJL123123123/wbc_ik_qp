// AUTOMATICALLY GENERATED CODE FOR CUSADI

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__constant__ int nnz_in[] = {19};
__constant__ int nnz_out[] = {3};
__constant__ int n_w = 31;

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
        work[env_idx + 0] = inputs[0][idx * nnz_in[0] + 0];
        work[env_idx + 1] = -0.2770000000000000;
        work[env_idx + 2] = 1.0000000000000000;
        work[env_idx + 3] = inputs[0][idx * nnz_in[0] + 4];
        work[env_idx + 4] = 2.*(work[env_idx + 3]);
        work[env_idx + 5] = work[env_idx + 4] * work[env_idx + 3];
        work[env_idx + 6] = inputs[0][idx * nnz_in[0] + 5];
        work[env_idx + 7] = 2.*(work[env_idx + 6]);
        work[env_idx + 6] = work[env_idx + 7] * work[env_idx + 6];
        work[env_idx + 8] = work[env_idx + 5] + work[env_idx + 6];
        work[env_idx + 8] = work[env_idx + 2] - work[env_idx + 8];
        work[env_idx + 9] = work[env_idx + 1] * work[env_idx + 8];
        work[env_idx + 10] = 0.1160000000000000;
        work[env_idx + 11] = inputs[0][idx * nnz_in[0] + 3];
        work[env_idx + 12] = work[env_idx + 4] * work[env_idx + 11];
        work[env_idx + 13] = inputs[0][idx * nnz_in[0] + 6];
        work[env_idx + 14] = work[env_idx + 7] * work[env_idx + 13];
        work[env_idx + 15] = work[env_idx + 12] - work[env_idx + 14];
        work[env_idx + 16] = work[env_idx + 10] * work[env_idx + 15];
        work[env_idx + 9] = work[env_idx + 9] + work[env_idx + 16];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 9];
        work[env_idx + 9] = -0.0635000000000000;
        work[env_idx + 16] = work[env_idx + 9] * work[env_idx + 8];
        work[env_idx + 17] = 0.0410000000000000;
        work[env_idx + 18] = inputs[0][idx * nnz_in[0] + 10];
        work[env_idx + 19] = cos(work[env_idx + 18]);
        work[env_idx + 20] = work[env_idx + 15] * work[env_idx + 19];
        work[env_idx + 21] = work[env_idx + 7] * work[env_idx + 11];
        work[env_idx + 4] = work[env_idx + 4] * work[env_idx + 13];
        work[env_idx + 22] = work[env_idx + 21] + work[env_idx + 4];
        work[env_idx + 18] = sin(work[env_idx + 18]);
        work[env_idx + 23] = work[env_idx + 22] * work[env_idx + 18];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 23];
        work[env_idx + 23] = work[env_idx + 17] * work[env_idx + 20];
        work[env_idx + 16] = work[env_idx + 16] + work[env_idx + 23];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 16];
        work[env_idx + 16] = 0.1090000000000000;
        work[env_idx + 23] = work[env_idx + 16] * work[env_idx + 20];
        work[env_idx + 24] = -0.2500000000000000;
        work[env_idx + 25] = inputs[0][idx * nnz_in[0] + 11];
        work[env_idx + 26] = sin(work[env_idx + 25]);
        work[env_idx + 27] = work[env_idx + 8] * work[env_idx + 26];
        work[env_idx + 22] = work[env_idx + 22] * work[env_idx + 19];
        work[env_idx + 15] = work[env_idx + 15] * work[env_idx + 18];
        work[env_idx + 22] = work[env_idx + 22] - work[env_idx + 15];
        work[env_idx + 25] = cos(work[env_idx + 25]);
        work[env_idx + 15] = work[env_idx + 22] * work[env_idx + 25];
        work[env_idx + 27] = work[env_idx + 27] + work[env_idx + 15];
        work[env_idx + 15] = work[env_idx + 24] * work[env_idx + 27];
        work[env_idx + 23] = work[env_idx + 23] + work[env_idx + 15];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 23];
        work[env_idx + 23] = -0.1000000000000000;
        work[env_idx + 8] = work[env_idx + 8] * work[env_idx + 25];
        work[env_idx + 22] = work[env_idx + 22] * work[env_idx + 26];
        work[env_idx + 8] = work[env_idx + 8] - work[env_idx + 22];
        work[env_idx + 22] = inputs[0][idx * nnz_in[0] + 12];
        work[env_idx + 15] = cos(work[env_idx + 22]);
        work[env_idx + 28] = work[env_idx + 8] * work[env_idx + 15];
        work[env_idx + 22] = sin(work[env_idx + 22]);
        work[env_idx + 29] = work[env_idx + 27] * work[env_idx + 22];
        work[env_idx + 28] = work[env_idx + 28] - work[env_idx + 29];
        work[env_idx + 28] = work[env_idx + 23] * work[env_idx + 28];
        work[env_idx + 29] = -0.0200000000000000;
        work[env_idx + 20] = work[env_idx + 29] * work[env_idx + 20];
        work[env_idx + 30] = -0.3212500000000000;
        work[env_idx + 8] = work[env_idx + 8] * work[env_idx + 22];
        work[env_idx + 27] = work[env_idx + 27] * work[env_idx + 15];
        work[env_idx + 8] = work[env_idx + 8] + work[env_idx + 27];
        work[env_idx + 8] = work[env_idx + 30] * work[env_idx + 8];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 8];
        work[env_idx + 28] = work[env_idx + 28] + work[env_idx + 20];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 28];
        outputs[0][idx * nnz_out[0] + 0] = work[env_idx + 0];
        work[env_idx + 0] = inputs[0][idx * nnz_in[0] + 1];
        work[env_idx + 12] = work[env_idx + 12] + work[env_idx + 14];
        work[env_idx + 14] = work[env_idx + 1] * work[env_idx + 12];
        work[env_idx + 28] = 2.*(work[env_idx + 11]);
        work[env_idx + 11] = work[env_idx + 28] * work[env_idx + 11];
        work[env_idx + 6] = work[env_idx + 11] + work[env_idx + 6];
        work[env_idx + 6] = work[env_idx + 2] - work[env_idx + 6];
        work[env_idx + 20] = work[env_idx + 10] * work[env_idx + 6];
        work[env_idx + 14] = work[env_idx + 14] + work[env_idx + 20];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 14];
        work[env_idx + 14] = work[env_idx + 9] * work[env_idx + 12];
        work[env_idx + 20] = work[env_idx + 6] * work[env_idx + 19];
        work[env_idx + 7] = work[env_idx + 7] * work[env_idx + 3];
        work[env_idx + 28] = work[env_idx + 28] * work[env_idx + 13];
        work[env_idx + 13] = work[env_idx + 7] - work[env_idx + 28];
        work[env_idx + 3] = work[env_idx + 13] * work[env_idx + 18];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 3];
        work[env_idx + 3] = work[env_idx + 17] * work[env_idx + 20];
        work[env_idx + 14] = work[env_idx + 14] + work[env_idx + 3];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 14];
        work[env_idx + 14] = work[env_idx + 16] * work[env_idx + 20];
        work[env_idx + 3] = work[env_idx + 12] * work[env_idx + 26];
        work[env_idx + 13] = work[env_idx + 13] * work[env_idx + 19];
        work[env_idx + 6] = work[env_idx + 6] * work[env_idx + 18];
        work[env_idx + 13] = work[env_idx + 13] - work[env_idx + 6];
        work[env_idx + 6] = work[env_idx + 13] * work[env_idx + 25];
        work[env_idx + 3] = work[env_idx + 3] + work[env_idx + 6];
        work[env_idx + 6] = work[env_idx + 24] * work[env_idx + 3];
        work[env_idx + 14] = work[env_idx + 14] + work[env_idx + 6];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 14];
        work[env_idx + 12] = work[env_idx + 12] * work[env_idx + 25];
        work[env_idx + 13] = work[env_idx + 13] * work[env_idx + 26];
        work[env_idx + 12] = work[env_idx + 12] - work[env_idx + 13];
        work[env_idx + 13] = work[env_idx + 12] * work[env_idx + 15];
        work[env_idx + 14] = work[env_idx + 3] * work[env_idx + 22];
        work[env_idx + 13] = work[env_idx + 13] - work[env_idx + 14];
        work[env_idx + 13] = work[env_idx + 23] * work[env_idx + 13];
        work[env_idx + 20] = work[env_idx + 29] * work[env_idx + 20];
        work[env_idx + 12] = work[env_idx + 12] * work[env_idx + 22];
        work[env_idx + 3] = work[env_idx + 3] * work[env_idx + 15];
        work[env_idx + 12] = work[env_idx + 12] + work[env_idx + 3];
        work[env_idx + 12] = work[env_idx + 30] * work[env_idx + 12];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 12];
        work[env_idx + 13] = work[env_idx + 13] + work[env_idx + 20];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 13];
        outputs[0][idx * nnz_out[0] + 1] = work[env_idx + 0];
        work[env_idx + 0] = inputs[0][idx * nnz_in[0] + 2];
        work[env_idx + 21] = work[env_idx + 21] - work[env_idx + 4];
        work[env_idx + 1] = work[env_idx + 1] * work[env_idx + 21];
        work[env_idx + 7] = work[env_idx + 7] + work[env_idx + 28];
        work[env_idx + 10] = work[env_idx + 10] * work[env_idx + 7];
        work[env_idx + 1] = work[env_idx + 1] + work[env_idx + 10];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 1];
        work[env_idx + 9] = work[env_idx + 9] * work[env_idx + 21];
        work[env_idx + 1] = work[env_idx + 7] * work[env_idx + 19];
        work[env_idx + 11] = work[env_idx + 11] + work[env_idx + 5];
        work[env_idx + 2] = work[env_idx + 2] - work[env_idx + 11];
        work[env_idx + 11] = work[env_idx + 2] * work[env_idx + 18];
        work[env_idx + 1] = work[env_idx + 1] + work[env_idx + 11];
        work[env_idx + 17] = work[env_idx + 17] * work[env_idx + 1];
        work[env_idx + 9] = work[env_idx + 9] + work[env_idx + 17];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 9];
        work[env_idx + 16] = work[env_idx + 16] * work[env_idx + 1];
        work[env_idx + 9] = work[env_idx + 21] * work[env_idx + 26];
        work[env_idx + 2] = work[env_idx + 2] * work[env_idx + 19];
        work[env_idx + 7] = work[env_idx + 7] * work[env_idx + 18];
        work[env_idx + 2] = work[env_idx + 2] - work[env_idx + 7];
        work[env_idx + 7] = work[env_idx + 2] * work[env_idx + 25];
        work[env_idx + 9] = work[env_idx + 9] + work[env_idx + 7];
        work[env_idx + 24] = work[env_idx + 24] * work[env_idx + 9];
        work[env_idx + 16] = work[env_idx + 16] + work[env_idx + 24];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 16];
        work[env_idx + 21] = work[env_idx + 21] * work[env_idx + 25];
        work[env_idx + 2] = work[env_idx + 2] * work[env_idx + 26];
        work[env_idx + 21] = work[env_idx + 21] - work[env_idx + 2];
        work[env_idx + 2] = work[env_idx + 21] * work[env_idx + 15];
        work[env_idx + 26] = work[env_idx + 9] * work[env_idx + 22];
        work[env_idx + 2] = work[env_idx + 2] - work[env_idx + 26];
        work[env_idx + 23] = work[env_idx + 23] * work[env_idx + 2];
        work[env_idx + 29] = work[env_idx + 29] * work[env_idx + 1];
        work[env_idx + 21] = work[env_idx + 21] * work[env_idx + 22];
        work[env_idx + 9] = work[env_idx + 9] * work[env_idx + 15];
        work[env_idx + 21] = work[env_idx + 21] + work[env_idx + 9];
        work[env_idx + 30] = work[env_idx + 30] * work[env_idx + 21];
        work[env_idx + 29] = work[env_idx + 29] + work[env_idx + 30];
        work[env_idx + 23] = work[env_idx + 23] + work[env_idx + 29];
        work[env_idx + 0] = work[env_idx + 0] + work[env_idx + 23];
        outputs[0][idx * nnz_out[0] + 2] = work[env_idx + 0];
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