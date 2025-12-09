// AUTOMATICALLY GENERATED CODE FOR CUSADI

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__constant__ int nnz_in[] = {19};
__constant__ int nnz_out[] = {3};
__constant__ int n_w = 35;

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
        work[env_idx + 7] = -0.0329246506683976;
        work[env_idx + 8] = -0.0919111929717844;
        work[env_idx + 9] = inputs[0][idx * nnz_in[0] + 17];
        work[env_idx + 10] = cos(work[env_idx + 9]);
        work[env_idx + 11] = 0.0063730860831871;
        work[env_idx + 12] = -0.0204024622805245;
        work[env_idx + 13] = inputs[0][idx * nnz_in[0] + 18];
        work[env_idx + 14] = cos(work[env_idx + 13]);
        work[env_idx + 15] = work[env_idx + 12] * work[env_idx + 14];
        work[env_idx + 16] = -0.0790949304856612;
        work[env_idx + 13] = sin(work[env_idx + 13]);
        work[env_idx + 17] = work[env_idx + 16] * work[env_idx + 13];
        work[env_idx + 15] = work[env_idx + 15] + work[env_idx + 17];
        work[env_idx + 15] = work[env_idx + 11] + work[env_idx + 15];
        work[env_idx + 17] = work[env_idx + 10] * work[env_idx + 15];
        work[env_idx + 9] = sin(work[env_idx + 9]);
        work[env_idx + 18] = -0.3508387663642895;
        work[env_idx + 14] = work[env_idx + 16] * work[env_idx + 14];
        work[env_idx + 13] = work[env_idx + 12] * work[env_idx + 13];
        work[env_idx + 14] = work[env_idx + 14] - work[env_idx + 13];
        work[env_idx + 13] = -0.0868437672500000;
        work[env_idx + 14] = work[env_idx + 14] + work[env_idx + 13];
        work[env_idx + 14] = work[env_idx + 18] + work[env_idx + 14];
        work[env_idx + 19] = work[env_idx + 9] * work[env_idx + 14];
        work[env_idx + 17] = work[env_idx + 17] + work[env_idx + 19];
        work[env_idx + 19] = -0.1258793225360000;
        work[env_idx + 17] = work[env_idx + 17] + work[env_idx + 19];
        work[env_idx + 17] = work[env_idx + 8] + work[env_idx + 17];
        work[env_idx + 20] = -0.9437312927520001;
        work[env_idx + 17] = work[env_idx + 17] + work[env_idx + 20];
        work[env_idx + 7] = work[env_idx + 7] + work[env_idx + 17];
        work[env_idx + 17] = 0.0919111929717844;
        work[env_idx + 21] = inputs[0][idx * nnz_in[0] + 14];
        work[env_idx + 22] = cos(work[env_idx + 21]);
        work[env_idx + 23] = -0.0063730860831871;
        work[env_idx + 24] = 0.0204024622805245;
        work[env_idx + 25] = inputs[0][idx * nnz_in[0] + 15];
        work[env_idx + 26] = cos(work[env_idx + 25]);
        work[env_idx + 27] = work[env_idx + 24] * work[env_idx + 26];
        work[env_idx + 25] = sin(work[env_idx + 25]);
        work[env_idx + 28] = work[env_idx + 16] * work[env_idx + 25];
        work[env_idx + 27] = work[env_idx + 27] + work[env_idx + 28];
        work[env_idx + 27] = work[env_idx + 23] + work[env_idx + 27];
        work[env_idx + 28] = work[env_idx + 22] * work[env_idx + 27];
        work[env_idx + 21] = sin(work[env_idx + 21]);
        work[env_idx + 26] = work[env_idx + 16] * work[env_idx + 26];
        work[env_idx + 25] = work[env_idx + 24] * work[env_idx + 25];
        work[env_idx + 26] = work[env_idx + 26] - work[env_idx + 25];
        work[env_idx + 26] = work[env_idx + 26] + work[env_idx + 13];
        work[env_idx + 26] = work[env_idx + 18] + work[env_idx + 26];
        work[env_idx + 25] = work[env_idx + 21] * work[env_idx + 26];
        work[env_idx + 28] = work[env_idx + 28] + work[env_idx + 25];
        work[env_idx + 25] = 0.1258793225360000;
        work[env_idx + 28] = work[env_idx + 28] + work[env_idx + 25];
        work[env_idx + 28] = work[env_idx + 17] + work[env_idx + 28];
        work[env_idx + 29] = 0.9437312927520001;
        work[env_idx + 28] = work[env_idx + 28] + work[env_idx + 29];
        work[env_idx + 7] = work[env_idx + 7] + work[env_idx + 28];
        work[env_idx + 28] = inputs[0][idx * nnz_in[0] + 11];
        work[env_idx + 30] = cos(work[env_idx + 28]);
        work[env_idx + 31] = inputs[0][idx * nnz_in[0] + 12];
        work[env_idx + 32] = cos(work[env_idx + 31]);
        work[env_idx + 33] = work[env_idx + 12] * work[env_idx + 32];
        work[env_idx + 31] = sin(work[env_idx + 31]);
        work[env_idx + 34] = work[env_idx + 16] * work[env_idx + 31];
        work[env_idx + 33] = work[env_idx + 33] + work[env_idx + 34];
        work[env_idx + 11] = work[env_idx + 11] + work[env_idx + 33];
        work[env_idx + 33] = work[env_idx + 30] * work[env_idx + 11];
        work[env_idx + 28] = sin(work[env_idx + 28]);
        work[env_idx + 32] = work[env_idx + 16] * work[env_idx + 32];
        work[env_idx + 12] = work[env_idx + 12] * work[env_idx + 31];
        work[env_idx + 32] = work[env_idx + 32] - work[env_idx + 12];
        work[env_idx + 32] = work[env_idx + 32] + work[env_idx + 13];
        work[env_idx + 32] = work[env_idx + 18] + work[env_idx + 32];
        work[env_idx + 12] = work[env_idx + 28] * work[env_idx + 32];
        work[env_idx + 33] = work[env_idx + 33] + work[env_idx + 12];
        work[env_idx + 33] = work[env_idx + 33] + work[env_idx + 19];
        work[env_idx + 8] = work[env_idx + 8] + work[env_idx + 33];
        work[env_idx + 8] = work[env_idx + 8] + work[env_idx + 20];
        work[env_idx + 7] = work[env_idx + 7] + work[env_idx + 8];
        work[env_idx + 8] = inputs[0][idx * nnz_in[0] + 8];
        work[env_idx + 20] = cos(work[env_idx + 8]);
        work[env_idx + 33] = inputs[0][idx * nnz_in[0] + 9];
        work[env_idx + 19] = cos(work[env_idx + 33]);
        work[env_idx + 12] = work[env_idx + 24] * work[env_idx + 19];
        work[env_idx + 33] = sin(work[env_idx + 33]);
        work[env_idx + 31] = work[env_idx + 16] * work[env_idx + 33];
        work[env_idx + 12] = work[env_idx + 12] + work[env_idx + 31];
        work[env_idx + 23] = work[env_idx + 23] + work[env_idx + 12];
        work[env_idx + 12] = work[env_idx + 20] * work[env_idx + 23];
        work[env_idx + 8] = sin(work[env_idx + 8]);
        work[env_idx + 16] = work[env_idx + 16] * work[env_idx + 19];
        work[env_idx + 24] = work[env_idx + 24] * work[env_idx + 33];
        work[env_idx + 16] = work[env_idx + 16] - work[env_idx + 24];
        work[env_idx + 16] = work[env_idx + 16] + work[env_idx + 13];
        work[env_idx + 18] = work[env_idx + 18] + work[env_idx + 16];
        work[env_idx + 16] = work[env_idx + 8] * work[env_idx + 18];
        work[env_idx + 12] = work[env_idx + 12] + work[env_idx + 16];
        work[env_idx + 12] = work[env_idx + 12] + work[env_idx + 25];
        work[env_idx + 17] = work[env_idx + 17] + work[env_idx + 12];
        work[env_idx + 17] = work[env_idx + 17] + work[env_idx + 29];
        work[env_idx + 7] = work[env_idx + 7] + work[env_idx + 17];
        work[env_idx + 6] = work[env_idx + 6] * work[env_idx + 7];
        work[env_idx + 17] = inputs[0][idx * nnz_in[0] + 3];
        work[env_idx + 29] = work[env_idx + 2] * work[env_idx + 17];
        work[env_idx + 12] = inputs[0][idx * nnz_in[0] + 6];
        work[env_idx + 25] = work[env_idx + 5] * work[env_idx + 12];
        work[env_idx + 16] = work[env_idx + 29] - work[env_idx + 25];
        work[env_idx + 13] = -0.0237328831644008;
        work[env_idx + 24] = -0.1986442323264954;
        work[env_idx + 33] = inputs[0][idx * nnz_in[0] + 16];
        work[env_idx + 19] = cos(work[env_idx + 33]);
        work[env_idx + 31] = work[env_idx + 24] * work[env_idx + 19];
        work[env_idx + 33] = sin(work[env_idx + 33]);
        work[env_idx + 34] = -0.0002168050202306;
        work[env_idx + 10] = work[env_idx + 10] * work[env_idx + 14];
        work[env_idx + 9] = work[env_idx + 9] * work[env_idx + 15];
        work[env_idx + 10] = work[env_idx + 10] - work[env_idx + 9];
        work[env_idx + 10] = work[env_idx + 34] + work[env_idx + 10];
        work[env_idx + 9] = work[env_idx + 33] * work[env_idx + 10];
        work[env_idx + 31] = work[env_idx + 31] - work[env_idx + 9];
        work[env_idx + 9] = -0.3952087724160000;
        work[env_idx + 31] = work[env_idx + 31] + work[env_idx + 9];
        work[env_idx + 13] = work[env_idx + 13] + work[env_idx + 31];
        work[env_idx + 31] = inputs[0][idx * nnz_in[0] + 13];
        work[env_idx + 15] = cos(work[env_idx + 31]);
        work[env_idx + 14] = work[env_idx + 24] * work[env_idx + 15];
        work[env_idx + 31] = sin(work[env_idx + 31]);
        work[env_idx + 22] = work[env_idx + 22] * work[env_idx + 26];
        work[env_idx + 21] = work[env_idx + 21] * work[env_idx + 27];
        work[env_idx + 22] = work[env_idx + 22] - work[env_idx + 21];
        work[env_idx + 22] = work[env_idx + 34] + work[env_idx + 22];
        work[env_idx + 21] = work[env_idx + 31] * work[env_idx + 22];
        work[env_idx + 14] = work[env_idx + 14] - work[env_idx + 21];
        work[env_idx + 14] = work[env_idx + 14] + work[env_idx + 9];
        work[env_idx + 13] = work[env_idx + 13] + work[env_idx + 14];
        work[env_idx + 14] = 0.1986442323264954;
        work[env_idx + 9] = inputs[0][idx * nnz_in[0] + 10];
        work[env_idx + 21] = cos(work[env_idx + 9]);
        work[env_idx + 27] = work[env_idx + 14] * work[env_idx + 21];
        work[env_idx + 9] = sin(work[env_idx + 9]);
        work[env_idx + 30] = work[env_idx + 30] * work[env_idx + 32];
        work[env_idx + 28] = work[env_idx + 28] * work[env_idx + 11];
        work[env_idx + 30] = work[env_idx + 30] - work[env_idx + 28];
        work[env_idx + 30] = work[env_idx + 34] + work[env_idx + 30];
        work[env_idx + 28] = work[env_idx + 9] * work[env_idx + 30];
        work[env_idx + 27] = work[env_idx + 27] - work[env_idx + 28];
        work[env_idx + 28] = 0.3952087724160000;
        work[env_idx + 27] = work[env_idx + 27] + work[env_idx + 28];
        work[env_idx + 13] = work[env_idx + 13] + work[env_idx + 27];
        work[env_idx + 27] = inputs[0][idx * nnz_in[0] + 7];
        work[env_idx + 11] = cos(work[env_idx + 27]);
        work[env_idx + 32] = work[env_idx + 14] * work[env_idx + 11];
        work[env_idx + 27] = sin(work[env_idx + 27]);
        work[env_idx + 20] = work[env_idx + 20] * work[env_idx + 18];
        work[env_idx + 8] = work[env_idx + 8] * work[env_idx + 23];
        work[env_idx + 20] = work[env_idx + 20] - work[env_idx + 8];
        work[env_idx + 34] = work[env_idx + 34] + work[env_idx + 20];
        work[env_idx + 20] = work[env_idx + 27] * work[env_idx + 34];
        work[env_idx + 32] = work[env_idx + 32] - work[env_idx + 20];
        work[env_idx + 32] = work[env_idx + 32] + work[env_idx + 28];
        work[env_idx + 13] = work[env_idx + 13] + work[env_idx + 32];
        work[env_idx + 16] = work[env_idx + 16] * work[env_idx + 13];
        work[env_idx + 32] = work[env_idx + 5] * work[env_idx + 17];
        work[env_idx + 2] = work[env_idx + 2] * work[env_idx + 12];
        work[env_idx + 28] = work[env_idx + 32] + work[env_idx + 2];
        work[env_idx + 20] = 0.8431537489713424;
        work[env_idx + 33] = work[env_idx + 24] * work[env_idx + 33];
        work[env_idx + 19] = work[env_idx + 19] * work[env_idx + 10];
        work[env_idx + 33] = work[env_idx + 33] + work[env_idx + 19];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 33];
        work[env_idx + 24] = work[env_idx + 24] * work[env_idx + 31];
        work[env_idx + 15] = work[env_idx + 15] * work[env_idx + 22];
        work[env_idx + 24] = work[env_idx + 24] + work[env_idx + 15];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 24];
        work[env_idx + 9] = work[env_idx + 14] * work[env_idx + 9];
        work[env_idx + 21] = work[env_idx + 21] * work[env_idx + 30];
        work[env_idx + 9] = work[env_idx + 9] + work[env_idx + 21];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 9];
        work[env_idx + 14] = work[env_idx + 14] * work[env_idx + 27];
        work[env_idx + 11] = work[env_idx + 11] * work[env_idx + 34];
        work[env_idx + 14] = work[env_idx + 14] + work[env_idx + 11];
        work[env_idx + 20] = work[env_idx + 20] + work[env_idx + 14];
        work[env_idx + 28] = work[env_idx + 28] * work[env_idx + 20];
        work[env_idx + 16] = work[env_idx + 16] + work[env_idx + 28];
        work[env_idx + 6] = work[env_idx + 6] + work[env_idx + 16];
        work[env_idx + 16] = 30.4213964620000006;
        work[env_idx + 28] = inputs[0][idx * nnz_in[0] + 0];
        work[env_idx + 28] = work[env_idx + 16] * work[env_idx + 28];
        work[env_idx + 6] = work[env_idx + 6] + work[env_idx + 28];
        work[env_idx + 6] = work[env_idx + 6] / work[env_idx + 16];
        outputs[0][idx * nnz_out[0] + 0] = work[env_idx + 6];
        work[env_idx + 29] = work[env_idx + 29] + work[env_idx + 25];
        work[env_idx + 29] = work[env_idx + 29] * work[env_idx + 7];
        work[env_idx + 25] = 2.*(work[env_idx + 17]);
        work[env_idx + 17] = work[env_idx + 25] * work[env_idx + 17];
        work[env_idx + 4] = work[env_idx + 17] + work[env_idx + 4];
        work[env_idx + 4] = work[env_idx + 0] - work[env_idx + 4];
        work[env_idx + 4] = work[env_idx + 4] * work[env_idx + 13];
        work[env_idx + 5] = work[env_idx + 5] * work[env_idx + 1];
        work[env_idx + 25] = work[env_idx + 25] * work[env_idx + 12];
        work[env_idx + 12] = work[env_idx + 5] - work[env_idx + 25];
        work[env_idx + 12] = work[env_idx + 12] * work[env_idx + 20];
        work[env_idx + 4] = work[env_idx + 4] + work[env_idx + 12];
        work[env_idx + 29] = work[env_idx + 29] + work[env_idx + 4];
        work[env_idx + 4] = inputs[0][idx * nnz_in[0] + 1];
        work[env_idx + 4] = work[env_idx + 16] * work[env_idx + 4];
        work[env_idx + 29] = work[env_idx + 29] + work[env_idx + 4];
        work[env_idx + 29] = work[env_idx + 29] / work[env_idx + 16];
        outputs[0][idx * nnz_out[0] + 1] = work[env_idx + 29];
        work[env_idx + 32] = work[env_idx + 32] - work[env_idx + 2];
        work[env_idx + 32] = work[env_idx + 32] * work[env_idx + 7];
        work[env_idx + 5] = work[env_idx + 5] + work[env_idx + 25];
        work[env_idx + 5] = work[env_idx + 5] * work[env_idx + 13];
        work[env_idx + 17] = work[env_idx + 17] + work[env_idx + 3];
        work[env_idx + 0] = work[env_idx + 0] - work[env_idx + 17];
        work[env_idx + 0] = work[env_idx + 0] * work[env_idx + 20];
        work[env_idx + 5] = work[env_idx + 5] + work[env_idx + 0];
        work[env_idx + 32] = work[env_idx + 32] + work[env_idx + 5];
        work[env_idx + 5] = inputs[0][idx * nnz_in[0] + 2];
        work[env_idx + 5] = work[env_idx + 16] * work[env_idx + 5];
        work[env_idx + 32] = work[env_idx + 32] + work[env_idx + 5];
        work[env_idx + 32] = work[env_idx + 32] / work[env_idx + 16];
        outputs[0][idx * nnz_out[0] + 2] = work[env_idx + 32];
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