// AUTOMATICALLY GENERATED CODE FOR CUSADI

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

__constant__ int nnz_in[] = {19};
__constant__ int nnz_out[] = {108};
__constant__ int n_w = 2;

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
        outputs[0][idx * nnz_out[0] + 0] = work[env_idx + 0];
        work[env_idx + 1] = 0.0000000000000000;
        outputs[0][idx * nnz_out[0] + 1] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 2] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 3] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 4] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 5] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 6] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 7] = work[env_idx + 0];
        outputs[0][idx * nnz_out[0] + 8] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 9] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 10] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 11] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 12] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 13] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 14] = work[env_idx + 0];
        outputs[0][idx * nnz_out[0] + 15] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 16] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 17] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 18] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 19] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 20] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 21] = work[env_idx + 0];
        outputs[0][idx * nnz_out[0] + 22] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 23] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 24] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 25] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 26] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 27] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 28] = work[env_idx + 0];
        outputs[0][idx * nnz_out[0] + 29] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 30] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 31] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 32] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 33] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 34] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 35] = work[env_idx + 0];
        outputs[0][idx * nnz_out[0] + 36] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 37] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 38] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 39] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 40] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 41] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 42] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 43] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 44] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 45] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 46] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 47] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 48] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 49] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 50] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 51] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 52] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 53] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 54] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 55] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 56] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 57] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 58] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 59] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 60] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 61] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 62] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 63] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 64] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 65] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 66] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 67] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 68] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 69] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 70] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 71] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 72] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 73] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 74] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 75] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 76] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 77] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 78] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 79] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 80] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 81] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 82] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 83] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 84] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 85] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 86] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 87] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 88] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 89] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 90] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 91] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 92] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 93] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 94] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 95] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 96] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 97] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 98] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 99] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 100] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 101] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 102] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 103] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 104] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 105] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 106] = work[env_idx + 1];
        outputs[0][idx * nnz_out[0] + 107] = work[env_idx + 1];
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