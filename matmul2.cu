#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float SA[BM * BK];
    __shared__ float SB[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0.};

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            SA[(a_tile_row + i) * BK + a_tile_col] = 
                A[(a_tile_row + i) * K + a_tile_col];
        }

        for (int i = 0; i < BK; i += b_tile_stride) {
            SB[(b_tile_row + i) * BN + b_tile_col] = 
                B[(b_tile_row + i) * N + b_tile_col];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (int i = 0; i < BK; i ++) {
            for (int m = 0; m < TM; m ++) {
                for (int n = 0; n < TN; n ++) {
                    tmp[m][n] += SA[(ty + m) * BK + i] * SB[tx + n + i * BN];
                }
            }
        }

        __syncthreads();

          for (int m = 0; m < TM; m ++) {
            for (int n = 0; n < TN; n ++) {
                C[(ty + m) * N + tx + n] =
                    alpha * tmp[m][n] + beta * C[(ty + m) * N + tx + n];
            }
        }
    }

    for (int m = 0; m < TM; m ++) {
        for (int n = 0; n < TN; n++) {
            C[(ty + m) * N + tx + n] =
                alpha * tmp[m][n] + beta * C[(ty + m) * N + tx + n];
        }
    }
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    for (int N : sizes) {
        size_t size = static_cast<size_t>(N) * N * sizeof(float);

        float *A = (float*)malloc(size);
        float *B = (float*)malloc(size);
        float *C_cublas = (float*)malloc(size);
        float *C_mysgemm = (float*)malloc(size);

        if (A == nullptr || B == nullptr ||
            C_cublas == nullptr || C_mysgemm == nullptr) {
            fprintf(stderr, "Host malloc failed (N = %d)\n", N);
            return 1;
        }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        // 初始化矩阵
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                A[row * N + col] = static_cast<float>((row + col) % 5 - 2);
                B[row * N + col] = static_cast<float>((row * 2 + col * 3) % 5 - 2);
            }
        }

        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // cuBLAS warmup
        int warmup_time = 10;
        for (int i = 0; i < warmup_time; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, d_B, N, d_A, N, &beta, d_C, N);
        }

        // cuBLAS SGEMM
        int repeat_time = 5;
        cudaEventRecord(start);

        for (int i = 0; i < repeat_time; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                        &alpha, d_B, N, d_A, N, &beta, d_C, N);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float cublas_time = 0.0f;
        cudaEventElapsedTime(&cublas_time, start, stop);

        // 拷贝 cuBLAS 结果
        cudaMemcpy(C_cublas, d_C, size, cudaMemcpyDeviceToHost);

        cudaMemset(d_C, 0, size);

        dim3 block(16 * 16);
        dim3 grid((N + 127 - 1) / 128,
                  (N + 127 - 1) / 128);

        // MySGEMM warmup
        for (int i = 0; i < warmup_time; i++) {
            mysgemm<128, 128, 8, 8, 8>
                <<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_C);
        }

        cudaDeviceSynchronize();

        // MySGEMM
        cudaEventRecord(start);

        for (int i = 0; i < repeat_time; i++) {
            mysgemm<128, 128, 8, 8, 8>
                <<<grid, block>>>(N, N, N, alpha, d_A, d_B, beta, d_C);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float mysgemm_time = 0.0f;
        cudaEventElapsedTime(&mysgemm_time, start, stop);

        // 拷贝 MySGEMM 结果
        cudaMemcpy(C_mysgemm, d_C, size, cudaMemcpyDeviceToHost);

        // 结果比较
        int error_count = 0;
        for (int i = 0; i < N * N && error_count < 10; i++) {
            if (C_cublas[i] != C_mysgemm[i]) {
                printf("Mismatch at %d: cuBLAS = %f, mysgemm = %f\n",
                       i, C_cublas[i], C_mysgemm[i]);
                error_count++;
            }
        }

        const char *matched = error_count == 0 ? "Yes" : "No";

        // 计算性能并打印结果
        double cublas_gflops = repeat_time * 2.0 * N * N * N / (cublas_time * 1e6);
        double mysgemm_gflops = repeat_time * 2.0 * N * N * N / (mysgemm_time * 1e6);

        printf("Size = %4d | CUBLAS_GFLOPS = %7.2f | MySGEMM_GFLOPS = %7.2f | Matched: %s\n",
               N, cublas_gflops, mysgemm_gflops, matched);

        // 释放资源
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(A);
        free(B);
        free(C_cublas);
        free(C_mysgemm);
    }

    return 0;
}