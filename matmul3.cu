#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;

    float accum[TM][TN] = {0.};

    float ldg_a_reg[4 * ldg_a_num] = {0.};

    float a_frag[TM];
    float b_frag[TN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;
            
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }

        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();
        
        A += BK;
        B += BK * N;

        for (int i = 0; i < BK; i++) {
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]);
            }

            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]);
            }

            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
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