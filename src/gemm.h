#ifndef GEMM_H
#define GEMM_H

void static gemm_nn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void static gemm_nt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void static gemm_tn(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (k = 0; k < K; ++k)
        {
            float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void static gemm_tt(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            float sum = 0;
            for (k = 0; k < K; ++k)
            {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

/**
 * gemm - matrix c = matrix a * matrix b
 * @TA: transpose matrix a
 * @TB: transpose matrix b
 * @M: a row
 * @N: b col
 * @K: a col
 * @ALPHA: 1
 * @A: matrix a
 * @lda: a col
 * @B: matrix b
 * @ldb: b col
 * @BETA: 1
 * @C: matrix c
 * @ldc: c col
*/
void static inline gemm(int TA, int TB, int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float BETA, float *C, int ldc)
{
    int i, j;
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

#endif