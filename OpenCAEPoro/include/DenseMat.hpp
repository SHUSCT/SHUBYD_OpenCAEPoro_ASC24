/*! \file    DenseMat.hpp
 *  \brief   Operations about small dense mat
 *  \author  Shizhe Li
 *  \date    Oct/24/2021
 *
 *-----------------------------------------------------------------------------------
 *  Copyright (C) 2021--present by the OpenCAEPoroX team. All rights reserved.
 *  Released under the terms of the GNU Lesser General Public License 3.0 or later.
 *-----------------------------------------------------------------------------------
 */

#ifndef __DENSEMAT_HEADER__
#define __DENSEMAT_HEADER__

// Standard header files
#include "OCPDataType.hpp"
#include "UtilError.hpp"
#include <algorithm>
#include <execution>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <omp.h>
#include <string>

using namespace std;

extern "C" {

////// BLAS functions

/// Scales a vector by a constant.
void dscal_(const int* n, const double* alpha, double* x, const int* incx);

/// Forms the dot product of two vectors.
double ddot_(const int* n, double* a, const int* inca, double* b, const int* incb);

/// Copies a vector, src, to a vector, dst.
int dcopy_(
    const int* n, const double* src, const int* incx, double* dst, const int* incy);

/// Constant times a vector plus a vector.
int daxpy_(const int* n,
           const double* alpha,
           const double* x,
           const int* incx,
           double* y,
           const int* incy);

/// Computes the Euclidean norm of a vector.
double dnrm2_(const int* n, double* x, const int* incx);

/// Computes the sum of the absolute values of a vector.
double dasum_(const int* n, double* x, const int* incx);

/// Finds the index of element having max absolute value.
int idamax_(const int* n, double* x, const int* incx);

/// Performs matrix-matrix operations C : = alpha * op(A) * op(B) + beta * C.
int dgemm_(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const double* alpha,
           const double* A,
           const int* lda,
           const double* B,
           const int* ldb,
           const double* beta,
           double* C,
           const int* ldc);

////// LAPACK functions

/// Computes the solution to system of linear equations A * X = B for general matrices.
void dgesv_(const int* n,
            const int* nrhs,
            double* A,
            const int* lda,
            int* ipiv,
            double* b,
            const int* ldb,
            int* info);

/// Computes the solution to system of linear equations A * X = B for symm matrices.
void dsysv_(const char* uplo,
            const int* n,
            const int* nrhs,
            double* A,
            const int* lda,
            int* ipiv,
            double* b,
            const int* ldb,
            double* work,
            const int* lwork,
            int* info);

/// Computes the eigenvalues and, optionally, the leftand /or right eigenvectors for SY
/// matrices
void ssyevd_(const char* jobz,
             const char* uplo,
             const int* n,
             float* A,
             const int* lda,
             float* w,
             float* work,
             const int* lwork,
             int* iwork,
             const int* liwork,
             int* info);
}

/// Computes L1-norm of a vector.
OCP_DBL Dnorm1(const INT& N, OCP_DBL* x);

/// Computes L1-norm of a vector.
template <typename T1, typename T2>
T2 OCP_norm1(const T1& n, const T2* x) {
    T2 tmp = 0;
    for (T1 i = 0; i < n; i++) {
        tmp += fabs(x[i]);
    }
    return tmp;
}

/// Computes L2-norm of a vector.
OCP_DBL Dnorm2(const INT& N, OCP_DBL* x);

/// Computes L2-norm of a vector.
template <typename T1, typename T2>
T2 OCP_norm2(const T1& n, const T2* x) {
    T2 tmp = 0;
    for (T1 i = 0; i < n; i++) {
        tmp += x[i] * x[i];
    }
    return sqrt(tmp);
}

/// Scales a vector by a constant.
void Dscalar(const INT& n, const OCP_DBL& alpha, OCP_DBL* x);

/// Computes x = ax
template <typename T1, typename T2>
void OCP_scale(const T1& n, const T2& a, T2* x) {
    for (T1 i = 0; i < n; i++) {
        x[i] *= a;
    }
}

/// Constant times a vector plus a vector.
void Daxpy(const INT& n, const OCP_DBL& alpha, const OCP_DBL* x, OCP_DBL* y);

/// Computes y = ax + y
template <typename T1, typename T2>
void OCP_axpy(const T1& n, const T2& a, const T2* x, T2* y) {
    for (T1 i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

namespace byd {
    // [Note]
    //   This block is some implementations of ABpC with different methods.
    //   It is for sure that A(4*12) B(12*4) C(4*4), aka, m=4, n=4, k=12.

    inline void DaABpbC_mkl(const int& m,
                            const int& n,
                            const int& k,
                            const double& alpha,
                            const double* A,
                            const double* B,
                            const double& beta,
                            double* C) {
        // A : m * k
        // B : k * n
        // C : m * n
        // Call dgemm to perform the operation C = alpha*A*B + beta*C
        const char transa = 'N', transb = 'N';
        dgemm_(&transa, &transb, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);
    }

    // @brief Computes C = AB + C with openmp and sse
    inline void DaABpbC_openmp_sse(const double* a, const double* b, double* c, const int row1, const int col1, const int col2) {
        #pragma omp parallel for shared(a, b, c)
        for (int i = 0; i < row1; i++) {

            for (int k = 0; k < col1; k++) {
                double r = a[i * col1 + k];

                // for(int j = 0; j < col2; j++){
                // c[i * col2 + j] += (r * b[k * col2 + j]);
                // }
                for (int j = 0; j + 2 <= col2; j += 2) {
                    __m128d k1 = _mm_set1_pd(r);
                    __m128d k2 = _mm_loadu_pd(b + (k * col2 + j));

                    __m128d k3 = _mm_mul_pd(k1, k2);

                    k1 = _mm_loadu_pd(&c[i * col2 + j]);
                    k2 = _mm_add_pd(k1, k3);
                    _mm_storeu_pd(&c[i * col2 + j], k2);
                    // c[i * col2 + j] += k3[0];
                    // c[i * col2 + j + 1] += k3[1];
                }

                for (int j = col2 - col2 % 2; j < col2; j++) {
                    c[i * col2 + j] += (r * b[k * col2 + j]);
                }
            }
        }
    }


    // @brief Computes C = AB + C with openmp
    inline double calcuPartOfMatrixMulti(const double* A, const double* B, const int i, const int j, const int k, const int n) {
        double sum = 0;
        for (int l = 0; l < k; l++) {
            sum += (double)A[i * k + l] * B[l * n + j];
        }
        return sum;
    }

    inline void DaABpbC_openmp(const int m,        // 4
                               const int n,        // 4
                               const int k,        // 12
                               const double alpha, // 1
                               const double* A,
                               const double* B,
                               const double beta, // 1
                               double* C) {
        #pragma omp parallel for collapse(2) shared(A, B, C)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) { // C(i,j) = alpha * A(i,k) * B(k,j) + beta * C(i,j)
                C[i * n + j] += calcuPartOfMatrixMulti(A, B, i, j, k, n);
            }
        }
    }
    // @brief Computes C = AB + C with openmp and simd
    inline double calcuPartOfMatrixMulti_simd(const double* A, const double* B, const int i, const int j, const int k, const int n) {
        double sum = 0;
        #pragma omp simd reduction(+ : sum)
        for (int l = 0; l < k; l++) {
            sum += A[i * k + l] * B[l * n + j];
        }
        return sum;
    }
    inline void DaABpbC_openmp_simd(const int m,        // 4
                                    const int n,        // 4
                                    const int k,        // 12
                                    const double alpha, // 1
                                    const double* A,
                                    const double* B,
                                    const double beta, // 1
                                    double* C) {
        #pragma omp parallel for collapse(2) shared(A, B, C)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) { // C(i,j) = alpha * A(i,k) * B(k,j) + beta * C(i,j)
                C[i * n + j] += calcuPartOfMatrixMulti_simd(A, B, i, j, k, n);
            }
        }
    }


    // @brief Computes C = ax with openmp
    inline void Dscal_openmp_simd(const int n, const double& alpha, OCP_DBL* x, const int incx) {
        const __m128d k1 = _mm_set1_pd(alpha);
        #pragma omp parallel for shared(x, k1)
        for (int i = 0; i + 2 <= n; i += 2 * incx)
        {
            __m128d k2 = _mm_loadu_pd(x + i);
            _mm_storeu_pd(&x[i], _mm_mul_pd(k1, k2));
        }
        for (int i = n - n % 2; i < n; ++i) {
            x[i] = (alpha * x[i]);
        }
    }


    template <int I, int J, int Row, int Col, typename T>
    struct Transpose
    {
        static void run_unroll(const T* A, T* B) {
            B[J * Row + I] = A[I * Col + J];
            if constexpr (J + 1 < Col) {
                Transpose<I, J + 1, Row, Col, T>::run_unroll(A, B);
            } else if constexpr (I + 1 < Row) {
                Transpose<I + 1, 0, Row, Col, T>::run_unroll(A, B);
            }
        }

        static void run_openmp(const T* A, T* B) {
            #pragma omp parallel for
            for (int i = 0; i < Row; i++) {
                for (int j = 0; j < Col; j++) {
                    B[j * Row + i] = A[i * Col + j];
                }
            }
        }
    };
    template<int I, int L, int M, int N, int K, typename T>
    struct DaABpbCHelper2 {
        static void compute(const T* A, const T* B, T* C) {
            T r = A[I * K + L];
            __m128d n1 = _mm_set1_pd(r);
            for (int j = 0; j + 2 <= N; j += 2) {
                __m128d n2 = _mm_loadu_pd(B + (N * L + j));
                __m128d n3 = _mm_loadu_pd(C + (I * N + j));
                _mm_storeu_pd(C + (I * N + j), _mm_add_pd(_mm_mul_pd(n1, n2), n3));
            }
            for (int j = N - N % 2; j < N; ++j) {
                C[I * N + j] += (r * B[L * N +j]);
            }
            if constexpr (L + 1 < K)
                DaABpbCHelper2<I, L + 1, M, N, K, T>::compute(A, B, C);
            else if constexpr (I + 1 < M)
                DaABpbCHelper2<I + 1, 0, M, N, K, T>::compute(A, B, C);
        }
    };

    template <typename T>
    void DaABpbC_unroll_simd(const int m, const int n, const int k, const T& alpha, const T* A, const T* B, const T& beta, T* C) {
        DaABpbCHelper2<0, 0, 4, 4, 12, T>::compute(A, B, C);
    }


    // Specializations for terminating the recursion
    template <int I, int J, int K, int M, int N, int L, typename T>
    struct DaABpbC_unroll_core
    {
        static void compute(const T* A, const T* B, T* C) {
            T sum = 0;
            // #pragma omp parallel for reduction(+ : sum)
            for (int l = 0; l < K; ++l) {
                sum += A[I * L + l] * B[l * N + J];
            }
            C[I * N + J] = sum + C[I * N + J];
            if constexpr (J + 1 < N) {
                DaABpbC_unroll_core<I, J + 1, K, M, N, L, T>::compute(A, B, C);
            } else if constexpr (I + 1 < M) {
                DaABpbC_unroll_core<I + 1, 0, K, M, N, L, T>::compute(A, B, C);
            }
        }

        static void compute_transposed_B(const T* A, const T* B, T* C) {
            T sum = 0;
            // #pragma omp parallel for reduction(+ : sum)
            for (int l = 0; l < K; ++l) {
                sum += A[I * L + l] * B[J * L + l];
            }
            C[I * N + J] = sum + C[I * N + J];
            if constexpr (J + 1 < N) {
                DaABpbC_unroll_core<I, J + 1, K, M, N, L, T>::compute_transposed_B(A, B, C);
            } else if constexpr (I + 1 < M) {
                DaABpbC_unroll_core<I + 1, 0, K, M, N, L, T>::compute_transposed_B(A, B, C);
            }
        }
    };

    template <typename T>
    void DaABpbC_unroll(const int m, const int n, const int k, const T& alpha, const T* A, const T* B, const T& beta, T* C) {
        DaABpbC_unroll_core<0, 0, 12, 4, 4, 12, T>::compute(A, B, C);
    }

    template <typename T, typename _Pred>
    void DaABpbc_unroll_transpose_B(const int m, const int n, const int k, const T& alpha, const T* A, const T* B, const T& beta, T* C, _Pred _pred) {
        T* transposed_B = new T[12 * 4];
        _pred(B, transposed_B);
        DaABpbC_unroll_core<0, 0, 12, 4, 4, 12, T>::compute_transposed_B(A, transposed_B, C);
    }
}

/// Computes C' = alpha B'A' + beta C', all matrices are column-major.
inline void DaABpbC(const INT m,
                    const INT n,
                    const INT k,
                    const OCP_DBL& alpha,
                    const OCP_DBL* A,
                    const OCP_DBL* B,
                    const OCP_DBL& beta,
                    OCP_DBL* C) {
#if OCPFLOATTYPEWIDTH == 64
    // const char transa = 'N', transb = 'N';
    // dgemm_(&transa, &transb, &n, &m, &k, &alpha, B, &n, A, &k, &beta, C, &n);  //参数顺序example
    // byd::DaABpbC_mkl(m, n, k, alpha, A, B, beta, C);
    // byd::DaABpbC_unroll(m, n, k, alpha, A, B, beta, C);
    byd::DaABpbC_openmp(m, n, k, alpha, A, B, beta, C);
    // byd::DaABpbc_unroll_transpose_B(m, n, k, alpha, A, B, beta, C, byd::Transpose<0, 0, 12, 4, OCP_DBL>::run_unroll);
    // byd::DaABpbc_unroll_transpose_B(m, n, k, alpha, A, B, beta, C, byd::Transpose<0, 0, 12, 4, OCP_DBL>::run_openmp);
    // byd::DaABpbC_unroll_simd(m, n, k, alpha, A, B, beta, C);


#else
    OCP_ABpC(m, n, k, A, B, C);
#endif
}

/// Computes C = AB + C
template <typename T1, typename T2>
void OCP_ABpC(const T1& m, const T1& n, const T1& k, const T2* A, const T2* B, T2* C) {
    // C = AB + C
    // A: m*k  B:k*n  C:m*n
    // all matrix are row majored matrices

    for (T1 i = 0; i < m; i++) {
        for (T1 j = 0; j < n; j++) {
            for (T1 l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

// [Jan,9,2024 jamesnulliu] >>>>>>



// @brief Computes C = AB + C with openmp
template <typename T1, typename T2>
void OCP_ABpC_openmp(const T1& m, const T1& n, const T1& k, const T2* A, const T2* B, T2* C) {
    #pragma omp parallel for collapse(2)
    for (T1 i = 0; i < m; i++) {
        for (T1 j = 0; j < n; j++) {
            T2 sum = 0;
            for (T1 l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] += sum;
        }
    }
}

// [Jan,9,2024 jamesnulliu] <<<<<<

/// Computes y = a A x + b y
template <typename T1, typename T2>
void OCP_aAxpby(const T1& m, const T1& n, const T2& a, const T2* A, const T2* x, const T2& b, T2* y) {
    for (T1 i = 0; i < m; i++) {
        y[i] = b * y[i];
        for (T1 j = 0; j < n; j++) {
            y[i] += a * A[i * n + j] * x[j];
        }
    }
}

// [Jan,9,2024 jamesnulliu] >>>>>>


// @brief Computes y = a A x + b y with openmp
template <typename T1, typename T2>
void OCP_aAxpby_openmp(const T1& m, const T1& n, const T2& a, const T2* A, const T2* x, const T2& b, T2* y) {
#pragma omp parallel for
    for (T1 i = 0; i < m; i++) {
        T2 temp = 0;
        for (T1 j = 0; j < n; j++) {
            temp += a * A[i * n + j] * x[j];
        }
        y[i] = b * y[i] + temp;
    }
}

// [Jan,9,2024 jamesnulliu] <<<<<<

// [Jan,11,2024 //ctrl] >>>>>>>>
inline void Dscalar(const INT& n, const OCP_DBL& alpha, OCP_DBL* x) {

#if OCPFLOATTYPEWIDTH == 64
    // x = a x
    const int incx = 1;
    // dscal_(&n, &alpha, x, &incx);
    byd::Dscal_openmp_simd(n, alpha, x, incx);
#else
    OCP_scale(n, alpha, x);
#endif
}

// [Jan,11,2024 //ctrl] <<<<<<<<

/// Calls dgesv to solve the linear system for general matrices.
void LUSolve(const INT& nrhs, const INT& N, OCP_DBL* A, OCP_DBL* b, INT* pivot);

/// Calls dsysy to solve the linear system for symm matrices.
INT SYSSolve(const INT& nrhs, const OCP_CHAR* uplo, const INT& N, OCP_DBL* A, OCP_DBL* b, INT* pivot, OCP_DBL* work, const INT& lwork);

/// Calculate the minimal eigenvalue for symmetric matrix with mkl lapack
void CalEigenSY(const INT& N, OCP_SIN* A, OCP_SIN* w, OCP_SIN* work, const INT& lwork);

void myDABpCp(const int& m,
              const int& n,
              const int& k,
              const double* A,
              const double* B,
              double* C,
              const int* flag,
              const int N);

void myDABpCp1(const int& m,
               const int& n,
               const int& k,
               const double* A,
               const double* B,
               double* C,
               const int* flag,
               const int N);

void myDABpCp2(const int& m,
               const int& n,
               const int& k,
               const double* A,
               const double* B,
               double* C,
               const int* flag,
               const int N);

/// Prints a vector.
// >>>>>> [Jan,9,2024 jamesnulliu] >>>>>>
template <typename T>
void PrintDX(const int& N, const T* x) {
    for (int i = 0; i < N; i++) {
        std::cout << std::format("{:d}   {:f}\n", i, x[i]);
    }
    cout << endl;
}
/// check NaN
template <typename T>
bool CheckNan(const int& N, const T* x) {
    for (int i = 0; i < N; i++) {
        if (!isfinite(x[i])) {
            return false;
        }
    }
    return true;
}
/// swap value instead of pointer
template <typename T>
inline void OCPSwap(T a, T b, const int& n, T w) {
#pragma omp parallel for shared(w, a, b)
    for (int i = 0; i < n; i++) {
        w[i] = a[i];
        a[i] = b[i];
        b[i] = w[i];
    }
}
// <<<<<< [Jan,9,2024 jamesnulliu] <<<<<<

#endif

/*----------------------------------------------------------------------------*/
/*  Brief Change History of This File                                         */
/*----------------------------------------------------------------------------*/
/*  Author              Date             Actions                              */
/*----------------------------------------------------------------------------*/
/*  Shizhe Li           Oct/24/2021      Create file                          */
/*  Chensong Zhang      Jan/16/2022      Update Doxygen                       */
/*----------------------------------------------------------------------------*/