#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <cstring>
#include <string>

// [OPTIMIZATION] AVX2 Intrinsics
#include <immintrin.h>

// ============================================================================
// CONFIGURATION
// ============================================================================
#define ENABLE_LAPACK_BENCHMARK 1
#define EPSILON 1e-13
#define MAX_ITER_MULTIPLIER 30

// Compiler hint for restrict pointers (MSVC vs GCC/Clang)
#if defined(_WIN32) || defined(_MSC_VER)
    #include <malloc.h>
    #define RESTRICT __restrict
    #define ALIGNED_MALLOC(size, align) _aligned_malloc(size, align)
    #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <stdlib.h>
    #define RESTRICT __restrict__
    #define ALIGNED_MALLOC(size, align) aligned_alloc(align, size)
    #define ALIGNED_FREE(ptr) free(ptr)
#endif

// ============================================================================
// LAPACK BINDINGS
// ============================================================================
#if ENABLE_LAPACK_BENCHMARK
extern "C" {
    void dgesvd_(char* jobu, char* jobvt, int* m, int* n,
                 double* a, int* lda,
                 double* s,
                 double* u, int* ldu,
                 double* vt, int* ldvt,
                 double* work, int* lwork, int* info);
}
#endif

// ============================================================================
// OPTIMIZED MATRIX CLASS
// ============================================================================
class Matrix {
public:
    int rows;
    int cols;
    int stride; // Optimization: Leading dimension with padding
    double* data;

    Matrix(int r, int c) : rows(r), cols(c) {
        // [OPTIMIZATION] Pad rows to 64-byte boundary (8 doubles)
        // This ensures every row starts at a cache-line boundary.
        stride = (r + 7) & ~7;
        if (stride < 8) stride = 8;

        size_t bytes = (size_t)stride * c * sizeof(double);

        // Ensure total allocation is also aligned
        data = (double*)ALIGNED_MALLOC(bytes, 64);
        if (!data) { std::exit(1); }
        std::memset(data, 0, bytes);
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), stride(other.stride) {
        size_t bytes = (size_t)stride * cols * sizeof(double);
        data = (double*)ALIGNED_MALLOC(bytes, 64);
        if (!data) std::exit(1);
        std::memcpy(data, other.data, bytes);
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (data) ALIGNED_FREE(data);
            rows = other.rows;
            cols = other.cols;
            stride = other.stride;
            size_t bytes = (size_t)stride * cols * sizeof(double);
            data = (double*)ALIGNED_MALLOC(bytes, 64);
            if (!data) std::exit(1);
            std::memcpy(data, other.data, bytes);
        }
        return *this;
    }

    ~Matrix() {
        if (data) ALIGNED_FREE(data);
    }

    static Matrix identity(int n) {
        Matrix res(n, n);
        for (int i = 0; i < n; i++) res.data[i * res.stride + i] = 1.0;
        return res;
    }

    // Unsafe access for speed
    inline double* ptr() { return data; }
    inline const double* ptr() const { return data; }

    // Stride-aware access
    inline double& operator()(int r, int c) { return data[c * stride + r]; }
    inline double* col_ptr(int c) { return data + c * stride; }
    inline const double* col_ptr(int c) const { return data + c * stride; }

    void randomize(double min_val, double max_val) {
        static std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(min_val, max_val);
        // Fill taking stride into account
        for (int c = 0; c < cols; ++c) {
            double* col = col_ptr(c);
            for (int r = 0; r < rows; ++r) {
                col[r] = dis(gen);
            }
        }
    }
};

// ============================================================================
// OPTIMIZED C++ IMPLEMENTATION (Sequential + AVX2)
// ============================================================================

namespace OptSVD {

    struct GivensRotation {
        int k;
        double c;
        double s;
    };

    Matrix transpose(const Matrix& A) {
        Matrix T(A.cols, A.rows);
        for (int j = 0; j < A.cols; ++j) {
            const double* A_col = A.col_ptr(j);
            for (int i = 0; i < A.rows; ++i) {
                T.col_ptr(i)[j] = A_col[i];
            }
        }
        return T;
    }

    inline double sign(double x) {
        return (x >= 0.0) ? 1.0 : -1.0;
    }

    // ------------------------------------------------------------------------
    // AVX Helper: Horizontal Sum
    // ------------------------------------------------------------------------
    inline double hsum_avx(__m256d v) {
        __m128d low = _mm256_castpd256_pd128(v);
        __m128d high = _mm256_extractf128_pd(v, 1);
        low = _mm_add_pd(low, high);
        return _mm_cvtsd_f64(_mm_hadd_pd(low, low));
    }

    // ------------------------------------------------------------------------
    // Householder Reflections
    // ------------------------------------------------------------------------

    double householderVector(double* RESTRICT x, int n, double& v0_out) {
        double norm_sq = 0.0;

        int i = 1;

        // [OPTIMIZATION] AVX2 Dot Product Accumulation
        if (n >= 5) {
            __m256d sum_vec = _mm256_setzero_pd();
            for (; i <= n - 4; i += 4) {
                __m256d val = _mm256_loadu_pd(&x[i]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(val, val));
            }
            norm_sq = hsum_avx(sum_vec);
        }

        for (; i < n; ++i) norm_sq += x[i] * x[i];

        double x0 = x[0];
        double norm = std::sqrt(x0*x0 + norm_sq);

        if (norm <= EPSILON) {
            v0_out = x0;
            return 0.0;
        }

        double alpha = (x0 >= 0) ? -norm : norm;
        double v0 = x0 - alpha;

        if (std::abs(v0) < EPSILON) {
             v0_out = x0;
             return 0.0;
        }

        double inv_v0 = 1.0 / v0;

        // [OPTIMIZATION] AVX2 Scaling
        i = 1;
        if (n >= 5) {
            __m256d inv_v0_vec = _mm256_set1_pd(inv_v0);
            for(; i <= n - 4; i += 4) {
                __m256d val = _mm256_loadu_pd(&x[i]);
                val = _mm256_mul_pd(val, inv_v0_vec);
                _mm256_storeu_pd(&x[i], val);
            }
        }
        for(; i < n; ++i) x[i] *= inv_v0;

        // Recompute norm for beta (stability)
        double v_norm_sq = 1.0;
        i = 1;
        if (n >= 5) {
             __m256d sum_vec = _mm256_setzero_pd();
            for (; i <= n - 4; i += 4) {
                __m256d val = _mm256_loadu_pd(&x[i]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(val, val));
            }
            v_norm_sq += hsum_avx(sum_vec);
        }
        else {
             // Fallback for very small vectors if skipped above
             // (though loop structure handles it, explicit else ensures clarity)
             // The loop structure `for(; i <= n-4...)` handles `i` correctly.
        }
        // Cleanup loop
        for(; i < n; ++i) v_norm_sq += x[i]*x[i];

        v0_out = alpha;
        return 2.0 / v_norm_sq;
    }

    // Apply Householder from Left
    void applyHouseholderLeft(Matrix& A, int rowStart, int colStart,
                             const std::vector<double>& v, double beta,
                             std::vector<double>& work) {
        int m = A.rows;
        int n = A.cols;
        int v_len = m - rowStart;
        int w_len = n - colStart;

        std::memset(work.data(), 0, w_len * sizeof(double));

        const double* RESTRICT v_ptr = v.data();
        double* RESTRICT work_ptr = work.data();

        // 1. Compute w = A(rowStart:m, colStart:n)^T * v
        for (int j = 0; j < w_len; ++j) {
            const double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double sum = A_col[0] * 1.0;
            int i = 1;

            // [OPTIMIZATION] AVX2 Dot Product
            if (v_len >= 5) {
                __m256d v_sum = _mm256_setzero_pd();
                for (; i <= v_len - 4; i += 4) {
                    __m256d a_val = _mm256_loadu_pd(&A_col[i]);
                    __m256d v_val = _mm256_loadu_pd(&v_ptr[i]);
                    v_sum = _mm256_add_pd(v_sum, _mm256_mul_pd(a_val, v_val));
                }
                sum += hsum_avx(v_sum);
            }
            for (; i < v_len; ++i) sum += A_col[i] * v_ptr[i];
            work_ptr[j] = sum;
        }

        // 2. A = A - beta * v * w^T
        for (int j = 0; j < w_len; ++j) {
            double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double val = beta * work_ptr[j];
            A_col[0] -= val;

            int i = 1;
            // [OPTIMIZATION] AVX2 Vector Update
            if (v_len >= 5) {
                __m256d val_vec = _mm256_set1_pd(val);
                for (; i <= v_len - 4; i += 4) {
                    __m256d a_vec = _mm256_loadu_pd(&A_col[i]);
                    __m256d v_vec = _mm256_loadu_pd(&v_ptr[i]);
                    // a = a - v * val
                    a_vec = _mm256_sub_pd(a_vec, _mm256_mul_pd(v_vec, val_vec));
                    _mm256_storeu_pd(&A_col[i], a_vec);
                }
            }
            for (; i < v_len; ++i) A_col[i] -= v_ptr[i] * val;
        }
    }

    // Apply Householder from Right
    void applyHouseholderRight(Matrix& A, int rowStart, int colStart,
                              const std::vector<double>& v, double beta,
                              std::vector<double>& work) {
        int m = A.rows;
        int n = A.cols;
        int v_len = n - colStart;
        int w_len = m - rowStart;

        double* RESTRICT work_ptr = work.data();
        std::memset(work_ptr, 0, w_len * sizeof(double));

        // 1. Compute w = A * v
        const double* RESTRICT A_col0 = A.col_ptr(colStart) + rowStart;
        int i = 0;

        // Initial copy
        if (w_len >= 4) {
             for (; i <= w_len - 4; i += 4) {
                _mm256_storeu_pd(&work_ptr[i], _mm256_loadu_pd(&A_col0[i]));
             }
        }
        for (; i < w_len; ++i) work_ptr[i] = A_col0[i];

        const double* RESTRICT v_ptr = v.data();
        for (int j = 1; j < v_len; ++j) {
            const double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double vj = v_ptr[j];
            __m256d vj_vec = _mm256_set1_pd(vj);

            i = 0;
            // [OPTIMIZATION] AVX2 Accumulate
            if (w_len >= 4) {
                for (; i <= w_len - 4; i += 4) {
                    __m256d w_vec = _mm256_loadu_pd(&work_ptr[i]);
                    __m256d a_vec = _mm256_loadu_pd(&A_col[i]);
                    w_vec = _mm256_add_pd(w_vec, _mm256_mul_pd(a_vec, vj_vec));
                    _mm256_storeu_pd(&work_ptr[i], w_vec);
                }
            }
            for (; i < w_len; ++i) work_ptr[i] += A_col[i] * vj;
        }

        // 2. A = A - beta * w * v^T
        // Column 0 update
        __m256d beta_vec = _mm256_set1_pd(beta);
        i = 0;
        if (w_len >= 4) {
            for (; i <= w_len - 4; i += 4) {
                __m256d a_vec = _mm256_loadu_pd(&A.col_ptr(colStart)[rowStart + i]);
                __m256d w_vec = _mm256_loadu_pd(&work_ptr[i]);
                a_vec = _mm256_sub_pd(a_vec, _mm256_mul_pd(beta_vec, w_vec));
                _mm256_storeu_pd(&A.col_ptr(colStart)[rowStart + i], a_vec);
            }
        }
        for (; i < w_len; ++i) A.col_ptr(colStart)[rowStart + i] -= beta * work_ptr[i];

        // Rest of columns
        for (int j = 1; j < v_len; ++j) {
            double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double factor = beta * v_ptr[j];
            __m256d factor_vec = _mm256_set1_pd(factor);

            i = 0;
            if (w_len >= 4) {
                for (; i <= w_len - 4; i += 4) {
                    __m256d a_vec = _mm256_loadu_pd(&A_col[i]);
                    __m256d w_vec = _mm256_loadu_pd(&work_ptr[i]);
                    a_vec = _mm256_sub_pd(a_vec, _mm256_mul_pd(w_vec, factor_vec));
                    _mm256_storeu_pd(&A_col[i], a_vec);
                }
            }
            for (; i < w_len; ++i) A_col[i] -= work_ptr[i] * factor;
        }
    }

    // ------------------------------------------------------------------------
    // Givens Rotations
    // ------------------------------------------------------------------------

    void givens(double a, double b, double& c, double& s) {
        if (b == 0.0) {
            c = 1.0; s = 0.0;
        } else {
            if (std::abs(b) > std::abs(a)) {
                double t = -a / b;
                s = 1.0 / std::sqrt(1.0 + t * t);
                c = s * t;
            } else {
                double t = -b / a;
                c = 1.0 / std::sqrt(1.0 + t * t);
                s = c * t;
            }
        }
    }

    // [OPTIMIZATION] Batched Givens Application with AVX2
    void flushGivens(Matrix& M, const std::vector<GivensRotation>& batch) {
        if (batch.empty()) return;

        int m = M.rows;
        const int ROW_CHUNK = 256;

        for (int r_base = 0; r_base < m; r_base += ROW_CHUNK) {
            int r_end = std::min(m, r_base + ROW_CHUNK);
            int len = r_end - r_base;

            for (const auto& rot : batch) {
                double* RESTRICT col_i = M.col_ptr(rot.k) + r_base;
                double* RESTRICT col_j = M.col_ptr(rot.k + 1) + r_base;
                double c = rot.c;
                double s = rot.s;

                __m256d c_vec = _mm256_set1_pd(c);
                __m256d s_vec = _mm256_set1_pd(s);

                int i = 0;
                // [OPTIMIZATION] AVX2 Rotation
                for (; i <= len - 4; i += 4) {
                    __m256d a_vec = _mm256_loadu_pd(&col_i[i]);
                    __m256d b_vec = _mm256_loadu_pd(&col_j[i]);

                    // new_a = c * a - s * b
                    __m256d new_a = _mm256_sub_pd(
                        _mm256_mul_pd(c_vec, a_vec),
                        _mm256_mul_pd(s_vec, b_vec)
                    );

                    // new_b = s * a + c * b
                    __m256d new_b = _mm256_add_pd(
                        _mm256_mul_pd(s_vec, a_vec),
                        _mm256_mul_pd(c_vec, b_vec)
                    );

                    _mm256_storeu_pd(&col_i[i], new_a);
                    _mm256_storeu_pd(&col_j[i], new_b);
                }
                for (; i < len; ++i) {
                    double a = col_i[i];
                    double b = col_j[i];
                    col_i[i] = c * a - s * b;
                    col_j[i] = s * a + c * b;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Helper: Matrix Multiplication with AVX2
    // ------------------------------------------------------------------------
    Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
        // C = A * B
        Matrix C(A.rows, B.cols);
        int m = C.rows;
        int n = C.cols;
        int k_dim = A.cols;

        // Optimized Cache Loop: j-k-i
        for (int j = 0; j < n; ++j) {
            double* RESTRICT C_col = C.col_ptr(j);
            const double* RESTRICT B_col = B.col_ptr(j);

            for (int k = 0; k < k_dim; ++k) {
                double b_val = B_col[k];
                if (b_val == 0.0) continue;

                const double* RESTRICT A_col = A.col_ptr(k);
                __m256d b_vec = _mm256_set1_pd(b_val);

                int i = 0;
                // [OPTIMIZATION] AVX2 FMA (simulated with mul+add)
                for (; i <= m - 4; i += 4) {
                    __m256d c_vec = _mm256_loadu_pd(&C_col[i]);
                    __m256d a_vec = _mm256_loadu_pd(&A_col[i]);

                    // c += a * b
                    c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));

                    _mm256_storeu_pd(&C_col[i], c_vec);
                }
                for (; i < m; ++i) {
                    C_col[i] += A_col[i] * b_val;
                }
            }
        }
        return C;
    }

    // ------------------------------------------------------------------------
    // Helper: Thin QR Decomposition (A = Q * R)
    // ------------------------------------------------------------------------
    void thinQR(const Matrix& A, Matrix& Q, Matrix& R) {
        int m = A.rows;
        int n = A.cols;

        Matrix H = A; // Copy of A to transform
        R = Matrix(n, n);
        Q = Matrix(m, n);

        std::vector<std::vector<double>> Vs;
        std::vector<double> Betas;
        Vs.reserve(n);
        Betas.reserve(n);

        std::vector<double> v_work(m);
        std::vector<double> work_buff(n);

        // 1. Compute Householder Vectors
        for (int k = 0; k < n; ++k) {
            int len = m - k;
            if (len <= 0) break;

            const double* H_col = H.col_ptr(k);
            for(int i=0; i<len; ++i) v_work[i] = H_col[k+i];

            double alpha;
            double beta = householderVector(v_work.data(), len, alpha);

            // Store for Q reconstruction
            std::vector<double> v_store(v_work.begin(), v_work.begin() + len);
            Vs.push_back(v_store);
            Betas.push_back(beta);

            if (beta != 0.0) {
                // Apply to remaining columns of H
                if (k + 1 < n) {
                    applyHouseholderLeft(H, k, k+1, v_store, beta, work_buff);
                }
            }
            // Update R diagonal (explicitly)
            H.col_ptr(k)[k] = alpha;
        }

        // 2. Extract R (Upper Triangular from H)
        for (int j = 0; j < n; ++j) {
            double* r_col = R.col_ptr(j);
            double* h_col = H.col_ptr(j);
            for (int i = 0; i <= j; ++i) r_col[i] = h_col[i];
        }

        // 3. Form Q (Explicit MxN)
        // Initialize Q to Thin Identity (MxN)
        for (int j = 0; j < n; ++j) {
            Q.col_ptr(j)[j] = 1.0;
        }

        // Apply reflectors in reverse order to form Q
        std::vector<double> q_work(n);
        for (int k = n - 1; k >= 0; --k) {
            if (Betas[k] == 0.0) continue;
            // Apply H_k to columns k..n of Q
            applyHouseholderLeft(Q, k, k, Vs[k], Betas[k], q_work);
        }
    }

    // ------------------------------------------------------------------------
    // Main Algorithm
    // ------------------------------------------------------------------------
    void compute(const Matrix& inputA, Matrix& U, std::vector<double>& S, Matrix& V);

    void compute(const Matrix& inputA, Matrix& U, std::vector<double>& S, Matrix& V) {
        // [OPTIMIZATION] Fat Matrix Handling (Row > Col)
        if (inputA.rows < inputA.cols) {
            Matrix AT = transpose(inputA);
            Matrix U_t(AT.rows, AT.rows);
            Matrix V_t(AT.cols, AT.cols);

            compute(AT, U_t, S, V_t);

            U = V_t;
            V = U_t;
            return;
        }

        // [OPTIMIZATION] Tall Matrix Handling (M >> N)
        // Use Thin QR to reduce M x N problem to N x N problem
        if (inputA.rows > 2 * inputA.cols) {
            Matrix Q_rect(inputA.rows, inputA.cols);
            Matrix R_small(inputA.cols, inputA.cols);

            // 1. Decompose A = Q * R
            thinQR(inputA, Q_rect, R_small);

            // 2. Compute SVD of R (Small Square)
            Matrix U_small(inputA.cols, inputA.cols);
            Matrix V_small(inputA.cols, inputA.cols);
            compute(R_small, U_small, S, V_small);

            // 3. Reconstruct U = Q * U_small
            U = matrixMultiply(Q_rect, U_small);
            V = V_small;
            return;
        }

        Matrix A = inputA;
        int m = A.rows;
        int n = A.cols;
        int min_mn = std::min(m, n);

        U = Matrix::identity(m);
        V = Matrix::identity(n);
        S.resize(min_mn);

        std::vector<double> v_work(std::max(m, n));
        std::vector<double> scratch_work(std::max(m, n));

        // 1. Bidiagonalization
        for (int k = 0; k < min_mn; ++k) {
            // Left Reflector
            int len_l = m - k;
            if (len_l > 0) {
                const double* RESTRICT A_col_k = A.col_ptr(k);
                for(int i=0; i<len_l; ++i) v_work[i] = A_col_k[k+i];

                double alpha;
                double beta = householderVector(v_work.data(), len_l, alpha);
                if (beta != 0.0) {
                    applyHouseholderLeft(A, k, k+1, v_work, beta, scratch_work);
                    A.col_ptr(k)[k] = alpha;
                    applyHouseholderRight(U, 0, k, v_work, beta, scratch_work);
                }
            }
            // Right Reflector
            if (k < n - 2) {
                int len_r = n - (k + 1);
                for(int j=0; j<len_r; ++j) v_work[j] = A.col_ptr(k+1+j)[k];

                double alpha;
                double beta = householderVector(v_work.data(), len_r, alpha);
                if (beta != 0.0) {
                    applyHouseholderRight(A, k+1, k+1, v_work, beta, scratch_work);
                    A.col_ptr(k+1)[k] = alpha;
                    applyHouseholderRight(V, 0, k+1, v_work, beta, scratch_work);
                }
            }
        }

        // Extract Bidiagonal d and f
        std::vector<double> d(min_mn);
        std::vector<double> f(min_mn - 1);
        for (int i = 0; i < min_mn; ++i) d[i] = A.col_ptr(i)[i];
        for (int i = 0; i < min_mn - 1; ++i) f[i] = A.col_ptr(i+1)[i];

        // [OPTIMIZATION] Batched Givens Buffers
        std::vector<GivensRotation> batchU, batchV;
        batchU.reserve(128);
        batchV.reserve(128);
        const size_t BATCH_LIMIT = 64;

        // 2. Golub-Kahan Iterations
        int iter = 0;
        int max_iter = MAX_ITER_MULTIPLIER * min_mn;

        while (iter < max_iter) {
            // Find small superdiagonal element
            int m_idx = min_mn - 1;
            while (m_idx > 0) {
                if (std::abs(f[m_idx-1]) <= EPSILON * (std::abs(d[m_idx-1]) + std::abs(d[m_idx]))) {
                    f[m_idx-1] = 0.0;
                    break;
                }
                m_idx--;
            }

            if (m_idx == min_mn - 1) {
                min_mn--;
                if (min_mn == 0) break;
                continue;
            }

            int q = min_mn - 1;
            int p = q - 1;
            while (p >= 0) {
                if (p == 0) break;
                if (std::abs(f[p-1]) <= EPSILON * (std::abs(d[p-1]) + std::abs(d[p]))) {
                    f[p-1] = 0.0;
                    break;
                }
                p--;
            }

            // Wilkinson Shift
            double d_q = d[q];
            double d_qm1 = d[q-1];
            double f_qm1 = f[q-1];
            double f_qm2 = (q-2 >= p) ? f[q-2] : 0.0;

            double a11 = d_qm1*d_qm1 + f_qm2*f_qm2;
            double a12 = d_qm1*f_qm1;
            double a22 = d_q*d_q + f_qm1*f_qm1;

            double diff = (a11 - a22) * 0.5;
            double mu = a22 - (a12*a12) / (diff + sign(diff) * std::sqrt(diff*diff + a12*a12));

            double y = d[p]*d[p] - mu;
            double z = d[p]*f[p];

            // Chase the bulge
            for (int k = p; k < q; ++k) {
                double c, s;
                givens(y, z, c, s);

                // Right Rotation (affects B and V)
                double dk = d[k];
                double fk = f[k];
                double dkp1 = d[k+1];

                d[k] = c*dk - s*fk;
                f[k] = s*dk + c*fk;
                d[k+1] = c*dkp1;
                double bulge = -s*dkp1;

                if (k > p) f[k-1] = c * y - s * z;

                // [OPTIMIZATION] Batch V update
                batchV.push_back({k, c, s});
                if (batchV.size() >= BATCH_LIMIT) {
                    flushGivens(V, batchV);
                    batchV.clear();
                }

                // Left Rotation (affects B and U)
                y = d[k];
                z = bulge;
                givens(y, z, c, s);

                d[k] = c*y - s*z;
                double old_fk = f[k];
                double old_dkp1 = d[k+1];

                f[k] = c*old_fk - s*old_dkp1;
                d[k+1] = s*old_fk + c*old_dkp1;

                if (k < q - 1) {
                    double fkp1 = f[k+1];
                    f[k+1] = c * fkp1;
                    y = f[k];
                    z = -s * fkp1;
                }

                // [OPTIMIZATION] Batch U update
                batchU.push_back({k, c, s});
                if (batchU.size() >= BATCH_LIMIT) {
                    flushGivens(U, batchU);
                    batchU.clear();
                }
            }
            iter++;
        }

        // Flush remaining rotations
        flushGivens(V, batchV);
        flushGivens(U, batchU);

        // Final Sort
        for(int i=0; i<(int)S.size(); ++i) {
            S[i] = d[i];
            if (S[i] < 0) {
                S[i] = -S[i];
                double* v_col = V.col_ptr(i);
                for(int r=0; r<n; ++r) v_col[r] = -v_col[r];
            }
        }

        for (int i = 0; i < (int)S.size() - 1; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < (int)S.size(); ++j) {
                if (S[j] > S[max_idx]) max_idx = j;
            }
            if (max_idx != i) {
                std::swap(S[i], S[max_idx]);
                // Swap columns of U
                double* u_i = U.col_ptr(i);
                double* u_max = U.col_ptr(max_idx);
                for(int r=0; r<m; ++r) std::swap(u_i[r], u_max[r]);

                // Swap columns of V
                double* v_i = V.col_ptr(i);
                double* v_max = V.col_ptr(max_idx);
                for(int r=0; r<n; ++r) std::swap(v_i[r], v_max[r]);
            }
        }
    }
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

void run_test(int rows, int cols) {
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Testing Size: " << rows << "x" << cols << "\n";

    Matrix A(rows, cols);
    A.randomize(0.0, 1.0);
    Matrix A_copy = A;

    Matrix U(rows, rows);
    Matrix V(cols, cols);
    std::vector<double> S;

    auto start_opt = std::chrono::high_resolution_clock::now();
    OptSVD::compute(A, U, S, V);
    auto end_opt = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_opt = end_opt - start_opt;
    std::cout << "  Custom SVD Time:   " << std::fixed << std::setprecision(5)
              << elapsed_opt.count() << " s\n";

    // Reconstruction Check (Manual)
    Matrix Recon(rows, cols);
    Matrix VT = OptSVD::transpose(V);

    // R = U * S * V^T
    // We compute this carefully. R = (U*S) * V^T
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            int k_lim = (int)S.size();
            for (int k = 0; k < k_lim; ++k) {
                // U(i,k) * S[k] * VT(k,j)
                // U is col-major: U.col_ptr(k)[i]
                // VT is col-major: VT.col_ptr(j)[k]
                sum += U.col_ptr(k)[i] * S[k] * VT.col_ptr(j)[k];
            }
            Recon.col_ptr(j)[i] = sum;
        }
    }

    double err_recon = 0.0;
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            double diff = A_copy.col_ptr(c)[r] - Recon.col_ptr(c)[r];
            err_recon += diff * diff;
        }
    }
    err_recon = std::sqrt(err_recon);
    std::cout << "  Error (Reconstruct): " << std::scientific << err_recon << "\n";
    std::cout << std::defaultfloat;

#if ENABLE_LAPACK_BENCHMARK
    std::vector<double> A_col_major(rows * cols);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            A_col_major[c * rows + r] = A_copy.col_ptr(c)[r];
        }
    }

    char jobu = 'A';
    char jobvt = 'A';
    int m = rows;
    int n = cols;
    int lda = m;
    int ldu = m;
    int ldvt = n;
    int info = 0;

    std::vector<double> s_lapack(std::min(m, n));
    std::vector<double> u_lapack(m * m);
    std::vector<double> vt_lapack(n * n);

    double work_query;
    int lwork = -1;
    dgesvd_(&jobu, &jobvt, &m, &n, A_col_major.data(), &lda,
            s_lapack.data(), u_lapack.data(), &ldu,
            vt_lapack.data(), &ldvt, &work_query, &lwork, &info);

    if (info == 0) {
        lwork = (int)work_query;
        std::vector<double> work(lwork);

        auto start_lapack = std::chrono::high_resolution_clock::now();
        dgesvd_(&jobu, &jobvt, &m, &n, A_col_major.data(), &lda,
                s_lapack.data(), u_lapack.data(), &ldu,
                vt_lapack.data(), &ldvt, work.data(), &lwork, &info);
        auto end_lapack = std::chrono::high_resolution_clock::now();

        if (info == 0) {
            std::chrono::duration<double> elapsed_lapack = end_lapack - start_lapack;
            std::cout << "  LAPACK SVD Time:   " << std::fixed << std::setprecision(5)
                      << elapsed_lapack.count() << " s\n";

            double ratio = elapsed_opt.count() / elapsed_lapack.count();
            std::cout << "  Slowdown Factor:   " << std::setprecision(2) << ratio << "x\n";
        }
    }
#endif
}

int main() {
    std::cout << "=== Optimized Serial QR SVD Benchmark ===\n";
    std::cout << "Optimizations: Thin-QR, Padded Columns, Explicit AVX2 Intrinsics\n";

    run_test(50, 50);
    run_test(200, 200);
    run_test(500, 500);

    std::cout << "\n--- Heavy Tests ---\n";
    run_test(1000, 1000);
     run_test(2000, 2000); // Uncomment for long test

    std::cout << "\n--- Rectangular Tests ---\n";
    run_test(1000, 200);
    run_test(200, 1000);

    return 0;
}