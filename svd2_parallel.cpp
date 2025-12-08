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

// [OPTIMIZATION] OpenMP
#include <omp.h>

// ============================================================================
// CONFIGURATION
// ============================================================================
#define ENABLE_LAPACK_BENCHMARK 1
#define EPSILON 1e-13
#define MAX_ITER_MULTIPLIER 30

// Tuning: Only use threads if total elements > threshold.
// 64*64 = 4096 elements is roughly where parallel overhead breaks even.
#define PARALLEL_THRESHOLD_ELEMENTS 16384 // e.g. 128x128 matrix
#define PARALLEL_THRESHOLD_GIVENS_ROWS 512 // Only parallelize rotations on tall matrices

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
// OPTIMIZED C++ IMPLEMENTATION (Sequential Base + Smart OpenMP)
// ============================================================================

namespace OptSVD {

    struct GivensRotation {
        int k;
        double c;
        double s;
    };

    Matrix transpose(const Matrix& A) {
        Matrix T(A.cols, A.rows);
        long long size = (long long)A.rows * A.cols;

        // [OPTIMIZATION] Parallel Transpose (Conditional)
        #pragma omp parallel for collapse(2) if(size > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 0; j < A.cols; ++j) {
            for (int i = 0; i < A.rows; ++i) {
                T.col_ptr(i)[j] = A.col_ptr(j)[i];
            }
        }
        return T;
    }

    inline double sign(double x) {
        return (x >= 0.0) ? 1.0 : -1.0;
    }

    // ------------------------------------------------------------------------
    // Householder Reflections
    // ------------------------------------------------------------------------

    // Keep Householder vector calc serial (too fine-grained)
    double householderVector(double* RESTRICT x, int n, double& v0_out) {
        double norm_sq = 0.0;

        // [OPTIMIZATION] Manual 4x Unroll
        int i = 1;
        for (; i <= n - 4; i += 4) {
            norm_sq += x[i] * x[i];
            norm_sq += x[i+1] * x[i+1];
            norm_sq += x[i+2] * x[i+2];
            norm_sq += x[i+3] * x[i+3];
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

        i = 1;
        for(; i <= n - 4; i += 4) {
            x[i]   *= inv_v0;
            x[i+1] *= inv_v0;
            x[i+2] *= inv_v0;
            x[i+3] *= inv_v0;
        }
        for(; i < n; ++i) x[i] *= inv_v0;

        double v_norm_sq = 1.0;
        i = 1;
        for(; i <= n - 4; i += 4) {
            v_norm_sq += x[i] * x[i];
            v_norm_sq += x[i+1] * x[i+1];
            v_norm_sq += x[i+2] * x[i+2];
            v_norm_sq += x[i+3] * x[i+3];
        }
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
        long long ops_estimate = (long long)v_len * w_len;

        std::memset(work.data(), 0, w_len * sizeof(double));

        const double* RESTRICT v_ptr = v.data();
        double* RESTRICT work_ptr = work.data();

        // 1. Compute w
        #pragma omp parallel for schedule(static) if(ops_estimate > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 0; j < w_len; ++j) {
            const double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double sum = A_col[0] * 1.0;
            int i = 1;
            for (; i <= v_len - 4; i += 4) {
                sum += A_col[i]   * v_ptr[i];
                sum += A_col[i+1] * v_ptr[i+1];
                sum += A_col[i+2] * v_ptr[i+2];
                sum += A_col[i+3] * v_ptr[i+3];
            }
            for (; i < v_len; ++i) sum += A_col[i] * v_ptr[i];
            work_ptr[j] = sum;
        }

        // 2. A = A - beta * v * w^T
        #pragma omp parallel for schedule(static) if(ops_estimate > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 0; j < w_len; ++j) {
            double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double val = beta * work_ptr[j];
            A_col[0] -= val;
            int i = 1;
            for (; i <= v_len - 4; i += 4) {
                A_col[i]   -= v_ptr[i]   * val;
                A_col[i+1] -= v_ptr[i+1] * val;
                A_col[i+2] -= v_ptr[i+2] * val;
                A_col[i+3] -= v_ptr[i+3] * val;
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
        long long ops_estimate = (long long)v_len * w_len;

        double* RESTRICT work_ptr = work.data();
        std::memset(work_ptr, 0, w_len * sizeof(double));

        // 1. Compute w = A * v (Serial to prevent race conditions on work_ptr)
        const double* RESTRICT A_col0 = A.col_ptr(colStart) + rowStart;
        int i = 0;
        for (; i <= w_len - 4; i += 4) {
            work_ptr[i]   = A_col0[i];
            work_ptr[i+1] = A_col0[i+1];
            work_ptr[i+2] = A_col0[i+2];
            work_ptr[i+3] = A_col0[i+3];
        }
        for (; i < w_len; ++i) work_ptr[i] = A_col0[i];

        const double* RESTRICT v_ptr = v.data();
        for (int j = 1; j < v_len; ++j) {
            const double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double vj = v_ptr[j];
            i = 0;
            for (; i <= w_len - 4; i += 4) {
                work_ptr[i]   += A_col[i]   * vj;
                work_ptr[i+1] += A_col[i+1] * vj;
                work_ptr[i+2] += A_col[i+2] * vj;
                work_ptr[i+3] += A_col[i+3] * vj;
            }
            for (; i < w_len; ++i) work_ptr[i] += A_col[i] * vj;
        }

        // 2. A = A - beta * w * v^T
        // Column 0 separate
        i = 0;
        for (; i <= w_len - 4; i += 4) {
            double* RESTRICT A_col = A.col_ptr(colStart) + rowStart;
            A_col[i]   -= beta * work_ptr[i];
            A_col[i+1] -= beta * work_ptr[i+1];
            A_col[i+2] -= beta * work_ptr[i+2];
            A_col[i+3] -= beta * work_ptr[i+3];
        }
        for (; i < w_len; ++i) {
            double* RESTRICT A_col = A.col_ptr(colStart) + rowStart;
            A_col[i] -= beta * work_ptr[i];
        }

        // Parallelize remaining columns ONLY if large enough
        #pragma omp parallel for schedule(static) if(ops_estimate > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 1; j < v_len; ++j) {
            double* RESTRICT A_col = A.col_ptr(colStart + j) + rowStart;
            double factor = beta * v_ptr[j];
            int local_i = 0;
            for (; local_i <= w_len - 4; local_i += 4) {
                A_col[local_i]   -= work_ptr[local_i]   * factor;
                A_col[local_i+1] -= work_ptr[local_i+1] * factor;
                A_col[local_i+2] -= work_ptr[local_i+2] * factor;
                A_col[local_i+3] -= work_ptr[local_i+3] * factor;
            }
            for (; local_i < w_len; ++local_i) A_col[local_i] -= work_ptr[local_i] * factor;
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

    // [OPTIMIZATION] Parallel Batched Givens Application
    void flushGivens(Matrix& M, const std::vector<GivensRotation>& batch) {
        if (batch.empty()) return;

        int m = M.rows;
        // Removed manual ROW_CHUNK to allow OpenMP to utilize all cores on full range

        // Parallelize over ALL rows directly. OpenMP chunks this automatically.
        #pragma omp parallel for schedule(static) if(m > PARALLEL_THRESHOLD_GIVENS_ROWS)
        for (int r = 0; r < m; ++r) {
            // Apply all rotations in the batch to row 'r'
            // Since we operate on columns (rot.k, rot.k+1), row operations are independent.
            double* RESTRICT row_ptr = M.col_ptr(0) + r; // Base offset for row r? No, M is col-major.

            // In col-major, M(r, c) = M.data[c * stride + r]
            // We need to access M(r, k) and M(r, k+1)

            for (const auto& rot : batch) {
                double* RESTRICT col_i = M.col_ptr(rot.k);
                double* RESTRICT col_j = M.col_ptr(rot.k + 1);

                double a = col_i[r];
                double b = col_j[r];

                // Rotation
                col_i[r] = rot.c * a - rot.s * b;
                col_j[r] = rot.s * a + rot.c * b;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Helper: Matrix Multiplication
    // ------------------------------------------------------------------------
    Matrix matrixMultiply(const Matrix& A, const Matrix& B) {
        Matrix C(A.rows, B.cols);
        int m = C.rows;
        int n = C.cols;
        int k_dim = A.cols;
        long long ops_estimate = (long long)m * n * k_dim;

        // [OPTIMIZATION] Parallel Matrix Multiply
        #pragma omp parallel for schedule(static) if(ops_estimate > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 0; j < n; ++j) {
            double* RESTRICT C_col = C.col_ptr(j);
            const double* RESTRICT B_col = B.col_ptr(j);

            for (int k = 0; k < k_dim; ++k) {
                double b_val = B_col[k];
                if (b_val == 0.0) continue;

                const double* RESTRICT A_col = A.col_ptr(k);

                int i = 0;
                for (; i <= m - 4; i += 4) {
                    C_col[i]   += A_col[i]   * b_val;
                    C_col[i+1] += A_col[i+1] * b_val;
                    C_col[i+2] += A_col[i+2] * b_val;
                    C_col[i+3] += A_col[i+3] * b_val;
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

        Matrix H = A;
        R = Matrix(n, n);
        Q = Matrix(m, n);

        std::vector<std::vector<double>> Vs;
        std::vector<double> Betas;
        Vs.reserve(n);
        Betas.reserve(n);

        std::vector<double> v_work(m);
        std::vector<double> work_buff(n);

        for (int k = 0; k < n; ++k) {
            int len = m - k;
            if (len <= 0) break;

            const double* H_col = H.col_ptr(k);
            for(int i=0; i<len; ++i) v_work[i] = H_col[k+i];

            double alpha;
            double beta = householderVector(v_work.data(), len, alpha);

            std::vector<double> v_store(v_work.begin(), v_work.begin() + len);
            Vs.push_back(v_store);
            Betas.push_back(beta);

            if (beta != 0.0) {
                if (k + 1 < n) {
                    applyHouseholderLeft(H, k, k+1, v_store, beta, work_buff);
                }
            }
            H.col_ptr(k)[k] = alpha;
        }

        // Extract R
        #pragma omp parallel for schedule(static) if(n*n > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 0; j < n; ++j) {
            double* r_col = R.col_ptr(j);
            double* h_col = H.col_ptr(j);
            for (int i = 0; i <= j; ++i) r_col[i] = h_col[i];
        }

        // Initialize Q
        #pragma omp parallel for schedule(static) if(n*n > PARALLEL_THRESHOLD_ELEMENTS)
        for (int j = 0; j < n; ++j) {
            Q.col_ptr(j)[j] = 1.0;
        }

        std::vector<double> q_work(n);
        for (int k = n - 1; k >= 0; --k) {
            if (Betas[k] == 0.0) continue;
            applyHouseholderLeft(Q, k, k, Vs[k], Betas[k], q_work);
        }
    }

    // ------------------------------------------------------------------------
    // Main Algorithm
    // ------------------------------------------------------------------------
    void compute(const Matrix& inputA, Matrix& U, std::vector<double>& S, Matrix& V);

    void compute(const Matrix& inputA, Matrix& U, std::vector<double>& S, Matrix& V) {
        if (inputA.rows < inputA.cols) {
            Matrix AT = transpose(inputA);
            Matrix U_t(AT.rows, AT.rows);
            Matrix V_t(AT.cols, AT.cols);

            compute(AT, U_t, S, V_t);

            U = V_t;
            V = U_t;
            return;
        }

        if (inputA.rows > 2 * inputA.cols) {
            Matrix Q_rect(inputA.rows, inputA.cols);
            Matrix R_small(inputA.cols, inputA.cols);

            thinQR(inputA, Q_rect, R_small);

            Matrix U_small(inputA.cols, inputA.cols);
            Matrix V_small(inputA.cols, inputA.cols);
            compute(R_small, U_small, S, V_small);

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

        // Extract Bidiagonal
        std::vector<double> d(min_mn);
        std::vector<double> f(min_mn - 1);
        for (int i = 0; i < min_mn; ++i) d[i] = A.col_ptr(i)[i];
        for (int i = 0; i < min_mn - 1; ++i) f[i] = A.col_ptr(i+1)[i];

        std::vector<GivensRotation> batchU, batchV;
        batchU.reserve(128);
        batchV.reserve(128);

        // [OPTIMIZATION] Increase batch limit to reduce synchronization frequency
        const size_t BATCH_LIMIT = 256;

        // 2. Golub-Kahan Iterations
        int iter = 0;
        int max_iter = MAX_ITER_MULTIPLIER * min_mn;

        while (iter < max_iter) {
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

            for (int k = p; k < q; ++k) {
                double c, s;
                givens(y, z, c, s);

                double dk = d[k];
                double fk = f[k];
                double dkp1 = d[k+1];

                d[k] = c*dk - s*fk;
                f[k] = s*dk + c*fk;
                d[k+1] = c*dkp1;
                double bulge = -s*dkp1;

                if (k > p) f[k-1] = c * y - s * z;

                batchV.push_back({k, c, s});
                if (batchV.size() >= BATCH_LIMIT) {
                    flushGivens(V, batchV);
                    batchV.clear();
                }

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

                batchU.push_back({k, c, s});
                if (batchU.size() >= BATCH_LIMIT) {
                    flushGivens(U, batchU);
                    batchU.clear();
                }
            }
            iter++;
        }

        flushGivens(V, batchV);
        flushGivens(U, batchU);

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
                double* u_i = U.col_ptr(i);
                double* u_max = U.col_ptr(max_idx);
                for(int r=0; r<m; ++r) std::swap(u_i[r], u_max[r]);

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

    // Check parallel threshold for verification logic too
    long long verify_ops = (long long)rows * cols * S.size();

    #pragma omp parallel for schedule(static) if(verify_ops > PARALLEL_THRESHOLD_ELEMENTS)
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            int k_lim = (int)S.size();
            for (int k = 0; k < k_lim; ++k) {
                sum += U.col_ptr(k)[i] * S[k] * VT.col_ptr(j)[k];
            }
            Recon.col_ptr(j)[i] = sum;
        }
    }

    double err_recon = 0.0;
    #pragma omp parallel for reduction(+:err_recon) if(verify_ops > PARALLEL_THRESHOLD_ELEMENTS)
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
    std::cout << "=== Optimized Parallel QR SVD Benchmark ===\n";
    std::cout << "Optimizations: Thin-QR, Padded Columns, Smart Parallel (OpenMP), No Explicit AVX\n";

    run_test(50, 50);
    run_test(200, 200);
    run_test(500, 500);

    std::cout << "\n--- Heavy Tests ---\n";
    run_test(2000, 2000);
    // run_test(1500, 1500); // Uncomment for long test

    std::cout << "\n--- Rectangular Tests ---\n";
    run_test(1000, 200);
    run_test(200, 1000);

    return 0;
}