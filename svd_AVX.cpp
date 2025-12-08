#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h> // REQUIRED: AVX/AVX2 Intrinsics
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h> // Required for _aligned_malloc on Windows
#define RESTRICT __restrict
#else
#include <stdlib.h> // Required for posix_memalign on Linux/Mac
#define RESTRICT __restrict__
#endif

// -------------------------------------------------------------------------
// LAPACK Prototype
// -------------------------------------------------------------------------
extern "C" {
void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
             double *s, double *u, int *ldu, double *vt, int *ldvt,
             double *work, int *lwork, int *info);
}
extern "C" {
// Keep dgesvd_ if you want, but add this:
void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda, double *s,
             double *u, int *ldu, double *vt, int *ldvt, double *work,
             int *lwork, int *iwork, int *info);
}
extern "C" {
void openblas_set_num_threads(int num_threads);
int openblas_get_num_threads();
}
// -------------------------------------------------------------------------
// ALIGNED MEMORY UTILS
// -------------------------------------------------------------------------
void *allocate_aligned(size_t size) {
  // 64-byte alignment matches Cache Line size on most modern CPUs
#if defined(_WIN32) || defined(_MSC_VER)
  return _aligned_malloc(size, 64);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 64, size) != 0)
    return nullptr;
  return ptr;
#endif
}

void free_aligned(void *ptr) {
#if defined(_WIN32) || defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

// -------------------------------------------------------------------------
// Matrix Class (Manual Memory Management for Alignment)
// -------------------------------------------------------------------------
struct Matrix {
  int rows;
  int cols;
  double *data;

  Matrix(int r, int c) : rows(r), cols(c) {
    size_t bytes = (size_t)r * c * sizeof(double);
    data = (double *)allocate_aligned(bytes);
    if (!data) {
      std::cerr << "Memory allocation failed!" << std::endl;
      std::exit(1);
    }
    std::memset(data, 0, bytes);
  }

  Matrix(const Matrix &other) : rows(other.rows), cols(other.cols) {
    size_t bytes = (size_t)rows * cols * sizeof(double);
    data = (double *)allocate_aligned(bytes);
    if (!data)
      std::exit(1);
    std::memcpy(data, other.data, bytes);
  }

  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      if (data)
        free_aligned(data);
      rows = other.rows;
      cols = other.cols;
      size_t bytes = (size_t)rows * cols * sizeof(double);
      data = (double *)allocate_aligned(bytes);
      if (!data)
        std::exit(1);
      std::memcpy(data, other.data, bytes);
    }
    return *this;
  }

  ~Matrix() {
    if (data)
      free_aligned(data);
  }

  inline double *col_ptr(int c) { return data + c * rows; }
  inline const double *col_ptr(int c) const { return data + c * rows; }

  inline double &operator()(int r, int c) { return data[r + c * rows]; }
  inline const double &operator()(int r, int c) const {
    return data[r + c * rows];
  }
  double *ptr() { return data; }
  const double *ptr() const { return data; }
};

// -------------------------------------------------------------------------
// BLOCK CONFIGURATION
// -------------------------------------------------------------------------
// 32 columns * 500 rows * 8 bytes = 128 KB.
// Two blocks fit comfortably in a 256KB or 512KB L2 cache.
const int BLOCK_SIZE = 32;

// -------------------------------------------------------------------------
// BLOCKED SEQUENTIAL SVD WITH AVX2 VECTORIZATION
// -------------------------------------------------------------------------
int svd_jacobi_one_sided(const Matrix &input, Matrix &U, std::vector<double> &S,
                         Matrix &V) {
  int m = input.rows;
  int n = input.cols;

  U = input;
  V = Matrix(n, n);

  // Init V to Identity
  std::memset(V.data, 0, (size_t)n * n * sizeof(double));
  for (int i = 0; i < n; ++i)
    V(i, i) = 1.0;

  const double tol = 1e-15;
  const int max_sweeps = 100;

  // Fix: Lowered threshold to 1e-32 (approx machine epsilon squared).
  const double small_val = 1e-32;

  // Cache Norms
  std::vector<double> col_norms_sq(n);
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    const double *RESTRICT ptr = U.col_ptr(i);

    // AVX2 Norm Calculation
    __m256d v_sum = _mm256_setzero_pd();
    int k = 0;
    for (; k <= m - 4; k += 4) {
      __m256d v_val = _mm256_loadu_pd(ptr + k);
      v_sum = _mm256_fmadd_pd(v_val, v_val, v_sum); // sum += val * val
    }

    // Horizontal reduction of v_sum
    double temp[4];
    _mm256_storeu_pd(temp, v_sum);
    sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Tail cleanup
    for (; k < m; ++k)
      sum += ptr[k] * ptr[k];

    col_norms_sq[i] = sum;
  }

  int sweep = 0;
  for (; sweep < max_sweeps; ++sweep) {
    int changed_cnt = 0;

    // --- BLOCK ITERATION ---
    for (int bi = 0; bi < n; bi += BLOCK_SIZE) {
      int bi_end = std::min(bi + BLOCK_SIZE, n);

      for (int bj = bi; bj < n; bj += BLOCK_SIZE) {
        int bj_end = std::min(bj + BLOCK_SIZE, n);

        // --- INNER KERNEL: PROCESS BLOCK PAIR ---
        for (int i = bi; i < bi_end; ++i) {
          double a = col_norms_sq[i];
          if (a < small_val)
            continue;

          double *RESTRICT p_ui = U.col_ptr(i);
          double *RESTRICT p_vi = V.col_ptr(i);

          int j_start = (bi == bj) ? (i + 1) : bj;

          for (int j = j_start; j < bj_end; ++j) {
            double b = col_norms_sq[j];
            if (b < small_val)
              continue;

            double *RESTRICT p_uj = U.col_ptr(j);
            double *RESTRICT p_vj = V.col_ptr(j);

            // --- AVX2 DOT PRODUCT ---
            __m256d v_dot = _mm256_setzero_pd();
            int k = 0;

            // Process 4 doubles at a time
            for (; k <= m - 4; k += 4) {
              __m256d v_a = _mm256_loadu_pd(p_ui + k);
              __m256d v_b = _mm256_loadu_pd(p_uj + k);
              v_dot = _mm256_fmadd_pd(v_a, v_b, v_dot);
            }

            // Horizontal Sum
            double dot_temp[4];
            _mm256_storeu_pd(dot_temp, v_dot);
            double g = dot_temp[0] + dot_temp[1] + dot_temp[2] + dot_temp[3];

            // Tail loop for remaining elements
            for (; k < m; ++k)
              g += p_ui[k] * p_uj[k];

            // --- CHECK ORTHOGONALITY ---
            if (g * g < (tol * tol) * a * b)
              continue;

            changed_cnt++;

            // --- ROTATION PARAMS ---
            double zeta = (b - a) / (2.0 * g);
            double t = std::copysign(
                1.0 / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta)), zeta);
            double c = 1.0 / std::sqrt(1.0 + t * t);
            double s = c * t;

            // --- AVX2 APPLY ROTATION U ---
            __m256d v_c = _mm256_set1_pd(c);
            __m256d v_s = _mm256_set1_pd(s);

            k = 0;
            for (; k <= m - 4; k += 4) {
              // Load columns
              __m256d v_col_i = _mm256_loadu_pd(p_ui + k);
              __m256d v_col_j = _mm256_loadu_pd(p_uj + k);

              // Explicit FMA:
              // new_i = fmsub(c, col_i, s*col_j) -> (c*col_i) - (s*col_j)
              // new_j = fmadd(s, col_i, c*col_j) -> (s*col_i) + (c*col_j)

              __m256d v_new_i =
                  _mm256_fmsub_pd(v_c, v_col_i, _mm256_mul_pd(v_s, v_col_j));
              __m256d v_new_j =
                  _mm256_fmadd_pd(v_s, v_col_i, _mm256_mul_pd(v_c, v_col_j));

              _mm256_storeu_pd(p_ui + k, v_new_i);
              _mm256_storeu_pd(p_uj + k, v_new_j);
            }
            // Tail cleanup
            for (; k < m; ++k) {
              double tmp = p_ui[k];
              p_ui[k] = c * tmp - s * p_uj[k];
              p_uj[k] = s * tmp + c * p_uj[k];
            }

            // --- AVX2 APPLY ROTATION V ---
            k = 0;
            for (; k <= n - 4; k += 4) {
              __m256d v_vi = _mm256_loadu_pd(p_vi + k);
              __m256d v_vj = _mm256_loadu_pd(p_vj + k);

              __m256d v_new_vi =
                  _mm256_fmsub_pd(v_c, v_vi, _mm256_mul_pd(v_s, v_vj));
              __m256d v_new_vj =
                  _mm256_fmadd_pd(v_s, v_vi, _mm256_mul_pd(v_c, v_vj));

              _mm256_storeu_pd(p_vi + k, v_new_vi);
              _mm256_storeu_pd(p_vj + k, v_new_vj);
            }
            for (; k < n; ++k) {
              double tmp = p_vi[k];
              p_vi[k] = c * tmp - s * p_vj[k];
              p_vj[k] = s * tmp + c * p_vj[k];
            }

            // --- UPDATE NORMS (Algebraic) ---
            double c2 = c * c;
            double s2 = s * s;
            double two_cs_g = 2.0 * c * s * g;

            double new_a =
                c2 * col_norms_sq[i] + s2 * col_norms_sq[j] - two_cs_g;
            double new_b =
                s2 * col_norms_sq[i] + c2 * col_norms_sq[j] + two_cs_g;

            if (new_a < small_val)
              new_a = 0.0;
            if (new_b < small_val)
              new_b = 0.0;

            col_norms_sq[i] = new_a;
            col_norms_sq[j] = new_b;
            a = new_a;
          }
        }
      }
    }

    if (changed_cnt == 0)
      break;

    // Refresh Norms (AVX2)
    if (sweep % 4 == 0) {
      for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        const double *RESTRICT ptr = U.col_ptr(i);
        __m256d v_sum = _mm256_setzero_pd();
        int k = 0;
        for (; k <= m - 4; k += 4) {
          __m256d v_val = _mm256_loadu_pd(ptr + k);
          v_sum = _mm256_fmadd_pd(v_val, v_val, v_sum);
        }
        double temp[4];
        _mm256_storeu_pd(temp, v_sum);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
        for (; k < m; ++k)
          sum += ptr[k] * ptr[k];
        col_norms_sq[i] = sum;
      }
    }
  }

  // Extract S and Normalize U (AVX2)
  S.resize(n);
  for (int i = 0; i < n; ++i) {
    double *RESTRICT ptr = U.col_ptr(i);
    double sum = 0.0;

    // Norm calc
    __m256d v_sum = _mm256_setzero_pd();
    int k = 0;
    for (; k <= m - 4; k += 4) {
      __m256d v_val = _mm256_loadu_pd(ptr + k);
      v_sum = _mm256_fmadd_pd(v_val, v_val, v_sum);
    }
    double temp[4];
    _mm256_storeu_pd(temp, v_sum);
    sum = temp[0] + temp[1] + temp[2] + temp[3];
    for (; k < m; ++k)
      sum += ptr[k] * ptr[k];

    double norm = std::sqrt(sum);
    S[i] = norm;

    if (norm > 1e-20) {
      double inv_norm = 1.0 / norm;
      __m256d v_inv = _mm256_set1_pd(inv_norm);
      k = 0;
      for (; k <= m - 4; k += 4) {
        __m256d v_val = _mm256_loadu_pd(ptr + k);
        v_val = _mm256_mul_pd(v_val, v_inv);
        _mm256_storeu_pd(ptr + k, v_val);
      }
      for (; k < m; ++k)
        ptr[k] *= inv_norm;
    }
  }

  return sweep;
}

// -------------------------------------------------------------------------
// RECONSTRUCTION ERROR (Fast)
// -------------------------------------------------------------------------
double check_reconstruction_error(const Matrix &A, const Matrix &U,
                                  const std::vector<double> &S,
                                  const Matrix &V_or_VT, bool is_VT) {
  int m = A.rows;
  int n = A.cols;
  double error_sum = 0.0;
  int k_lim = S.size();

  // Reconstruct A(r,c) individually
  for (int c = 0; c < n; ++c) {
    for (int r = 0; r < m; ++r) {
      double reconstructed = 0.0;
      for (int k = 0; k < k_lim; ++k) {
        double u_val = U(r, k);
        double s_val = S[k];
        double v_val = is_VT ? V_or_VT(k, c) : V_or_VT(c, k);
        reconstructed += u_val * s_val * v_val;
      }
      double diff = A(r, c) - reconstructed;
      error_sum += diff * diff;
    }
  }
  return std::sqrt(error_sum);
}

// -------------------------------------------------------------------------
// LAPACK WRAPPER
// -------------------------------------------------------------------------
void svd_lapack(const Matrix &input, Matrix &U, std::vector<double> &S,
                Matrix &VT) {
  int m = input.rows;
  int n = input.cols;
  int lda = m, ldu = m, ldvt = n, info;
  Matrix A_copy = input;
  S.resize(std::min(m, n));
  U = Matrix(m, m);
  VT = Matrix(n, n);
  double wkopt;
  int lwork = -1;
  char jobu = 'S', jobvt = 'S';
  dgesvd_(&jobu, &jobvt, &m, &n, A_copy.ptr(), &lda, S.data(), U.ptr(), &ldu,
          VT.ptr(), &ldvt, &wkopt, &lwork, &info);
  lwork = (int)wkopt;
  std::vector<double> work(lwork);
  dgesvd_(&jobu, &jobvt, &m, &n, A_copy.ptr(), &lda, S.data(), U.ptr(), &ldu,
          VT.ptr(), &ldvt, work.data(), &lwork, &info);
}
void svd_lapack_divide_conquer(const Matrix &input, Matrix &U,
                               std::vector<double> &S, Matrix &VT) {
  int m = input.rows;
  int n = input.cols;
  int lda = m, ldu = m, ldvt = n, info;
  Matrix A_copy = input;

  int min_dim = std::min(m, n);
  S.resize(min_dim);
  U = Matrix(m, m); // Note: dgesdd usually requires jobz='A' or 'S'
  VT = Matrix(n, n);

  // dgesdd uses 'S' for singular vectors (min(m,n) columns of U)
  // or 'A' for all. Let's use 'A' to match your previous setup for safety,
  // or 'S' for speed. Let's stick to 'A' for direct comparison.
  char jobz = 'S';

  // Workspace query
  double wkopt;
  int lwork = -1;
  // dgesdd needs an Integer Workspace (iwork)
  int iwork_len = 8 * min_dim;
  std::vector<int> iwork(iwork_len);

  dgesdd_(&jobz, &m, &n, A_copy.ptr(), &lda, S.data(), U.ptr(), &ldu, VT.ptr(),
          &ldvt, &wkopt, &lwork, iwork.data(), &info);

  lwork = (int)wkopt;
  std::vector<double> work(lwork);

  // Real Compute
  dgesdd_(&jobz, &m, &n, A_copy.ptr(), &lda, S.data(), U.ptr(), &ldu, VT.ptr(),
          &ldvt, work.data(), &lwork, iwork.data(), &info);
}

// -------------------------------------------------------------------------
// TEST GENERATOR
// -------------------------------------------------------------------------
enum MatrixType {
  RANDOM,
  IDENTITY,
  HILBERT,
  RANK_DEFICIENT,
  CLUSTERED_SIGMA,
  ZERO,
  DIAGONAL
};

void generate_matrix(Matrix &M, MatrixType type) {
  int m = M.rows;
  int n = M.cols;
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  if (type == RANDOM) {
    for (int i = 0; i < m * n; ++i)
      M.data[i] = dis(gen);
  } else if (type == CLUSTERED_SIGMA) {
    Matrix Temp(m, n);
    std::memset(Temp.data, 0, (size_t)m * n * sizeof(double));
    for (int i = 0; i < std::min(m, n); ++i)
      Temp(i, i) = 1.0 + (i * 1e-5);
    for (int i = 0; i < m * n; ++i)
      M.data[i] = Temp.data[i] + dis(gen) * 0.01;
  } else if (type == IDENTITY) {
    std::memset(M.data, 0, (size_t)m * n * sizeof(double));
    for (int i = 0; i < std::min(m, n); ++i)
      M(i, i) = 1.0;
  } else if (type == HILBERT) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        M(i, j) = 1.0 / (double)(i + j + 1);
      }
    }
  } else if (type == RANK_DEFICIENT) {
    for (int i = 0; i < m * n; ++i)
      M.data[i] = dis(gen);
    // Make last 20% of columns copies
    int limit = n * 0.2;
    if (limit < 1)
      limit = 1;
    for (int k = 0; k < limit; ++k)
      std::memcpy(M.col_ptr(n - 1 - k), M.col_ptr(k),
                  (size_t)m * sizeof(double));
  } else if (type == ZERO) {
    std::memset(M.data, 0, (size_t)m * n * sizeof(double));
  } else if (type == DIAGONAL) {
    std::memset(M.data, 0, (size_t)m * n * sizeof(double));
    for (int i = 0; i < std::min(m, n); ++i) {
      M(i, i) = (double)(i + 1);
    }
  }
}
double check_orthogonality(const Matrix &U, const std::vector<double> &S) {
  int rows = U.rows;
  int cols = U.cols;
  double max_error = 0.0;
  const double epsilon = 1e-15;

  for (int i = 0; i < cols; ++i) {
    double s_val = (i < S.size()) ? S[i] : 0.0;
    if (s_val < epsilon)
      continue;

    for (int j = i; j < cols; ++j) {
      double s_val_j = (j < S.size()) ? S[j] : 0.0;
      if (s_val_j < epsilon)
        continue;

      double dot = 0.0;
      for (int k = 0; k < rows; ++k)
        dot += U(k, i) * U(k, j);

      double expected = (i == j) ? 1.0 : 0.0;
      double error = std::abs(dot - expected);
      if (error > max_error)
        max_error = error;
    }
  }
  return max_error;
}
// -------------------------------------------------------------------------
// HEAVY TEST RUNNER
// -------------------------------------------------------------------------
void run_heavy_test(int rows, int cols, MatrixType type,
                    std::string test_name) {
  std::cout << "\n=== TEST: " << test_name << " (" << rows << "x" << cols
            << ") ===" << std::endl;

  Matrix A(rows, cols);
  generate_matrix(A, type);

  // --- CUSTOM SVD ---
  Matrix U_custom(rows, cols);
  Matrix V_custom(cols, cols);
  std::vector<double> S_custom;

  auto start_c = std::chrono::high_resolution_clock::now();
  int sweeps = svd_jacobi_one_sided(A, U_custom, S_custom, V_custom);
  auto end_c = std::chrono::high_resolution_clock::now();

  // --- TIMING ---
  std::chrono::duration<double> diff_c = end_c - start_c;

  std::cout << "Time Custom: " << std::fixed << std::setprecision(5)
            << diff_c.count() << "s (Sweeps: " << sweeps << ")" << std::endl;
}

int main(int argc, char **argv) {
  int num_threads = 1;
  int N = 500;

  if (argc > 1)
    N = std::atoi(argv[1]);
  if (argc > 2)
    num_threads = std::atoi(argv[2]);
  openblas_set_num_threads(num_threads);

  std::cout << "Starting HYPER-OPTIMIZED Sequential SVD Benchmark" << std::endl;
  std::cout << "Optimizations: Tiled (Blocked) Jacobi, Aligned Malloc, AVX2 "
               "FMA Vectorization, Rank-Guard"
            << std::endl;
  std::cout << "OpenBLAS Thread Count: " << openblas_get_num_threads()
            << " threads active." << std::endl;
  std::cout << "---------------------------------------------------------------"
               "--------------------"
            << std::endl;

  // --- Heavy Performance Tests ---
  std::cout << "\n[PERFORMANCE TESTS]" << std::endl;
  run_heavy_test(1500, 1500, RANDOM, "Large Cache Breaker");

  return 0;
}