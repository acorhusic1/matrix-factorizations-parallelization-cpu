#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h> // REQUIRED: OpenMP
#include <random>
#include <vector>


#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#define RESTRICT __restrict
#else
#include <stdlib.h>
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

// -------------------------------------------------------------------------
// ALIGNED MEMORY UTILS
// -------------------------------------------------------------------------
void *allocate_aligned(size_t size) {
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
// Matrix Class
// -------------------------------------------------------------------------
struct Matrix {
  int rows;
  int cols;
  double *data;

  Matrix(int r, int c) : rows(r), cols(c) {
    size_t bytes = (size_t)r * c * sizeof(double);
    data = (double *)allocate_aligned(bytes);
    if (!data) {
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
// CONSTANTS
// -------------------------------------------------------------------------
const int BLOCK_SIZE = 32; // Matches L1/L2 cache well

// -------------------------------------------------------------------------
// CORE KERNEL: PROCESS BLOCK PAIR
// -------------------------------------------------------------------------
// Processes interactions between Block I and Block J.
// If bi == bj, it processes the internal diagonal block.
// Returns the number of rotations performed.
int process_block_pair(int bi, int bj, int m, int n, Matrix &U, Matrix &V,
                       std::vector<double> &col_norms_sq, double tol,
                       double small_val) {
  int changed = 0;
  int bi_end = std::min(bi + BLOCK_SIZE, n);
  int bj_end = std::min(bj + BLOCK_SIZE, n);

  for (int i = bi; i < bi_end; ++i) {
    double a = col_norms_sq[i];
    if (a < small_val)
      continue;

    double *RESTRICT p_ui = U.col_ptr(i);
    double *RESTRICT p_vi = V.col_ptr(i);

    // If same block, start j at i+1. If different blocks, start at bj.
    int j_start = (bi == bj) ? (i + 1) : bj;

    for (int j = j_start; j < bj_end; ++j) {
      double b = col_norms_sq[j];
      if (b < small_val)
        continue;

      double *RESTRICT p_uj = U.col_ptr(j);
      double *RESTRICT p_vj = V.col_ptr(j);

      // --- DOT PRODUCT (Unrolled 8x with 4 Accumulators) ---
      double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0;
      int k = 0;
      for (; k <= m - 8; k += 8) {
        g0 += p_ui[k] * p_uj[k];
        g1 += p_ui[k + 1] * p_uj[k + 1];
        g2 += p_ui[k + 2] * p_uj[k + 2];
        g3 += p_ui[k + 3] * p_uj[k + 3];

        g0 += p_ui[k + 4] * p_uj[k + 4];
        g1 += p_ui[k + 5] * p_uj[k + 5];
        g2 += p_ui[k + 6] * p_uj[k + 6];
        g3 += p_ui[k + 7] * p_uj[k + 7];
      }
      double g = (g0 + g1) + (g2 + g3);
      for (; k < m; ++k)
        g += p_ui[k] * p_uj[k];

      // --- CHECK ORTHOGONALITY ---
      if (g * g < (tol * tol) * a * b)
        continue;

      changed++;

      // --- ROTATION PARAMS ---
      double zeta = (b - a) / (2.0 * g);
      double t = std::copysign(
          1.0 / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta)), zeta);
      double c = 1.0 / std::sqrt(1.0 + t * t);
      double s = c * t;

      // --- APPLY ROTATION U (Unrolled 8x) ---
      k = 0;
      for (; k <= m - 8; k += 8) {
        double u0 = p_ui[k];
        double v0 = p_uj[k];
        double u1 = p_ui[k + 1];
        double v1 = p_uj[k + 1];
        double u2 = p_ui[k + 2];
        double v2 = p_uj[k + 2];
        double u3 = p_ui[k + 3];
        double v3 = p_uj[k + 3];
        double u4 = p_ui[k + 4];
        double v4 = p_uj[k + 4];
        double u5 = p_ui[k + 5];
        double v5 = p_uj[k + 5];
        double u6 = p_ui[k + 6];
        double v6 = p_uj[k + 6];
        double u7 = p_ui[k + 7];
        double v7 = p_uj[k + 7];

        p_ui[k] = c * u0 - s * v0;
        p_uj[k] = s * u0 + c * v0;
        p_ui[k + 1] = c * u1 - s * v1;
        p_uj[k + 1] = s * u1 + c * v1;
        p_ui[k + 2] = c * u2 - s * v2;
        p_uj[k + 2] = s * u2 + c * v2;
        p_ui[k + 3] = c * u3 - s * v3;
        p_uj[k + 3] = s * u3 + c * v3;
        p_ui[k + 4] = c * u4 - s * v4;
        p_uj[k + 4] = s * u4 + c * v4;
        p_ui[k + 5] = c * u5 - s * v5;
        p_uj[k + 5] = s * u5 + c * v5;
        p_ui[k + 6] = c * u6 - s * v6;
        p_uj[k + 6] = s * u6 + c * v6;
        p_ui[k + 7] = c * u7 - s * v7;
        p_uj[k + 7] = s * u7 + c * v7;
      }
      for (; k < m; ++k) {
        double tmp = p_ui[k];
        p_ui[k] = c * tmp - s * p_uj[k];
        p_uj[k] = s * tmp + c * p_uj[k];
      }

      // --- APPLY ROTATION V (Unrolled 4x) ---
      k = 0;
      for (; k <= n - 4; k += 4) {
        double v_i0 = p_vi[k];
        double v_j0 = p_vj[k];
        double v_i1 = p_vi[k + 1];
        double v_j1 = p_vj[k + 1];
        double v_i2 = p_vi[k + 2];
        double v_j2 = p_vj[k + 2];
        double v_i3 = p_vi[k + 3];
        double v_j3 = p_vj[k + 3];

        p_vi[k] = c * v_i0 - s * v_j0;
        p_vj[k] = s * v_i0 + c * v_j0;
        p_vi[k + 1] = c * v_i1 - s * v_j1;
        p_vj[k + 1] = s * v_i1 + c * v_j1;
        p_vi[k + 2] = c * v_i2 - s * v_j2;
        p_vj[k + 2] = s * v_i2 + c * v_j2;
        p_vi[k + 3] = c * v_i3 - s * v_j3;
        p_vj[k + 3] = s * v_i3 + c * v_j3;
      }
      for (; k < n; ++k) {
        double tmp = p_vi[k];
        p_vi[k] = c * tmp - s * p_vj[k];
        p_vj[k] = s * tmp + c * p_vj[k];
      }

      // --- UPDATE NORMS ---
      double c2 = c * c;
      double s2 = s * s;
      double two_cs_g = 2.0 * c * s * g;

      double new_a = c2 * col_norms_sq[i] + s2 * col_norms_sq[j] - two_cs_g;
      double new_b = s2 * col_norms_sq[i] + c2 * col_norms_sq[j] + two_cs_g;

      if (new_a < small_val)
        new_a = 0.0;
      if (new_b < small_val)
        new_b = 0.0;

      col_norms_sq[i] = new_a;
      col_norms_sq[j] = new_b;
      a = new_a;
    }
  }
  return changed;
}

// -------------------------------------------------------------------------
// PARALLEL BLOCKED SVD (Round Robin / Chess Tournament)
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
  const double small_val = 1e-32;

  // Cache Norms (Parallel)
  std::vector<double> col_norms_sq(n);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    const double *RESTRICT ptr = U.col_ptr(i);
    for (int k = 0; k < m; ++k)
      sum += ptr[k] * ptr[k];
    col_norms_sq[i] = sum;
  }

  // Determine blocks
  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Round Robin Scheduler setup
  // We treat block indices as teams in a tournament.
  // If num_blocks is odd, we add a dummy block (handled by index check)
  int n_teams = num_blocks;
  if (n_teams % 2 != 0)
    n_teams++;

  std::vector<int> teams(n_teams);
  for (int i = 0; i < num_blocks; ++i)
    teams[i] = i * BLOCK_SIZE;
  for (int i = num_blocks; i < n_teams; ++i)
    teams[i] = -1; // Dummy

  int sweep = 0;
  for (; sweep < max_sweeps; ++sweep) {
    int changed_cnt = 0;

// PHASE 1: Intra-Block Parallelism
// Each thread cleans up the columns inside one block
#pragma omp parallel for reduction(+ : changed_cnt) schedule(dynamic)
    for (int b = 0; b < num_blocks; ++b) {
      int bi = b * BLOCK_SIZE;
      changed_cnt +=
          process_block_pair(bi, bi, m, n, U, V, col_norms_sq, tol, small_val);
    }

    // PHASE 2: Inter-Block Parallelism (Tournament)
    // We do (n_teams - 1) rounds to ensure every block plays every other block
    for (int round = 0; round < n_teams - 1; ++round) {

// Pair teams[0] with teams[round_idx], etc.
// Standard Round Robin: Fixed first element, rotate the rest.

// In parallel, process the N/2 pairings for this round
#pragma omp parallel for reduction(+ : changed_cnt) schedule(dynamic)
      for (int k = 0; k < n_teams / 2; ++k) {
        int idx1 = k;
        int idx2 = n_teams - 1 - k;

        int b1_start = teams[idx1];
        int b2_start = teams[idx2];

        // If valid pair (neither is dummy), process interaction
        if (b1_start != -1 && b2_start != -1) {
          // Ensure consistent ordering to avoid logic bugs, though technically
          // symmetric process_block_pair handles (bi, bj) generally. We must
          // treat them as off-diagonal blocks.
          changed_cnt += process_block_pair(std::min(b1_start, b2_start),
                                            std::max(b1_start, b2_start), m, n,
                                            U, V, col_norms_sq, tol, small_val);
        }
      }

      // Rotate teams for next round: Keep teams[0], rotate
      // teams[1]...teams[end] std::rotate(begin, new_begin, end)
      if (n_teams > 2) {
        int temp = teams[n_teams - 1];
        for (int k = n_teams - 1; k > 1; --k) {
          teams[k] = teams[k - 1];
        }
        teams[1] = temp;
      }
    }

    if (changed_cnt == 0)
      break;

    // Refresh Norms (Periodic)
    if (sweep % 4 == 0) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        const double *RESTRICT ptr = U.col_ptr(i);
        int k = 0;
        for (; k <= m - 8; k += 8) { // Reuse unrolled logic
          sum += ptr[k] * ptr[k] + ptr[k + 1] * ptr[k + 1] +
                 ptr[k + 2] * ptr[k + 2] + ptr[k + 3] * ptr[k + 3] +
                 ptr[k + 4] * ptr[k + 4] + ptr[k + 5] * ptr[k + 5] +
                 ptr[k + 6] * ptr[k + 6] + ptr[k + 7] * ptr[k + 7];
        }
        for (; k < m; ++k)
          sum += ptr[k] * ptr[k];
        col_norms_sq[i] = sum;
      }
    }
  }

  // Extract S and Normalize U (Parallel)
  S.resize(n);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    double *RESTRICT ptr = U.col_ptr(i);
    double sum = 0.0;
    int k = 0;
    for (; k <= m - 8; k += 8) {
      sum += ptr[k] * ptr[k] + ptr[k + 1] * ptr[k + 1] +
             ptr[k + 2] * ptr[k + 2] + ptr[k + 3] * ptr[k + 3] +
             ptr[k + 4] * ptr[k + 4] + ptr[k + 5] * ptr[k + 5] +
             ptr[k + 6] * ptr[k + 6] + ptr[k + 7] * ptr[k + 7];
    }
    for (; k < m; ++k)
      sum += ptr[k] * ptr[k];

    double norm = std::sqrt(sum);
    S[i] = norm;

    if (norm > 1e-20) {
      double inv_norm = 1.0 / norm;
      for (int x = 0; x < m; ++x)
        ptr[x] *= inv_norm;
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

// Reconstruct A(r,c) individually - Parallelize check
#pragma omp parallel for reduction(+ : error_sum) schedule(static)
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
  char jobu = 'S', jobvt = 'S'; // 'S' for min(M,N) cols of U, 'A' for full VT
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

  // Parallelize check? Tricky with max_error reduction, stick to serial for
  // correctness check
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
  int N = 500;
  if (argc > 1)
    N = std::atoi(argv[1]);

  std::cout << "Starting HYPER-OPTIMIZED Sequential SVD Benchmark" << std::endl;
  std::cout << "Optimizations: OpenMP Parallel (Round-Robin Blocks), Aligned "
               "Malloc, Unrolling + Multiple Accumulators, Rank-Guard"
            << std::endl;
  std::cout << "---------------------------------------------------------------"
               "--------------------"
            << std::endl;

  // --- Heavy Performance Tests ---
  std::cout << "\n[PERFORMANCE TESTS]" << std::endl;
  run_heavy_test(1000, 1000, RANDOM, "Large Cache Breaker");

  return 0;
}