// =========================================================================
// SVD OPTIMIZED V6 - AVX2 (Frequency Friendly) + Cache Fixes
// Target: High Frequency (2.8GHz+) & Reduced Cache Thrashing
// =========================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h> // AVX2, FMA
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

// _mm_malloc header availability
#ifdef _MSC_VER
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

// Define Restrict
#if defined(_WIN32) || defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

// -------------------------------------------------------------------------
// MEMORY MANAGEMENT (Aligned 64-byte)
// -------------------------------------------------------------------------
struct Matrix {
  int rows, cols;
  int stride;
  double *data;

  Matrix(int r, int c) : rows(r), cols(c) {
    stride = (r + 7) & ~7; // Align columns to 64-byte boundary
    size_t total_size = (size_t)stride * cols * sizeof(double);
    data = (double *)_mm_malloc(total_size, 64); // 64-byte alignment
    if (!data)
      exit(1);
    std::memset(data, 0, total_size);
  }

  // Copy Constructor
  Matrix(const Matrix &o) : rows(o.rows), cols(o.cols), stride(o.stride) {
    size_t total_size = (size_t)stride * cols * sizeof(double);
    data = (double *)_mm_malloc(total_size, 64);
    if (data)
      std::memcpy(data, o.data, total_size);
  }

  // Assignment Operator
  Matrix &operator=(const Matrix &o) {
    if (this != &o) {
      if (data)
        _mm_free(data);
      rows = o.rows;
      cols = o.cols;
      stride = o.stride;
      size_t total_size = (size_t)stride * cols * sizeof(double);
      data = (double *)_mm_malloc(total_size, 64);
      if (data)
        std::memcpy(data, o.data, total_size);
    }
    return *this;
  }

  ~Matrix() {
    if (data)
      _mm_free(data);
  }

  inline double *col_ptr(int c) { return data + (size_t)c * stride; }
  inline const double *col_ptr(int c) const {
    return data + (size_t)c * stride;
  }
  inline double &operator()(int r, int c) {
    return data[r + (size_t)c * stride];
  }
};

// Larger block size usually benefits AVX2 to amortize overhead
const int BLOCK_SIZE = 64;

// -------------------------------------------------------------------------
// HELPER: Horizontal Sum (AVX2)
// -------------------------------------------------------------------------
static inline double hsum_pd_256(__m256d v) {
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1);
  vlow = _mm_add_pd(vlow, vhigh);
  vlow = _mm_hadd_pd(vlow, vlow);
  return _mm_cvtsd_f64(vlow);
}

// -------------------------------------------------------------------------
// CORE KERNEL: AVX2 (256-bit) + PREFETCHING
// -------------------------------------------------------------------------
int process_block_pair(int bi, int bj, int m, int n, Matrix &U, Matrix &V,
                       double *col_norms_sq, double tol_sq, double small_val,
                       bool force_rotate) {
  int changed = 0;
  const int bi_end = std::min(bi + BLOCK_SIZE, n);
  const int bj_end = std::min(bj + BLOCK_SIZE, n);

  for (int i = bi; i < bi_end; ++i) {
    double a = col_norms_sq[i];
    double *RESTRICT p_ui = U.col_ptr(i);
    double *RESTRICT p_vi = V.col_ptr(i);

    const int j_start = (bi == bj) ? (i + 1) : bj;

    for (int j = j_start; j < bj_end; ++j) {
      double b = col_norms_sq[j];
      double *RESTRICT p_uj = U.col_ptr(j);
      double *RESTRICT p_vj = V.col_ptr(j);

      // --- DOT PRODUCT (AVX2) ---
      __m256d sum0 = _mm256_setzero_pd();
      __m256d sum1 = _mm256_setzero_pd();
      __m256d sum2 = _mm256_setzero_pd();
      __m256d sum3 = _mm256_setzero_pd();

      int k = 0;
      const int m16 = m & ~15;

      for (; k < m16; k += 16) {
        // PREFETCH: Look ahead 2 cache lines (128 bytes)
        _mm_prefetch((const char *)(p_ui + k + 32), _MM_HINT_T0);
        _mm_prefetch((const char *)(p_uj + k + 32), _MM_HINT_T0);

        __m256d u0 = _mm256_load_pd(p_ui + k);
        __m256d u1 = _mm256_load_pd(p_ui + k + 4);
        __m256d u2 = _mm256_load_pd(p_ui + k + 8);
        __m256d u3 = _mm256_load_pd(p_ui + k + 12);

        __m256d v0 = _mm256_load_pd(p_uj + k);
        __m256d v1 = _mm256_load_pd(p_uj + k + 4);
        __m256d v2 = _mm256_load_pd(p_uj + k + 8);
        __m256d v3 = _mm256_load_pd(p_uj + k + 12);

        sum0 = _mm256_fmadd_pd(u0, v0, sum0);
        sum1 = _mm256_fmadd_pd(u1, v1, sum1);
        sum2 = _mm256_fmadd_pd(u2, v2, sum2);
        sum3 = _mm256_fmadd_pd(u3, v3, sum3);
      }

      // Reduction
      sum0 = _mm256_add_pd(sum0, sum1);
      sum2 = _mm256_add_pd(sum2, sum3);
      sum0 = _mm256_add_pd(sum0, sum2);
      double g = hsum_pd_256(sum0);

      // Scalar cleanup
      for (; k < m; ++k)
        g += p_ui[k] * p_uj[k];

      if (!force_rotate && g * g < tol_sq * a * b)
        continue;

      changed++;

      // --- ROTATION PARAMETERS ---
      double zeta = (b - a) / (2.0 * g);
      double t = std::copysign(
          1.0 / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta)), zeta);
      double c = 1.0 / std::sqrt(1.0 + t * t);
      double s = c * t;

      __m256d v_c = _mm256_set1_pd(c);
      __m256d v_s = _mm256_set1_pd(s);
      __m256d v_ns = _mm256_set1_pd(-s);

      // --- APPLY ROTATION U (AVX2) ---
      k = 0;
      for (; k < m16; k += 16) {
        // PREFETCH for writes
        _mm_prefetch((const char *)(p_ui + k + 32), _MM_HINT_T0);
        _mm_prefetch((const char *)(p_uj + k + 32), _MM_HINT_T0);

        __m256d u0 = _mm256_load_pd(p_ui + k);
        __m256d uj0 = _mm256_load_pd(p_uj + k);
        __m256d u1 = _mm256_load_pd(p_ui + k + 4);
        __m256d uj1 = _mm256_load_pd(p_uj + k + 4);
        __m256d u2 = _mm256_load_pd(p_ui + k + 8);
        __m256d uj2 = _mm256_load_pd(p_uj + k + 8);
        __m256d u3 = _mm256_load_pd(p_ui + k + 12);
        __m256d uj3 = _mm256_load_pd(p_uj + k + 12);

        _mm256_store_pd(p_ui + k,
                        _mm256_fmadd_pd(v_c, u0, _mm256_mul_pd(v_ns, uj0)));
        _mm256_store_pd(p_ui + k + 4,
                        _mm256_fmadd_pd(v_c, u1, _mm256_mul_pd(v_ns, uj1)));
        _mm256_store_pd(p_ui + k + 8,
                        _mm256_fmadd_pd(v_c, u2, _mm256_mul_pd(v_ns, uj2)));
        _mm256_store_pd(p_ui + k + 12,
                        _mm256_fmadd_pd(v_c, u3, _mm256_mul_pd(v_ns, uj3)));

        _mm256_store_pd(p_uj + k,
                        _mm256_fmadd_pd(v_s, u0, _mm256_mul_pd(v_c, uj0)));
        _mm256_store_pd(p_uj + k + 4,
                        _mm256_fmadd_pd(v_s, u1, _mm256_mul_pd(v_c, uj1)));
        _mm256_store_pd(p_uj + k + 8,
                        _mm256_fmadd_pd(v_s, u2, _mm256_mul_pd(v_c, uj2)));
        _mm256_store_pd(p_uj + k + 12,
                        _mm256_fmadd_pd(v_s, u3, _mm256_mul_pd(v_c, uj3)));
      }
      for (; k < m; ++k) {
        double tmp = p_ui[k];
        p_ui[k] = c * tmp - s * p_uj[k];
        p_uj[k] = s * tmp + c * p_uj[k];
      }

      // --- APPLY ROTATION V (AVX2) ---
      const int n16 = n & ~15;
      k = 0;
      for (; k < n16; k += 16) {
        // Prefetch V
        _mm_prefetch((const char *)(p_vi + k + 32), _MM_HINT_T0);
        _mm_prefetch((const char *)(p_vj + k + 32), _MM_HINT_T0);

        __m256d v0 = _mm256_load_pd(p_vi + k);
        __m256d vj0 = _mm256_load_pd(p_vj + k);
        __m256d v1 = _mm256_load_pd(p_vi + k + 4);
        __m256d vj1 = _mm256_load_pd(p_vj + k + 4);
        __m256d v2 = _mm256_load_pd(p_vi + k + 8);
        __m256d vj2 = _mm256_load_pd(p_vj + k + 8);
        __m256d v3 = _mm256_load_pd(p_vi + k + 12);
        __m256d vj3 = _mm256_load_pd(p_vj + k + 12);

        _mm256_store_pd(p_vi + k,
                        _mm256_fmadd_pd(v_c, v0, _mm256_mul_pd(v_ns, vj0)));
        _mm256_store_pd(p_vi + k + 4,
                        _mm256_fmadd_pd(v_c, v1, _mm256_mul_pd(v_ns, vj1)));
        _mm256_store_pd(p_vi + k + 8,
                        _mm256_fmadd_pd(v_c, v2, _mm256_mul_pd(v_ns, vj2)));
        _mm256_store_pd(p_vi + k + 12,
                        _mm256_fmadd_pd(v_c, v3, _mm256_mul_pd(v_ns, vj3)));

        _mm256_store_pd(p_vj + k,
                        _mm256_fmadd_pd(v_s, v0, _mm256_mul_pd(v_c, vj0)));
        _mm256_store_pd(p_vj + k + 4,
                        _mm256_fmadd_pd(v_s, v1, _mm256_mul_pd(v_c, vj1)));
        _mm256_store_pd(p_vj + k + 8,
                        _mm256_fmadd_pd(v_s, v2, _mm256_mul_pd(v_c, vj2)));
        _mm256_store_pd(p_vj + k + 12,
                        _mm256_fmadd_pd(v_s, v3, _mm256_mul_pd(v_c, vj3)));
      }
      for (; k < n; ++k) {
        double tmp = p_vi[k];
        p_vi[k] = c * tmp - s * p_vj[k];
        p_vj[k] = s * tmp + c * p_vj[k];
      }

      // Update Norms...
      double c_sq = c * c;
      double s_sq = s * s;
      double two_cs_g = 2.0 * c * s * g;
      double new_a = c_sq * col_norms_sq[i] + s_sq * col_norms_sq[j] - two_cs_g;
      double new_b = s_sq * col_norms_sq[i] + c_sq * col_norms_sq[j] + two_cs_g;

      col_norms_sq[i] = (new_a < small_val) ? 0.0 : new_a;
      col_norms_sq[j] = (new_b < small_val) ? 0.0 : new_b;
      a = col_norms_sq[i];
    }
  }
  return changed;
}

// -------------------------------------------------------------------------
// PARALLEL SVD
// -------------------------------------------------------------------------
int svd_jacobi_one_sided(const Matrix &input, Matrix &U, std::vector<double> &S,
                         Matrix &V) {
  const int m = input.rows;
  const int n = input.cols;

  U = input;
  V = Matrix(n, n);
  for (int i = 0; i < n; ++i)
    V(i, i) = 1.0;

  const double tol_sq = 1e-30;
  const double small_val = 1e-32;
  const int max_sweeps = 100;

  // Setup Teams
  const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int n_teams = num_blocks + (num_blocks % 2);
  std::vector<int> teams(n_teams, -1);
  for (int i = 0; i < num_blocks; ++i)
    teams[i] = i * BLOCK_SIZE;

  std::vector<double> col_norms_sq(n);
  int sweep = 0;
  bool finished = false;

// Initial Norms
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const double *RESTRICT ptr = U.col_ptr(i);
    __m256d s0 = _mm256_setzero_pd();
    __m256d s1 = _mm256_setzero_pd();
    int k = 0;
    const int m8 = m & ~7;
    for (; k < m8; k += 8) {
      __m256d v0 = _mm256_load_pd(ptr + k);
      __m256d v1 = _mm256_load_pd(ptr + k + 4);
      s0 = _mm256_fmadd_pd(v0, v0, s0);
      s1 = _mm256_fmadd_pd(v1, v1, s1);
    }
    s0 = _mm256_add_pd(s0, s1);
    double sum = hsum_pd_256(s0);
    for (; k < m; ++k)
      sum += ptr[k] * ptr[k];
    col_norms_sq[i] = sum;
  }

// MAIN PARALLEL LOOP
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

    // Static Scheduling Ranges
    int pair_cnt = n_teams / 2;
    int pair_chunk = (pair_cnt + num_threads - 1) / num_threads;
    int pair_start = std::min(tid * pair_chunk, pair_cnt);
    int pair_end = std::min(pair_start + pair_chunk, pair_cnt);

    int diag_chunk = (num_blocks + num_threads - 1) / num_threads;
    int diag_start = std::min(tid * diag_chunk, num_blocks);
    int diag_end = std::min(diag_start + diag_chunk, num_blocks);

    int local_changes = 0;

    while (!finished && sweep < max_sweeps) {
      local_changes = 0;
      bool force_rotate = (sweep < 3);

      // 1. Diagonal Blocks
      for (int b = diag_start; b < diag_end; ++b) {
        int bi = b * BLOCK_SIZE;
        local_changes +=
            process_block_pair(bi, bi, m, n, U, V, col_norms_sq.data(), tol_sq,
                               small_val, force_rotate);
      }
#pragma omp barrier

      // 2. Off-Diagonal Blocks (Tournament)
      for (int round = 0; round < n_teams - 1; ++round) {
        for (int k = pair_start; k < pair_end; ++k) {
          int b1 = teams[k];
          int b2 = teams[n_teams - 1 - k];
          if (b1 != -1 && b2 != -1) {
            int r1 = std::min(b1, b2);
            int r2 = std::max(b1, b2);
            local_changes +=
                process_block_pair(r1, r2, m, n, U, V, col_norms_sq.data(),
                                   tol_sq, small_val, force_rotate);
          }
        }
#pragma omp barrier
#pragma omp single
        {
          int t = teams[n_teams - 1];
          for (int k = n_teams - 1; k > 1; --k)
            teams[k] = teams[k - 1];
          teams[1] = t;
        }
      }

      // Convergence Check
      static int total_changes;
#pragma omp atomic
      total_changes += local_changes;
#pragma omp barrier
#pragma omp single
      {
        if (total_changes == 0)
          finished = true;
        if (!finished) {
          sweep++;
          total_changes = 0;
        }
      }
#pragma omp barrier

      if (finished)
        break;

      // 3. Recalculate Norms (Every 4 sweeps)
      if (sweep % 4 == 0) {
        int n_chunk = (n + num_threads - 1) / num_threads;
        int n_s = std::min(tid * n_chunk, n);
        int n_e = std::min(n_s + n_chunk, n);
        for (int i = n_s; i < n_e; ++i) {
          const double *RESTRICT ptr = U.col_ptr(i);
          __m256d s0 = _mm256_setzero_pd();
          int k = 0;
          for (; k < (m & ~3); k += 4) {
            __m256d v0 = _mm256_load_pd(ptr + k);
            s0 = _mm256_fmadd_pd(v0, v0, s0);
          }
          double sum = hsum_pd_256(s0);
          for (; k < m; ++k)
            sum += ptr[k] * ptr[k];
          col_norms_sq[i] = sum;
        }
#pragma omp barrier
      }
    }
  }

  // Extract S
  S.resize(n);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    double *RESTRICT ptr = U.col_ptr(i);

    // 1. Calculate Square Norm of the column
    __m256d s0 = _mm256_setzero_pd();
    int k = 0;
    for (; k < (m & ~3); k += 4) {
      __m256d v0 = _mm256_load_pd(ptr + k);
      s0 = _mm256_fmadd_pd(v0, v0, s0);
    }
    double sum = hsum_pd_256(s0);
    for (; k < m; ++k)
      sum += ptr[k] * ptr[k];

    // 2. Store Singular Value
    double sigma = std::sqrt(sum);
    S[i] = sigma;

    // 3. Normalize U (Critical for correctness)
    // U = U / sigma
    if (sigma > 1e-32) {
      double inv_sig = 1.0 / sigma;
      __m256d v_inv = _mm256_set1_pd(inv_sig);

      k = 0;
      for (; k < (m & ~3); k += 4) {
        __m256d v0 = _mm256_load_pd(ptr + k);
        _mm256_store_pd(ptr + k, _mm256_mul_pd(v0, v_inv));
      }
      for (; k < m; ++k) {
        ptr[k] *= inv_sig;
      }
    } else {
      // Handle zero singular value (prevent NaN)
      k = 0;
      for (; k < m; ++k)
        ptr[k] = 0.0;
    }
  }
  return sweep;
}

// -------------------------------------------------------------------------
// CHECK: Reconstruction Error || A - U*S*Vt ||
// -------------------------------------------------------------------------
double check_reconstruction_error(Matrix &A, Matrix &U, std::vector<double> &S,
                                  Matrix &V) {
  double max_err = 0.0;
  long double sum_sq_err = 0.0;

// We compute A_reconstructed[i][j] = Sum_k ( U[i][k] * S[k] * V[j][k] )
// Note: V is stored as V (not V^T), so we access V.col_ptr(k)[j] typically.
// However, our Matrix structure is column-major.
// U.col_ptr(k)[i] is U[i][k]
// V.col_ptr(k)[j] is V[j][k]

// This check is expensive (O(N^3)), so we parallelize it.
#pragma omp parallel for reduction(+ : sum_sq_err) schedule(static)
  for (int c = 0; c < A.cols; ++c) {
    for (int r = 0; r < A.rows; ++r) {
      double recon_val = 0.0;
      for (int k = 0; k < A.cols; ++k) {
        // U(r, k) * S[k] * V(c, k)  <-- V is transposed in formula so V_ik ->
        // V_ki
        recon_val += U(r, k) * S[k] * V(c, k);
      }
      double diff = A(r, c) - recon_val;
      sum_sq_err += diff * diff;
    }
  }

  return std::sqrt((double)sum_sq_err);
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------
void generate_matrix(Matrix &M) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (int c = 0; c < M.cols; ++c)
    for (int r = 0; r < M.rows; ++r)
      M(r, c) = dis(gen);
}

int main(int argc, char **argv) {
  // CRITICAL: Limit threads to physical cores to reduce cache thrashing
  omp_set_num_threads(4);

  int N = 1500;
  if (argc > 1)
    N = std::atoi(argv[1]);

  std::cout << "=== SVD OPTIMIZED V6 (AVX2 + Thread Limit) ===" << std::endl;
  std::cout << "Testing N=" << N << " | Threads=8 " << std::endl;

  Matrix A(N, N);
  generate_matrix(A);

  Matrix U(N, N), V(N, N);
  std::vector<double> S;

  auto start = std::chrono::high_resolution_clock::now();
  int sweeps = svd_jacobi_one_sided(A, U, S, V);
  double time = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();

  double gflops = 5.0 * N * N * N * sweeps / time / 1e9;
  std::cout << "Time: " << std::fixed << std::setprecision(3) << time << "s"
            << " | Sweeps: " << sweeps << " | GFLOPS: " << gflops << std::endl;

  // Verify Correctness

  return 0;
}