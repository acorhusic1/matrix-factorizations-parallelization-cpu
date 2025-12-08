#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h> // AVX2
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

// --- TUNING CONSTANTS ---
#define CACHE_LINE_SIZE 64

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

class Matrix {
public:
  int rows, cols, stride;
  double *data;

  Matrix(int r, int c) : rows(r), cols(c) {
    // Avoid power-of-2 strides to prevent cache set conflicts.
    int padded_rows = (r + 7) & ~7;
    stride = padded_rows + 8;

    size_t bytes = (size_t)stride * c * sizeof(double);
    data = (double *)ALIGNED_MALLOC(bytes, 64);

// Touch memory to fault pages in (First Touch Policy)
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < bytes / sizeof(double); i += 4096) {
      data[i] = 0.0;
    }
    std::memset(data, 0, bytes);
  }

  Matrix(const Matrix &o) : rows(o.rows), cols(o.cols), stride(o.stride) {
    size_t bytes = (size_t)stride * cols * sizeof(double);
    data = (double *)ALIGNED_MALLOC(bytes, 64);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < cols; ++c) {
      const double *src = o.col_ptr(c);
      double *dst = col_ptr(c);
      std::memcpy(dst, src, stride * sizeof(double));
    }
  }

  Matrix &operator=(const Matrix &o) {
    if (this != &o) {
      if (data)
        ALIGNED_FREE(data);
      rows = o.rows;
      cols = o.cols;
      stride = o.stride;
      size_t bytes = (size_t)stride * cols * sizeof(double);
      data = (double *)ALIGNED_MALLOC(bytes, 64);
#pragma omp parallel for schedule(static)
      for (int c = 0; c < cols; ++c) {
        const double *src = o.col_ptr(c);
        double *dst = col_ptr(c);
        std::memcpy(dst, src, stride * sizeof(double));
      }
    }
    return *this;
  }

  ~Matrix() {
    if (data)
      ALIGNED_FREE(data);
  }

  static Matrix identity(int n) {
    Matrix res(n, n);
    for (int i = 0; i < n; i++)
      res.data[i * res.stride + i] = 1.0;
    return res;
  }

  inline double *col_ptr(int c) { return data + c * stride; }
  inline const double *col_ptr(int c) const { return data + c * stride; }

  void randomize(double lo, double hi) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(lo, hi);
    for (int c = 0; c < cols; ++c)
      for (int r = 0; r < rows; ++r)
        data[c * stride + r] = dis(gen);
  }
};

namespace OptSVD {

inline double hsum_avx2(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  lo = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}

inline double sign(double x) { return x >= 0.0 ? 1.0 : -1.0; }

struct GivensRotation {
  int k;
  double c, s;
};

// --- HOUSEHOLDER GEN ---
double householderVector(double *RESTRICT x, int n, double &alpha_out) {
  if (n <= 0) {
    alpha_out = 0;
    return 0;
  }

  __m256d vsum = _mm256_setzero_pd();
  int i = 1;
  for (; i <= n - 4; i += 4) {
    __m256d v = _mm256_loadu_pd(x + i);
    vsum = _mm256_fmadd_pd(v, v, vsum);
  }
  double tail_sq = hsum_avx2(vsum);
  for (; i < n; ++i)
    tail_sq += x[i] * x[i];

  double x0 = x[0];
  double norm = std::sqrt(x0 * x0 + tail_sq);
  if (norm <= 1e-13) {
    alpha_out = x0;
    return 0;
  }

  double alpha = (x0 >= 0) ? -norm : norm;
  double v0 = x0 - alpha;
  if (std::abs(v0) < 1e-13) {
    alpha_out = x0;
    return 0;
  }

  double inv = 1.0 / v0;
  __m256d vinv = _mm256_set1_pd(inv);
  i = 1;
  for (; i <= n - 4; i += 4) {
    _mm256_storeu_pd(x + i, _mm256_mul_pd(_mm256_loadu_pd(x + i), vinv));
  }
  for (; i < n; ++i)
    x[i] *= inv;

  // Quick Recompute norm sq for stability
  vsum = _mm256_setzero_pd();
  i = 1;
  for (; i <= n - 4; i += 4)
    vsum =
        _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(x + i), vsum);
  double v_sq = 1.0 + hsum_avx2(vsum);
  for (; i < n; ++i)
    v_sq += x[i] * x[i];

  alpha_out = alpha;
  return 2.0 / v_sq;
}

// --- LEFT UPDATE (CACHE OPTIMIZED) ---
// REPLACED: Streaming stores -> Standard stores (to use L2/L3 cache)
void applyHouseholderLeft(Matrix &A, int rs, int cs, const double *v, int vlen,
                          double beta) {
  if (beta == 0)
    return;
  int wlen = A.cols - cs;

#pragma omp parallel for schedule(static) proc_bind(close)
  for (int j = 0; j < wlen; ++j) {
    double *Ac = A.col_ptr(cs + j) + rs;

    // Dot Product
    __m256d vsum = _mm256_setzero_pd();
    double sum = Ac[0];
    int i = 1;
    for (; i <= vlen - 4; i += 4) {
      vsum = _mm256_fmadd_pd(_mm256_loadu_pd(Ac + i), _mm256_loadu_pd(v + i),
                             vsum);
    }
    sum += hsum_avx2(vsum);
    for (; i < vlen; ++i)
      sum += Ac[i] * v[i];

    // Update
    double val = beta * sum;
    Ac[0] -= val;
    __m256d vval = _mm256_set1_pd(val);

    i = 1;
    bool aligned = ((uintptr_t)(Ac + 1) % 32) == 0;

    // CRITICAL FIX: Use store_pd instead of stream_pd
    if (aligned) {
      for (; i <= vlen - 4; i += 4) {
        __m256d va = _mm256_load_pd(Ac + i);
        __m256d vv = _mm256_loadu_pd(v + i);
        _mm256_store_pd(Ac + i, _mm256_fnmadd_pd(vv, vval, va));
      }
    } else {
      for (; i <= vlen - 4; i += 4) {
        __m256d va = _mm256_loadu_pd(Ac + i);
        __m256d vv = _mm256_loadu_pd(v + i);
        _mm256_storeu_pd(Ac + i, _mm256_fnmadd_pd(vv, vval, va));
      }
    }
    for (; i < vlen; ++i)
      Ac[i] -= v[i] * val;
  }
}

// --- RIGHT UPDATE (CACHE OPTIMIZED) ---
// REPLACED: Streaming stores -> Standard stores
void applyHouseholderRight(Matrix &A, int rs, int cs, const double *v, int vlen,
                           double beta) {
  if (beta == 0)
    return;
  int m_rows = A.rows - rs;

#pragma omp parallel proc_bind(close)
  {
    int n_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    int chunk = (m_rows + n_threads - 1) / n_threads;
    int r_start = std::min(tid * chunk, m_rows);
    int r_end = std::min(r_start + chunk, m_rows);
    int len = r_end - r_start;

    if (len > 0) {
      std::vector<double> y(len);

      double *col0 = A.col_ptr(cs) + rs + r_start;
      std::memcpy(y.data(), col0, len * sizeof(double));

      for (int j = 1; j < vlen; ++j) {
        double vj = v[j];
        double *colj = A.col_ptr(cs + j) + rs + r_start;
        __m256d vvj = _mm256_set1_pd(vj);

        int i = 0;
        for (; i <= len - 4; i += 4) {
          __m256d vy = _mm256_loadu_pd(y.data() + i);
          __m256d va = _mm256_loadu_pd(colj + i);
          _mm256_storeu_pd(y.data() + i, _mm256_fmadd_pd(va, vvj, vy));
        }
        for (; i < len; ++i)
          y[i] += colj[i] * vj;
      }

      for (int j = 0; j < vlen; ++j) {
        double fact = beta * (j == 0 ? 1.0 : v[j]);
        double *colj = A.col_ptr(cs + j) + rs + r_start;
        __m256d vfact = _mm256_set1_pd(fact);

        int i = 0;
        bool aligned = ((uintptr_t)colj % 32) == 0;

        // CRITICAL FIX: Use store_pd instead of stream_pd
        if (aligned) {
          for (; i <= len - 4; i += 4) {
            __m256d va = _mm256_load_pd(colj + i);
            __m256d vy = _mm256_loadu_pd(y.data() + i);
            _mm256_store_pd(colj + i, _mm256_fnmadd_pd(vy, vfact, va));
          }
        } else {
          for (; i <= len - 4; i += 4) {
            __m256d va = _mm256_loadu_pd(colj + i);
            __m256d vy = _mm256_loadu_pd(y.data() + i);
            _mm256_storeu_pd(colj + i, _mm256_fnmadd_pd(vy, vfact, va));
          }
        }
        for (; i < len; ++i)
          colj[i] -= y[i] * fact;
      }
    }
  }
}

// --- SEQUENTIAL GIVENS ---
void givens(double a, double b, double &c, double &s) {
  if (b == 0) {
    c = 1;
    s = 0;
  } else if (std::abs(b) > std::abs(a)) {
    double t = -a / b;
    s = 1.0 / std::sqrt(1.0 + t * t);
    c = s * t;
  } else {
    double t = -b / a;
    c = 1.0 / std::sqrt(1.0 + t * t);
    s = c * t;
  }
}

// --- PARALLEL GIVENS FLUSH ---
// CRITICAL OPTIMIZATION: Process rows in parallel
void flushGivens(Matrix &M, const std::vector<GivensRotation> &batch) {
  if (batch.empty())
    return;
  int m = M.rows;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i += 8) {
    int remaining = std::min(8, m - i);

    for (const auto &rot : batch) {
      double *RESTRICT ci = M.col_ptr(rot.k) + i;
      double *RESTRICT cj = M.col_ptr(rot.k + 1) + i;
      double c = rot.c, s = rot.s;

      __m256d vc = _mm256_set1_pd(c);
      __m256d vs = _mm256_set1_pd(s);

      // Process 4 or 8 doubles at a time
      if (remaining >= 4) {
        __m256d a0 = _mm256_loadu_pd(ci);
        __m256d b0 = _mm256_loadu_pd(cj);
        _mm256_storeu_pd(ci, _mm256_fmsub_pd(vc, a0, _mm256_mul_pd(vs, b0)));
        _mm256_storeu_pd(cj, _mm256_fmadd_pd(vs, a0, _mm256_mul_pd(vc, b0)));
      }
      if (remaining == 8) {
        __m256d a1 = _mm256_loadu_pd(ci + 4);
        __m256d b1 = _mm256_loadu_pd(cj + 4);
        _mm256_storeu_pd(ci + 4,
                         _mm256_fmsub_pd(vc, a1, _mm256_mul_pd(vs, b1)));
        _mm256_storeu_pd(cj + 4,
                         _mm256_fmadd_pd(vs, a1, _mm256_mul_pd(vc, b1)));
      }

      // Cleanup
      for (int k_rem = (remaining >= 4 ? (remaining == 8 ? 8 : 4) : 0);
           k_rem < remaining; ++k_rem) {
        double a = ci[k_rem];
        double b = cj[k_rem];
        ci[k_rem] = c * a - s * b;
        cj[k_rem] = s * a + c * b;
      }
    }
  }
}

// --- UTILS ---
Matrix matmul(const Matrix &A, const Matrix &B) {
  Matrix C(A.rows, B.cols);
  int m = C.rows, n = C.cols, K = A.cols;
#pragma omp parallel for schedule(static) proc_bind(close)
  for (int j = 0; j < n; ++j) {
    double *Cc = C.col_ptr(j);
    const double *Bc = B.col_ptr(j);
    for (int k = 0; k < K; ++k) {
      double bv = Bc[k];
      if (bv == 0)
        continue;
      const double *Ac = A.col_ptr(k);
      __m256d vb = _mm256_set1_pd(bv);
      int i = 0;
      for (; i <= m - 4; i += 4) {
        _mm256_storeu_pd(Cc + i, _mm256_fmadd_pd(_mm256_loadu_pd(Ac + i), vb,
                                                 _mm256_loadu_pd(Cc + i)));
      }
      for (; i < m; ++i)
        Cc[i] += Ac[i] * bv;
    }
  }
  return C;
}

Matrix transpose(const Matrix &A) {
  Matrix T(A.cols, A.rows);
#pragma omp parallel for schedule(static)
  for (int j = 0; j < A.cols; ++j) {
    const double *Ac = A.col_ptr(j);
    for (int i = 0; i < A.rows; ++i)
      T.col_ptr(i)[j] = Ac[i];
  }
  return T;
}

void thinQR(const Matrix &A, Matrix &Q, Matrix &R) {
  int m = A.rows, n = A.cols;
  Matrix H = A;
  R = Matrix(n, n);
  Q = Matrix(m, n);
  std::vector<std::vector<double>> Vs(n);
  std::vector<double> Betas(n);
  std::vector<double> vw(m);
  for (int k = 0; k < n; ++k) {
    int len = m - k;
    if (len <= 0)
      break;
    for (int i = 0; i < len; ++i)
      vw[i] = H.col_ptr(k)[k + i];
    double alpha;
    double beta = householderVector(vw.data(), len, alpha);
    Vs[k].assign(vw.begin(), vw.begin() + len);
    Betas[k] = beta;
    if (beta != 0 && k + 1 < n)
      applyHouseholderLeft(H, k, k + 1, vw.data(), len, beta);
    H.col_ptr(k)[k] = alpha;
  }
#pragma omp parallel for
  for (int j = 0; j < n; ++j) {
    double *rc = R.col_ptr(j), *hc = H.col_ptr(j);
    for (int i = 0; i <= j; ++i)
      rc[i] = hc[i];
  }
#pragma omp parallel for
  for (int j = 0; j < n; ++j)
    Q.col_ptr(j)[j] = 1.0;
  for (int k = n - 1; k >= 0; --k) {
    if (Betas[k] == 0)
      continue;
    applyHouseholderLeft(Q, k, k, Vs[k].data(), m - k, Betas[k]);
  }
}

void compute(const Matrix &in, Matrix &U, std::vector<double> &S, Matrix &V) {
  if (in.rows < in.cols) {
    Matrix AT = transpose(in);
    Matrix Ut(1, 1), Vt(1, 1);
    compute(AT, Ut, S, Vt);
    U = Vt;
    V = Ut;
    return;
  }
  if (in.rows > 2 * in.cols) {
    Matrix Qr(1, 1), Rs(1, 1);
    thinQR(in, Qr, Rs);
    Matrix Us(1, 1), Vs(1, 1);
    compute(Rs, Us, S, Vs);
    U = matmul(Qr, Us);
    V = Vs;
    return;
  }

  Matrix A = in;
  int m = A.rows, n = A.cols, mn = std::min(m, n);
  U = Matrix::identity(m);
  V = Matrix::identity(n);
  S.resize(mn);
  std::vector<double> vw(std::max(m, n));

  // BIDIAGONALIZATION
  for (int k = 0; k < mn; ++k) {
    // Left
    int len_l = m - k;
    if (len_l > 0) {
      for (int i = 0; i < len_l; ++i)
        vw[i] = A.col_ptr(k)[k + i];
      double alpha;
      double beta = householderVector(vw.data(), len_l, alpha);
      if (beta != 0) {
        applyHouseholderLeft(A, k, k + 1, vw.data(), len_l, beta);
        A.col_ptr(k)[k] = alpha;
        applyHouseholderRight(U, 0, k, vw.data(), len_l, beta);
      }
    }
    // Right
    if (k < n - 2) {
      int len_r = n - k - 1;
      for (int j = 0; j < len_r; ++j)
        vw[j] = A.col_ptr(k + 1 + j)[k];
      double alpha;
      double beta = householderVector(vw.data(), len_r, alpha);
      if (beta != 0) {
        applyHouseholderRight(A, k + 1, k + 1, vw.data(), len_r, beta);
        A.col_ptr(k + 1)[k] = alpha;
        applyHouseholderRight(V, 0, k + 1, vw.data(), len_r, beta);
      }
    }
  }

  // DIAGONALIZATION
  std::vector<double> d(mn), f(std::max(1, mn - 1));
  for (int i = 0; i < mn; ++i)
    d[i] = A.col_ptr(i)[i];
  for (int i = 0; i < mn - 1; ++i)
    f[i] = A.col_ptr(i + 1)[i];
  std::vector<GivensRotation> bU, bV;
  bU.reserve(1024);
  bV.reserve(1024);
  const size_t BLIM = 256;
  int iter = 0, maxiter = 30 * mn;

  while (iter < maxiter) {
    int midx = mn - 1;
    while (midx > 0) {
      if (std::abs(f[midx - 1]) <=
          1e-13 * (std::abs(d[midx - 1]) + std::abs(d[midx]))) {
        f[midx - 1] = 0;
        break;
      }
      midx--;
    }
    if (midx == mn - 1) {
      mn--;
      if (mn == 0)
        break;
      continue;
    }

    int q = mn - 1, p = q - 1;
    while (p > 0 &&
           std::abs(f[p - 1]) > 1e-13 * (std::abs(d[p - 1]) + std::abs(d[p])))
      p--;

    double dq = d[q], dq1 = d[q - 1], fq1 = f[q - 1];
    double fq2 = (q - 2 >= p) ? f[q - 2] : 0;
    double a11 = dq1 * dq1 + fq2 * fq2, a12 = dq1 * fq1,
           a22 = dq * dq + fq1 * fq1;
    double diff = (a11 - a22) * 0.5;
    double mu =
        a22 -
        (a12 * a12) / (diff + sign(diff) * std::sqrt(diff * diff + a12 * a12));
    double y = d[p] * d[p] - mu, z = d[p] * f[p];

    for (int k = p; k < q; ++k) {
      double c, s;
      givens(y, z, c, s);
      double dk = d[k], fk = f[k], dk1 = d[k + 1];
      d[k] = c * dk - s * fk;
      f[k] = s * dk + c * fk;
      d[k + 1] = c * dk1;
      double bulge = -s * dk1;
      if (k > p)
        f[k - 1] = c * y - s * z;
      bV.push_back({k, c, s});
      if (bV.size() >= BLIM) {
        flushGivens(V, bV);
        bV.clear();
      }
      y = d[k];
      z = bulge;
      givens(y, z, c, s);
      d[k] = c * y - s * z;
      double ofk = f[k], odk1 = d[k + 1];
      f[k] = c * ofk - s * odk1;
      d[k + 1] = s * ofk + c * odk1;
      if (k < q - 1) {
        double fk1 = f[k + 1];
        f[k + 1] = c * fk1;
        y = f[k];
        z = -s * fk1;
      }
      bU.push_back({k, c, s});
      if (bU.size() >= BLIM) {
        flushGivens(U, bU);
        bU.clear();
      }
    }
    iter++;
  }
  flushGivens(V, bV);
  flushGivens(U, bU);

  for (int i = 0; i < (int)S.size(); ++i) {
    S[i] = d[i];
    if (S[i] < 0) {
      S[i] = -S[i];
      double *vc = V.col_ptr(i);
#pragma omp parallel for
      for (int r = 0; r < n; ++r)
        vc[r] = -vc[r];
    }
  }
  for (int i = 0; i < (int)S.size() - 1; ++i) {
    int mi = i;
    for (int j = i + 1; j < (int)S.size(); ++j)
      if (S[j] > S[mi])
        mi = j;
    if (mi != i) {
      std::swap(S[i], S[mi]);
#pragma omp parallel for
      for (int r = 0; r < m; ++r)
        std::swap(U.data[r + i * U.stride], U.data[r + mi * U.stride]);
#pragma omp parallel for
      for (int r = 0; r < n; ++r)
        std::swap(V.data[r + i * V.stride], V.data[r + mi * V.stride]);
    }
  }
}
} // namespace OptSVD

void run_test(int rows, int cols) {
  std::cout << "Testing: " << rows << "x" << cols << "\n";
  Matrix A(rows, cols);
  A.randomize(0, 1);
  Matrix U(rows, rows), V(cols, cols);
  std::vector<double> S;
  auto t0 = std::chrono::high_resolution_clock::now();
  OptSVD::compute(A, U, S, V);
  double t = std::chrono::duration<double>(
                 std::chrono::high_resolution_clock::now() - t0)
                 .count();
  std::cout << "  Time: " << std::fixed << std::setprecision(3) << t << "s\n";
}

int main() {
  std::cout << "=== SVD V5: Stride Padding + Streaming Stores ===\n";
// Important: Bind threads to cores to prevent migration
#pragma omp parallel
  {
    // warm up
  }
  run_test(1000, 1000);
  return 0;
}