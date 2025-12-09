#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <numeric>
#include <cstring> 
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::LLT;

using namespace std;
using namespace std::chrono;

struct AlignedMatrix {
    size_t n;
    size_t stride;
    double* data;

    AlignedMatrix(size_t n_) : n(n_) {
       
        size_t rem = n_ % 4;
        stride = (rem == 0) ? n_ : (n_ + 4 - rem);

        data = (double*)_mm_malloc(stride * n_ * sizeof(double), 32);
        memset(data, 0, stride * n_ * sizeof(double));
    }

    ~AlignedMatrix() {
        if (data) _mm_free(data);
    }

    AlignedMatrix(const AlignedMatrix& other) : n(other.n), stride(other.stride) {
        data = (double*)_mm_malloc(stride * n * sizeof(double), 32);
        memcpy(data, other.data, stride * n * sizeof(double));
    }

    inline double& operator()(size_t i, size_t j) { return data[i * stride + j]; }
    inline const double& operator()(size_t i, size_t j) const { return data[i * stride + j]; }
};

inline double hsum_avx(__m256d v) {
    __m128d low = _mm256_extractf128_pd(v, 0);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(low) + _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
}

double computeError(const AlignedMatrix& L_my, const MatrixXd& L_eigen) {
    double err = 0.0;
    size_t n = L_my.n;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double diff = L_my(i, j) - L_eigen(i, j);
            err += diff * diff;
        }
    }
    return sqrt(err);
}

AlignedMatrix generateSPD(size_t n) {
    AlignedMatrix A(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double v = (rand() % 100) / 100.0;
            A(i, j) = A(j, i) = v;
        }
        A(i, i) += n;
    }
    return A;
}

AlignedMatrix Cholesky_SIMD_OMP(const AlignedMatrix& A, size_t blockSize = 64) {
    size_t n = A.n;
    size_t stride = A.stride; 
    AlignedMatrix L = A;
    double* data = L.data;

    for (size_t k = 0; k < n; k += blockSize) {
        size_t b = std::min(blockSize, n - k);

        for (size_t i = k; i < k + b; i++) {
            double* rowI = data + i * stride;

            double sum = 0.0;
            for (size_t p = k; p < i; p++) {
                sum += rowI[p] * rowI[p];
            }

            double val = rowI[i] - sum;
            double lii = sqrt(val);
            rowI[i] = lii;
            double invLii = 1.0 / lii;

            for (size_t j = i + 1; j < k + b; j++) {
                double* rowJ = data + j * stride;
                double sum2 = 0.0;
                for (size_t p = k; p < i; p++) {
                    sum2 += rowJ[p] * rowI[p];
                }
                rowJ[i] = (rowJ[i] - sum2) * invLii;
            }
        }

#pragma omp parallel for schedule(static)
        for (long long i = k + b; i < n; i++) {
            double* __restrict rowI = data + i * stride;

            for (size_t j = k; j < k + b; j++) {
                const double* __restrict rowJ = data + j * stride;

                __m256d vs1 = _mm256_setzero_pd();
                __m256d vs2 = _mm256_setzero_pd();

                size_t p = k;
                for (; p + 7 < j; p += 8) {
                    vs1 = _mm256_fmadd_pd(_mm256_load_pd(rowI + p), _mm256_load_pd(rowJ + p), vs1);
                    vs2 = _mm256_fmadd_pd(_mm256_load_pd(rowI + p + 4), _mm256_load_pd(rowJ + p + 4), vs2);
                }

                double sum = hsum_avx(_mm256_add_pd(vs1, vs2));

                for (; p < j; p++)
                    sum += rowI[p] * rowJ[p];

                rowI[j] = (rowI[j] - sum) / rowJ[j];
            }
        }

#pragma omp parallel for schedule(dynamic, 4)
        for (long long i = k + b; i < n; i++) {
            double* __restrict rowI = data + i * stride;
            size_t limit = k + b;

            for (size_t j = k + b; j <= i; j++) {
                const double* __restrict rowJ = data + j * stride;

                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();

                size_t p = k;

                for (; p + 15 < limit; p += 16) {
                    __m256d rI0 = _mm256_load_pd(rowI + p);
                    __m256d rJ0 = _mm256_load_pd(rowJ + p);
                    sum0 = _mm256_fmadd_pd(rI0, rJ0, sum0);

                    __m256d rI1 = _mm256_load_pd(rowI + p + 4);
                    __m256d rJ1 = _mm256_load_pd(rowJ + p + 4);
                    sum1 = _mm256_fmadd_pd(rI1, rJ1, sum1);

                    __m256d rI2 = _mm256_load_pd(rowI + p + 8);
                    __m256d rJ2 = _mm256_load_pd(rowJ + p + 8);
                    sum2 = _mm256_fmadd_pd(rI2, rJ2, sum2);

                    __m256d rI3 = _mm256_load_pd(rowI + p + 12);
                    __m256d rJ3 = _mm256_load_pd(rowJ + p + 12);
                    sum3 = _mm256_fmadd_pd(rI3, rJ3, sum3);
                }

                sum0 = _mm256_add_pd(sum0, sum2);
                sum1 = _mm256_add_pd(sum1, sum3);

                while (p + 7 < limit) {
                    sum0 = _mm256_fmadd_pd(_mm256_load_pd(rowI + p), _mm256_load_pd(rowJ + p), sum0);
                    sum1 = _mm256_fmadd_pd(_mm256_load_pd(rowI + p + 4), _mm256_load_pd(rowJ + p + 4), sum1);
                    p += 8;
                }

                double sum = hsum_avx(_mm256_add_pd(sum0, sum1));

                for (; p < limit; p++)
                    sum += rowI[p] * rowJ[p];

                rowI[j] -= sum;
            }
        }
    }

#pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            L(i, j) = 0.0;
        }
    }

    return L;
}

int main() {
    int n = 3000;
    int blockSize = 64;
   
    AlignedMatrix A = generateSPD(n);
    AlignedMatrix A_copy = A;

    auto start = std::chrono::high_resolution_clock::now();

    AlignedMatrix L = Cholesky_SIMD_OMP(A_copy, blockSize);

    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Vrijeme izvrsavanja: " << ms << " ms\n";

    return 0;
}

/*
int main() {

    srand(42);

    int n = 3000;
    int blockSize = 64;
    int runs = 100;

    AlignedMatrix A = generateSPD(n);

   
    cout << "\n Mjerenje Eigen LLT (3 puta, pravilno) =====\n";

    MatrixXd AE(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AE(i, j) = A(i, j);

    LLT<MatrixXd> llt;  
    vector<double> eigenTimes;

    for (int r = 0; r < 3; r++) {

        MatrixXd AE_copy = AE;  

        auto t1 = high_resolution_clock::now();
        llt.compute(AE_copy);   
        auto t2 = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        eigenTimes.push_back(ms);

        cout << "   Eigen iteracija " << r + 1 << ": " << ms << " ms\n";
    }

    double eigenBest = *min_element(eigenTimes.begin(), eigenTimes.end());
    cout << ">> Najbolje Eigen vrijeme: " << eigenBest << " ms\n\n";

    MatrixXd L_eigen = llt.matrixL();

    

    vector<double> simdTimes;
    simdTimes.reserve(runs);

    for (int r = 0; r < runs; r++) {

        AlignedMatrix A_copy = A;

        auto t1 = high_resolution_clock::now();
        AlignedMatrix L_my = Cholesky_SIMD_OMP(A_copy, blockSize);
        auto t2 = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        simdTimes.push_back(ms);

        if (r == 0) {
            double err = computeError(L_my, L_eigen);
            cout << "   [Validacija] greska = " << err
                << (err < 1e-5 ? " (OK)\n" : " (FAIL)\n");
        }

        cout << " Run " << r + 1 << ": " << ms << " ms\n";
    }

    vector<double> trimmed(simdTimes.begin() + 10, simdTimes.end() - 10);

    double avg = accumulate(trimmed.begin(), trimmed.end(), 0.0) / trimmed.size();
    double minT = *min_element(trimmed.begin(), trimmed.end());
    double maxT = *max_element(trimmed.begin(), trimmed.end());

    cout << "\n STATISTIKA (80 mjerenja, trimovana) =====\n";
    cout << "Prosjek:  " << avg << " ms\n";
    cout << "Minimum:  " << minT << " ms\n";
    cout << "Maksimum: " << maxT << " ms\n\n";

    return 0;
}
*/
