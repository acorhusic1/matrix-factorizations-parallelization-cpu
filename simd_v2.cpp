#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <immintrin.h>
#include <Eigen/Dense>
#include <iomanip>
#include <numeric>

using namespace std;
using namespace std::chrono;
using Eigen::MatrixXd;
using Eigen::LLT;


struct Matrix {
    size_t n;
    vector<double> data;

    Matrix() : n(0) {}
    explicit Matrix(size_t n_) : n(n_), data(n_* n_, 0.0) {}

    inline double& operator()(size_t i, size_t j) { return data[i * n + j]; }
    inline const double& operator()(size_t i, size_t j) const { return data[i * n + j]; }
};



inline double hsum_avx(__m256d v) {
    __m128d low = _mm256_extractf128_pd(v, 0);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(low) + _mm_cvtsd_f64(_mm_unpackhi_pd(low, low));
}

Matrix generateSPD(size_t n) {
    Matrix A(n);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double v = (rand() % 100) / 100.0;
            A(i, j) = A(j, i) = v;
        }
        A(i, i) += n; 
    }
    return A;
}

double checkError(const Matrix& L_custom, const MatrixXd& L_eigen) {
    double error = 0.0;
    size_t n = L_custom.n;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double diff = L_custom(i, j) - L_eigen(i, j);
            error += diff * diff;
        }
    }
    return sqrt(error);
}


Matrix Cholesky_SIMD_Optimized(const Matrix& A, size_t blockSize) {
    size_t n = A.n;
    Matrix L = A;
    double* data = L.data.data();

    for (size_t k = 0; k < n; k += blockSize) {
        size_t b = min(blockSize, n - k);

        for (size_t i = k; i < k + b; i++) {
            double* rowI = data + i * n;
            double sum = 0;
            for (size_t p = k; p < i; p++)
                sum += rowI[p] * rowI[p];
            double lii = sqrt(rowI[i] - sum);
            rowI[i] = lii;
            double invLii = 1.0 / lii;



            for (size_t j = i + 1; j < k + b; j++) {
                double* rowJ = data + j * n;
                double sum2 = 0;
                for (size_t p = k; p < i; p++)
                    sum2 += rowJ[p] * rowI[p];
                rowJ[i] = (rowJ[i] - sum2) * invLii;
            }
        }


        for (size_t i = k + b; i < n; i++) {
            double* rowI = data + i * n;
            for (size_t j = k; j < k + b; j++) {
                double* rowJ = data + j * n;
                __m256d vsum1 = _mm256_setzero_pd();
                __m256d vsum2 = _mm256_setzero_pd();
                size_t p = k;
                for (; p + 7 < j; p += 8) {
                    vsum1 = _mm256_fmadd_pd(_mm256_loadu_pd(rowI + p), _mm256_loadu_pd(rowJ + p), vsum1);
                    vsum2 = _mm256_fmadd_pd(_mm256_loadu_pd(rowI + p + 4), _mm256_loadu_pd(rowJ + p + 4), vsum2);
                }
                double sum = hsum_avx(_mm256_add_pd(vsum1, vsum2));
                for (; p < j; p++) sum += rowI[p] * rowJ[p];
                rowI[j] = (rowI[j] - sum) / rowJ[j];
            }
        }

        for (size_t i = k + b; i < n; i++) {
            double* rowI = data + i * n;
            size_t j = k + b;
            size_t limit_j = i;

            for (; j + 3 <= limit_j; j += 4) {
                double* rowJ0 = data + (j + 0) * n;
                double* rowJ1 = data + (j + 1) * n;
                double* rowJ2 = data + (j + 2) * n;
                double* rowJ3 = data + (j + 3) * n;

                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();

                size_t p = k;
                size_t limit_p = k + b;

                for (; p + 3 < limit_p; p += 4) {
                    __m256d vi = _mm256_loadu_pd(rowI + p);
                    sum0 = _mm256_fmadd_pd(vi, _mm256_loadu_pd(rowJ0 + p), sum0);
                    sum1 = _mm256_fmadd_pd(vi, _mm256_loadu_pd(rowJ1 + p), sum1);
                    sum2 = _mm256_fmadd_pd(vi, _mm256_loadu_pd(rowJ2 + p), sum2);
                    sum3 = _mm256_fmadd_pd(vi, _mm256_loadu_pd(rowJ3 + p), sum3);
                }
                rowI[j + 0] -= hsum_avx(sum0);
                rowI[j + 1] -= hsum_avx(sum1);
                rowI[j + 2] -= hsum_avx(sum2);
                rowI[j + 3] -= hsum_avx(sum3);

                for (; p < limit_p; p++) {
                    double valI = rowI[p];
                    rowI[j + 0] -= valI * rowJ0[p];
                    rowI[j + 1] -= valI * rowJ1[p];
                    rowI[j + 2] -= valI * rowJ2[p];
                    rowI[j + 3] -= valI * rowJ3[p];
                }
            }

            for (; j <= limit_j; j++) {
                double* rowJ = data + j * n;
                __m256d vsum = _mm256_setzero_pd();
                size_t p = k;
                size_t limit_p = k + b;
                for (; p + 3 < limit_p; p += 4) {
                    vsum = _mm256_fmadd_pd(_mm256_loadu_pd(rowI + p), _mm256_loadu_pd(rowJ + p), vsum);
                }
                double sum = hsum_avx(vsum);
                for (; p < limit_p; p++) sum += rowI[p] * rowJ[p];
                rowI[j] -= sum;
            }
        }
    }
    for (size_t i = 0; i < n; ++i) for (size_t j = i + 1; j < n; ++j) L(i, j) = 0.0;
    return L;
}

/*
int main() {

    srand(42);

    int n = 3000;
    int blockSize = 64;
    int ITER = 100;

    Matrix A = generateSPD(n);

    MatrixXd AE(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AE(i, j) = A(i, j);

    double eigen_min_time = 1e18;

    LLT<MatrixXd> llt;  

    for (int r = 0; r < 3; r++) {

        MatrixXd AE_copy = AE;   
        auto e1 = high_resolution_clock::now();
        llt.compute(AE_copy);    
        auto e2 = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(e2 - e1).count() / 1000.0;

        cout << "Eigen iteracija " << r + 1 << ": " << ms << " ms\n";

        eigen_min_time = min(eigen_min_time, ms);
    }

    cout << "\nNajbolje Eigen vrijeme: " << eigen_min_time << " ms\n\n";



    vector<double> times;
    times.reserve(ITER);

    cout << " Mjerenje SIMD optimizovane implementacije (100 puta)\n";

    LLT<MatrixXd> llt_full(AE);
    MatrixXd L_eigen_full = llt_full.matrixL();

    for (int iter = 0; iter < ITER; iter++) {

        Matrix A_copy = A;

        auto t1 = high_resolution_clock::now();
        Matrix L = Cholesky_SIMD_Optimized(A_copy, blockSize);
        auto t2 = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        times.push_back(ms);

        if (iter == 0) {
            double err = checkError(L, L_eigen_full);
            cout << "   [Validacija] Greska = " << err
                << (err < 1e-5 ? " (OK)" : " (FAIL)") << "\n";
        }

        cout << "Iteracija " << iter + 1 << "/" << ITER << " = " << ms << " ms\n";
    }



    sort(times.begin(), times.end());
    vector<double> trimmed(times.begin() + 10, times.end() - 10);

    double sum = accumulate(trimmed.begin(), trimmed.end(), 0.0);
    double avg_trimmed = sum / trimmed.size();
    double min_t = trimmed.front();
    double max_t = trimmed.back();



    cout << "\n===== REZULTATI (Trimmed) =====\n";
    cout << "Minimum   : " << min_t << " ms\n";
    cout << "Maximum   : " << max_t << " ms\n";
    cout << "AVG (80)  : " << avg_trimmed << " ms\n\n";

    cout << "Najbolje Eigen vrijeme: " << eigen_min_time << " ms\n";
    cout << "Speedup (Eigen / SIMD_AVG): " << eigen_min_time / avg_trimmed << "x\n";

    return 0;
}*/

int main() {

    srand(42);

    int n = 3000;
    int blockSize = 64;

    Matrix A = generateSPD(n);

    Matrix A_copy = A;

    auto t1 = chrono::high_resolution_clock::now();

    Matrix L = Cholesky_SIMD_Optimized(A_copy, blockSize);

    auto t2 = chrono::high_resolution_clock::now();

    double ms = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

    cout << "SIMD Cholesky izvrseno u: " << ms << " ms\n";

    return 0;
}
