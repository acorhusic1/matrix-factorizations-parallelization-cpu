#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <Eigen/Dense>


using namespace std;
using Eigen::MatrixXd;
using Eigen::LLT;
using namespace std::chrono;



struct Matrix {
    size_t n;
    vector<double> data;

    Matrix() : n(0) {}
    explicit Matrix(size_t n_) : n(n_), data(n_* n_, 0.0) {}

    double& operator()(size_t i, size_t j) { return data[i * n + j]; }
    const double& operator()(size_t i, size_t j) const { return data[i * n + j]; }
};

Matrix Cholesky_Parallel_Fixed(const Matrix& A, size_t blockSize) {
    size_t n = A.n;
    if (n == 0) return Matrix();

    std::vector<double> L_data(n * n);
    double* data = L_data.data();

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            data[i * n + j] = A(i, j);
        }
    }

    const double EPS = 1e-12;

  
#pragma omp parallel
    {
        for (size_t k = 0; k < n; k += blockSize) {
            size_t b = std::min(blockSize, n - k);

#pragma omp single
            {
                for (size_t i = k; i < k + b; i++) {
                    double* rowI = data + i * n;
                    double sum = 0.0;
                    for (size_t p = k; p < i; p++) sum += rowI[p] * rowI[p];

                    double val = rowI[i] - sum;
                    if (val < EPS) val = EPS;

                    double lii = sqrt(val);
                    rowI[i] = lii;
                    double invLii = 1.0 / lii;

                    for (size_t j = i + 1; j < k + b; j++) {
                        double* rowJ = data + j * n;
                        double sum_inner = 0.0;
                        for (size_t p = k; p < i; p++) sum_inner += rowJ[p] * rowI[p];
                        rowJ[i] = (rowJ[i] - sum_inner) * invLii;
                    }
                }
            }


           
#pragma omp for schedule(static)
            for (long long ii = (long long)k + (long long)b; ii < (long long)n; ii++) {
                size_t i = (size_t)ii;
                double* rowI = data + i * n;
                for (size_t j = k; j < k + b; j++) {
                    double* rowJ = data + j * n;
                    double sum = 0.0;

                    size_t p = k;
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    for (; p + 3 < j; p += 4) {
                        s0 += rowI[p] * rowJ[p];
                        s1 += rowI[p + 1] * rowJ[p + 1];
                        s2 += rowI[p + 2] * rowJ[p + 2];
                        s3 += rowI[p + 3] * rowJ[p + 3];
                    }
                    sum = s0 + s1 + s2 + s3;
                    for (; p < j; ++p) sum += rowI[p] * rowJ[p];

                    rowI[j] = (rowI[j] - sum) / rowJ[j];
                }
            }

          
#pragma omp for schedule(dynamic, 4)
            for (long long ii = (long long)k + (long long)b; ii < (long long)n; ii++) {
                size_t i = (size_t)ii;
                double* rowI = data + i * n;

                for (size_t j = k + b; j <= i; j++) {
                    double* rowJ = data + j * n;
                    double sum = 0.0;

                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    size_t p = k;
                    size_t limit = k + b;
                    const double* ptrI = rowI + k;
                    const double* ptrJ = rowJ + k;

                    size_t steps = (limit - k) / 4;
                    while (steps--) {
                        s0 += ptrI[0] * ptrJ[0];
                        s1 += ptrI[1] * ptrJ[1];
                        s2 += ptrI[2] * ptrJ[2];
                        s3 += ptrI[3] * ptrJ[3];
                        ptrI += 4; ptrJ += 4;
                    }
                    sum = s0 + s1 + s2 + s3;
                    p = k + ((limit - k) / 4) * 4;
                    while (p < limit) {
                        sum += rowI[p] * rowJ[p];
                        p++;
                    }
                    rowI[j] -= sum;
                }
            }
        }
    } 

    Matrix L_final(n);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; ++i) {
        for (size_t j = 0; j <= (size_t)i; ++j) {
            L_final(i, j) = L_data[i * n + j];
        }
    }
    return L_final;
}

Matrix generisiPozitivnoDefinitnu(size_t n) {
    Matrix A(n);
#pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double val = static_cast<double>((i * j + i + j) % 100) / 100.0;
            A(i, j) = val;
            A(j, i) = val;
        }
        A(i, i) += n;
    }
    return A;
}


/*
int main() {

    srand(42);

    int n = 3000;
    int blockSize = 48;   
    int ITER = 100;       

    Matrix A = generisiPozitivnoDefinitnu(n);



    MatrixXd AE(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AE(i, j) = A(i, j);

    double eigen_min_time = 1e18;

    cout << "Mjerenje Eigen LLT (3 puta)\n";
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

    cout << "Najbolje Eigen vrijeme: " << eigen_min_time << " ms\n\n";


    
    vector<double> times;
    times.reserve(ITER);

    cout << " Mjerenje paralelne implementacije (100 puta)\n";

    for (int iter = 0; iter < ITER; iter++) {

        Matrix A_copy = A; 

        auto t1 = high_resolution_clock::now();
        Matrix L = Cholesky_Parallel_Fixed(A_copy, blockSize);
        auto t2 = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
        times.push_back(ms);

        cout << "Iteracija " << iter + 1 << "/" << ITER << " = " << ms << " ms\n";
    }


    sort(times.begin(), times.end());

    vector<double> trimmed(times.begin() + 10, times.end() - 10);  

    double sum = 0;
    for (double t : trimmed) sum += t;

    double avg_trimmed = sum / trimmed.size();
    double min_t = trimmed.front();
    double max_t = trimmed.back();


    cout << "\n REZULTATI 100 ITERACIJA (sa odbacivanjem prvih 10 i zadnjih 10 iteracija))\n";
    cout << "Minimum (trimmed): " << min_t << " ms\n";
    cout << "Maximum (trimmed): " << max_t << " ms\n";
    cout << "AVG (trimmed 80):  " << avg_trimmed << " ms\n\n";

    cout << "Najbolje Eigen vrijeme: " << eigen_min_time << " ms\n";
    cout << "Speedup (Eigen / Tvoj AVG): " << (eigen_min_time / avg_trimmed) << "x\n";

    return 0;
}

*/


int main() {
    int n = 3000;
    int blockSize = 48; 
    omp_set_num_threads(2);

    Matrix A = generisiPozitivnoDefinitnu(n);
    auto start = chrono::high_resolution_clock::now();

    Matrix L = Cholesky_Parallel_Fixed(A, blockSize); 

    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    cout << "Vrijeme izvrsavanja: " << ms << " ms\n";


    






    return 0;
}