#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>   
#include <Eigen/Dense>


using namespace std;
using namespace std::chrono;

using Eigen::MatrixXd;
using Eigen::LLT;

struct Matrix {
    size_t n;
    vector<double> data;

    Matrix() : n(0) {}
    explicit Matrix(size_t n_) : n(n_), data(n_* n_, 0.0) {}

    double& operator()(size_t i, size_t j) {
        return data[i * n + j];
    }
    const double& operator()(size_t i, size_t j) const {
        return data[i * n + j];
    }
};

inline double& AT(vector<double>& data, size_t n, size_t i, size_t j) {
    return data[i * n + j];
}

Matrix Cholesky_blocked_optimizovana(const Matrix& A, size_t blockSize = 48) {
    size_t n = A.n;
    if (n == 0) return Matrix();

    vector<double> L_data(n * n);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            L_data[i * n + j] = A(i, j);


    const double EPS = 1e-12;
    double* data = L_data.data();


    for (size_t k = 0; k < n; k += blockSize) {
        size_t b = std::min(blockSize, n - k);


        for (size_t i = k; i < k + b; i++) {
            double* rowI = data + i * n;

            double sum = 0.0;
            for (size_t p = k; p < i; p++)
                sum += rowI[p] * rowI[p];

            double val = rowI[i] - sum;
            if (val <= EPS) {
                cout << "Greska: Matrica nije pozitivno definitna (red " << i << ")!\n";
                exit(1);
            }

            double lii = sqrt(val);
            rowI[i] = lii;
            double invLii = 1.0 / lii;




            for (size_t j = i + 1; j < k + b; j++) {
                double* rowJ = data + j * n;
                double sum_inner = 0.0;
                for (size_t p = k; p < i; p++)
                    sum_inner += rowJ[p] * rowI[p];

                rowJ[i] = (rowJ[i] - sum_inner) * invLii;
            }
        }



        for (size_t i = k + b; i < n; i++) {
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

                for (; p < j; ++p)
                    sum += rowI[p] * rowJ[p];

                rowI[j] = (rowI[j] - sum) / rowJ[j];
            }
        }



        for (size_t i = k + b; i < n; i++) {
            double* rowI = data + i * n;

            for (size_t j = k + b; j <= i; j++) {
                double* rowJ = data + j * n;

                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                size_t p = k;
                size_t limit = k + b;

                const double *ptrI = rowI + k;
                const double *ptrJ = rowJ + k;

                size_t steps = (limit - k) / 4;
                while (steps--) {
                    s0 += ptrI[0] * ptrJ[0];
                    s1 += ptrI[1] * ptrJ[1];
                    s2 += ptrI[2] * ptrJ[2];
                    s3 += ptrI[3] * ptrJ[3];

                    ptrI += 4;
                    ptrJ += 4;
                    p += 4;
                }

                double sum = s0 + s1 + s2 + s3;

                while (p < limit) {
                    sum += rowI[p] * rowJ[p];
                    p++;
                }

                rowI[j] -= sum;
            }
        }
    }

    Matrix L_final(A.n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            L_final(i, j) = (j > i) ? 0.0 : L_data[i * n + j];

    return L_final;
}


Matrix generisiPozitivnoDefinitnu(size_t n) {
    Matrix A(n);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            double val = static_cast<double>(rand()) / RAND_MAX;
            A(i, j) = val;
            A(j, i) = val;
        }
        A(i, i) += n;  
    }
    cout << "Gotovo.\n";
    return A;
}

double mean(const vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}




int main() {
    
   srand(42);

    int n = 3000;
    int blockSize = 48;

    Matrix A = generisiPozitivnoDefinitnu(n);

    auto t1 = high_resolution_clock::now();
    Matrix L = Cholesky_blocked_optimizovana(A, blockSize);
    auto t2 = high_resolution_clock::now();

    double ms = duration_cast<microseconds>(t2 - t1).count() / 1000.0;

    cout << "Vrijeme jedne iteracije: " << ms << " ms\n";

    return 0;
    

    /*
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

    cout << "\n Mjerenje Eigen LLT (3 puta)\n";

    for (int r = 0; r < 3; r++) {
        auto e1 = high_resolution_clock::now();
        LLT<MatrixXd> llt(AE);
        auto e2 = high_resolution_clock::now();

        double ms = duration_cast<microseconds>(e2 - e1).count() / 1000.0;
        eigen_min_time = min(eigen_min_time, ms);

        cout << "Eigen iteracija " << r + 1 << ": " << ms << " ms\n";
    }

    cout << "Najbolje Eigen vrijeme: " << eigen_min_time << " ms\n\n";

    vector<double> times;
    times.reserve(ITER);

    cout << " Mjerenje sekv. implementacije (100 puta) ---\n";

    for (int iter = 0; iter < ITER; iter++) {

        Matrix A_copy = A; 

        auto t1 = high_resolution_clock::now();
        Matrix L = Cholesky_blocked_optimizovana(A_copy, blockSize);
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

    double min_t = times[10];
    double max_t = times[times.size() - 11];

  
    cout << "\n REZULTATI 100 ITERACIJA (trimmed) \n";
    cout << "Minimum (posle odbacivanja): " << min_t << " ms\n";
    cout << "Maximum (posle odbacivanja): " << max_t << " ms\n";
    cout << "AVG prosjek (80 iteracija):   " << avg_trimmed << " ms\n\n";

    cout << "Eigen (najbolji od 3):        " << eigen_min_time << " ms\n\n";


    return 0;
 */ 
}
