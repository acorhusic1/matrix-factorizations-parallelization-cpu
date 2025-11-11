#include <iostream>
#include <vector>
#include <cmath> 
#include <iomanip>
#include <chrono>
#include <omp.h>

using namespace std;
using Matrix = vector<vector<double>>;


Matrix Cholesky(const Matrix& A) {
    size_t n = A.size();
    if (A.empty()) {
       cout << "Greska: Matrica je prazna!" << endl;
       return {};
    }

    for (const auto& row : A) {
        if (row.size() != n) {
            cout<< "Greska: Matrica nije kvadratna!" << endl;
            return {};
        }
    }

    Matrix L(n, vector<double>(n, 0.0));
    omp_set_num_threads(2);
    for (size_t i = 0; i < n; i++) {
        double suma = 0.0;
        for (size_t k = 0; k < i; k++)
            suma += L[i][k] * L[i][k];
        double val = A[i][i] - suma;
        if (val <= 0) {
            cout << "Matrica nije pozitivno definitna!" << endl;
            exit(1);
        }
        L[i][i] = sqrt(val);

#pragma omp parallel for schedule(static)
        for (int j = i + 1; j < (int)n; j++) {
            double suma2 = 0.0;
            for (size_t k = 0; k < i; k++)
                suma2 += L[j][k] * L[i][k];
            L[j][i] = (A[j][i] - suma2) / L[i][i];
        }
    }

    return L;
}

Matrix generisiPozitivnoDefinitnu(size_t n) {
    Matrix A(n, vector<double>(n));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = i; j < n; j++) {
            double val = (double)rand() / RAND_MAX;
            A[i][j] = A[j][i] = val;
        }
        A[i][i] += n;
    }
    return A;
}


int main() {
    srand(0);

    vector<int> velicine = { 500, 1000, 2000, 3000, 5000, 6000 };

    cout << "Broj jezgara (niti): " << omp_get_max_threads() << endl;

    for (int n : velicine) {
        cout << "\nTest za matricu " << n << "x" << n << endl;
        Matrix A = generisiPozitivnoDefinitnu(n);

        auto start = chrono::high_resolution_clock::now();
        Matrix L = Cholesky(A);
        auto end = chrono::high_resolution_clock::now();

        auto trajanje = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        cout << "Vrijeme izvrsavanja: " << trajanje << " ms" << endl;
    }

    return 0;
}
