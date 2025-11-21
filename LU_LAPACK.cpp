#include <immintrin.h>
#include <random>
#include <memory>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>

extern "C" {
    // LAPACK LU factorization
    void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
}


int main() {
    int n = 8192;
    
    std::vector<double> A(1LL * n * n);

    // Popuni matricu A random vrijednostima
    for (int i = 0; i < n * n; i++)
        A[i] = (rand() % 100) / 10.0 + 1.0;

    // Kopija matrice jer LAPACK mijenja ulaz
    std::vector<double> A_copy = A;

    std::vector<int> ipiv(n); // pivot array
    int info;

    // --- LAPACK dgetrf ---
    auto t0 = std::chrono::high_resolution_clock::now();
    dgetrf_(&n, &n, A_copy.data(), &n, ipiv.data(), &info);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (info != 0) {
        std::cout << "LAPACK dgetrf error, info = " << info << "\n";
        return 1;
    }

    std::cout << "LAPACK dgetrf LU factorization (n=" << n << "): "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";
            
    return 0;
}