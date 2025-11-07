#include <immintrin.h>
#include <random>
#include <memory>
#include <chrono>
#include <iostream>
#include <vector>

void LU_1(double* A, double* L, double* U, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			L[i * n + j] = 0.0;
			U[i * n + j] = 0.0;
		}
	}

	for (int i = 0; i < n; i++) {
        // Gornja matrica U
        for (int j = i; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += L[i * n + k] * U[k * n + j];

            U[i * n + j] = A[i * n + j] - sum;
        }

        // Donja matrica L
        for (int j = i; j < n; j++) {
            if (i == j)
                L[i * n + j] = 1.0;
            else {
                double sum = 0.0;
                for (int k = 0; k < i; k++)
                    sum += L[j * n + k] * U[k * n + i];

                L[j * n + i] = (A[j * n + i] - sum) / U[i * n + i];
            }
        }
    }
}

void LU_2(double* A, double* L, double* U, int n)
{
    // Postavi L kao jediničnu matricu i U kao nule
    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }

    for (int k = 0; k < n; ++k)
    {
        // 1. Izračunaj red k matrice U
        const int kn = k * n;
        for (int j = k; j < n; ++j)
        {
            double sum = 0.0;
            const int jn = j;  // smanjuje ponavljanje množenja
            for (int p = 0; p < k; ++p)
                sum += L[kn + p] * U[p * n + jn];

            U[kn + jn] = A[kn + jn] - sum;
        }

        // 2. Izračunaj kolonu k matrice L
        for (int i = k; i < n; ++i)
        {
            const int in = i * n;
            if (i == k)
                L[in + k] = 1.0;
            else
            {
                double sum = 0.0;
                for (int p = 0; p < k; ++p)
                    sum += L[in + p] * U[p * n + k];

                L[in + k] = (A[in + k] - sum) / U[k * n + k];
            }
        }
    }
}

int main() {
	int n = 1024;

	std::vector<double> A(1LL * n * n);
	std::vector<double> L(1LL * n * n);
	std::vector<double> U(1LL * n * n);

	// Popuni matricu A random vrijednostima
	for (int i = 0; i < n * n; i++)
		A[i] = (rand() % 100) / 10.0 + 1.0;

	auto t0 = std::chrono::high_resolution_clock::now();
	LU_1(A.data(), L.data(), U.data(), n);
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> t = t1 - t0;
	std::cout << "LU_1: " << t.count() << " s\n";

	std::vector<double> A2(1LL * n * n);
	std::vector<double> L2(1LL * n * n);
	std::vector<double> U2(1LL * n * n);

	// Popuni matricu A2 random vrijednostima
	for (int i = 0; i < n * n; i++)
		A2[i] = (rand() % 100) / 10.0 + 1.0;

	t0 = std::chrono::high_resolution_clock::now();
	LU_2(A2.data(), L2.data(), U2.data(), n);
	t1 = std::chrono::high_resolution_clock::now();

	t = t1 - t0;
	std::cout << "LU_2: " << t.count() << " s\n";

	return 0;
}
