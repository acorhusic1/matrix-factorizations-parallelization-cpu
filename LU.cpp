#include <immintrin.h>
#include <random>
#include <memory>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>

void LU_naivna(double* A, double* L, double* U, int n) {
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

void LU_optimizovana(double* A, double* L, double* U, int n)
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


void LU_blokovska(double* A, double* L, double* U, int n) {
    int B = 96;

    std::vector<double> A_copy(A, A + n*n); // kopija matrice za Schur update

    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }

    for (int kb = 0; kb < n; kb += B) {
        int kend = std::min(kb + B, n);

        // 1) Dijagonalni blok
        for (int k = kb; k < kend; ++k) {
            int kn = k * n;
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                for (int p = kb; p < k; ++p) sum += L[k * n + p] * U[p * n + j];
                U[kn + j] = A_copy[kn + j] - sum;
            }
            for (int i = k; i < kend; ++i) {
                int in = i * n;
                if (i == k)
                    L[in + k] = 1.0;
                else {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_copy[in + k] - sum) / U[kn + k];
                }
            }
        }

        // 2) U blokovi desno
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                for (int j = colb; j < colend; ++j) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) sum += L[kn + p] * U[p * n + j];
                    U[kn + j] = A_copy[kn + j] - sum;
                }
            }
        }

        // 3) L blokovi ispod dijagonale
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int i = rowb; i < rowend; ++i) {
                int in = i * n;
                for (int k = kb; k < kend; ++k) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_copy[in + k] - sum) / U[k * n + k];
                }
            }
        }

        // 4) Schur update
        for (int p = kb; p < kend; ++p) {
            int pn = p * n;
            for (int i = kend; i < n; ++i) {
                double Lip = L[i * n + p];
                int in = i * n;
                for (int j = kend; j < n; ++j) {
                    A_copy[in + j] -= Lip * U[pn + j];
                }
            }
        }
    }
}



// Funkcija za provjeru je li A = L*U
bool checkLU(const std::vector<double>& A, const std::vector<double>& L, const std::vector<double>& U, int n, double tol = 1e-6) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += L[i * n + k] * U[k * n + j];
            }
            if (std::abs(A[i * n + j] - sum) > tol) {
                std::cout << "Mismatch at (" << i << ", " << j << "): A=" << A[i * n + j] << " L*U=" << sum << "\n";
                return false;
            }
        }
    }
    return true;
}


void LU_blokovska_omp(double* A, double* L, double* U, int n) {
    int B = 96;
    std::vector<double> A_copy(A, A + n*n);

    // Paralelizovana inicijalizacija
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }

    for (int kb = 0; kb < n; kb += B) {
        int kend = std::min(kb + B, n);

        // Dijagonalni blok - mora biti sekvencijalno
        for (int k = kb; k < kend; ++k) {
            int kn = k * n;
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                for (int p = kb; p < k; ++p) sum += L[k * n + p] * U[p * n + j];
                U[kn + j] = A_copy[kn + j] - sum;
            }
            for (int i = k; i < kend; ++i) {
                int in = i * n;
                if (i == k) L[in + k] = 1.0;
                else {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_copy[in + k] - sum) / U[kn + k];
                }
            }
        }


        // 2) U blokovi desno
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                for (int j = colb; j < colend; ++j) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) sum += L[kn + p] * U[p * n + j];
                    U[kn + j] = A_copy[kn + j] - sum;
                }
            }
        }

        // 3) L blokovi ispod dijagonale
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int i = rowb; i < rowend; ++i) {
                int in = i * n;
                for (int k = kb; k < kend; ++k) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_copy[in + k] - sum) / U[k * n + k];
                }
            }
        }

        // Schur update - sigurno paralelizovati
        #pragma omp parallel for collapse(2)
        for (int i = kend; i < n; ++i) {
            for (int j = kend; j < n; ++j) {
                double val = 0.0;
                for (int p = kb; p < kend; ++p) val += L[i * n + p] * U[p * n + j];
                A_copy[i * n + j] -= val;
            }
        }
    }
}




int main() {
	int n = 4096;
    
	std::vector<double> A(1LL * n * n);
	std::vector<double> L(1LL * n * n);
	std::vector<double> U(1LL * n * n);

	// Popuni matricu A random vrijednostima
	for (int i = 0; i < n * n; i++)
		A[i] = (rand() % 100) / 10.0 + 1.0;

    // --- LU_naivna ---
	/*auto t0 = std::chrono::high_resolution_clock::now();
	LU_naivna(A.data(), L.data(), U.data(), n);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "LU_naivna: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    if (checkLU(A, L, U, n)) 
        std::cout << "LU_naivna verification: PASS\n";
    else 
        std::cout << "LU_naivna verification: FAIL\n";

	// --- LU_optimizovana ---
	std::vector<double> A2 = A, L2(n * n), U2(n * n);
	t0 = std::chrono::high_resolution_clock::now();
	LU_optimizovana(A2.data(), L2.data(), U2.data(), n);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "LU_optimizovana: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    if (checkLU(A2, L2, U2, n)) 
        std::cout << "LU_optimizovana verification: PASS\n";
    else 
        std::cout << "LU_optimizovana verification: FAIL\n";*/

	// --- LU_blokovska ---
	std::vector<double> A3 = A, L3(n * n), U3(n * n);
	auto t0 = std::chrono::high_resolution_clock::now();
	LU_blokovska(A3.data(), L3.data(), U3.data(), n);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "LU_blokovska: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    
    /*if (checkLU(A3, L3, U3, n))
        std::cout << "LU_blokovska verification: PASS\n";
    else 
        std::cout << "LU_blokovska verification: FAIL\n";*/


    std::vector<double> A4 = A, L4(n * n), U4(n * n);
	t0 = std::chrono::high_resolution_clock::now();
	LU_blokovska_omp(A4.data(), L4.data(), U4.data(), n);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "LU_blokovska_omp: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
    
    /*if (checkLU(A4, L4, U4, n))
        std::cout << "LU_blokovska verification: PASS\n";
    else 
        std::cout << "LU_blokovska verification: FAIL\n";*/


	/* for (int B : {16, 32, 64, 96, 128, 256}) {
		auto start = std::chrono::high_resolution_clock::now();
		LU_blokovska(A.data(), L.data(), U.data(), n, B);
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "B = " << B << ": " << std::chrono::duration<double>(end - start).count() << " s\n";
    } */
	return 0;
}
