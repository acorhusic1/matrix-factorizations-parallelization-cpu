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


void LU_blokovska_V1(double* A, double* L, double* U, int n) {
    // B: veličina bloka (block size).
    int B = 96;

    // Napravimo kopiju matrice A jer ćemo je modificirati tokom Schur update-a.
    // Razlog: Schur update-a (A -= L[:,p] * U[p,:]) mijenja A; ako želimo da
    // originalnu A ostavimo nepromijenjenom ili da koristimo A za izračun u drugim fazama,
    // koristimo radnu kopiju. Također to čini elementarne pristupe jasnijim.
    std::vector<double> A_copy(A, A + n*n); // kopija matrice za Schur update

    // Inicijalizacija L i U na 0. Potrebno, jer ih tokom algoritma popunjavamo.
    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }

    // Glavna petlja po blokovima duž glavne dijagonale.
    // kb označava "k-block" početak, tj. početni red/kolonu trenutnog dijagonalnog bloka.
    for (int kb = 0; kb < n; kb += B) {
        // kend: indeks koji je krajnji za trenutni blok; zadnji blok može biti manji od B.
        int kend = std::min(kb + B, n);

        // ========================
        // 1) Obrada dijagonalnog bloka (izračun U i L unutar bloka)
        // ========================
        // Cilj: izračunati U[k, j] za sve k, j u bloku i L[i, k] za i u bloku, k u bloku,
        // koristeći već izračunate kolone/retke iz ranijih blokova.
        for (int k = kb; k < kend; ++k) {
            // kn je offset početka k-tog reda (row-major layout: red * n)
            int kn = k * n;

            // Izračun U[k][j] unutar bloka za j >= k
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                // sum = Σ_{p=kb..k-1} L[k][p] * U[p][j]
                // Napomena: p kreće od početka bloka (kb) jer smo prethodne blokove već
                // uklonili kroz Schur update, odnosno A_copy već sadrži ažurirane vrijednosti.
                for (int p = kb; p < k; ++p) sum += L[k * n + p] * U[p * n + j];

                // U[k][j] = A_copy[k][j] - sum
                // Koristimo A_copy jer je A modificiran u Schur fazi.
                U[kn + j] = A_copy[kn + j] - sum;
            }

            // Izračun L[i][k] za i >= k unutar bloka (L je lower-triangular)
            for (int i = k; i < kend; ++i) {
                int in = i * n;
                if (i == k)
                    // Dijagonala L je 1 (u Doolittleovoj formi).
                    L[in + k] = 1.0;
                else {
                    double sum = 0.0;
                    // sum = Σ_{p=kb..k-1} L[i][p] * U[p][k]
                    for (int p = kb; p < k; ++p) sum += L[in + p] * U[p * n + k];

                    // L[i][k] = (A_copy[i][k] - sum) / U[k][k]
                    // Djelimo kroz U[k][k] (mora biti nenulti — vidi napomenu o pivotiranju niže).
                    L[in + k] = (A_copy[in + k] - sum) / U[kn + k];
                }
            }
        }

        // ========================
        // 2) Popunjavanje U-blokova desno od dijagonalnog bloka
        // ========================
        // Za sve blokove koji su u istom "bloku retka" ali desno od dijagonale,
        // popunimo U elemente koristeći L unutar dijagonalnog bloka i prethodno izračunate U.
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                for (int j = colb; j < colend; ++j) {
                    double sum = 0.0;
                    // sum = Σ_{p=kb..k-1} L[k][p] * U[p][j]
                    for (int p = kb; p < k; ++p) sum += L[kn + p] * U[p * n + j];

                    // U[k][j] = A_copy[k][j] - sum
                    U[kn + j] = A_copy[kn + j] - sum;
                }
            }
        }

        // ========================
        // 3) Popunjavanje L-blokova ispod dijagonale
        // ========================
        // Za blokove ispod dijagonale: popuni L[i][k] za k u dijagonalnom bloku.
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int i = rowb; i < rowend; ++i) {
                int in = i * n;
                for (int k = kb; k < kend; ++k) {
                    double sum = 0.0;
                    // sum = Σ_{p=kb..k-1} L[i][p] * U[p][k]
                    for (int p = kb; p < k; ++p) sum += L[in + p] * U[p * n + k];

                    // L[i][k] = (A_copy[i][k] - sum) / U[k][k]
                    L[in + k] = (A_copy[in + k] - sum) / U[k * n + k];
                }
            }
        }

        // ========================
        // 4) Schur update (ažuriranje donjeg-desnog dijela matrice A_copy)
        // ========================
        // Što zapravo radimo ovdje: uklanjamo doprinos L[:,p] * U[p,:] za p u trenutnom bloku
        // iz preostale (donje-desne) podmatrice. To priprema A_copy za sljedeći dijagonalni blok.
        for (int p = kb; p < kend; ++p) {
            int pn = p * n;
            for (int i = kend; i < n; ++i) {
                // L[i][p] je fiksna tokom unutrašnje petlje po j, pa je dobro izvući je u var.
                double Lip = L[i * n + p];
                int in = i * n;
                for (int j = kend; j < n; ++j) {
                    // A_copy[i][j] -= L[i][p] * U[p][j]
                    // Ovo je matrično množenje "rank-1" tipa koje se primjenjuje B puta.
                    A_copy[in + j] -= Lip * U[pn + j];
                }
            }
        }
        // Nakon Schur update-a, podmatrica [kend:n, kend:n] sadrži A - L_block*U_block
        // i koristi se u sljedećim iteracijama.
    } // kraj petlje po blokovima kb
}


void LU_blokovska_V2(double* A, double* L, double* U, int n) {
    const int B = 96;
    
    // Moramo imati kopiju jer radimo Schur update
    std::vector<double> A_work(n * n);
    std::memcpy(A_work.data(), A, n * n * sizeof(double));
    
    std::memset(L, 0, n * n * sizeof(double));
    std::memset(U, 0, n * n * sizeof(double));
    
    for (int kb = 0; kb < n; kb += B) {
        int kend = std::min(kb + B, n);
        
        // 1) Dijagonalni blok
        for (int k = kb; k < kend; ++k) {
            int kn = k * n;
            
            // U red
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                for (int p = kb; p < k; ++p) 
                    sum += L[kn + p] * U[p * n + j];
                U[kn + j] = A_work[kn + j] - sum;
            }
            
            // L kolona
            L[kn + k] = 1.0;
            double inv_Ukk = 1.0 / U[kn + k];
            for (int i = k + 1; i < kend; ++i) {
                int in = i * n;
                double sum = 0.0;
                for (int p = kb; p < k; ++p) 
                    sum += L[in + p] * U[p * n + k];
                L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
            }
        }
        
        // 2) U blokovi desno
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                
                // Loop unrolling - 4 elementa odjednom
                int j = colb;
                for (; j + 3 < colend; j += 4) {
                    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (int p = kb; p < k; ++p) {
                        double Lkp = L[kn + p];
                        int pn = p * n;
                        sum0 += Lkp * U[pn + j];
                        sum1 += Lkp * U[pn + j + 1];
                        sum2 += Lkp * U[pn + j + 2];
                        sum3 += Lkp * U[pn + j + 3];
                    }
                    U[kn + j]     = A_work[kn + j]     - sum0;
                    U[kn + j + 1] = A_work[kn + j + 1] - sum1;
                    U[kn + j + 2] = A_work[kn + j + 2] - sum2;
                    U[kn + j + 3] = A_work[kn + j + 3] - sum3;
                }
                for (; j < colend; ++j) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) 
                        sum += L[kn + p] * U[p * n + j];
                    U[kn + j] = A_work[kn + j] - sum;
                }
            }
        }
        
        // 3) L blokovi ispod
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                double inv_Ukk = 1.0 / U[kn + k];
                
                for (int i = rowb; i < rowend; ++i) {
                    int in = i * n;
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p) 
                        sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
                }
            }
        }
        
        // 4) Schur update - optimiziran sa dvostrukim blokiranjem
        const int SB = 64;
        for (int ib = kend; ib < n; ib += SB) {
            int iend = std::min(ib + SB, n);
            for (int jb = kend; jb < n; jb += SB) {
                int jend = std::min(jb + SB, n);
                
                for (int p = kb; p < kend; ++p) {
                    int pn = p * n;
                    for (int i = ib; i < iend; ++i) {
                        double Lip = L[i * n + p];
                        int in = i * n;
                        
                        // Loop unrolling
                        int j = jb;
                        for (; j + 3 < jend; j += 4) {
                            A_work[in + j]     -= Lip * U[pn + j];
                            A_work[in + j + 1] -= Lip * U[pn + j + 1];
                            A_work[in + j + 2] -= Lip * U[pn + j + 2];
                            A_work[in + j + 3] -= Lip * U[pn + j + 3];
                        }
                        for (; j < jend; ++j) {
                            A_work[in + j] -= Lip * U[pn + j];
                        }
                    }
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


void LU_blokovska_V1_omp(double* A, double* L, double* U, int n) {
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

void LU_blokovska_V2_omp(double* A, double* L, double* U, int n) {
    const int B = 96;
    
    std::vector<double> A_work(n * n);
    std::memcpy(A_work.data(), A, n * n * sizeof(double));
    
    // Paralelizovana inicijalizacija
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }
    
    for (int kb = 0; kb < n; kb += B) {
        int kend = std::min(kb + B, n);
        
        // ========================
        // 1) Dijagonalni blok - MORA biti sekvencijalno
        // ========================
        for (int k = kb; k < kend; ++k) {
            int kn = k * n;
            
            // U red
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                for (int p = kb; p < k; ++p)
                    sum += L[kn + p] * U[p * n + j];
                U[kn + j] = A_work[kn + j] - sum;
            }
            
            // L kolona
            L[kn + k] = 1.0;
            double inv_Ukk = 1.0 / U[kn + k];
            for (int i = k + 1; i < kend; ++i) {
                int in = i * n;
                double sum = 0.0;
                for (int p = kb; p < k; ++p)
                    sum += L[in + p] * U[p * n + k];
                L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
            }
        }
        
        // ========================
        // 2) U blokovi desno - PARALELIZOVANO po blokovima
        // ========================
        #pragma omp parallel for schedule(dynamic)
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                
                // Loop unrolling - 4 elementa odjednom
                int j = colb;
                for (; j + 3 < colend; j += 4) {
                    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                    for (int p = kb; p < k; ++p) {
                        double Lkp = L[kn + p];
                        int pn = p * n;
                        sum0 += Lkp * U[pn + j];
                        sum1 += Lkp * U[pn + j + 1];
                        sum2 += Lkp * U[pn + j + 2];
                        sum3 += Lkp * U[pn + j + 3];
                    }
                    U[kn + j]     = A_work[kn + j]     - sum0;
                    U[kn + j + 1] = A_work[kn + j + 1] - sum1;
                    U[kn + j + 2] = A_work[kn + j + 2] - sum2;
                    U[kn + j + 3] = A_work[kn + j + 3] - sum3;
                }
                
                // Ostatak
                for (; j < colend; ++j) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p)
                        sum += L[kn + p] * U[p * n + j];
                    U[kn + j] = A_work[kn + j] - sum;
                }
            }
        }
        
        // ========================
        // 3) L blokovi ispod - PARALELIZOVANO po blokovima
        // ========================
        #pragma omp parallel for schedule(dynamic)
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                double inv_Ukk = 1.0 / U[kn + k];
                for (int i = rowb; i < rowend; ++i) {
                    int in = i * n;
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p)
                        sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
                }
            }
        }
        
        // ========================
        // 4) Schur update - PARALELIZOVANO po blokovima
        // ========================
        const int SB = 64;
        
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ib = kend; ib < n; ib += SB) {
            for (int jb = kend; jb < n; jb += SB) {
                int iend = std::min(ib + SB, n);
                int jend = std::min(jb + SB, n);
                
                for (int p = kb; p < kend; ++p) {
                    int pn = p * n;
                    for (int i = ib; i < iend; ++i) {
                        double Lip = L[i * n + p];
                        int in = i * n;
                        
                        // Loop unrolling
                        int j = jb;
                        for (; j + 3 < jend; j += 4) {
                            A_work[in + j]     -= Lip * U[pn + j];
                            A_work[in + j + 1] -= Lip * U[pn + j + 1];
                            A_work[in + j + 2] -= Lip * U[pn + j + 2];
                            A_work[in + j + 3] -= Lip * U[pn + j + 3];
                        }
                        for (; j < jend; ++j) {
                            A_work[in + j] -= Lip * U[pn + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// SAMO SIMD verzija
// ============================================================================
void LU_blokovska_V2_simd(double* A, double* L, double* U, int n) {
    const int B = 96;
    
    std::vector<double> A_work(n * n);
    std::memcpy(A_work.data(), A, n * n * sizeof(double));
    
    // SIMD inicijalizacija
    __m256d zero = _mm256_setzero_pd();
    int i = 0;
    for (; i + 3 < n * n; i += 4) {
        _mm256_storeu_pd(&L[i], zero);
        _mm256_storeu_pd(&U[i], zero);
    }
    for (; i < n * n; ++i) {
        L[i] = 0.0;
        U[i] = 0.0;
    }
    
    for (int kb = 0; kb < n; kb += B) {
        int kend = std::min(kb + B, n);
        
        // 1) Dijagonalni blok - sekvencijalno (zavisan kod)
        for (int k = kb; k < kend; ++k) {
            int kn = k * n;
            
            // U red
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                for (int p = kb; p < k; ++p)
                    sum += L[kn + p] * U[p * n + j];
                U[kn + j] = A_work[kn + j] - sum;
            }
            
            // L kolona
            L[kn + k] = 1.0;
            double inv_Ukk = 1.0 / U[kn + k];
            for (int i = k + 1; i < kend; ++i) {
                int in = i * n;
                double sum = 0.0;
                for (int p = kb; p < k; ++p)
                    sum += L[in + p] * U[p * n + k];
                L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
            }
        }
        
        // 2) U blokovi desno - SIMD optimizovano
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                
                // SIMD petlja - 4 elementa odjednom
                int j = colb;
                for (; j + 3 < colend; j += 4) {
                    __m256d sum_vec = _mm256_setzero_pd();
                    
                    for (int p = kb; p < k; ++p) {
                        __m256d Lkp_vec = _mm256_set1_pd(L[kn + p]);
                        int pn = p * n;
                        __m256d U_vec = _mm256_loadu_pd(&U[pn + j]);
                        sum_vec = _mm256_fmadd_pd(Lkp_vec, U_vec, sum_vec);
                    }
                    
                    __m256d A_vec = _mm256_loadu_pd(&A_work[kn + j]);
                    __m256d result = _mm256_sub_pd(A_vec, sum_vec);
                    _mm256_storeu_pd(&U[kn + j], result);
                }
                
                // Ostatak
                for (; j < colend; ++j) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p)
                        sum += L[kn + p] * U[p * n + j];
                    U[kn + j] = A_work[kn + j] - sum;
                }
            }
        }
        
        // 3) L blokovi ispod
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                double inv_Ukk = 1.0 / U[kn + k];
                __m256d inv_Ukk_vec = _mm256_set1_pd(inv_Ukk);
                
                for (int i = rowb; i < rowend; ++i) {
                    int in = i * n;
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p)
                        sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
                }
            }
        }
        
        // 4) Schur update - SIMD optimizovano
        const int SB = 64;
        for (int ib = kend; ib < n; ib += SB) {
            int iend = std::min(ib + SB, n);
            for (int jb = kend; jb < n; jb += SB) {
                int jend = std::min(jb + SB, n);
                
                for (int p = kb; p < kend; ++p) {
                    int pn = p * n;
                    for (int i = ib; i < iend; ++i) {
                        double Lip = L[i * n + p];
                        __m256d Lip_vec = _mm256_set1_pd(Lip);
                        int in = i * n;
                        
                        // SIMD petlja
                        int j = jb;
                        for (; j + 3 < jend; j += 4) {
                            __m256d A_vec = _mm256_loadu_pd(&A_work[in + j]);
                            __m256d U_vec = _mm256_loadu_pd(&U[pn + j]);
                            __m256d prod = _mm256_mul_pd(Lip_vec, U_vec);
                            __m256d result = _mm256_sub_pd(A_vec, prod);
                            _mm256_storeu_pd(&A_work[in + j], result);
                        }
                        
                        // Ostatak
                        for (; j < jend; ++j) {
                            A_work[in + j] -= Lip * U[pn + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// KOMBINOVANA OMP + SIMD verzija
// ============================================================================
void LU_blokovska_V2_omp_simd(double* A, double* L, double* U, int n) {
    const int B = 96;
    
    std::vector<double> A_work(n * n);
    std::memcpy(A_work.data(), A, n * n * sizeof(double));
    
    // Paralelizovana SIMD inicijalizacija
    int total = n * n;
    int vec_limit = total - 3; // Umesto i + 3 < total koristimo i < vec_limit
    
    #pragma omp parallel
    {
        __m256d zero = _mm256_setzero_pd();
        
        #pragma omp for nowait
        for (int i = 0; i < vec_limit; i += 4) {
            _mm256_storeu_pd(&L[i], zero);
            _mm256_storeu_pd(&U[i], zero);
        }
        
        #pragma omp for
        for (int i = (vec_limit / 4) * 4; i < total; ++i) {
            L[i] = 0.0;
            U[i] = 0.0;
        }
    }
    
    for (int kb = 0; kb < n; kb += B) {
        int kend = std::min(kb + B, n);
        
        // 1) Dijagonalni blok - sekvencijalno
        for (int k = kb; k < kend; ++k) {
            int kn = k * n;
            
            // U red
            for (int j = k; j < kend; ++j) {
                double sum = 0.0;
                for (int p = kb; p < k; ++p)
                    sum += L[kn + p] * U[p * n + j];
                U[kn + j] = A_work[kn + j] - sum;
            }
            
            // L kolona
            L[kn + k] = 1.0;
            double inv_Ukk = 1.0 / U[kn + k];
            for (int i = k + 1; i < kend; ++i) {
                int in = i * n;
                double sum = 0.0;
                for (int p = kb; p < k; ++p)
                    sum += L[in + p] * U[p * n + k];
                L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
            }
        }
        
        // 2) U blokovi desno - OMP + SIMD
        #pragma omp parallel for schedule(dynamic)
        for (int colb = kend; colb < n; colb += B) {
            int colend = std::min(colb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                
                // SIMD petlja
                int j = colb;
                int j_limit = colend - 3;
                for (; j < j_limit; j += 4) {
                    __m256d sum_vec = _mm256_setzero_pd();
                    
                    for (int p = kb; p < k; ++p) {
                        __m256d Lkp_vec = _mm256_set1_pd(L[kn + p]);
                        int pn = p * n;
                        __m256d U_vec = _mm256_loadu_pd(&U[pn + j]);
                        sum_vec = _mm256_fmadd_pd(Lkp_vec, U_vec, sum_vec);
                    }
                    
                    __m256d A_vec = _mm256_loadu_pd(&A_work[kn + j]);
                    __m256d result = _mm256_sub_pd(A_vec, sum_vec);
                    _mm256_storeu_pd(&U[kn + j], result);
                }
                
                // Ostatak
                for (; j < colend; ++j) {
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p)
                        sum += L[kn + p] * U[p * n + j];
                    U[kn + j] = A_work[kn + j] - sum;
                }
            }
        }
        
        // 3) L blokovi ispod - OMP + SIMD
        #pragma omp parallel for schedule(dynamic)
        for (int rowb = kend; rowb < n; rowb += B) {
            int rowend = std::min(rowb + B, n);
            for (int k = kb; k < kend; ++k) {
                int kn = k * n;
                double inv_Ukk = 1.0 / U[kn + k];
                
                for (int i = rowb; i < rowend; ++i) {
                    int in = i * n;
                    double sum = 0.0;
                    for (int p = kb; p < k; ++p)
                        sum += L[in + p] * U[p * n + k];
                    L[in + k] = (A_work[in + k] - sum) * inv_Ukk;
                }
            }
        }
        
        // 4) Schur update - OMP + SIMD
        const int SB = 64;
        
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ib = kend; ib < n; ib += SB) {
            for (int jb = kend; jb < n; jb += SB) {
                int iend = std::min(ib + SB, n);
                int jend = std::min(jb + SB, n);
                
                for (int p = kb; p < kend; ++p) {
                    int pn = p * n;
                    for (int i = ib; i < iend; ++i) {
                        double Lip = L[i * n + p];
                        __m256d Lip_vec = _mm256_set1_pd(Lip);
                        int in = i * n;
                        
                        // SIMD petlja
                        int j = jb;
                        int j_limit = jend - 3;
                        for (; j < j_limit; j += 4) {
                            __m256d A_vec = _mm256_loadu_pd(&A_work[in + j]);
                            __m256d U_vec = _mm256_loadu_pd(&U[pn + j]);
                            __m256d prod = _mm256_mul_pd(Lip_vec, U_vec);
                            __m256d result = _mm256_sub_pd(A_vec, prod);
                            _mm256_storeu_pd(&A_work[in + j], result);
                        }
                        
                        // Ostatak
                        for (; j < jend; ++j) {
                            A_work[in + j] -= Lip * U[pn + j];
                        }
                    }
                }
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
    auto t0_naivna = std::chrono::steady_clock::now();
	LU_naivna(A.data(), L.data(), U.data(), n);
	auto t1_naivna = std::chrono::steady_clock::now();
	std::cout << "LU_naivna: " << std::chrono::duration<double>(t1_naivna - t0_naivna).count() << " s\n";
    /* if (checkLU(A, L, U, n)) 
        std::cout << "LU_naivna verification: PASS\n";
    else 
        std::cout << "LU_naivna verification: FAIL\n"; */

	// --- LU_optimizovana ---
	std::vector<double> A2 = A, L2(n * n), U2(n * n);
	auto t0_opt = std::chrono::steady_clock::now();
	LU_optimizovana(A2.data(), L2.data(), U2.data(), n);
	auto t1_opt = std::chrono::steady_clock::now();
	std::cout << "LU_optimizovana: " << std::chrono::duration<double>(t1_opt - t0_opt).count() << " s\n";
    /* if (checkLU(A2, L2, U2, n)) 
        std::cout << "LU_optimizovana verification: PASS\n";
    else 
        std::cout << "LU_optimizovana verification: FAIL\n"; */

	// --- LU_blokovska_V1 ---
	std::vector<double> A3 = A, L3(n * n), U3(n * n);
	auto t0_b1 = std::chrono::steady_clock::now();
	LU_blokovska_V1(A3.data(), L3.data(), U3.data(), n);
	auto t1_b1 = std::chrono::steady_clock::now();
	std::cout << "LU_blokovska_V1: " << std::chrono::duration<double>(t1_b1 - t0_b1).count() << " s\n";
    
    /*if (checkLU(A3, L3, U3, n))
        std::cout << "LU_blokovska_V1 verification: PASS\n";
    else 
        std::cout << "LU_blokovska_V1 verification: FAIL\n";*/

	// --- LU_blokovska_V2 ---
    std::vector<double> A4 = A, L4(n * n), U4(n * n);
	auto t0_b2 = std::chrono::steady_clock::now();
	LU_blokovska_V2(A4.data(), L4.data(), U4.data(), n);
	auto t1_b2 = std::chrono::steady_clock::now();
	std::cout << "LU_blokovska_V2: " << std::chrono::duration<double>(t1_b2 - t0_b2).count() << " s\n";
    /*if (checkLU(A4, L4, U4, n))
        std::cout << "LU_blokovska_V2 verification: PASS\n";
    else 
        std::cout << "LU_blokovska_V2 verification: FAIL\n";*/

    // --- LU_blokovska_V1_omp ---
    std::vector<double> A5 = A, L5(n * n), U5(n * n);
	auto t0_b1omp = std::chrono::steady_clock::now();
	LU_blokovska_V1_omp(A5.data(), L5.data(), U5.data(), n);
	auto t1_b1omp = std::chrono::steady_clock::now();
	std::cout << "LU_blokovska_V1_omp: " << std::chrono::duration<double>(t1_b1omp - t0_b1omp).count() << " s\n";
    
   /*if (checkLU(A5, L5, U5, n))
        std::cout << "LU_blokovska_V1_omp verification: PASS\n";
    else 
        std::cout << "LU_blokovska_V1_omp verification: FAIL\n";*/

    // --- LU_blokovska_V2_omp ---
    std::vector<double> A6 = A, L6(n * n), U6(n * n);
	auto t0_b2omp = std::chrono::steady_clock::now();
	LU_blokovska_V2_omp(A6.data(), L6.data(), U6.data(), n);
	auto t1_b2omp = std::chrono::steady_clock::now();
	std::cout << "LU_blokovska_V2_omp: " << std::chrono::duration<double>(t1_b2omp - t0_b2omp).count() << " s\n";
    
	/*if (checkLU(A6, L6, U6, n))
        std::cout << "LU_blokovska_V2_omp verification: PASS\n";
    else 
        std::cout << "LU_blokovska_V2_omp verification: FAIL\n";*/

    // --- LU_blokovska_V2_SIMD ---
    std::vector<double> A7 = A, L7(n * n), U7(n * n);
	auto t0_b2simd = std::chrono::steady_clock::now();
	LU_blokovska_V2_simd(A7.data(), L7.data(), U7.data(), n);
	auto t1_b2simd = std::chrono::steady_clock::now();
	std::cout << "LU_blokovska_V2_simd: " << std::chrono::duration<double>(t1_b2simd - t0_b2simd).count() << " s\n";
    
	/*if (checkLU(A7, L7, U7, n))
        std::cout << "LU_blokovska_V2_simd verification: PASS\n";
    else 
        std::cout << "LU_blokovska_V2_simd verification: FAIL\n";*/

    // --- LU_blokovska_V2_omp ---
    std::vector<double> A8 = A, L8(n * n), U8(n * n);
	auto t0_b2ompsimd = std::chrono::steady_clock::now();
	LU_blokovska_V2_omp_simd(A8.data(), L8.data(), U8.data(), n);
	auto t1_b2ompsimd = std::chrono::steady_clock::now();
	std::cout << "LU_blokovska_V2_omp_simd: " << std::chrono::duration<double>(t1_b2ompsimd - t0_b2ompsimd).count() << " s\n";
    
	/*if (checkLU(A8, L8, U8, n))
        std::cout << "LU_blokovska_V2_omp_simd verification: PASS\n";
    else 
        std::cout << "LU_blokovska_V2_omp_simd verification: FAIL\n"; */
    
    /* for (int B : {16, 32, 64, 96, 128, 256}) {
		auto start = std::chrono::high_resolution_clock::now();
		LU_blokovska_V1(A.data(), L.data(), U.data(), n, B);
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "B = " << B << ": " << std::chrono::duration<double>(end - start).count() << " s\n";
    } */
       // ===================== SPEEDUP =====================
    double T_naivna = std::chrono::duration<double>(t1_naivna - t0_naivna).count();
    double T_opt    = std::chrono::duration<double>(t1_opt    - t0_opt).count();
    double T_b1     = std::chrono::duration<double>(t1_b1     - t0_b1).count();
    double T_b2     = std::chrono::duration<double>(t1_b2     - t0_b2).count();
    double T_b1omp  = std::chrono::duration<double>(t1_b1omp  - t0_b1omp).count();
    double T_b2omp  = std::chrono::duration<double>(t1_b2omp  - t0_b2omp).count();
    double T_b2simd  = std::chrono::duration<double>(t1_b2simd  - t0_b2simd).count();
    double T_b2ompsimd  = std::chrono::duration<double>(t1_b2ompsimd  - t0_b2ompsimd).count();


    std::cout << "\n===== SPEEDUP (T1 = LU_naivna) =====\n";
    std::cout << "Speedup_optimizovana    = " << T_naivna / T_opt   << "\n";
    std::cout << "Speedup_blokovska_V1    = " << T_naivna / T_b1    << "\n";
    std::cout << "Speedup_blokovska_V2    = " << T_naivna / T_b2    << "\n";
    std::cout << "Speedup_blokovska_V1_omp= " << T_naivna / T_b1omp << "\n";
    std::cout << "Speedup_blokovska_V2_omp= " << T_naivna / T_b2omp << "\n";
    std::cout << "Speedup_blokovska_V2_simd= " << T_naivna / T_b2simd << "\n";
    std::cout << "Speedup_blokovska_V2_omp_simd= " << T_naivna / T_b2ompsimd << "\n";

    std::cout << "====================================\n";

	return 0;
}