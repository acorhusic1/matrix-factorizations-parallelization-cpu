#include "QR_paralelizovana.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <cblas.h>
#include <lapacke.h>

std::vector<double> GenerateRandomMatrix(int rows, int cols) {
    std::vector<double> m(rows * cols);
    // Column-Major: Popunjavanje po kolonama
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            m[j * rows + i] = (double)rand() / RAND_MAX * 200.0 - 100.0;
        }
    }
    return m;
}

// OPTIMIZOVANO: Koristi BLAS za množenje matrica (MNOGO brže!)
std::vector<double> MatrixMultiplyBLAS(const std::vector<double>& A, int m, int k,
    const std::vector<double>& B, int k2, int n) {
    if (k != k2) throw std::domain_error("Incompatible matrix dimensions");

    std::vector<double> C(m * n, 0.0);

    // cblas_dgemm: C = alpha*A*B + beta*C
    // CblasColMajor: Matrice su u Column-Major formatu
    // CblasNoTrans: Koristimo A i B bez transpozicije
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,           // Dimenzije: m×k * k×n = m×n
        1.0,               // alpha = 1.0
        A.data(), m,       // A matrica, leading dimension = m
        B.data(), k,       // B matrica, leading dimension = k
        0.0,               // beta = 0.0
        C.data(), m);      // C matrica, leading dimension = m

    return C;
}

// OPTIMIZOVANO: Koristi BLAS za Frobenius normu razlike ||A - Q*R||_F
double MatrixDifferenceFrobenius(const std::vector<double>& A,
    const std::vector<double>& QR,
    int rows, int cols) {
    // Računanje: ||A - QR||_F = sqrt(sum((A[i] - QR[i])^2))
    double sum = 0.0;

    // SIMD-optimized petlja
#pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < A.size(); i++) {
        double diff = A[i] - QR[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

// Brza provjera korištenjem relativne greške
bool MatrixEqualToFast(const std::vector<double>& A,
    const std::vector<double>& QR,
    int rows, int cols, double eps) {
    // Računanje Frobenius norme originalne matrice ||A||_F
    double norm_A = cblas_dnrm2(A.size(), A.data(), 1);

    // Računanje ||A - QR||_F
    double norm_diff = MatrixDifferenceFrobenius(A, QR, rows, cols);

    // Relativna greška: ||A - QR||_F / ||A||_F
    double relative_error = norm_diff / norm_A;

    return relative_error < eps;
}

int main() {
    // Postavi seed za random brojeve i preciznost ispisa
    srand(time(0));
    std::cout << std::fixed << std::setprecision(8);
    const double CHECK_EPS = 1e-9; // Strožiji epsilon za relativnu grešku

    // Ispis informacija o OpenMP i BLAS konfiguraciji
    std::cout << "=========================================================" << std::endl;
    std::cout << " PARALELNA QR FAKTORIZACIJA (OpenMP + BLAS)" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Broj dostupnih threadova: " << omp_get_max_threads() << std::endl;

    // Provjeri koja BLAS biblioteka se koristi
#ifdef OPENBLAS_VERSION
    std::cout << "BLAS biblioteka: OpenBLAS " << OPENBLAS_VERSION << std::endl;
#else
    std::cout << "BLAS biblioteka: Detektovana" << std::endl;
#endif

    std::cout << "=========================================================" << std::endl;

    // --- Test performansi ---
    const int N = 1000;
    const int TEST_RUNS = 10;
    std::cout << "\n\n--- Test performansi (" << N << "x" << N << " matrica) ---" << std::endl;

    double total_duration_moja = 0.0;
    double total_duration_lapack = 0.0;
    double total_duration_verification = 0.0;
    int failed_checks = 0;
    int valid_runs = 0;

    for (int i = 0; i < TEST_RUNS; ++i) {
        std::vector<double> A_large = GenerateRandomMatrix(N, N);

        // ===== TEST 1: MOJA PARALELNA IMPLEMENTACIJA =====
        std::vector<double> QRmat, R_diag;

        auto start_moja = std::chrono::high_resolution_clock::now();
        try {
            QRFactorizationParallel(A_large, N, N, QRmat, R_diag);
        }
        catch (const std::exception& e) {
            std::cerr << "Upozorenje: Pokretanje " << i + 1 << " preskočeno: " << e.what() << std::endl;
            continue;
        }
        auto end_moja = std::chrono::high_resolution_clock::now();
        double duration_moja = std::chrono::duration<double>(end_moja - start_moja).count();
        total_duration_moja += duration_moja;

        // ===== TEST 2: LAPACK IMPLEMENTACIJA (za poređenje) =====
        std::vector<double> A_lapack = A_large; // Kopija za LAPACK
        std::vector<double> tau(N);

        auto start_lapack = std::chrono::high_resolution_clock::now();
        int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, N, N, A_lapack.data(), N, tau.data());
        auto end_lapack = std::chrono::high_resolution_clock::now();
        double duration_lapack = std::chrono::duration<double>(end_lapack - start_lapack).count();
        total_duration_lapack += duration_lapack;

        if (info != 0) {
            std::cerr << "LAPACK dgeqrf neuspješan (info=" << info << ")" << std::endl;
        }

        valid_runs++;

        std::cout << "Pokretanje " << i + 1 << ": "
            << "Moja=" << duration_moja * 1000.0 << " ms | "
            << "LAPACK=" << duration_lapack * 1000.0 << " ms | ";

        // ===== BRZA PROVJERA REZULTATA (OPTIMIZOVANO!) =====
        auto start_verify = std::chrono::high_resolution_clock::now();

        std::vector<double> Q = GetQParallel(QRmat, N, N);
        std::vector<double> R = GetRParallel(QRmat, R_diag, N, N);

        // Koristi BLAS za množenje (MNOGO brže!)
        std::vector<double> A_reconstructed = MatrixMultiplyBLAS(Q, N, N, R, N, N);

        // Brza provjera sa relativnom greškom
        bool is_correct = MatrixEqualToFast(A_large, A_reconstructed, N, N, CHECK_EPS);

        auto end_verify = std::chrono::high_resolution_clock::now();
        double duration_verify = std::chrono::duration<double>(end_verify - start_verify).count();
        total_duration_verification += duration_verify;

        if (is_correct) {
            std::cout << "✓ OK";
        }
        else {
            std::cout << "✗ GREŠKA";
            failed_checks++;
        }

        std::cout << " (verif: " << duration_verify * 1000.0 << " ms)" << std::endl;
        std::cout << std::endl;
    }

    // --- Ispis konačnih rezultata ---
    if (valid_runs > 0) {
        double mean_moja = total_duration_moja / valid_runs;
        double mean_lapack = total_duration_lapack / valid_runs;
        double mean_verify = total_duration_verification / valid_runs;
        double speedup = mean_lapack / mean_moja;

        std::cout << "\n============================================================" << std::endl;
        std::cout << " FINALNI REZULTATI" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Srednje vrijeme (MOJA impl):       " << mean_moja * 1000.0 << " ms" << std::endl;
        std::cout << "Srednje vrijeme (LAPACK):          " << mean_lapack * 1000.0 << " ms" << std::endl;
        std::cout << "Srednje vrijeme (verifikacija):    " << mean_verify * 1000.0 << " ms" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;

        if (speedup >= 1.0) {
            std::cout << "MOJA implementacija je " << speedup << "x BRŽA od LAPACK! 🚀" << std::endl;
        }
        else {
            std::cout << "LAPACK je " << (1.0 / speedup) << "x brži od MOJE impl." << std::endl;
        }

        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "Ukupan broj validnih pokretanja:  " << valid_runs << std::endl;
        std::cout << "Broj pogrešnih rezultata:          " << failed_checks << std::endl;
        std::cout << "Postotak pogrešnih:                "
            << ((double)failed_checks / valid_runs * 100.0) << " %" << std::endl;
        std::cout << "============================================================" << std::endl;
    }

    return 0;
}