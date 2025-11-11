#include "QR_sekvencijalna.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>

std::vector<double> GenerateRandomMatrix(int rows, int cols) {
    std::vector<double> m(rows * cols);
    // Postavlja elemente iz opsega [-100, 100]
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i * cols + j] = (double)rand() / RAND_MAX * 200.0 - 100.0;
        }
    }
    return m;
}

void CreateMatrixFromList(std::initializer_list<std::vector<double>> l, std::vector<double>& M, int& rows, int& cols) {
    if (l.size() == 0) throw std::range_error("Bad dimension");
    rows = l.size();
    cols = l.begin()->size();
    if (cols == 0) throw std::range_error("Bad dimension");
    M.resize(rows * cols);

    int k = 0;
    for (const auto& red : l) {
        if ((int)red.size() != cols) throw std::logic_error("Bad matrix");
        for (double val : red) {
            M[k++] = val;
        }
    }
}

int main() {

    // Postavi seed za random brojeve i preciznost ispisa
    srand(time(0));
    std::cout << std::fixed << std::setprecision(8);
    const double CHECK_EPS = 1e-6; // Epsilon za provjeru jednakosti A = Q*R

    // --- Test validacije (3x3 matrica) ---
    std::cout << "\n--- Test 3x3 Matrice i QR Faktorizacije (REKONSTRUKCIJA A = Q * R) ---" << std::endl;

    // Matrica A za koju se računa Q i R
    std::vector<double> A_small_vec;
    int Arows, Acols;
    CreateMatrixFromList({ {0, 3, 2}, {4, 6, 1}, {3, 1, 7} }, A_small_vec, Arows, Acols);

    try {
        std::cout << "Originalna matrica A (3x3):" << std::endl;
        MatrixPrint(A_small_vec, Arows, Acols);

        // Izracunavanje Q i R
        std::vector<double> QRmat, R_diag;
        QRFactorization(A_small_vec, Arows, Acols, QRmat, R_diag);

        std::vector<double> Q = GetQ(QRmat, Arows, Acols);
        std::vector<double> R = GetR(QRmat, R_diag, Arows, Acols);

        std::cout << "\nMatrica Q (Ortogonalna):" << std::endl;
        MatrixPrint(Q, Arows, Acols);
        std::cout << "\nMatrica R (Gornje trougaona):" << std::endl;
        MatrixPrint(R, Arows, Acols);

        // Rekonstrukcija A = Q * R
        std::vector<double> A_reconstructed = MatrixMultiply(Q, Arows, Acols, R, Arows, Acols);
        std::cout << "\nRekonstruisana matrica Q * R (Provjera):" << std::endl;
        MatrixPrint(A_reconstructed, Arows, Acols);

        if (MatrixEqualTo(A_small_vec, Arows, Acols, A_reconstructed, Arows, Acols, CHECK_EPS)) {
            std::cout << "\n-> Rekonstrukcija USPJESNA: Q * R je jednako A." << std::endl;
        }
        else {
            std::cout << "\n-> Rekonstrukcija NEUSPJESNA: Q * R nije jednako A." << std::endl;
        }

    }
    catch (const std::exception& e) {
        std::cout << "\nIzuzetak u validacionom testu: " << e.what() << std::endl;
    }


    // --- Test performansi ---
    const int N = 1000;
    const int TEST_RUNS = 10;
    std::cout << "\n\n--- Test performansi (" << N << "x" << N << " matrica) ---" << std::endl;

    double total_duration = 0.0;
    int failed_checks = 0; // Brojač pogrešnih rezultata

    for (int i = 0; i < TEST_RUNS; ++i) {
        std::vector<double> A_large = GenerateRandomMatrix(N, N);
        std::vector<double> QRmat, R_diag;

        // --- START TIMING (Samo QR faktorizacija) ---
        auto start = std::chrono::high_resolution_clock::now();

        try {
            QRFactorization(A_large, N, N, QRmat, R_diag);
        }
        catch (const std::exception& e) {
            std::cerr << "Upozorenje: Pokretanje " << i + 1 << " preskočeno zbog izuzetka: " << e.what() << std::endl;
            // Preskočene izuzetke (singularne matrice) ne računamo ni u uspjeh ni u neuspjeh testa
            continue;
        }

        auto end = std::chrono::high_resolution_clock::now();
        // --- END TIMING ---
        std::chrono::duration<double> duration = end - start;

        total_duration += duration.count();
        std::cout << "Pokretanje " << i + 1 << ": " << duration.count() * 1000.0 << " ms. ";

        // --- PROVJERA REKONSTRUKCIJE (NE ULAZI U VRIJEME) ---
        try {
            std::vector<double> Q = GetQ(QRmat, N, N);
            std::vector<double> R = GetR(QRmat, R_diag, N, N);
            std::vector<double> A_reconstructed = MatrixMultiply(Q, N, N, R, N, N);

            if (MatrixEqualTo(A_large, N, N, A_reconstructed, N, N, CHECK_EPS)) {
                std::cout << "Rezultat ispravan." << std::endl;
            }
            else {
                std::cout << "Pogresan rezultat." << std::endl;
                failed_checks++;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Provjera rekonstrukcije neuspješna zbog izuzetka: " << e.what() << std::endl;
            failed_checks++; // Računaj kao neuspjeh
        }
    }

    // --- Ispis konačnih rezultata ---
    double mean_duration = total_duration / (TEST_RUNS - (double)failed_checks);
    if (TEST_RUNS - failed_checks == 0) mean_duration = 0; // Izbjegavanje dijeljenja s nulom

    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "Srednje vrijeme (MEAN) QR faktorizacije: " << mean_duration * 1000.0 << " ms" << std::endl;

    double failure_rate = (double)failed_checks / TEST_RUNS * 100.0;
    std::cout << "Ukupan broj pokretanja: " << TEST_RUNS << std::endl;
    std::cout << "Broj pogresnih rezultata (A != Q*R): " << failed_checks << std::endl;
    std::cout << "Postotak pogresnih izracunavanja: " << failure_rate << " %" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    return 0;
}