#ifndef QR_PARALELIZOVANA_H
#define QR_PARALELIZOVANA_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>

// Konstanta za nulti prag
const double GLOBAL_EPSILON_PARALLEL = 1e-12;

// --- Pomoćne Inline Funkcije za Rad sa Matričnom Memorijom ---
/*
    Brz pristup elementu M[i][j] u Column-Major rasporedu.
    Pristup: index = j * rows + i
*/
inline double& AtParallel(std::vector<double>& M, int rows, int i, int j) {
    return M[j * rows + i];
}

inline const double& AtParallel(const std::vector<double>& M, int rows, int i, int j) {
    return M[j * rows + i];
}

// --- Matrične Funkcije (Prototipi) ---
std::vector<double> MatrixMultiplyParallel(const std::vector<double>& M1, int M1rows, int M1cols,
    const std::vector<double>& M2, int M2rows, int M2cols);

void MatrixPrintParallel(const std::vector<double>& M, int rows, int cols,
    int width = 10, double eps = 1e-9, int precision = 3);

bool MatrixEqualToParallel(const std::vector<double>& M1, int M1rows, int M1cols,
    const std::vector<double>& M2, int M2rows, int M2cols, double eps);

// --- QR Glavne Funkcije (Prototipi) ---
void QRFactorizationParallel(const std::vector<double>& A, int rows, int cols,
    std::vector<double>& QRmat, std::vector<double>& R_diag);

void ApplyHkParallel(const std::vector<double>& QRmat, int rows, int cols,
    std::vector<double>& res, int k);

std::vector<double> GetQParallel(const std::vector<double>& QRmat, int rows, int cols);

std::vector<double> GetRParallel(const std::vector<double>& QRmat,
    const std::vector<double>& R_diag, int rows, int cols);

#endif // QR_PARALELIZOVANA_H