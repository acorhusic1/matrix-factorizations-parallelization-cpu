#ifndef QR_SEKVENCIJALNA_BLOKIRANA_H
#define QR_SEKVENCIJALNA_BLOKIRANA_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm> // za std::min
#include <cblas.h>
#include "QR_sekvencijalna.h" // Uključujemo neblokirani header da bismo dobili GLOBAL_EPSILON i At funkcije

// Konstante specifične za blokiranje
const int BLOCK_SIZE = 64;

// --- QR Blokirane Glavne Funkcije (Prototipi) ---
// Napomena: Ove funkcije koriste At() i GLOBAL_EPSILON iz QR_sekvencijalna.h
void QRFactorization_Blocked(const std::vector<double>& A, int rows, int cols,
    std::vector<double>& QRmat, std::vector<double>& R_diag, int block_size);
std::vector<double> GetQ_Blocked(const std::vector<double>& QRmat, int rows, int cols);
std::vector<double> GetR_Blocked(const std::vector<double>& QRmat, const std::vector<double>& R_diag, int rows, int cols);

#endif // QR_SEKVENCIJALNA_BLOKIRANA_H