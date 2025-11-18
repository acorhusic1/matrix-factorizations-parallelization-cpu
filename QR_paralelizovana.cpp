#include "QR_paralelizovana.h"
#include <chrono>
#include <omp.h>

// --- Implementacija Matričnih Funkcija ---

std::vector<double> MatrixMultiplyParallel(const std::vector<double>& M1, int M1rows, int M1cols,
    const std::vector<double>& M2, int M2rows, int M2cols) {
    if (M1cols != M2rows) throw std::domain_error("Incompatible formats (Matrix dimensions for multiplication)");

    int Rrows = M1rows;
    int Rcols = M2cols;
    std::vector<double> rez(Rrows * Rcols, 0.0);

    // Optimizovano množenje za Column-Major (j-k-i petlje)
    for (int j = 0; j < Rcols; ++j) {
        for (int k = 0; k < M1cols; ++k) {
            double r = AtParallel(M2, M2rows, k, j);
            for (int i = 0; i < Rrows; ++i) {
                AtParallel(rez, Rrows, i, j) += AtParallel(M1, M1rows, i, k) * r;
            }
        }
    }
    return rez;
}

void MatrixPrintParallel(const std::vector<double>& M, int rows, int cols, int width, double eps, int precision) {
    std::cout << std::fixed << std::setprecision(precision);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = AtParallel(M, rows, i, j);
            if (std::abs(val) < eps) std::cout << std::setw(width) << 0.0;
            else std::cout << std::setw(width) << val;
        }
        std::cout << std::endl;
    }
}

bool MatrixEqualToParallel(const std::vector<double>& M1, int M1rows, int M1cols,
    const std::vector<double>& M2, int M2rows, int M2cols,
    double eps) {
    if (M1rows != M2rows || M1cols != M2cols) return false;
    for (size_t i = 0; i < M1.size(); i++)
        if (std::abs(M1[i] - M2[i]) > eps) return false;
    return true;
}

// --- Implementacija QR Glavnih Funkcija ---

void QRFactorizationParallel(const std::vector<double>& A, int rows, int cols,
    std::vector<double>& QRmat, std::vector<double>& R_diag) {

    if (rows < cols) throw std::domain_error("Invalid matrix format: rows < cols");

    double total_durationV1 = 0.0;
    double total_durationV2 = 0.0;
    double total_durationV3 = 0.0;
    double total_durationU1 = 0.0;
    double total_durationU2 = 0.0;

    QRmat = A; // Kopiraj A u QRmat 
    R_diag.resize(cols);

    for (int k = 0; k < cols; k++) {
        // 1. Računanje norme k-te kolone A(k:end, k)
        double s = 0;

        auto startV1 = std::chrono::high_resolution_clock::now();

        // Pristup je Stride-1 (sekvencijalan)
        for (int i = k; i < rows; i++) {
            auto temp = AtParallel(QRmat, rows, i, k);
            s += temp * temp;
        }

        auto endV1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationV1 = endV1 - startV1;
        total_durationV1 += durationV1.count();

        s = std::sqrt(s);

        if (AtParallel(QRmat, rows, k, k) < 0) s = -s;

        double v1 = AtParallel(QRmat, rows, k, k) + s;
        AtParallel(QRmat, rows, k, k) = v1;
        R_diag[k] = -s;

        // 2. Računanje kvadrata norme Householderovog vektora ||v||^2
        double v_norm_sq = v1 * v1;

        auto startV2 = std::chrono::high_resolution_clock::now();

        // Pristup je Stride-1 (sekvencijalan)
        for (int i = k + 1; i < rows; i++) {
            auto temp = AtParallel(QRmat, rows, i, k);
            v_norm_sq += temp * temp;
        }

        auto endV2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> durationV2 = endV2 - startV2;
        total_durationV2 += durationV2.count();

        // 3. Primjena transformacije na preostalu podmatricu (Kolone j > k)
        if (std::abs(v_norm_sq) > GLOBAL_EPSILON_PARALLEL * GLOBAL_EPSILON_PARALLEL) {

            auto startV3 = std::chrono::high_resolution_clock::now();

            // PARALELIZACIJA: Vanjska j-petlja - svaki thread obrađuje različite kolone
            // schedule(dynamic) omogućava bolju load balancing jer kolone mogu imati različite workload
#pragma omp parallel for schedule(dynamic)
            for (int j = k + 1; j < cols; j++) {
                double u = 0;

                // Unutrašnja petlja 1: Skalarni proizvod (Stride-1 pristup)
                for (int i = k; i < rows; i++)
                    u += AtParallel(QRmat, rows, i, k) * AtParallel(QRmat, rows, i, j);

                u *= (2.0 / v_norm_sq);

                // Unutrašnja petlja 2: SAXPY operacija (Stride-1 pristup)
                for (int i = k; i < rows; i++)
                    AtParallel(QRmat, rows, i, j) -= u * AtParallel(QRmat, rows, i, k);
            }
            auto endV3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> durationV3 = endV3 - startV3;
            total_durationV3 += durationV3.count();
        }
        else if (std::abs(R_diag[k]) < GLOBAL_EPSILON_PARALLEL) {
            throw std::domain_error("Matrix is singular");
        }
    }

    std::cout << "Ukupno trajanje vanjska petlja 1: " << total_durationV1 * 1000.0 << " ms. " << std::endl;
    std::cout << "Ukupno trajanje vanjska petlja 2: " << total_durationV2 * 1000.0 << " ms. " << std::endl;
    std::cout << "Ukupno trajanje vanjska petlja 3: " << total_durationV3 * 1000.0 << " ms. " << std::endl;
}

// Pomoćna metoda: Primjena Householderove refleksije H_k na vektor v
void ApplyHkParallel(const std::vector<double>& QRmat, int rows, int cols, std::vector<double>& res, int k) {

    double v_norm_sq = AtParallel(QRmat, rows, k, k) * AtParallel(QRmat, rows, k, k);
    for (int i = k + 1; i < rows; i++)
        v_norm_sq += AtParallel(QRmat, rows, i, k) * AtParallel(QRmat, rows, i, k);

    if (std::abs(v_norm_sq) < GLOBAL_EPSILON_PARALLEL * GLOBAL_EPSILON_PARALLEL) return;

    double s = AtParallel(QRmat, rows, k, k) * res[k];
    for (int i = k + 1; i < rows; i++)
        s += AtParallel(QRmat, rows, i, k) * res[i];

    s *= (2.0 / v_norm_sq);

    res[k] -= s * AtParallel(QRmat, rows, k, k);
    for (int i = k + 1; i < rows; i++)
        res[i] -= s * AtParallel(QRmat, rows, i, k);
}

std::vector<double> GetQParallel(const std::vector<double>& QRmat, int rows, int cols) {
    int Qrows = rows;
    int Qcols = rows;
    std::vector<double> Q(Qrows * Qcols, 0.0);
    for (int i = 0; i < rows; i++) AtParallel(Q, Qrows, i, i) = 1.0; // Jedinična matrica (Column-Major)

    for (int k = cols - 1; k >= 0; k--) {
        for (int j = 0; j < rows; ++j) {
            std::vector<double> col(rows);
            // Izdvajanje kolone j iz Q (Stride-1 pristup)
            for (int i = 0; i < rows; ++i) col[i] = AtParallel(Q, Qrows, i, j);

            ApplyHkParallel(QRmat, rows, cols, col, k);

            // Vraćanje kolone u matricu (Stride-1 pristup)
            for (int i = 0; i < rows; ++i) AtParallel(Q, Qrows, i, j) = col[i];
        }
    }
    return Q;
}

std::vector<double> GetRParallel(const std::vector<double>& QRmat, const std::vector<double>& R_diag, int rows, int cols) {
    std::vector<double> R(rows * cols, 0.0);
    for (int j = 0; j < cols; j++) { // Iteriramo po kolonama (Column-Major)
        for (int i = 0; i < rows; i++) {
            if (i == j) AtParallel(R, rows, i, j) = R_diag[i];
            else if (j > i) AtParallel(R, rows, i, j) = AtParallel(QRmat, rows, i, j);
        }
    }
    return R;
}