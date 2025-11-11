#include "QR_sekvencijalna.h"

// --- Implementacija Matričnih Funkcija ---

std::vector<double> MatrixMultiply(const std::vector<double>& M1, int M1rows, int M1cols,
    const std::vector<double>& M2, int M2rows, int M2cols) {
    if (M1cols != M2rows) throw std::domain_error("Incompatible formats (Matrix dimensions for multiplication)");

    int Rrows = M1rows;
    int Rcols = M2cols;
    std::vector<double> rez(Rrows * Rcols);

    for (int i = 0; i < Rrows; i++) {
        for (int j = 0; j < Rcols; j++) {
            double suma = 0;
            for (int k = 0; k < M1cols; k++) {
                suma += At(M1, M1cols, i, k) * At(M2, M2cols, k, j);
            }
            At(rez, Rcols, i, j) = suma;
        }
    }
    return rez;
}

void MatrixPrint(const std::vector<double>& M, int rows, int cols, int width, double eps, int precision) {
    std::cout << std::fixed << std::setprecision(precision);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = At(M, cols, i, j);
            if (std::abs(val) < eps) std::cout << std::setw(width) << 0.0;
            else std::cout << std::setw(width) << val;
        }
        std::cout << std::endl;
    }
}

bool MatrixEqualTo(const std::vector<double>& M1, int M1rows, int M1cols,
    const std::vector<double>& M2, int M2rows, int M2cols,
    double eps) {
    if (M1rows != M2rows || M1cols != M2cols) return false;
    for (size_t i = 0; i < M1.size(); i++)
        if (std::abs(M1[i] - M2[i]) > eps) return false;
    return true;
}

// --- Implementacija QR Glavnih Funkcija ---

void QRFactorization(const std::vector<double>& A, int rows, int cols,
    std::vector<double>& QRmat, std::vector<double>& R_diag) {

    if (rows < cols) throw std::domain_error("Invalid matrix format: rows < cols");

    QRmat = A; // Kopiraj A u QRmat (Householderova transformacija se vrši in-place)
    R_diag.resize(cols);

    for (int k = 0; k < cols; k++) {
        // 1. Računanje norme k-te kolone A(k:end, k)
        double s = 0;
        for (int i = k; i < rows; i++) s += At(QRmat, cols, i, k) * At(QRmat, cols, i, k);
        s = std::sqrt(s);

        // Određivanje predznaka za stabilnost
        if (At(QRmat, cols, k, k) < 0) s = -s;

        // Računanje prvog elementa Householderovog vektora v
        double v1 = At(QRmat, cols, k, k) + s;
        At(QRmat, cols, k, k) = v1; // Pohrani v1 u QRmat[k][k]
        R_diag[k] = -s;

        // 2. Računanje kvadrata norme Householderovog vektora ||v||^2
        double v_norm_sq = v1 * v1;
        for (int i = k + 1; i < rows; i++) v_norm_sq += At(QRmat, cols, i, k) * At(QRmat, cols, i, k);

        // 3. Primjena transformacije na preostalu podmatricu (Kolone j > k)
        if (std::abs(v_norm_sq) > GLOBAL_EPSILON * GLOBAL_EPSILON) {

            // Loop po kolonama desno od k
            for (int j = k + 1; j < cols; j++) {

                // Skalarni proizvod u = v^T * a_j 
                double u = 0;
                for (int i = k; i < rows; i++) u += At(QRmat, cols, i, k) * At(QRmat, cols, i, j);

                u *= (2.0 / v_norm_sq);

                // Ažuriranje a_j = a_j - u * v 
                for (int i = k; i < rows; i++) At(QRmat, cols, i, j) -= u * At(QRmat, cols, i, k);
            }
        }
        else if (std::abs(R_diag[k]) < GLOBAL_EPSILON) {
            throw std::domain_error("Matrix is singular");
        }
    }
}

// Pomoćna metoda: Primjena Householderove refleksije H_k na vektor v
void ApplyHk(const std::vector<double>& QRmat, int rows, int cols, std::vector<double>& res, int k) {

    // Izračunaj ||v||^2 Householderovog vektora v pohranjenog u k-toj koloni QRmat
    double v_norm_sq = At(QRmat, cols, k, k) * At(QRmat, cols, k, k);
    for (int i = k + 1; i < rows; i++) v_norm_sq += At(QRmat, cols, i, k) * At(QRmat, cols, i, k);

    if (std::abs(v_norm_sq) < GLOBAL_EPSILON * GLOBAL_EPSILON) return;

    // Skalarni proizvod s = v^T * res (od k-tog elementa na dalje)
    double s = 0;
    s += At(QRmat, cols, k, k) * res[k];
    for (int i = k + 1; i < rows; i++) s += At(QRmat, cols, i, k) * res[i];

    s *= (2.0 / v_norm_sq);

    // Ažuriranje res = res - s * v (In-place)
    res[k] -= s * At(QRmat, cols, k, k);
    for (int i = k + 1; i < rows; i++) res[i] -= s * At(QRmat, cols, i, k);
}

std::vector<double> GetQ(const std::vector<double>& QRmat, int rows, int cols) {
    int Qrows = rows;
    int Qcols = rows;
    std::vector<double> Q(Qrows * Qcols, 0.0);
    for (int i = 0; i < rows; i++) At(Q, Qcols, i, i) = 1.0; // Jedinična matrica (početni Q)

    // Primjenjujemo H_k obrnutim redoslijedom (H_n ... H_1) na kolone Jedinične matrice
    for (int k = cols - 1; k >= 0; k--) {
        for (int j = 0; j < rows; ++j) {
            // Izdvajanje kolone j iz Q
            std::vector<double> col(rows);
            for (int i = 0; i < rows; ++i) col[i] = At(Q, Qcols, i, j);

            // Primjena H_k na kolonu (Q <- H_k * Q)
            ApplyHk(QRmat, rows, cols, col, k);

            // Vraćanje kolone u matricu
            for (int i = 0; i < rows; ++i) At(Q, Qcols, i, j) = col[i];
        }
    }
    return Q;
}

std::vector<double> GetR(const std::vector<double>& QRmat, const std::vector<double>& R_diag, int rows, int cols) {
    std::vector<double> R(rows * cols, 0.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == j) R[i * cols + j] = R_diag[i];
            else if (j > i) R[i * cols + j] = At(QRmat, cols, i, j);
        }
    }
    return R;
}