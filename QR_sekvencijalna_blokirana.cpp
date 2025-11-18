#include "QR_sekvencijalna_blokirana.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cblas.h>
#include <algorithm>

void MatrixPrintDebug(const std::vector<double>& M, int rows, int cols, const std::string& name) {
    std::cout << "--- Matrica: " << name << " (" << rows << "x" << cols << ") ---" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(6) << At(M, rows, i, j) << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "---------------------------------------" << std::endl;
}

void ApplyHouseholderToColumn(std::vector<double>& QRmat, int rows, int k, double& r_diag_val, double& tau_val) {
    int M = rows;
    double s = 0;

    for (int i = k; i < M; i++) {
        s += At(QRmat, M, i, k) * At(QRmat, M, i, k);
    }
    s = std::sqrt(s);

    if (At(QRmat, M, k, k) < 0) s = -s;

    double v1 = At(QRmat, M, k, k) + s;
    r_diag_val = -s;

    if (std::abs(v1) < GLOBAL_EPSILON) {
        tau_val = 0.0;
        At(QRmat, M, k, k) = 1.0;
        for (int i = k + 1; i < M; i++) At(QRmat, M, i, k) = 0.0;
        return;
    }

    for (int i = k + 1; i < M; i++) {
        At(QRmat, M, i, k) /= v1;
    }

    At(QRmat, M, k, k) = 1.0;

    double v_norm_sq = 1.0;
    for (int i = k + 1; i < M; i++) {
        v_norm_sq += At(QRmat, M, i, k) * At(QRmat, M, i, k);
    }

    tau_val = 2.0 / v_norm_sq;
}

void Internal_Unblocked_Panel_QR(std::vector<double>& QRmat, int rows, int cols, int k, int B,
    std::vector<double>& R_diag_panel, std::vector<double>& W_panel) {

    R_diag_panel.resize(B);
    W_panel.assign(B * B, 0.0);

    int M = rows;
    int col_end = k + B;
    std::vector<double> w_j(M);

    MatrixPrintDebug(QRmat, rows, col_end, "Panel: ULaz A(k:M-1, k:k+B-1)");

    for (int j = k; j < col_end; j++) {
        int local_j = j - k;

        // Kopiramo trenutnu kolonu u w_j
        for (int i = 0; i < M; ++i) w_j[i] = At(QRmat, M, i, j);

        // Primjena prethodnih reflektora
        for (int l = k; l < j; l++) {
            int local_l = l - k;
            double tau_l = At(W_panel, B, local_l, local_l);
            if (std::abs(tau_l) < GLOBAL_EPSILON) continue;

            double dot_product = 0.0;
            for (int i = l; i < M; ++i) {
                dot_product += At(QRmat, M, i, l) * w_j[i];
            }

            double factor = tau_l * dot_product;
            for (int i = l; i < M; ++i) {
                w_j[i] -= factor * At(QRmat, M, i, l);
            }
        }

        // Kopiramo transformisanu kolonu nazad
        for (int i = 0; i < M; ++i) {
            At(QRmat, M, i, j) = w_j[i];
        }

        // Kreiranje Householder reflektora
        double r_val, tau_val;
        ApplyHouseholderToColumn(QRmat, M, j, r_val, tau_val);

        R_diag_panel[local_j] = r_val;
        At(W_panel, B, local_j, local_j) = tau_val;

        // RAČUNANJE VAN-DIJAGONALNIH ELEMENATA T
        for (int l = k; l < j; ++l) {
            int local_l = l - k;
            double tau_l = At(W_panel, B, local_l, local_l);
            if (std::abs(tau_l) < GLOBAL_EPSILON) continue;

            // v_l^T * v_j iz transformisanih vektora
            double dot_vl_vj = 0.0;
            for (int i = l; i < M; ++i) {
                dot_vl_vj += At(QRmat, M, i, l) * At(QRmat, M, i, j);
            }

            double T_l_j = -tau_l * dot_vl_vj;

            for (int i = l + 1; i < j; ++i) {
                T_l_j -= At(W_panel, B, local_l, i - k) * At(W_panel, B, i - k, local_j);
            }

            At(W_panel, B, local_l, local_j) = T_l_j;
        }

        std::cout << "\n--- Iteracija j=" << j << " (Kolona " << local_j << ") ---" << std::endl;
        std::cout << "R_diag[" << j << "]=" << r_val << ", Tau[" << local_j << "]=" << tau_val << std::endl;
        MatrixPrintDebug(W_panel, B, B, "W_panel (T matrica) nakon H_" + std::to_string(j));
    }
}

void Internal_Apply_Block_Reflector(std::vector<double>& QRmat, int rows, int cols, int k, int B,
    const std::vector<double>& W_panel) {

    int M_rows = rows - k;
    int N_cols = cols - k - B;
    if (N_cols <= 0) return;

    const int LD_A = rows;
    const int LD_T = B;
    const int LD_Y = B;

    const double* V_ptr = QRmat.data() + k + k * rows;
    double* X_ptr = QRmat.data() + k + (k + B) * rows;
    const double* T_ptr = W_panel.data();
    std::vector<double> Y_matrix(B * N_cols, 0.0);
    double* Y_Z_ptr = Y_matrix.data();

    std::cout << "\n>>> BLOKIRANO AZURIRANJE (k=" << k << ", B=" << B << ") <<<" << std::endl;

    std::vector<double> V_temp(M_rows * B);
    for (int j = 0; j < B; ++j) { for (int i = 0; i < M_rows; ++i) V_temp[i + j * M_rows] = V_ptr[i + j * LD_A]; }
    MatrixPrintDebug(V_temp, M_rows, B, "Submatrica V");

    std::vector<double> X_temp(M_rows * N_cols);
    for (int j = 0; j < N_cols; ++j) { for (int i = 0; i < M_rows; ++i) X_temp[i + j * M_rows] = X_ptr[i + j * LD_A]; }
    MatrixPrintDebug(X_temp, M_rows, N_cols, "Submatrica X (preostala)");

    MatrixPrintDebug(W_panel, B, B, "Submatrica T");

    cblas_dgemm(
        CblasColMajor, CblasTrans, CblasNoTrans,
        B, N_cols, M_rows,
        1.0, V_ptr, LD_A, X_ptr, LD_A,
        0.0, Y_Z_ptr, LD_Y
    );
    MatrixPrintDebug(Y_matrix, B, N_cols, "Rezultat KORAK 1: Matrica Y = V^T * X");

    cblas_dtrmm(
        CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        B, N_cols,
        1.0, T_ptr, LD_T,
        Y_Z_ptr, LD_Y
    );
    MatrixPrintDebug(Y_matrix, B, N_cols, "Rezultat KORAK 2: Matrica Z = T * Y");

    cblas_dgemm(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        M_rows, N_cols, B,
        -1.0, V_ptr, LD_A,
        Y_Z_ptr, LD_Y,
        1.0, X_ptr, LD_A
    );

    for (int j = 0; j < N_cols; ++j) { for (int i = 0; i < M_rows; ++i) X_temp[i + j * M_rows] = X_ptr[i + j * LD_A]; }
    MatrixPrintDebug(X_temp, M_rows, N_cols, "Rezultat KORAK 3: Azurirana X");
}

void QRFactorization_Blocked(const std::vector<double>& A, int rows, int cols,
    std::vector<double>& QRmat, std::vector<double>& R_diag, int block_size) {

    if (rows < cols) throw std::domain_error("Invalid matrix format: rows < cols");
    if (block_size <= 0) throw std::domain_error("Block size must be positive");

    QRmat = A;
    R_diag.resize(cols);

    auto start_total = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < cols; k += block_size) {
        int B = std::min(block_size, cols - k);

        std::vector<double> R_diag_panel;
        std::vector<double> W_panel;

        Internal_Unblocked_Panel_QR(QRmat, rows, cols, k, B, R_diag_panel, W_panel);

        for (int i = 0; i < B; i++) {
            R_diag[k + i] = R_diag_panel[i];
        }

        Internal_Apply_Block_Reflector(QRmat, rows, cols, k, B, W_panel);
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_total = end_total - start_total;
    std::cout << "Blokirana impl. (QR Faktorizacija): " << duration_total.count() * 1000.0 << " ms." << std::endl;
}

void ApplyHk_Blocked(const std::vector<double>& QRmat, int rows, int cols, std::vector<double>& res, int k) {
    const double v_k_diag = 1.0;
    double v_norm_sq = v_k_diag * v_k_diag;

    for (int i = k + 1; i < rows; i++) {
        double v_i = At(QRmat, rows, i, k);
        v_norm_sq += v_i * v_i;
    }

    if (std::abs(v_norm_sq) < GLOBAL_EPSILON) return;

    double tau_k = 2.0 / v_norm_sq;

    double s = 0;
    s += v_k_diag * res[k];
    for (int i = k + 1; i < rows; i++) {
        s += At(QRmat, rows, i, k) * res[i];
    }

    s *= tau_k;

    res[k] -= s * v_k_diag;
    for (int i = k + 1; i < rows; i++) {
        res[i] -= s * At(QRmat, rows, i, k);
    }
}

std::vector<double> GetQ_Blocked(const std::vector<double>& QRmat, int rows, int cols) {
    int Qrows = rows;
    int Qcols = rows;
    std::vector<double> Q(Qrows * Qcols, 0.0);
    for (int i = 0; i < rows; i++) At(Q, Qrows, i, i) = 1.0;

    for (int k = cols - 1; k >= 0; k--) {
        for (int j = 0; j < rows; ++j) {
            std::vector<double> col(rows);
            for (int i = 0; i < rows; ++i) col[i] = At(Q, Qrows, i, j);

            ApplyHk_Blocked(QRmat, rows, cols, col, k);

            for (int i = 0; i < rows; ++i) At(Q, Qrows, i, j) = col[i];
        }
    }
    return Q;
}

std::vector<double> GetR_Blocked(const std::vector<double>& QRmat, const std::vector<double>& R_diag, int rows, int cols) {
    std::vector<double> R(rows * cols, 0.0);
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            if (i == j) At(R, rows, i, j) = R_diag[i];
            else if (j > i) At(R, rows, i, j) = At(QRmat, rows, i, j);
        }
    }
    return R;
}