#include "QR_naivna.h"
#include <cstring> 

namespace NaiveQR {

    // ============================================================================
    // MATRIX IMPLEMENTATION
    // ============================================================================

    Matrix::Matrix(int m_, int n_) : m(m_), n(n_), own_data(true) {
        if (m > 0 && n > 0) {
            data = new double* [m];
            for (int i = 0; i < m; ++i) {
                data[i] = new double[n];
                // Initialize to 0.0
                for (int j = 0; j < n; ++j) data[i][j] = 0.0;
            }
        }
        else {
            data = nullptr;
        }
    }

    Matrix::Matrix(const Matrix& other) : m(other.m), n(other.n), own_data(true) {
        if (m > 0 && n > 0) {
            data = new double* [m];
            for (int i = 0; i < m; ++i) {
                data[i] = new double[n];
                for (int j = 0; j < n; ++j) {
                    data[i][j] = other.data[i][j];
                }
            }
        }
        else {
            data = nullptr;
        }
    }

    Matrix& Matrix::operator=(const Matrix& other) {
        if (this != &other) {
            // Free existing memory
            if (own_data && data) {
                for (int i = 0; i < m; ++i) delete[] data[i];
                delete[] data;
            }

            m = other.m; n = other.n; own_data = true;

            if (m > 0 && n > 0) {
                data = new double* [m];
                for (int i = 0; i < m; ++i) {
                    data[i] = new double[n];
                    for (int j = 0; j < n; ++j) {
                        data[i][j] = other.data[i][j];
                    }
                }
            }
            else {
                data = nullptr;
            }
        }
        return *this;
    }

    Matrix::~Matrix() {
        if (own_data && data) {
            for (int i = 0; i < m; ++i) delete[] data[i];
            delete[] data;
        }
    }

    void Matrix::Print() const {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << std::setw(11) << std::setprecision(6) << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // ============================================================================
    // QR IMPLEMENTATION (Unblocked / Naive)
    // ============================================================================

    void SimpleQR::factorize() {
        // Iterate over columns k
        for (int k = 0; k < n && k < m - 1; ++k) {

            // 1. Compute Norm of the column k below diagonal
            double norm = 0.0;
            for (int i = k; i < m; ++i) {
                norm += U(i, k) * U(i, k);
            }
            norm = std::sqrt(norm);

            if (norm < EPSILON) {
                tau[k] = 0.0;
                continue;
            }

            // 2. Form Householder vector
            double alpha = U(k, k);
            if (alpha < 0) norm = -norm;
            double v0 = alpha + norm;

            // Normalize column to get v (stored in U starting at k+1)
            for (int i = k + 1; i < m; ++i) {
                U(i, k) /= v0;
            }
            // Store v[0] implicitly as 1.0 
            // The diagonal element of R is -norm
            U(k, k) = -norm;

            // 3. Compute tau = 2 / (v . v)
            // v[k] is 1.0, so initial dot product is 1.0*1.0 = 1.0
            double v_dot_v = 1.0;
            for (int i = k + 1; i < m; ++i) {
                v_dot_v += U(i, k) * U(i, k);
            }
            tau[k] = 2.0 / v_dot_v;

            // 4. Update the TRAILING matrix (A' = (I - tau*v*v') * A)
            // For every remaining column j...
            for (int j = k + 1; j < n; ++j) {
                // Compute dot product: gamma = v' * A(:, j)
                double dot = U(k, j); // 1.0 * A(k, j) (implicit 1)
                for (int i = k + 1; i < m; ++i) {
                    dot += U(i, k) * U(i, j);
                }

                // Update: A(:, j) = A(:, j) - tau * gamma * v
                double factor = tau[k] * dot;

                U(k, j) -= factor; // Update row k (implicit v[k]=1)
                for (int i = k + 1; i < m; ++i) {
                    U(i, j) -= factor * U(i, k);
                }
            }
        }
    }

    Matrix SimpleQR::extract_R() const {
        Matrix R(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i <= j && i < m) R(i, j) = U(i, j);
                else R(i, j) = 0.0;
            }
        }
        return R;
    }

    Matrix SimpleQR::extract_Q() const {
        // Initialize Q to Identity
        int m_Q = m;
        int n_Q = std::min(m, n);
        Matrix Q(m_Q, n_Q);
        for (int i = 0; i < m_Q; ++i)
            for (int j = 0; j < n_Q; ++j)
                Q(i, j) = (i == j) ? 1.0 : 0.0;

        // Apply reflectors in reverse order
        for (int k = n - 1; k >= 0; --k) {
            if (tau[k] == 0.0) continue; // Skip zero taus

            // Apply H_k to Q: Q = (I - tau*v*v') * Q
            // We only need to update columns j from k to n_Q-1
            for (int j = k; j < n_Q; ++j) {
                // Dot product v' * Q(:, j)
                double dot = Q(k, j); // v[k] is implicit 1.
                for (int i = k + 1; i < m; ++i) {
                    dot += U(i, k) * Q(i, j);
                }

                // Update Q(:, j)
                double factor = tau[k] * dot;
                Q(k, j) -= factor;
                for (int i = k + 1; i < m; ++i) {
                    Q(i, j) -= factor * U(i, k);
                }
            }
        }
        return Q;
    }

    Matrix SimpleQR::multiply(const Matrix& A, const Matrix& B) {
        if (A.n != B.m) throw std::domain_error("Dimension mismatch");
        Matrix C(A.m, B.n);

        // Naive I-J-K loop (Row-Major optimized order, but no blocking)
        for (int i = 0; i < A.m; ++i) {
            for (int j = 0; j < B.n; ++j) {
                double sum = 0.0;
                for (int k = 0; k < A.n; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }

} // namespace NaiveQR