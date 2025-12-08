#include "SVD_naivna.h"
#include <cmath>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// JACOBI METODA za sopstvene vrednosti/vektore simetrične matrice
// Ovo je naivna implementacija - iterativno eliminišemo van-dijagonalne elemente
void SVDDecomposer::JacobiEigenSym(Matrix& A, Matrix& V) {
    int n = A.NRows();
    double eps = A.GetEpsilon();
    
    // V počinje kao jedinična matrica
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Povećaj broj iteracija za svaki slučaj
    const int MAX_ITER = 200; 
    const double TOLERANCE = 1e-10;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Pronađi najveći van-dijagonalni element
        double max_val = 0;
        int p = 0, q = 1;
        
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (std::abs(A[i][j]) > max_val) {
                    max_val = std::abs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        
        // Ako je najveći element dovoljno mali, završi
        if (max_val < TOLERANCE) break;
        
        // Izračunaj Jacobi rotaciju
        double theta;
        if (std::abs(A[p][p] - A[q][q]) < eps) {
            theta = (A[p][q] > 0) ? (M_PI / 4.0) : -(M_PI / 4.0);
        } else {
            theta = 0.5 * std::atan(2.0 * A[p][q] / (A[p][p] - A[q][q]));
        }
        
        double c = std::cos(theta);
        double s = std::sin(theta);
        
        // Ažuriraj A i V matricu sa Jacobi rotacijom
        // A' = J^T * A * J
        
        // Ažuriraj redove/kolone koji nisu p ili q
        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double aip = A[i][p];
                double aiq = A[i][q];
                A[i][p] = c * aip + s * aiq; // J^T deo
                A[p][i] = A[i][p];
                A[i][q] = -s * aip + c * aiq; // J^T deo
                A[q][i] = A[i][q];
            }
        }
        
        // Ažuriraj (p,q) blok
        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];
        
        A[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0; // (c*c-s*s)*apq + s*c*(app-aqq); // Trebalo bi biti 0
        A[q][p] = 0.0;
        
        // Ažuriraj sopstvene vektore u V
        // V' = V * J
        for (int i = 0; i < n; i++) {
            double vip = V[i][p];
            double viq = V[i][q];
            V[i][p] = c * vip + s * viq;
            V[i][q] = -s * vip + c * viq;
        }
    }
}

// Sortiranje singularnih vrednosti (i odgovarajućih vektora) u opadajućem redosledu
void SVDDecomposer::SortSingularValues() {
    int n = sigma.NElems();
    
    // Bubble sort (naivna implementacija)
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (sigma[j] < sigma[j + 1]) {
                // Zameni singularne vrednosti
                std::swap(sigma[j], sigma[j + 1]);
                
                // Zameni kolone u V^T (redove u V)
                for (int k = 0; k < VT.NRows(); k++) {
                    std::swap(VT[j][k], VT[j + 1][k]);
                }
                
                // Zameni kolone u U
                for (int k = 0; k < U.NRows(); k++) {
                    std::swap(U[k][j], U[k][j + 1]);
                }
            }
        }
    }
}

// KONSTRUKTOR - Glavni SVD algoritam
SVDDecomposer::SVDDecomposer(Matrix A) 
    : m_rows(A.NRows()), n_cols(A.NCols()), sigma(std::min(A.NRows(), A.NCols())) {
    
    int m = A.NRows();
    int n = A.NCols();
    int k = std::min(m, n);
    double eps = A.GetEpsilon();
    
    if (m >= n) {
        // *** Slučaj: više redova nego kolona ***
        // Računamo A^T * A
        Matrix ATA(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < m; k++) {
                    sum += A[k][i] * A[k][j];
                }
                ATA[i][j] = sum;
            }
        }
        
        // Nađi sopstvene vrednosti i vektore A^T*A
        Matrix V_matrix(n, n);
        JacobiEigenSym(ATA, V_matrix);
        
        // Izvuci singularne vrednosti (koreni sopstvenih vrednosti)
        for (int i = 0; i < k; i++) {
            double eigenval = ATA[i][i];
            sigma[i] = (eigenval > 0) ? std::sqrt(eigenval) : 0;
        }
        
        // V^T su redovi V matrice (transpozicija)
        VT = Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                VT[i][j] = V_matrix[j][i]; // Transponuj
            }
        }
        
        // Računaj U = A * V / σ
        U = Matrix(m, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int l = 0; l < n; l++) {
                    sum += A[i][l] * V_matrix[l][j];
                }
                if (sigma[j] > eps) {
                    U[i][j] = sum / sigma[j];
                } else {
                    U[i][j] = 0;
                }
            }
        }
        
        // Popuni ostatak U sa ortogonalnom bazom (Gram-Schmidt)
        for (int j = k; j < m; j++) {
            // Kreiraj random vektor i ortogonalizuj ga
            for (int i = 0; i < m; i++) {
                U[i][j] = (i == j) ? 1.0 : 0.0;
            }
            
            // Gram-Schmidt ortogonalizacija
            for (int l = 0; l < j; l++) {
                double dot = 0;
                for (int i = 0; i < m; i++) {
                    dot += U[i][j] * U[i][l];
                }
                for (int i = 0; i < m; i++) {
                    U[i][j] -= dot * U[i][l];
                }
            }
            
            // Normalizuj
            double norm = 0;
            for (int i = 0; i < m; i++) {
                norm += U[i][j] * U[i][j];
            }
            norm = std::sqrt(norm);
            if (norm > eps) {
                for (int i = 0; i < m; i++) {
                    U[i][j] /= norm;
                }
            }
        }
        
    } else {
        // *** Slučaj: više kolona nego redova ***
        // Računamo A * A^T
        Matrix AAT(m, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * A[j][k];
                }
                AAT[i][j] = sum;
            }
        }
        
        // Nađi sopstvene vrednosti i vektore A*A^T
        Matrix U_matrix(m, m);
        JacobiEigenSym(AAT, U_matrix);
        
        // Izvuci singularne vrednosti
        for (int i = 0; i < k; i++) {
            double eigenval = AAT[i][i];
            sigma[i] = (eigenval > 0) ? std::sqrt(eigenval) : 0;
        }
        
        U = U_matrix;
        
        // Računaj V = A^T * U / σ
        VT = Matrix(n, n);
        Matrix V_matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double sum = 0;
                for (int l = 0; l < m; l++) {
                    sum += A[l][i] * U_matrix[l][j];
                }
                if (sigma[j] > eps) {
                    V_matrix[i][j] = sum / sigma[j];
                } else {
                    V_matrix[i][j] = 0;
                }
            }
        }
        
        // Popuni ostatak V sa ortogonalnom bazom
        for (int j = k; j < n; j++) {
            for (int i = 0; i < n; i++) {
                V_matrix[i][j] = (i == j) ? 1.0 : 0.0;
            }
            
            for (int l = 0; l < j; l++) {
                double dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += V_matrix[i][j] * V_matrix[i][l];
                }
                for (int i = 0; i < n; i++) {
                    V_matrix[i][j] -= dot * V_matrix[i][l];
                }
            }
            
            double norm = 0;
            for (int i = 0; i < n; i++) {
                norm += V_matrix[i][j] * V_matrix[i][j];
            }
            norm = std::sqrt(norm);
            if (norm > eps) {
                for (int i = 0; i < n; i++) {
                    V_matrix[i][j] /= norm;
                }
            }
        }
        
        // Transponuj V u V^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                VT[i][j] = V_matrix[j][i];
            }
        }
    }
    
    // Sortiraj singularne vrednosti u opadajućem redosledu
    SortSingularValues();
}

// Vraća Sigma matricu (m x n dijagonalna sa singularnim vrednostima)
Matrix SVDDecomposer::GetSigma() const {
    Matrix S(m_rows, n_cols);
    int k = sigma.NElems();
    for (int i = 0; i < k; i++) {
        S[i][i] = sigma[i];
    }
    return S;
}

// Rešavanje sistema Ax = b korišćenjem SVD (sa tolerancijom za singularne vrednosti)
Vector SVDDecomposer::Solve(Vector b, double tolerance) const {
    if (m_rows != b.NElems()) {
        throw std::domain_error("Incompatible formats");
    }
    
    double tol = (tolerance < 0) ? (sigma[0] * std::max(m_rows, n_cols) * 1e-15) : tolerance;
    
    // x = V * Σ^(-1) * U^T * b
    // Korak 1: w = U^T * b
    Vector w(m_rows);
    for (int i = 0; i < m_rows; i++) {
        double sum = 0;
        for (int j = 0; j < m_rows; j++) {
            sum += U[j][i] * b[j];
        }
        w[i] = sum;
    }
    
    // Korak 2: y = Σ^(-1) * w (samo za nenulte singularne vrednosti)
    Vector y(n_cols);
    int k = sigma.NElems();
    for (int i = 0; i < k; i++) {
        if (sigma[i] > tol) {
            y[i] = w[i] / sigma[i];
        } else {
            y[i] = 0; 
        }
    }
    
    // Korak 3: x = V * y
    Vector x(n_cols);
    for (int i = 0; i < n_cols; i++) {
        double sum = 0;
        for (int j = 0; j < n_cols; j++) {
            sum += VT[j][i] * y[j]; // VT[j][i] je V[i][j]
        }
        x[i] = sum;
    }
    
    return x;
}

Matrix SVDDecomposer::Solve(Matrix B, double tolerance) const {
    if (m_rows != B.NRows()) {
        throw std::domain_error("Incompatible formats");
    }
    
    Matrix X(n_cols, B.NCols());
    for (int i = 0; i < B.NCols(); i++) {
        Vector b(B.NRows());
        for (int j = 0; j < B.NRows(); j++) {
            b[j] = B[j][i];
        }
        Vector x = Solve(b, tolerance);
        for (int j = 0; j < n_cols; j++) {
            X[j][i] = x[j];
        }
    }
    return X;
}

// Pseudoinverzna matrica: A^+ = V * Σ^+ * U^T
Matrix SVDDecomposer::PseudoInverse(double tolerance) const {
    double tol = (tolerance < 0) ? (sigma[0] * std::max(m_rows, n_cols) * 1e-15) : tolerance;
    
    // A^+ je (n x m)
    Matrix A_pinv(n_cols, m_rows);
    int k = sigma.NElems();
    
    for (int i = 0; i < n_cols; i++) {
        for (int j = 0; j < m_rows; j++) {
            double sum = 0;
            for (int l = 0; l < k; l++) {
                if (sigma[l] > tol) {
                    // V[i][l] * (1/sigma[l]) * U[j][l]
                    sum += VT[l][i] * (1.0 / sigma[l]) * U[j][l];
                }
            }
            A_pinv[i][j] = sum;
        }
    }
    
    return A_pinv;
}

// Rang matrice (broj nenultih singularnih vrednosti)
int SVDDecomposer::Rank(double tolerance) const {
    double tol = (tolerance < 0) ? (sigma[0] * std::max(m_rows, n_cols) * 1e-15) : tolerance;
    
    int rank = 0;
    for (int i = 0; i < sigma.NElems(); i++) {
        if (sigma[i] > tol) rank++;
    }
    return rank;
}

// Kondicioniranost: σ_max / σ_min
double SVDDecomposer::ConditionNumber() const {
    int k = sigma.NElems();
    if (k == 0) return 0;
    
    double sigma_max = sigma[0];
    double sigma_min = sigma[k - 1];
    
    if (sigma_min < 1e-15) {
        return std::numeric_limits<double>::infinity();
    }
    
    return sigma_max / sigma_min;
}