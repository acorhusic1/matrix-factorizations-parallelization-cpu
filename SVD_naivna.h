#ifndef SVD_NAIVNA_H
#define SVD_NAIVNA_H

#include "QR_naivna.h"

class SVDDecomposer {
    Matrix U;           // Levi singularni vektori (m x m)
    Vector sigma;       // Singularne vrednosti (min(m,n))
    Matrix VT;          // Desni singularni vektori transponovani (n x n)
    int m_rows;         // Broj redova originalne matrice
    int n_cols;         // Broj kolona originalne matrice
    
    // Pomoćna funkcija za Jacobi iteraciju na simetričnoj matrici
    void JacobiEigenSym(Matrix& A, Matrix& V);
    
    // Pomoćna funkcija za sortiranje singularnih vrednosti
    void SortSingularValues();

public:
    // Konstruktor - vrši SVD dekompoziciju
    SVDDecomposer(Matrix m);
    
    // Getteri za matrice
    Matrix GetU() const { return U; }
    Vector GetSingularValues() const { return sigma; }
    Matrix GetV() const { Matrix V = VT; V.Transpose(); return V; }
    Matrix GetVT() const { return VT; }
    
    // Vraća Sigma matricu (m x n dijagonalna)
    Matrix GetSigma() const;
    
    // Rešavanje sistema Ax = b korišćenjem SVD
    Vector Solve(Vector b, double tolerance = -1) const;
    Matrix Solve(Matrix b, double tolerance = -1) const;
    
    // Pseudoinverzna matrica
    Matrix PseudoInverse(double tolerance = -1) const;
    
    // Rang matrice
    int Rank(double tolerance = -1) const;
    
    // Kondicioniranost matrice (σ_max / σ_min)
    double ConditionNumber() const;
};

#endif // SVD_NAIVNA_H