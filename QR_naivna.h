#ifndef QR_NAIVNA_H
#define QR_NAIVNA_H

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>

namespace NaiveQR {

    const double EPSILON = 1e-9;

    // ============================================================================
    // NAIVE MATRIX CLASS (Row-Major, Array of Arrays - double**)
    // This represents the "standard" C++ way that is cache-inefficient.
    // ============================================================================
    class Matrix {
    public:
        int m, n;
        double** data; // Pointer to array of pointers
        bool own_data;

        // Constructor: Allocates row pointers, then rows
        Matrix(int m_ = 0, int n_ = 0);

        // Copy Constructor
        Matrix(const Matrix& other);

        // Assignment Operator
        Matrix& operator=(const Matrix& other);

        // Destructor
        ~Matrix();

        // Accessors (Row-Major: data[i][j])
        // NOTE: i is row, j is column.
        double& operator()(int i, int j) { return data[i][j]; }
        const double& operator()(int i, int j) const { return data[i][j]; }

        void Print() const;
    };

    // ============================================================================
    // NAIVE (UNBLOCKED) QR FACTORIZER
    // Uses standard Householder algorithm (BLAS-2 level)
    // ============================================================================
    class SimpleQR {
    public:
        int m, n;
        Matrix U; // Stores Householder vectors (lower) and R (upper)
        std::vector<double> tau;

        SimpleQR(int m_, int n_) : m(m_), n(n_), U(m_, n_), tau(n_) {}

        // Standard Unblocked Householder QR
        void factorize();

        // Helpers for verification
        Matrix extract_R() const;
        Matrix extract_Q() const;

        // Static Utility for main.cpp verification
        static Matrix multiply(const Matrix& A, const Matrix& B);
    };

} // namespace NaiveQR

#endif // QR_NAIVNA_H