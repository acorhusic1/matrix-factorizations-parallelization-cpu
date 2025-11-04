#ifndef QR_NAIVNA_H
#define QR_NAIVNA_H

#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>

class Vector {
    std::vector<double> v;
    static void TestirajDimenzije(const Vector& v1, const Vector& v2);

public:
    explicit Vector(int n = 0);
    Vector(std::initializer_list<double> l);
    int NElems() const;
    double& operator[](int i);
    double operator[](int i) const;
    double& operator()(int i);
    double operator()(int i) const;
    double Norm() const;
    double GetEpsilon() const;
    void Print(char separator = '\n', double eps = -1) const;
    friend Vector operator+(const Vector& v1, const Vector& v2);
    Vector& operator+=(const Vector& v);
    friend Vector operator-(const Vector& v1, const Vector& v2);
    Vector& operator-=(const Vector& v);
    friend Vector operator*(double s, const Vector& v);
    friend Vector operator*(const Vector& v, double s);
    Vector& operator*=(double s);
    friend double operator*(const Vector& v1, const Vector& v2);
    friend Vector operator/(const Vector& v, double s);
    Vector& operator/=(double s);
    void Chop(double eps = -1);
    bool EqualTo(const Vector& v, double eps = -1) const;
};

class Matrix {
    std::vector<std::vector<double>> m;
    int RankRREF();
public:
    Matrix(int m, int n);
    Matrix(const Vector& v);
    Matrix(std::initializer_list<std::vector<double>> l);
    int NRows() const;
    int NCols() const;
    double* operator[](int i);
    const double* operator[](int i) const;
    double& operator()(int i, int j);
    double operator()(int i, int j) const;
    double Norm() const;
    double GetEpsilon() const;
    void Print(int width = 10, double eps = -1) const;
    friend Matrix operator+(const Matrix& m1, const Matrix& m2);
    Matrix& operator+=(const Matrix& m);
    friend Matrix operator-(const Matrix& m1, const Matrix& m2);
    Matrix& operator-=(const Matrix& m);
    friend Matrix operator*(double s, const Matrix& m);
    Matrix& operator*=(double s);
    friend Matrix operator*(const Matrix& m1, const Matrix& m2);
    friend Vector operator*(const Matrix& m, const Vector& v);
    void Transpose();
    void Chop(double eps = -1);
    bool EqualTo(const Matrix& m, double eps = -1) const;
    friend Matrix LeftDiv(Matrix m1, Matrix m2);
    friend Vector LeftDiv(Matrix m, Vector v);
    friend Matrix operator/(const Matrix& m, double s);
    Matrix& operator/=(double s);
    friend Matrix operator/(Matrix m1, Matrix m2);
    Matrix& operator/=(Matrix m);
    double Det() const;
    void Invert();
    void ReduceToRREF();
    int Rank() const;
};

// Deklaracije prijateljskih funkcija za Matrix
void PrintMatrix(const Matrix& m, int width = 10, double eps = -1);
Matrix operator*(const Matrix& m, double s);
Matrix Transpose(const Matrix& m);
double Det(Matrix m);
Matrix Inverse(Matrix m);
Matrix RREF(Matrix m);
int Rank(Matrix m);

class QRDecomposer {
    Matrix QRmat;
    Vector R_diag;
    bool is_square;
public:
    QRDecomposer(Matrix m);
    void Solve(const Vector& b, Vector& x) const;
    Vector Solve(Vector b) const;
    void Solve(const Matrix& b, Matrix& x) const;
    Matrix Solve(Matrix b) const;
    Vector MulQWith(Vector v) const;
    Matrix MulQWith(Matrix m) const;
    Vector MulQTWith(Vector v) const;
    Matrix MulQTWith(Matrix m) const;
    Matrix GetQ() const;
    Matrix GetR() const;
};

#endif // QR_NAIVNA_H