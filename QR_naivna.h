#ifndef OPTIMIZED_MATRIX_H
#define OPTIMIZED_MATRIX_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>

// ============================================================================
// OPTIMIZED VECTOR CLASS
// ============================================================================
class Vector {
private:
    std::vector<double> v;
    
public:
    // Constructors
    Vector() = default;
    explicit Vector(int n) : v(n, 0.0) {
        if (n < 0) throw std::range_error("Bad dimension");
    }
    Vector(std::initializer_list<double> l) : v(l) {
        if (l.size() == 0) throw std::range_error("Bad dimension");
    }
    
    // Size
    int NElems() const { return v.size(); }
    
    // Access operators - optimized with inline
    inline double& operator[](int i) { return v[i]; }
    inline double operator[](int i) const { return v[i]; }
    
    double& operator()(int i) {
        if (i <= 0 || i > (int)v.size()) throw std::range_error("Invalid index");
        return v[i - 1];
    }
    double operator()(int i) const {
        if (i <= 0 || i > (int)v.size()) throw std::range_error("Invalid index");
        return v[i - 1];
    }
    
    // Direct data access for performance
    inline double* data() { return v.data(); }
    inline const double* data() const { return v.data(); }
    
    // Norm - optimized
    double Norm() const {
        double suma = 0.0;
        const int n = v.size();
        const double* ptr = v.data();
        
        // Unroll loop for better performance
        int i = 0;
        for (; i + 3 < n; i += 4) {
            suma += ptr[i] * ptr[i];
            suma += ptr[i+1] * ptr[i+1];
            suma += ptr[i+2] * ptr[i+2];
            suma += ptr[i+3] * ptr[i+3];
        }
        for (; i < n; i++) {
            suma += ptr[i] * ptr[i];
        }
        
        return std::sqrt(suma);
    }
    
    double GetEpsilon() const { 
        return 10.0 * Norm() * std::numeric_limits<double>::epsilon(); 
    }
    
    // Arithmetic operations - optimized
    Vector& operator+=(const Vector& other) {
        if (v.size() != other.v.size()) 
            throw std::domain_error("Incompatible formats");
        
        const int n = v.size();
        double* ptr1 = v.data();
        const double* ptr2 = other.v.data();
        
        for (int i = 0; i < n; i++) {
            ptr1[i] += ptr2[i];
        }
        return *this;
    }
    
    Vector& operator-=(const Vector& other) {
        if (v.size() != other.v.size()) 
            throw std::domain_error("Incompatible formats");
        
        const int n = v.size();
        double* ptr1 = v.data();
        const double* ptr2 = other.v.data();
        
        for (int i = 0; i < n; i++) {
            ptr1[i] -= ptr2[i];
        }
        return *this;
    }
    
    Vector& operator*=(double s) {
        const int n = v.size();
        double* ptr = v.data();
        
        for (int i = 0; i < n; i++) {
            ptr[i] *= s;
        }
        return *this;
    }
    
    Vector& operator/=(double s) {
        if (std::abs(s) < 1e-12) throw std::domain_error("Division by zero");
        const double inv_s = 1.0 / s;
        return (*this) *= inv_s;
    }
    
    // Dot product - optimized
    friend double operator*(const Vector& v1, const Vector& v2) {
        if (v1.v.size() != v2.v.size()) 
            throw std::domain_error("Incompatible formats");
        
        double sum = 0.0;
        const int n = v1.v.size();
        const double* ptr1 = v1.v.data();
        const double* ptr2 = v2.v.data();
        
        // Loop unrolling
        int i = 0;
        for (; i + 3 < n; i += 4) {
            sum += ptr1[i] * ptr2[i];
            sum += ptr1[i+1] * ptr2[i+1];
            sum += ptr1[i+2] * ptr2[i+2];
            sum += ptr1[i+3] * ptr2[i+3];
        }
        for (; i < n; i++) {
            sum += ptr1[i] * ptr2[i];
        }
        
        return sum;
    }
    
    void Chop(double eps = -1) {
        double prag = (eps < 0) ? GetEpsilon() : eps;
        for (auto& x : v) {
            if (std::abs(x) < prag) x = 0;
        }
    }
    
    bool EqualTo(const Vector& other, double eps = -1) const {
        if (NElems() != other.NElems()) return false;
        double prag = (eps < 0) ? GetEpsilon() : eps;
        
        for (int i = 0; i < NElems(); i++) {
            if (std::abs(v[i] - other.v[i]) > prag) return false;
        }
        return true;
    }
    
    void Print(char separator = ' ', double eps = -1) const {
        double prag = (eps < 0) ? GetEpsilon() : eps;
        for (int i = 0; i < (int)v.size(); i++) {
            if (std::abs(v[i]) < prag) std::cout << 0;
            else std::cout << v[i];
            if (i < (int)v.size() - 1) std::cout << separator;
        }
        if (separator == '\n') std::cout << std::endl;
    }
};

// Vector operations
inline Vector operator+(const Vector& v1, const Vector& v2) {
    Vector temp = v1;
    return temp += v2;
}

inline Vector operator-(const Vector& v1, const Vector& v2) {
    Vector temp = v1;
    return temp -= v2;
}

inline Vector operator*(double s, const Vector& v) {
    Vector temp = v;
    return temp *= s;
}

inline Vector operator*(const Vector& v, double s) {
    return s * v;
}

inline Vector operator/(const Vector& v, double s) {
    Vector temp = v;
    return temp /= s;
}

// ============================================================================
// OPTIMIZED MATRIX CLASS
// ============================================================================
class Matrix {
private:
    std::vector<double> data;  // Flat storage for cache efficiency
    int rows;
    int cols;
    
    inline int index(int i, int j) const { return i * cols + j; }
    
public:
    // Constructors
    Matrix() : rows(0), cols(0) {}
    
    Matrix(int m_rows, int n_cols) : rows(m_rows), cols(n_cols) {
        if (m_rows <= 0 || n_cols <= 0) throw std::range_error("Bad dimension");
        data.resize(m_rows * n_cols, 0.0);
    }
    
    Matrix(std::initializer_list<std::vector<double>> l) {
        if (l.size() == 0) throw std::range_error("Bad dimension");
        rows = l.size();
        cols = l.begin()->size();
        if (cols == 0) throw std::range_error("Bad dimension");
        
        data.reserve(rows * cols);
        for (const auto& row : l) {
            if (row.size() != cols) throw std::logic_error("Bad matrix");
            data.insert(data.end(), row.begin(), row.end());
        }
    }
    
    Matrix(const Vector& v) : rows(v.NElems()), cols(1) {
        data.resize(rows);
        for (int i = 0; i < rows; i++) {
            data[i] = v[i];
        }
    }
    
    // Size
    int NRows() const { return rows; }
    int NCols() const { return cols; }
    
    // Access operators - optimized with inline
    class RowProxy {
        double* row_ptr;
    public:
        RowProxy(double* ptr) : row_ptr(ptr) {}
        inline double& operator[](int j) { return row_ptr[j]; }
        inline double operator[](int j) const { return row_ptr[j]; }
    };
    
    class ConstRowProxy {
        const double* row_ptr;
    public:
        ConstRowProxy(const double* ptr) : row_ptr(ptr) {}
        inline double operator[](int j) const { return row_ptr[j]; }
    };
    
    inline RowProxy operator[](int i) { 
        return RowProxy(&data[i * cols]); 
    }
    
    inline ConstRowProxy operator[](int i) const { 
        return ConstRowProxy(&data[i * cols]); 
    }
    
    double& operator()(int i, int j) {
        if (i < 1 || i > rows || j < 1 || j > cols) 
            throw std::range_error("Invalid index");
        return data[(i-1) * cols + (j-1)];
    }
    
    double operator()(int i, int j) const {
        if (i < 1 || i > rows || j < 1 || j > cols) 
            throw std::range_error("Invalid index");
        return data[(i-1) * cols + (j-1)];
    }
    
    // Direct data access
    inline double* GetData() { return data.data(); }
    inline const double* GetData() const { return data.data(); }
    
    // Norm - optimized
    double Norm() const {
        double suma = 0.0;
        const int n = data.size();
        const double* ptr = data.data();
        
        int i = 0;
        for (; i + 3 < n; i += 4) {
            suma += ptr[i] * ptr[i];
            suma += ptr[i+1] * ptr[i+1];
            suma += ptr[i+2] * ptr[i+2];
            suma += ptr[i+3] * ptr[i+3];
        }
        for (; i < n; i++) {
            suma += ptr[i] * ptr[i];
        }
        
        return std::sqrt(suma);
    }
    
    double GetEpsilon() const { 
        return 10.0 * Norm() * std::numeric_limits<double>::epsilon(); 
    }
    
    // Arithmetic operations - optimized
    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) 
            throw std::domain_error("Incompatible formats");
        
        const int n = data.size();
        double* ptr1 = data.data();
        const double* ptr2 = other.data.data();
        
        for (int i = 0; i < n; i++) {
            ptr1[i] += ptr2[i];
        }
        return *this;
    }
    
    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) 
            throw std::domain_error("Incompatible formats");
        
        const int n = data.size();
        double* ptr1 = data.data();
        const double* ptr2 = other.data.data();
        
        for (int i = 0; i < n; i++) {
            ptr1[i] -= ptr2[i];
        }
        return *this;
    }
    
    Matrix& operator*=(double s) {
        const int n = data.size();
        double* ptr = data.data();
        
        for (int i = 0; i < n; i++) {
            ptr[i] *= s;
        }
        return *this;
    }
    
    Matrix& operator/=(double s) {
        if (std::abs(s) < 1e-12) throw std::domain_error("Division by zero");
        const double inv_s = 1.0 / s;
        return (*this) *= inv_s;
    }
    
    // Matrix multiplication - optimized
    friend Matrix operator*(const Matrix& m1, const Matrix& m2) {
        if (m1.cols != m2.rows) throw std::domain_error("Incompatible formats");
        
        Matrix result(m1.rows, m2.cols);
        
        // Cache-friendly matrix multiplication
        for (int i = 0; i < m1.rows; i++) {
            for (int k = 0; k < m1.cols; k++) {
                double m1_ik = m1.data[i * m1.cols + k];
                for (int j = 0; j < m2.cols; j++) {
                    result.data[i * result.cols + j] += 
                        m1_ik * m2.data[k * m2.cols + j];
                }
            }
        }
        
        return result;
    }
    
    // Matrix-vector multiplication - optimized
    friend Vector operator*(const Matrix& m, const Vector& v) {
        if (m.cols != v.NElems()) throw std::domain_error("Incompatible formats");
        
        Vector result(m.rows);
        const double* m_ptr = m.data.data();
        const double* v_ptr = v.data();
        double* r_ptr = result.data();
        
        for (int i = 0; i < m.rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < m.cols; j++) {
                sum += m_ptr[i * m.cols + j] * v_ptr[j];
            }
            r_ptr[i] = sum;
        }
        
        return result;
    }
    
    void Transpose() {
        if (rows == cols) {
            // In-place transpose for square matrices
            for (int i = 0; i < rows; i++) {
                for (int j = i + 1; j < cols; j++) {
                    std::swap(data[i * cols + j], data[j * cols + i]);
                }
            }
        } else {
            // Out-of-place transpose
            std::vector<double> new_data(cols * rows);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    new_data[j * rows + i] = data[i * cols + j];
                }
            }
            data = std::move(new_data);
            std::swap(rows, cols);
        }
    }
    
    void Chop(double eps = -1) {
        double prag = (eps < 0) ? GetEpsilon() : eps;
        for (auto& x : data) {
            if (std::abs(x) < prag) x = 0;
        }
    }
    
    bool EqualTo(const Matrix& other, double eps = -1) const {
        if (rows != other.rows || cols != other.cols) return false;
        double prag = (eps < 0) ? GetEpsilon() : eps;
        
        for (size_t i = 0; i < data.size(); i++) {
            if (std::abs(data[i] - other.data[i]) > prag) return false;
        }
        return true;
    }
    
    void Print(int width = 10, double eps = -1) const {
        double prag = (eps < 0) ? GetEpsilon() : eps;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = data[i * cols + j];
                if (std::abs(val) < prag) 
                    std::cout << std::setw(width) << 0;
                else 
                    std::cout << std::setw(width) << val;
            }
            std::cout << std::endl;
        }
    }
    
    // For compatibility with old code
    std::vector<std::vector<double>> ToNestedVector() const {
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i * cols + j];
            }
        }
        return result;
    }
    
    static Matrix FromNestedVector(const std::vector<std::vector<double>>& v) {
        if (v.empty() || v[0].empty()) throw std::range_error("Bad dimension");
        Matrix m(v.size(), v[0].size());
        for (size_t i = 0; i < v.size(); i++) {
            for (size_t j = 0; j < v[0].size(); j++) {
                m.data[i * m.cols + j] = v[i][j];
            }
        }
        return m;
    }
    
    int Rank() const;  // Forward declaration
    double Det() const;
    void Invert();
};

// Additional operations
inline Matrix operator+(const Matrix& m1, const Matrix& m2) {
    Matrix temp = m1;
    return temp += m2;
}

inline Matrix operator-(const Matrix& m1, const Matrix& m2) {
    Matrix temp = m1;
    return temp -= m2;
}

inline Matrix operator*(double s, const Matrix& m) {
    Matrix temp = m;
    return temp *= s;
}

inline Matrix operator*(const Matrix& m, double s) {
    return s * m;
}

inline Matrix operator/(const Matrix& m, double s) {
    Matrix temp = m;
    return temp /= s;
}

inline Matrix Transpose(const Matrix& m) {
    Matrix temp = m;
    temp.Transpose();
    return temp;
}

#endif // OPTIMIZED_MATRIX_H