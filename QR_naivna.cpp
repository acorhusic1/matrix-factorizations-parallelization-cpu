#include "QR_naivna.h"

void Vector::TestirajDimenzije(const Vector& v1, const Vector& v2) {
    if (v1.NElems() != v2.NElems())
        throw std::domain_error("Incompatible formats");
}
Vector::Vector(int n) {
    if (n < 0) throw std::range_error("Bad dimension");
    v = std::vector<double>(n);
}
Vector::Vector(std::initializer_list<double> l) {
    if (l.size() == 0) throw std::range_error("Bad dimension");
    v = l;
}
int Vector::NElems() const { return v.size(); }
double& Vector::operator[](int i) { return v[i]; }
double Vector::operator[](int i) const { return v[i]; }
double& Vector::operator()(int i) {
    if (i <= 0 || i > v.size()) throw std::range_error("Invalid index");
    return v[i - 1];
}
double Vector::operator()(int i) const {
    if (i <= 0 || i > v.size()) throw std::range_error("Invalid index");
    return v[i - 1];
}
double Vector::Norm() const {
    double suma{};
    for (const double& x : v) suma += x * x;
    return std::sqrt(suma);
}
double Vector::GetEpsilon() const { return 10 * Norm() * std::numeric_limits<double>::epsilon(); }
void Vector::Print(char separator, double eps) const {
    double prag = (eps < 0) ? (NElems() > 0 ? GetEpsilon() : 0) : eps;
    for (int i = 0; i < v.size(); i++) {
        if (std::abs(v[i]) < prag) std::cout << 0;
        else std::cout << v[i];
        if (i < v.size() - 1) std::cout << separator;
    }
    if (separator == '\n') std::cout << std::endl;
}
Vector& Vector::operator+=(const Vector& v) { TestirajDimenzije(*this, v); for (int i = 0; i < NElems(); i++) (*this)[i] += v[i]; return *this; }
Vector operator+(const Vector& v1, const Vector& v2) { Vector temp = v1; return temp += v2; }
Vector& Vector::operator-=(const Vector& v) { TestirajDimenzije(*this, v); for (int i = 0; i < NElems(); i++) (*this)[i] -= v[i]; return *this; }
Vector operator-(const Vector& v1, const Vector& v2) { Vector temp = v1; return temp -= v2; }
Vector operator*(double s, const Vector& v) { Vector temp = v; return temp *= s; }
Vector operator*(const Vector& v, double s) { return s * v; }
Vector& Vector::operator*=(double s) { for (auto& x : v) x *= s; return *this; }
double operator*(const Vector& v1, const Vector& v2) {
    Vector::TestirajDimenzije(v1, v2);
    double proizvod{};
    for (int i = 0; i < v1.NElems(); i++) proizvod += v1[i] * v2[i];
    return proizvod;
}
Vector operator/(const Vector& v, double s) { Vector temp = v; return temp /= s; }
Vector& Vector::operator/=(double s) {
    if (std::abs(s) < 1e-12) throw std::domain_error("Division by zero");
    for (auto& x : v) x /= s;
    return *this;
}
void Vector::Chop(double eps) {
    double prag = (eps < 0) ? GetEpsilon() : eps;
    for (auto& x : v) if (std::abs(x) < prag) x = 0;
}
bool Vector::EqualTo(const Vector& v, double eps) const {
    if (NElems() != v.NElems()) return false;
    double prag = (eps < 0) ? GetEpsilon() : eps;
    for (int i = 0; i < NElems(); i++) if (std::abs((*this)[i] - v[i]) > prag) return false;
    return true;
}


// --- Implementacije za Matrix ---
Matrix::Matrix(int m_rows, int n_cols) {
    if (m_rows <= 0 || n_cols <= 0) throw std::range_error("Bad dimension");
    m = std::vector<std::vector<double>>(m_rows, std::vector<double>(n_cols));
}
Matrix::Matrix(const Vector& v) {
    m.resize(v.NElems(), std::vector<double>(1));
    for (int i = 0; i < v.NElems(); i++) m[i][0] = v[i];
}
Matrix::Matrix(std::initializer_list<std::vector<double>> l) {
    if (l.size() == 0) throw std::range_error("Bad dimension");
    int n_cols = l.begin()->size();
    if (n_cols == 0) throw std::range_error("Bad dimension");
    m.reserve(l.size());
    for (const auto& red : l) {
        if (red.size() != n_cols) throw std::logic_error("Bad matrix");
        m.push_back(red);
    }
}
int Matrix::NRows() const { return m.size(); }
int Matrix::NCols() const { return m.empty() ? 0 : m[0].size(); }
double* Matrix::operator[](int i) { return &m[i][0]; }
const double* Matrix::operator[](int i) const { return &m[i][0]; }
double& Matrix::operator()(int i, int j) {
    if (i < 1 || i > NRows() || j < 1 || j > NCols()) throw std::range_error("Invalid index");
    return m[i - 1][j - 1];
}
double Matrix::operator()(int i, int j) const {
    if (i < 1 || i > NRows() || j < 1 || j > NCols()) throw std::range_error("Invalid index");
    return m[i - 1][j - 1];
}
double Matrix::Norm() const {
    double suma = 0;
    for (const auto& red : m) for (double x : red) suma += x * x;
    return std::sqrt(suma);
}
double Matrix::GetEpsilon() const { return 10 * Norm() * std::numeric_limits<double>::epsilon(); }
void Matrix::Print(int width, double eps) const {
    double prag = (eps < 0) ? GetEpsilon() : eps;
    for (int i = 0; i < NRows(); i++) {
        for (int j = 0; j < NCols(); j++) {
            if (std::abs(m[i][j]) < prag) std::cout << std::setw(width) << 0;
            else std::cout << std::setw(width) << m[i][j];
        }
        std::cout << std::endl;
    }
}
Matrix& Matrix::operator+=(const Matrix& mat) {
    if (NRows() != mat.NRows() || NCols() != mat.NCols()) throw std::domain_error("Incompatible formats");
    for (int i = 0; i < NRows(); i++) for (int j = 0; j < NCols(); j++) m[i][j] += mat.m[i][j];
    return *this;
}
Matrix operator+(const Matrix& m1, const Matrix& m2) { Matrix temp = m1; return temp += m2; }
Matrix& Matrix::operator-=(const Matrix& mat) {
    if (NRows() != mat.NRows() || NCols() != mat.NCols()) throw std::domain_error("Incompatible formats");
    for (int i = 0; i < NRows(); i++) for (int j = 0; j < NCols(); j++) m[i][j] -= mat.m[i][j];
    return *this;
}
Matrix operator-(const Matrix& m1, const Matrix& m2) { Matrix temp = m1; return temp -= m2; }
Matrix& Matrix::operator*=(double s) { for (auto& red : m) for (auto& x : red) x *= s; return *this; }
Matrix operator*(double s, const Matrix& m) { Matrix temp = m; return temp *= s; }
Matrix operator*(const Matrix& m1, const Matrix& m2) {
    if (m1.NCols() != m2.NRows()) throw std::domain_error("Incompatible formats");
    Matrix rez(m1.NRows(), m2.NCols());
    for (int i = 0; i < m1.NRows(); i++) {
        for (int j = 0; j < m2.NCols(); j++) {
            double suma = 0;
            for (int k = 0; k < m1.NCols(); k++) suma += m1[i][k] * m2[k][j];
            rez[i][j] = suma;
        }
    }
    return rez;
}
Vector operator*(const Matrix& m, const Vector& v) {
    if (m.NCols() != v.NElems()) throw std::domain_error("Incompatible formats");
    Vector rez(m.NRows());
    for (int i = 0; i < m.NRows(); i++) {
        double suma = 0;
        for (int j = 0; j < m.NCols(); j++) suma += m[i][j] * v[j];
        rez[i] = suma;
    }
    return rez;
}
void Matrix::Transpose() {
    int rows = NRows(), cols = NCols();
    if (rows == cols) {
        for (int i = 0; i < rows; i++) for (int j = i + 1; j < cols; j++) std::swap(m[i][j], m[j][i]);
    }
    else {
        Matrix temp(cols, rows);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) temp.m[j][i] = m[i][j];
        *this = temp;
    }
}
void Matrix::Chop(double eps) {
    double prag = (eps < 0) ? GetEpsilon() : eps;
    for (auto& red : m) for (auto& x : red) if (std::abs(x) < prag) x = 0;
}
bool Matrix::EqualTo(const Matrix& mat, double eps) const {
    if (NRows() != mat.NRows() || NCols() != mat.NCols()) return false;
    double prag = (eps < 0) ? GetEpsilon() : eps;
    for (int i = 0; i < NRows(); i++)
        for (int j = 0; j < NCols(); j++)
            if (std::abs(m[i][j] - mat.m[i][j]) > prag) return false;
    return true;
}
Vector LeftDiv(Matrix m, Vector v) {
    if (m.NRows() != m.NCols()) throw std::domain_error("Divisor matrix is not square");
    if (m.NRows() != v.NElems()) throw std::domain_error("Incompatible formats");
    int n = m.NRows();
    double prag = m.GetEpsilon();
    for (int k = 0; k < n; k++) {
        int p = k;
        for (int i = k + 1; i < n; i++) if (std::abs(m[i][k]) > std::abs(m[p][k])) p = i;
        if (std::abs(m[p][k]) < prag) throw std::domain_error("Divisor matrix is singular");
        if (p != k) {
            std::swap(m.m[k], m.m[p]); // ISPRAVKA: swap-uj vektore direktno
            std::swap(v[k], v[p]);
        }
        for (int i = k + 1; i < n; i++) {
            double u = m[i][k] / m[k][k];
            for (int j = k + 1; j < n; j++) m[i][j] -= u * m[k][j];
            v[i] -= u * v[k];
        }
    }
    Vector x(n);
    for (int i = n - 1; i >= 0; i--) {
        double s = v[i];
        for (int j = i + 1; j < n; j++) s -= m[i][j] * x[j];
        x[i] = s / m[i][i];
    }
    return x;
}
Matrix LeftDiv(Matrix m1, Matrix m2) {
    if (m1.NRows() != m1.NCols()) throw std::domain_error("Divisor matrix is not square");
    if (m1.NRows() != m2.NRows()) throw std::domain_error("Incompatible formats");
    Matrix x(m2.NRows(), m2.NCols());
    for (int i = 0; i < m2.NCols(); ++i) {
        Vector b(m2.NRows());
        for (int j = 0; j < m2.NRows(); ++j) b[j] = m2[j][i];
        Vector res = LeftDiv(m1, b);
        for (int j = 0; j < m2.NRows(); ++j) x[j][i] = res[j];
    }
    return x;
}
Matrix& Matrix::operator/=(double s) {
    if (std::abs(s) < 1e-12) throw std::domain_error("Division by zero");
    for (auto& red : m) for (auto& x : red) x /= s;
    return *this;
}
Matrix operator/(const Matrix& m, double s) { Matrix temp = m; return temp /= s; }
Matrix& Matrix::operator/=(Matrix m) {
    if (m.NRows() != m.NCols()) throw std::domain_error("Divisor matrix is not square");
    if (NCols() != m.NRows()) throw std::domain_error("Incompatible formats");
    Matrix At = *this; At.Transpose();
    Matrix Bt = m;     Bt.Transpose();
    Matrix Xt = LeftDiv(Bt, At);
    Xt.Transpose();
    *this = Xt;
    return *this;
}
Matrix operator/(Matrix m1, Matrix m2) { return m1 /= m2; }
double Matrix::Det() const {
    if (NRows() != NCols()) throw std::domain_error("Matrix is not square");
    Matrix temp = *this;
    int n = NRows();
    double d = 1;
    double prag = GetEpsilon();
    for (int k = 0; k < n; k++) {
        int p = k;
        for (int i = k + 1; i < n; i++) if (std::abs(temp[i][k]) > std::abs(temp[p][k])) p = i;
        if (std::abs(temp[p][k]) < prag) return 0;
        if (p != k) {
            std::swap(temp.m[k], temp.m[p]); // ISPRAVKA: swap-uj vektore direktno
            d = -d;
        }
        d *= temp[k][k];
        for (int i = k + 1; i < n; i++) {
            double u = temp[i][k] / temp[k][k];
            for (int j = k + 1; j < n; j++) temp[i][j] -= u * temp[k][j];
        }
    }
    return d;
}
void Matrix::Invert() {
    if (NRows() != NCols()) throw std::domain_error("Matrix is not square");
    int n = NRows();
    std::vector<int> p(n);
    double prag = GetEpsilon();

    for (int k = 0; k < n; k++) {
        int pivot = k;
        for (int i = k + 1; i < n; i++) if (std::abs(m[i][k]) > std::abs(m[pivot][k])) pivot = i;
        if (std::abs(m[pivot][k]) < prag) throw std::domain_error("Matrix is singular");
        if (pivot != k) std::swap(m[k], m[pivot]);
        p[k] = pivot;
        double u = m[k][k];
        m[k][k] = 1;
        for (int j = 0; j < n; j++) m[k][j] /= u;
        for (int i = 0; i < n; i++) {
            if (i != k) {
                u = m[i][k];
                m[i][k] = 0;
                for (int j = 0; j < n; j++) m[i][j] -= u * m[k][j];
            }
        }
    }
    for (int j = n - 1; j >= 0; j--) {
        if (p[j] != j) {
            for (int i = 0; i < n; i++) std::swap(m[i][j], m[i][p[j]]);
        }
    }
}
int Matrix::RankRREF() {
    int pivot_count = 0, lead = 0;
    int rows = NRows(), cols = NCols();
    double prag = GetEpsilon();
    // Direktno radimo na clanu 'm'
    while (lead < cols && pivot_count < rows) {
        int i = pivot_count;
        while (std::abs(m[i][lead]) < prag) {
            i++;
            if (i == rows) { i = pivot_count; lead++; if (lead == cols) return pivot_count; }
        }
        std::swap(m[i], m[pivot_count]); // ISPRAVKA: Ovo sada radi ispravno
        double val = m[pivot_count][lead];
        for (int j = 0; j < cols; j++) m[pivot_count][j] /= val;
        for (i = 0; i < rows; i++) {
            if (i != pivot_count) {
                val = m[i][lead];
                for (int j = 0; j < cols; j++) m[i][j] -= val * m[pivot_count][j];
            }
        }
        pivot_count++;
        lead++;
    }
    return pivot_count;
}
void Matrix::ReduceToRREF() { RankRREF(); }
int Matrix::Rank() const { Matrix temp = *this; return temp.RankRREF(); }
void PrintMatrix(const Matrix& m, int width, double eps) { m.Print(width, eps); }
Matrix operator*(const Matrix& m, double s) { return s * m; }
Matrix Transpose(const Matrix& m) { Matrix a = m; a.Transpose(); return a; }
double Det(Matrix m) { return m.Det(); }
Matrix Inverse(Matrix m) { m.Invert(); return m; }
Matrix RREF(Matrix m) { m.ReduceToRREF(); return m; }
int Rank(Matrix m) { return m.Rank(); }

// --- Implementacije za QRDecomposer ---
// ... (Ovaj dio je nepromijenjen i ispravan) ...
QRDecomposer::QRDecomposer(Matrix m) : QRmat(m), R_diag(m.NCols()) {
    int rows = QRmat.NRows(), cols = QRmat.NCols();
    is_square = (rows == cols);
    if (rows < cols) throw std::domain_error("Invalid matrix format");
    double prag = m.GetEpsilon();

    for (int k = 0; k < cols; k++) {
        double s = 0;
        for (int i = k; i < rows; i++) s += QRmat[i][k] * QRmat[i][k];
        s = std::sqrt(s);
        if (QRmat[k][k] < 0) s = -s;
        if (std::abs(s) < prag) throw std::domain_error("Matrix is singular");

        double v1 = QRmat[k][k] + s;
        QRmat[k][k] = v1;
        R_diag[k] = -s;

        double v_norm_sq = v1 * v1;
        for (int i = k + 1; i < rows; i++) v_norm_sq += QRmat[i][k] * QRmat[i][k];

        if (std::abs(v_norm_sq) > 1e-20) {
            for (int j = k + 1; j < cols; j++) {
                double u = 0;
                for (int i = k; i < rows; i++) u += QRmat[i][k] * QRmat[i][j];
                u *= (2.0 / v_norm_sq);
                for (int i = k; i < rows; i++) QRmat[i][j] -= u * QRmat[i][k];
            }
        }
    }
}
Vector QRDecomposer::MulQTWith(Vector v) const {
    if (QRmat.NRows() != v.NElems()) throw std::domain_error("Incompatible formats");
    Vector res = v;
    int rows = QRmat.NRows(), cols = QRmat.NCols();
    for (int k = 0; k < cols; k++) {
        double v_norm_sq = QRmat[k][k] * QRmat[k][k];
        for (int i = k + 1; i < rows; i++) v_norm_sq += QRmat[i][k] * QRmat[i][k];
        if (std::abs(v_norm_sq) < 1e-20) continue;
        double s = 0;
        s += QRmat[k][k] * res[k];
        for (int i = k + 1; i < rows; i++) s += QRmat[i][k] * res[i];
        s *= (2.0 / v_norm_sq);
        res[k] -= s * QRmat[k][k];
        for (int i = k + 1; i < rows; i++) res[i] -= s * QRmat[i][k];
    }
    return res;
}
Matrix QRDecomposer::MulQTWith(Matrix m) const {
    if (QRmat.NRows() != m.NRows()) throw std::domain_error("Incompatible formats");
    Matrix res = m;
    for (int j = 0; j < m.NCols(); ++j) {
        Vector col(m.NRows());
        for (int i = 0; i < m.NRows(); ++i) col[i] = m[i][j];
        Vector transformed_col = MulQTWith(col);
        for (int i = 0; i < m.NRows(); ++i) res[i][j] = transformed_col[i];
    }
    return res;
}
Vector QRDecomposer::MulQWith(Vector v) const {
    if (QRmat.NRows() != v.NElems()) throw std::domain_error("Incompatible formats");
    Vector res = v;
    int rows = QRmat.NRows(), cols = QRmat.NCols();
    for (int k = cols - 1; k >= 0; k--) {
        double v_norm_sq = QRmat[k][k] * QRmat[k][k];
        for (int i = k + 1; i < rows; i++) v_norm_sq += QRmat[i][k] * QRmat[i][k];
        if (std::abs(v_norm_sq) < 1e-20) continue;
        double s = 0;
        s += QRmat[k][k] * res[k];
        for (int i = k + 1; i < rows; i++) s += QRmat[i][k] * res[i];
        s *= (2.0 / v_norm_sq);
        res[k] -= s * QRmat[k][k];
        for (int i = k + 1; i < rows; i++) res[i] -= s * QRmat[i][k];
    }
    return res;
}
Matrix QRDecomposer::MulQWith(Matrix m) const {
    if (QRmat.NRows() != m.NRows()) throw std::domain_error("Incompatible formats");
    Matrix res = m;
    for (int j = 0; j < m.NCols(); ++j) {
        Vector col(m.NRows());
        for (int i = 0; i < m.NRows(); ++i) col[i] = m[i][j];
        Vector transformed_col = MulQWith(col);
        for (int i = 0; i < m.NRows(); ++i) res[i][j] = transformed_col[i];
    }
    return res;
}
Vector QRDecomposer::Solve(Vector b) const {
    if (!is_square) throw std::domain_error("Matrix is not square");
    if (QRmat.NRows() != b.NElems()) throw std::domain_error("Incompatible formats");
    Vector x = MulQTWith(b);
    int n = QRmat.NCols();
    for (int i = n - 1; i >= 0; i--) {
        double s = 0;
        for (int j = i + 1; j < n; j++) s += QRmat[i][j] * x[j];
        if (std::abs(R_diag[i]) < QRmat.GetEpsilon()) throw std::domain_error("Matrix is singular");
        x[i] = (x[i] - s) / R_diag[i];
    }
    return x;
}
void QRDecomposer::Solve(const Vector& b, Vector& x) const { x = Solve(b); }
Matrix QRDecomposer::Solve(Matrix b) const {
    if (!is_square) throw std::domain_error("Matrix is not square");
    if (QRmat.NRows() != b.NRows()) throw std::domain_error("Incompatible formats");
    Matrix x(b.NRows(), b.NCols());
    for (int j = 0; j < b.NCols(); ++j) {
        Vector col(b.NRows());
        for (int i = 0; i < b.NRows(); ++i) col[i] = b[i][j];
        Vector res_col = Solve(col);
        for (int i = 0; i < b.NRows(); ++i) x[i][j] = res_col[i];
    }
    return x;
}
void QRDecomposer::Solve(const Matrix& b, Matrix& x) const { x = Solve(b); }
Matrix QRDecomposer::GetQ() const {
    Matrix Q(QRmat.NRows(), QRmat.NRows());
    for (int i = 0; i < QRmat.NRows(); i++) Q[i][i] = 1;
    return MulQWith(Q);
}
Matrix QRDecomposer::GetR() const {
    Matrix R(QRmat.NRows(), QRmat.NCols());
    for (int i = 0; i < QRmat.NRows(); i++) {
        for (int j = 0; j < QRmat.NCols(); j++) {
            if (i == j) R[i][j] = R_diag[i];
            else if (j > i) R[i][j] = QRmat[i][j];
            else R[i][j] = 0;
        }
    }
    return R;
}