#ifndef QR_SEKVENCIJALNA_BLOKOVSKA_H
#define QR_SEKVENCIJALNA_BLOKOVSKA_H

#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <chrono>
#include <omp.h>

namespace SequentialBlockedQR {

	const double EPSILON = 1e-9;

	// ============================================================================
	// MATRIX CLASS
	// (Most methods are kept inline in the header)
	// ============================================================================

	class Matrix {
	public:
		int m, n, ld;
		double* data;
		bool own_data;

		Matrix(int m_ = 0, int n_ = 0) : m(m_), n(n_), ld(std::max(1, m_)), own_data(true) {
			data = (m_ > 0 && n_ > 0) ? new double[ld * n_] : nullptr;
			if (data) std::memset(data, 0, ld * n_ * sizeof(double));
		}

		Matrix(const Matrix& other) : m(other.m), n(other.n), ld(other.ld), own_data(true) {
			data = (m > 0 && n > 0) ? new double[ld * n] : nullptr;
			if (data) std::memcpy(data, other.data, ld * n * sizeof(double));
		}

		Matrix& operator=(const Matrix& other) {
			if (this != &other) {
				if (own_data && data) delete[] data;
				m = other.m; n = other.n; ld = other.ld; own_data = true;
				data = (m > 0 && n > 0) ? new double[ld * n] : nullptr;
				if (data) std::memcpy(data, other.data, ld * n * sizeof(double));
			}
			return *this;
		}

		~Matrix() { if (own_data && data) delete[] data; }

		double& operator()(int i, int j) { return data[j * ld + i]; }
		const double& operator()(int i, int j) const { return data[j * ld + i]; }

		// Prints the matrix to console using std::cout
		void Print() const {
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++)
					std::cout << std::setw(11) << std::setprecision(6) << (*this)(i, j) << " ";
				std::cout << std::endl;
			}
		}

		// Calculates and returns transpose of a matrix
		Matrix operator ~() const {
			Matrix Q_T(n, m);
#pragma omp parallel for
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					Q_T(j, i) = (*this)(i, j);

			return Q_T;
		}

		static Matrix GenerateRandom(int rows, int cols) {
			Matrix A(rows, cols);
			// Use time for seeding for better randomness across runs
			static bool seeded = false;
			if (!seeded) {
				std::srand(std::time(0));
				seeded = true;
			}

			for (int j = 0; j < cols; j++) {
				for (int i = 0; i < rows; i++) {
					// Generate double between -100.0 and 100.0
					A(i, j) = (double)std::rand() / RAND_MAX * 200.0 - 100.0;
				}
			}
			return A;
		}
	};

	// Function declaration for external use by BlockedQR::multiply
	void dgemm_parallel(char transa, char transb, int m, int n, int k,
		double alpha, const Matrix& A, const Matrix& B,
		double beta, Matrix& C);

	// ============================================================================
	// BLOCKED QR FACTORIZER
	// ============================================================================

	class BlockedQR {
	public:
		int m, n, b;
		Matrix U;
		std::vector<double> tau;

		BlockedQR(int m_, int n_, int b_ = 64) : m(m_), n(n_), b(b_), U(m_, n_), tau(n_) {}

		// Declarations for implementations in the CPP file
		void panel_qr(int k, int b_size);
		void compute_T(int k, int b_size, Matrix& T);
		void apply_block_reflector(int k, int b_size, int trail_start, int trail_cols);
		void factorize();

		// Helper function that explicitly forms R matrix and returns it
		Matrix extract_R() const {
			Matrix R(n, n);
#pragma omp parallel for
			for (int j = 0; j < n; ++j)
				for (int i = 0; i <= j && i < m; ++i)
					R(i, j) = U(i, j);
			return R;
		}

		// Helper function that explicitly forms Q matrix and returns it
		Matrix extract_Q() const {
			int m_Q = m;
			int n_Q = std::min(m, n);
			Matrix Q(m_Q, n_Q);

			// Initialization can be parallelized
#pragma omp parallel for
			for (int i = 0; i < m_Q; ++i)
				for (int j = 0; j < n_Q; ++j)
					Q(i, j) = (i == j) ? 1.0 : 0.0;

			// The outer loop (k) must be sequential
			for (int k = n - 1; k >= 0; --k) {
				if (tau[k] == 0.0) continue;

				// Parallelize the application of the current reflector H_k across all columns (j)
#pragma omp parallel for
				for (int j = 0; j < n_Q; ++j) {
					// Calculation of 's' (dot product) is local to this column
					double s = Q(k, j);
					for (int i = k + 1; i < m; ++i)
						s += U(i, k) * Q(i, j);

					// Update Q column
					Q(k, j) -= tau[k] * s;
					for (int i = k + 1; i < m; ++i)
						Q(i, j) -= tau[k] * s * U(i, k);
				}
			}

			return Q;
		}

		static Matrix multiply(const Matrix& A, const Matrix& B) {
			// --- OpenMP Verification ---
#ifdef _OPENMP
			static bool printed = false;
			if (!printed) {
				std::cout << "OpenMP Enabled: Using " << omp_get_max_threads() << " threads." << std::endl;
				printed = true;
			}
#endif
			// ---------------------------

			Matrix C(A.m, B.n);
			dgemm_parallel('N', 'N', A.m, B.n, A.n, 1.0, A, B, 0.0, C);
			return C;
		}

		static double rel_error(const Matrix& A, const Matrix& B) {
			double diff = 0.0, a = 0.0;
			int m_min = std::min(A.m, B.m);
			int n_min = std::min(A.n, B.n);
			for (int i = 0; i < m_min; ++i) {
				for (int j = 0; j < n_min; ++j) {
					double d = A(i, j) - B(i, j);
					diff += d * d;
					a += A(i, j) * A(i, j);
				}
			}
			return (a > EPSILON) ? std::sqrt(diff) / std::sqrt(a) : std::sqrt(diff);
		}

		static double rel_error_parallel(const Matrix& A, const Matrix& B) {
			double diff = 0.0, a = 0.0;
			int m_min = std::min(A.m, B.m);
			int n_min = std::min(A.n, B.n);

#pragma omp parallel for reduction(+: diff, a)
			for (int i = 0; i < m_min; ++i) {
				for (int j = 0; j < n_min; ++j) {
					double d = A(i, j) - B(i, j);
					diff += d * d;
					a += A(i, j) * A(i, j);
				}
			}
			return (a > EPSILON) ? std::sqrt(diff) / std::sqrt(a) : std::sqrt(diff);
		}

	};

} // namespace SequentialBlockedQR

#endif // QR_SEKVENCIJALNA_BLOKOVSKA_H