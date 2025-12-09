#include "QR_sekvencijalna_blokovska.h"
#include <iostream>

namespace SequentialBlockedQR {

	// ============================================================================
	// BLAS-3: Matrix-Matrix Multiply (Sequential version from original file, unused but kept for completeness)
	// ============================================================================
	void dgemm(char transa, char transb, int m, int n, int k,
		double alpha, const Matrix& A, const Matrix& B,
		double beta, Matrix& C) {

		if (beta == 0.0) {
			for (int j = 0; j < n; ++j)
				for (int i = 0; i < m; ++i)
					C(i, j) = 0.0;
		}
		else if (beta != 1.0) {
			for (int j = 0; j < n; ++j)
				for (int i = 0; i < m; ++i)
					C(i, j) *= beta;
		}

		if (alpha == 0.0) return;

		if (transa == 'N' && transb == 'N') {
			for (int j = 0; j < n; ++j)
				for (int i = 0; i < m; ++i)
					for (int p = 0; p < k; ++p)
						C(i, j) += alpha * A(i, p) * B(p, j);
		}
		else if (transa == 'T' && transb == 'N') {
			for (int j = 0; j < n; ++j)
				for (int i = 0; i < m; ++i)
					for (int p = 0; p < k; ++p)
						C(i, j) += alpha * A(p, i) * B(p, j);
		}
	}

	// ============================================================================
	// BLAS-3: Matrix-Matrix Multiply (OpenMP Parallelized)
	// ============================================================================

	void dgemm_parallel(char transa, char transb, int m, int n, int k,
		double alpha, const Matrix& A, const Matrix& B,
		double beta, Matrix& C) {

		// Parallelization for C = beta * C (Scaling C)
#pragma omp parallel for
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				if (beta == 0.0) {
					C(i, j) = 0.0;
				}
				else if (beta != 1.0) {
					C(i, j) *= beta;
				}
			}
		}

		if (alpha == 0.0) return;

		// Parallelization for C += alpha * A * B (Main calculation)
		if (transa == 'N' && transb == 'N') {
#pragma omp parallel for
			for (int j = 0; j < n; ++j) {
				for (int i = 0; i < m; ++i) {
					double temp_sum = 0.0;
					for (int p = 0; p < k; ++p)
						temp_sum += A(i, p) * B(p, j);
					C(i, j) += alpha * temp_sum;
				}
			}
		}
		else if (transa == 'T' && transb == 'N') {
#pragma omp parallel for
			for (int j = 0; j < n; ++j) {
				for (int i = 0; i < m; ++i) {
					double temp_sum = 0.0;
					for (int p = 0; p < k; ++p)
						temp_sum += A(p, i) * B(p, j);
					C(i, j) += alpha * temp_sum;
				}
			}
		}
	}

	// ============================================================================
	// BLOCKED QR FACTORIZER IMPLEMENTATIONS
	// ============================================================================

	void BlockedQR::panel_qr(int k, int b_size) {
		for (int j = 0; j < b_size; ++j) {
			int col = k + j;

			// Compute norm
			double norm = 0.0;
			for (int i = col; i < m; ++i)
				norm += U(i, col) * U(i, col);
			norm = std::sqrt(norm);

			if (norm < EPSILON) {
				tau[col] = 0.0;
				continue;
			}

			double alpha = U(col, col);
			if (alpha < 0) norm = -norm;
			double v0 = alpha + norm;

			if (std::abs(v0) < EPSILON * std::abs(norm)) {
				tau[col] = 0.0;
				U(col, col) = -norm;
				continue;
			}

			// Normalize u
			for (int i = col + 1; i < m; ++i)
				U(i, col) /= v0;

			// Compute tau
			double u_norm_sq = 0.0;
			for (int i = col + 1; i < m; ++i)
				u_norm_sq += U(i, col) * U(i, col);
			tau[col] = 2.0 / (1.0 + u_norm_sq);

			// Apply to remaining columns in panel [col+1 : k+b_size]
			for (int jj = col + 1; jj < k + b_size; ++jj) {
				double dot = U(col, jj);
				for (int i = col + 1; i < m; ++i)
					dot += U(i, col) * U(i, jj);

				U(col, jj) -= tau[col] * dot;
				for (int i = col + 1; i < m; ++i)
					U(i, jj) -= tau[col] * dot * U(i, col);
			}

			U(col, col) = -norm;
		}
	}

	/**
	 * LARFT: Compute upper triangular T factor
	 */
	void BlockedQR::compute_T(int k, int b_size, Matrix& T) {
		// Initialize T to zero
		for (int j = 0; j < b_size; ++j)
			for (int i = 0; i < b_size; ++i)
				T(i, j) = 0.0;

		// Temp buffer for matrix-vector multiply
		std::vector<double> t_col(b_size);

		for (int j = 0; j < b_size; ++j) {
			int col = k + j;
			if (tau[col] == 0.0) continue;

			// 1. Compute Raw Column: T(0:j-1, j) = -tau[j] * V(:, 0:j-1)' * V(:, j)
			for (int i = 0; i < j; ++i) {
				int col_i = k + i;
				double sum = 0.0;

				// V_i' * V_j
				sum += U(col, col_i); // V_i[col] * V_j[col] (1.0)

				for (int row = col + 1; row < m; ++row)
					sum += U(row, col_i) * U(row, col);

				T(i, j) = -tau[col] * sum;
			}

			// 2. Triangular Matrix-Vector Multiply: T(0:j-1, j) = T(0:j-1, 0:j-1) * T(0:j-1, j)
			// Copy current column to temp
			for (int i = 0; i < j; ++i) t_col[i] = T(i, j);

			// Standard TRMV logic (Upper Triangular)
			for (int i = 0; i < j; ++i) {
				double sum = 0.0;
				for (int l = i; l < j; ++l) {
					sum += T(i, l) * t_col[l];
				}
				T(i, j) = sum;
			}

			// Diagonal element
			T(j, j) = tau[col];
		}
	}

	/**
	 * LARFB: Apply block reflector (I - V*T*V^T) to trailing matrix
	 */
	void BlockedQR::apply_block_reflector(int k, int b_size, int trail_start, int trail_cols) {
		int m_sub = m - k;

		Matrix T(b_size, b_size);
		compute_T(k, b_size, T);

		// W = V^T * C
		Matrix W(b_size, trail_cols);

		for (int j = 0; j < trail_cols; ++j) {
			for (int i = 0; i < b_size; ++i) {
				// Part 1: Interaction with implicit 1 of V
				double sum = U(k + i, trail_start + j);

				// Part 2: Interaction with rest of V
				for (int row = i + 1; row < m_sub; ++row)
					sum += U(k + row, k + i) * U(k + row, trail_start + j);

				W(i, j) = sum;
			}
		}

		// W2 = T^T * W
		Matrix W2(b_size, trail_cols);
		for (int j = 0; j < trail_cols; ++j) {
			for (int i = 0; i < b_size; ++i) {
				double sum = 0.0;
				// T is Upper Triangular, so T^T is lower triangular
				for (int kk = 0; kk <= i; ++kk)
					sum += T(kk, i) * W(kk, j);
				W2(i, j) = sum;
			}
		}

		// C = C - V * W2
		for (int j = 0; j < trail_cols; ++j) {
			for (int i = 0; i < b_size; ++i) {
				double w_val = W2(i, j);

				// Apply 1.0 at row k+i
				U(k + i, trail_start + j) -= w_val;

				// Apply remaining vector
				for (int row = i + 1; row < m_sub; ++row)
					U(k + row, trail_start + j) -= U(k + row, k + i) * w_val;
			}
		}
	}

	void BlockedQR::factorize() {
		for (int k = 0; k < n; k += b) {
			int b_size = std::min(b, n - k);

		//	std::cout << "\n########## BLOCK " << (k / b) << ": columns " << k << " to " << (k + b_size - 1) << " ##########\n";

			panel_qr(k, b_size);

			if (k + b_size < n)
				apply_block_reflector(k, b_size, k + b_size, n - k - b_size);
		}
	}

} // namespace SequentialBlockedQR