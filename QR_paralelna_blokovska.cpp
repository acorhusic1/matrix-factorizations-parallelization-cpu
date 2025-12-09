#include "QR_paralelna_blokovska.h"
#include <iostream>

namespace ParallelBlockedQR {

	// ============================================================================
	// BLAS-3: Matrix-Matrix Multiply (Sequential version, unused but kept)
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
#pragma omp parallel for collapse(2)
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
#pragma omp parallel for collapse(2)
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
#pragma omp parallel for collapse(2)
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
	// BLOCKED QR FACTORIZER IMPLEMENTATIONS (Parallelized)
	// ============================================================================

	void BlockedQR::panel_qr(int k, int b_size) {
		// This loop must remain SEQUENTIAL because columns are applied one after another
		for (int j = 0; j < b_size; ++j) {
			int col = k + j;

			// Compute norm (minor parallelization, but better than none)
			double norm = 0.0;
#pragma omp parallel for reduction(+: norm)
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

			// Normalize u (parallelized)
#pragma omp parallel for
			for (int i = col + 1; i < m; ++i)
				U(i, col) /= v0;

			// Compute tau (parallelized)
			double u_norm_sq = 0.0;
#pragma omp parallel for reduction(+: u_norm_sq)
			for (int i = col + 1; i < m; ++i)
				u_norm_sq += U(i, col) * U(i, col);
			tau[col] = 2.0 / (1.0 + u_norm_sq);

			// Apply to remaining columns in panel [col+1 : k+b_size] (parallelized)
#pragma omp parallel for
			for (int jj = col + 1; jj < k + b_size; ++jj) {
				// Dot product
				double dot = U(col, jj);
				for (int i = col + 1; i < m; ++i)
					dot += U(i, col) * U(i, jj);

				// Update column (vector style)
				U(col, jj) -= tau[col] * dot;
				for (int i = col + 1; i < m; ++i)
					U(i, jj) -= tau[col] * dot * U(i, col);
			}

			U(col, col) = -norm;
		}
	}

	/**
	 * LARFT: Compute upper triangular T factor (Parallelized)
	 */
	void BlockedQR::compute_T(int k, int b_size, Matrix& T) {
		// Initialize T to zero (parallelized)
#pragma omp parallel for collapse(2)
		for (int j = 0; j < b_size; ++j)
			for (int i = 0; i < b_size; ++i)
				T(i, j) = 0.0;

		// Temp buffer for matrix-vector multiply (local, avoids shared memory)
		std::vector<double> t_col(b_size);

		for (int j = 0; j < b_size; ++j) {
			int col = k + j;
			if (tau[col] == 0.0) continue;

			// 1. Compute Raw Column: T(0:j-1, j) = -tau[j] * V(:, 0:j-1)' * V(:, j)
			// Parallelize the inner loop (over i)
#pragma omp parallel for
			for (int i = 0; i < j; ++i) {
				int col_i = k + i;
				double sum = 0.0;

				// V_i' * V_j (dot product)
				sum += U(col, col_i);

				for (int row = col + 1; row < m; ++row)
					sum += U(row, col_i) * U(row, col);

				T(i, j) = -tau[col] * sum;
			}

			// 2. Triangular Matrix-Vector Multiply: T(0:j-1, j) = T(0:j-1, 0:j-1) * T(0:j-1, j)
			// Copy current column to temp (sequential, minor work)
			for (int i = 0; i < j; ++i) t_col[i] = T(i, j);

			// Standard TRMV logic (Upper Triangular) - sequential due to dependency
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
	 * LARFB: Apply block reflector (I - V*T*V^T) to trailing matrix (Parallelized)
	 * Largest opportunity for parallelization (BLAS-3 operation)
	 */
	void BlockedQR::apply_block_reflector(int k, int b_size, int trail_start, int trail_cols) {
		int m_sub = m - k;

		Matrix T(b_size, b_size);
		compute_T(k, b_size, T);

		// W = V^T * C
		// W is b_size x trail_cols. Parallelization over columns (j)
		Matrix W(b_size, trail_cols);
#pragma omp parallel for
		for (int j = 0; j < trail_cols; ++j) {
			for (int i = 0; i < b_size; ++i) {
				// Part 1: Interaction with implicit 1 of V
				double sum = U(k + i, trail_start + j);

				// Part 2: Interaction with rest of V (dot product)
				for (int row = i + 1; row < m_sub; ++row)
					sum += U(k + row, k + i) * U(k + row, trail_start + j);

				W(i, j) = sum;
			}
		}

		// W2 = T^T * W
		// Standard DGEMM operation (T is Upper Triangular, T^T is Lower Triangular)
		Matrix W2(b_size, trail_cols);
		// Use DGEMM for full BLAS-3 optimization (M-sized columns)
		dgemm_parallel('T', 'N', b_size, trail_cols, b_size, 1.0, T, W, 0.0, W2);

		// C = C - V * W2
		// Parallelization over columns (j)
#pragma omp parallel for
		for (int j = 0; j < trail_cols; ++j) {
			for (int i = 0; i < b_size; ++i) {
				double w_val = W2(i, j);

				// Update column: C(:, j) -= V(:, i) * W2(i, j)
				// Apply 1.0 at row k+i
				U(k + i, trail_start + j) -= w_val;

				// Apply remaining vector (DGEMV style)
				for (int row = i + 1; row < m_sub; ++row)
					U(k + row, trail_start + j) -= U(k + row, k + i) * w_val;
			}
		}
	}

	void BlockedQR::factorize() {
		for (int k = 0; k < n; k += b) {
			int b_size = std::min(b, n - k);

		//	std::cout << "\n########## BLOCK " << (k / b) << ": columns " << k << " to " << (k + b_size - 1) << " ##########\n";

			// PANEL (LARFG + LAFM) - must be sequential across panels
			panel_qr(k, b_size);

			if (k + b_size < n)
				// LARFB (Block Reflector) - main opportunity for BLAS-3 parallelization
				apply_block_reflector(k, b_size, k + b_size, n - k - b_size);
		}
	}

} // namespace ParallelBlockedQR