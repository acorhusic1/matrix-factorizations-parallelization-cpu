#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cstring> 
#include <cblas.h>

#include "QR_naivna.h"
#include "QR_sekvencijalna_blokovska.h"
#include "QR_paralelna_blokovska.h"

namespace NQR = NaiveQR;
namespace SQB = SequentialBlockedQR;
namespace PQB = ParallelBlockedQR;

// Forward declare LAPACK functions
extern "C" {
	int dgeqrf_(int* M, int* N, double* A, int* LDA, double* TAU, double* WORK, int* LWORK, int* INFO);
	int dorgqr_(int* M, int* N, int* K, double* A, int* LDA, double* TAU, double* WORK, int* LWORK, int* INFO);
}

// ====================================================================
// STANDARD HELPER FUNCTIONS (UNCHANGED)
// ====================================================================

std::vector<double> GenerateRandomMatrix(int rows, int cols) {
	std::vector<double> m(rows * cols);
	for (int j = 0; j < cols; j++) {
		for (int i = 0; i < rows; i++) {
			m[j * rows + i] = (double)rand() / RAND_MAX * 200.0 - 100.0;
		}
	}
	return m;
}

std::vector<double> MatrixMultiplyBLAS(const std::vector<double>& A, int m, int k,
	const std::vector<double>& B, int k2, int n) {
	if (k != k2) throw std::domain_error("Incompatible matrix dimensions");
	std::vector<double> C(m * n, 0.0);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, 1.0, A.data(), m, B.data(), k, 0.0, C.data(), m);
	return C;
}

double RelativeErrorVector(const std::vector<double>& A, const std::vector<double>& B) {
	double diff_sq = 0.0;
	double a_sq = 0.0;
	for (size_t i = 0; i < A.size(); i++) {
		double diff = A[i] - B[i];
		diff_sq += diff * diff;
		a_sq += A[i] * A[i];
	}
	return (a_sq > SQB::EPSILON) ? std::sqrt(diff_sq) / std::sqrt(a_sq) : std::sqrt(diff_sq);
}

double OrthogonalityErrorVector(const std::vector<double>& QtQ, int N) {
	double err_sq = 0.0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			double identity_val = (i == j ? 1.0 : 0.0);
			double diff = QtQ[j * N + i] - identity_val;
			err_sq += diff * diff;
		}
	}
	return std::sqrt(err_sq);
}

std::vector<double> ExtractR_LAPACK(const std::vector<double>& A_lapack, int M, int N) {
	std::vector<double> R(N * N, 0.0);
	for (int j = 0; j < N; j++) {
		for (int i = 0; i <= j; i++) {
			R[j * N + i] = A_lapack[j * M + i];
		}
	}
	return R;
}

std::vector<double> ExtractQ_LAPACK(std::vector<double> A_lapack, const std::vector<double>& tau, int M, int N) {
	int m = M, n = N, k = N, lda = M, lwork = -1, info;
	double wkopt;
	dorgqr_(&m, &n, &k, A_lapack.data(), &lda, (double*)tau.data(), &wkopt, &lwork, &info);
	lwork = (int)wkopt;
	std::vector<double> work(lwork);
	dorgqr_(&m, &n, &k, A_lapack.data(), &lda, (double*)tau.data(), work.data(), &lwork, &info);
	return A_lapack;
}


// ====================================================================
// MAIN COMPARISON LOGIC
// ====================================================================

int main() {
	srand((unsigned int)time(0));
	std::cout << std::fixed << std::setprecision(3);
	const double CHECK_EPS = 1e-7;

	std::cout << "\n------------------------------------------------------------\n";
	std::cout << "  QR DECOMPOSITION: NAIVE vs SEQ vs PARALLEL vs LAPACK \n";
	std::cout << "============================================================\n\n";

	// ====================================================================
	// Generalized Randomized Matrix (Comparison)
	// ====================================================================
	{
		// --- USER DEFINED PARAMETERS ---
		const int M = 500; // Rows
		const int N = 500; // Columns 
		const int B = 32;   // Block size

		std::cout << "\n------------------------------------------------------------\n";
		std::cout << "Performance Comparison (" << M << "x" << N << ", Block Size = " << B << ")\n";
		std::cout << "------------------------------------------------------------\n\n";

		int TEST_RUNS = 100;

		double t_naive = 0, t_seq = 0, t_par = 0, t_lapack = 0;
		int failed = 0, runs = 0;

		for (int run = 0; run < TEST_RUNS; run++) {
			std::vector<double> A_large_vec = GenerateRandomMatrix(M, N);

			// =================================================
			// 0. NAIVE IMPLEMENTATION (Row-Major, Unblocked)
			// =================================================
			NQR::Matrix A_naive(M, N);
			// Copy data manually (Column-Major vector -> Row-Major Naive Matrix)
			for (int j = 0; j < N; ++j) {
				for (int i = 0; i < M; ++i) {
					// Naive Matrix is A(i,j), A_large_vec is [j*M + i]
					A_naive(i, j) = A_large_vec[j * M + i];
				}
			}

			NQR::SimpleQR qr_naive(M, N);
			qr_naive.U = A_naive; // Copy data in

			auto start_naive = std::chrono::high_resolution_clock::now();
			qr_naive.factorize();
			t_naive += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_naive).count();

			// =================================================
			// 1. SEQUENTIAL IMPLEMENTATION (SequentialBlockedQR)
			// =================================================
			SQB::Matrix A_seq_in(M, N);
			std::memcpy(A_seq_in.data, A_large_vec.data(), M * N * sizeof(double));

			SQB::BlockedQR qr_seq(M, N, B);
			qr_seq.U = A_seq_in;

			auto start_seq = std::chrono::high_resolution_clock::now();
			qr_seq.factorize();
			t_seq += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_seq).count();

			// =================================================
			// 2. PARALLEL IMPLEMENTATION (ParallelBlockedQR)
			// =================================================
			PQB::Matrix A_par_in(M, N);
			std::memcpy(A_par_in.data, A_large_vec.data(), M * N * sizeof(double));

			PQB::BlockedQR qr_par(M, N, B);
			qr_par.U = A_par_in;

			auto start_par = std::chrono::high_resolution_clock::now();
			qr_par.factorize();
			t_par += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_par).count();

			// =================================================
			// 3. LAPACK IMPLEMENTATION (dgeqrf)
			// =================================================
			std::vector<double> A_lap = A_large_vec;
			std::vector<double> tau_lapack(N);
			int m = M, n = N, lda = M, lwork = -1, info;
			double wkopt;

			dgeqrf_(&m, &n, A_lap.data(), &lda, tau_lapack.data(), &wkopt, &lwork, &info);
			lwork = (int)wkopt;
			std::vector<double> work(lwork);

			auto start_lapack = std::chrono::high_resolution_clock::now();
			dgeqrf_(&m, &n, A_lap.data(), &lda, tau_lapack.data(), work.data(), &lwork, &info);
			t_lapack += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_lapack).count();

			runs++;
	/*
			// =================================================
			// 4. VERIFICATION (Q*R = A and Q^T*Q = I)
			// =================================================

			bool ok_naive = true, ok_seq = true, ok_par = true, ok_lapack = true;

			// --- 4.0 NAIVE CHECK ---
			NQR::Matrix Q_nv = qr_naive.extract_Q();
			NQR::Matrix R_nv = qr_naive.extract_R();
			NQR::Matrix QR_nv = NQR::SimpleQR::multiply(Q_nv, R_nv);

			// Manual check for Row-Major Matrix
			double err_recon_nv = 0.0;
			double norm_A_nv = 0.0;
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					double diff = A_naive(i, j) - QR_nv(i, j);
					err_recon_nv += diff * diff;
					norm_A_nv += A_naive(i, j) * A_naive(i, j);
				}
			}
			err_recon_nv = (norm_A_nv > 1e-20) ? std::sqrt(err_recon_nv) / std::sqrt(norm_A_nv) : std::sqrt(err_recon_nv);

			if (err_recon_nv >= CHECK_EPS) ok_naive = false;

			// --- 4.1 SEQUENTIAL CHECK ---
			SQB::Matrix Q_seq = qr_seq.extract_Q();
			SQB::Matrix R_seq = qr_seq.extract_R();
			SQB::Matrix QR_seq = SQB::BlockedQR::multiply(Q_seq, R_seq);
			SQB::Matrix QtQ_seq = SQB::BlockedQR::multiply(~Q_seq, Q_seq);

			std::vector<double> A_rec_seq_vec(M * N);
			std::memcpy(A_rec_seq_vec.data(), QR_seq.data, M * N * sizeof(double));
			std::vector<double> QtQ_seq_vec(N * N);
			std::memcpy(QtQ_seq_vec.data(), QtQ_seq.data, N * N * sizeof(double));

			double err_recon_seq = RelativeErrorVector(A_large_vec, A_rec_seq_vec);
			double err_ortho_seq = OrthogonalityErrorVector(QtQ_seq_vec, N);

			if (err_recon_seq >= CHECK_EPS || err_ortho_seq >= CHECK_EPS) ok_seq = false;

			// --- 4.2 PARALLEL CHECK ---
			PQB::Matrix Q_par = qr_par.extract_Q();
			PQB::Matrix R_par = qr_par.extract_R();
			PQB::Matrix QR_par = PQB::BlockedQR::multiply(Q_par, R_par);
			PQB::Matrix QtQ_par = PQB::BlockedQR::multiply(~Q_par, Q_par);

			std::vector<double> A_rec_par_vec(M * N);
			std::memcpy(A_rec_par_vec.data(), QR_par.data, M * N * sizeof(double));
			std::vector<double> QtQ_par_vec(N * N);
			std::memcpy(QtQ_par_vec.data(), QtQ_par.data, N * N * sizeof(double));

			double err_recon_par = RelativeErrorVector(A_large_vec, A_rec_par_vec);
			double err_ortho_par = OrthogonalityErrorVector(QtQ_par_vec, N);

			if (err_recon_par >= CHECK_EPS || err_ortho_par >= CHECK_EPS) ok_par = false;

			// --- 4.3 LAPACK CHECK ---
			std::vector<double> Q_lapack_vec = ExtractQ_LAPACK(A_lap, tau_lapack, M, N);
			std::vector<double> R_lapack_vec = ExtractR_LAPACK(A_lap, M, N);
			std::vector<double> QtQ_lapack_vec(N * N, 0.0);
			cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
				N, N, M, 1.0, Q_lapack_vec.data(), M, Q_lapack_vec.data(), M, 0.0, QtQ_lapack_vec.data(), N);

			std::vector<double> A_rec_lapack_vec = MatrixMultiplyBLAS(Q_lapack_vec, M, N, R_lapack_vec, N, N);

			double err_recon_lapack = RelativeErrorVector(A_large_vec, A_rec_lapack_vec);
			double err_ortho_lapack = OrthogonalityErrorVector(QtQ_lapack_vec, N);

			if (err_recon_lapack >= CHECK_EPS || err_ortho_lapack >= CHECK_EPS) ok_lapack = false;
*/
			// --- PRINT RESULTS ---
			std::cout << std::fixed << std::setprecision(6);
			std::cout << "[RUN " << run + 1 << "]: ";
//			std::cout << "NAIVE=" << (ok_naive ? "OK" : "FAIL") << " | ";
//			std::cout << "SEQ=" << (ok_seq ? "OK" : "FAIL") << " | ";
//			std::cout << "PAR=" << (ok_par ? "OK" : "FAIL") << " | ";
//			std::cout << "LAP=" << (ok_lapack ? "OK" : "FAIL") << " | ";
			std::cout << "Time: N=" << (t_naive / runs) << "s | S=" << (t_seq / runs) << "s | P=" << (t_par / runs) << "s | L=" << (t_lapack / runs) << "s" << std::endl;

//			if (!ok_naive || !ok_seq || !ok_par || !ok_lapack) failed++;
		}

		if (runs > 0) {
			std::cout << "\n" << std::string(70, '=') << std::endl;
			std::cout << " RESULTS FOR " << M << "x" << N << " (BLOCK=" << B << ") - AVERAGE" << std::endl;
			std::cout << std::string(70, '=') << std::endl;

			std::cout << "Mean time elapsed (SEQUENTIAL): " << std::fixed << std::setprecision(6) << (t_seq / runs) << " s" << std::endl;
			std::cout << "Mean time elapsed (PARALLEL):   " << (t_par / runs) << " s" << std::endl;
			std::cout << "Mean time elapsed (LAPACK):     " << (t_lapack / runs) << " s" << std::endl;
			std::cout << "------------------------------------------------------------" << std::endl;
			std::cout << "Speedup (T_NAIVE / T_SEQ): " << std::fixed << std::setprecision(2) << (t_naive / t_seq) << "x" << std::endl;
			std::cout << "Speedup (T_SEQ / T_PAR): " << std::fixed << std::setprecision(2) << (t_seq / t_par) << "x" << std::endl;
			std::cout << "Speedup (T_PAR / T_LAP): " << (t_par / t_lapack) << "x" << std::endl;
			std::cout << "------------------------------------------------------------" << std::endl;
//			std::cout << "TOTAL ERRORS: " << failed << "/" << runs << std::endl;
			std::cout << std::string(70, '=') << std::endl;
		}

	}

	return 0;
}
