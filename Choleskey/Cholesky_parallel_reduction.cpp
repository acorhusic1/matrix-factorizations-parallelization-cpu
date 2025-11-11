#include <iostream>
#include <vector>
#include <cmath> 
#include <iomanip>
#include <chrono>
#include <omp.h>

using namespace std;
using Matrix = vector<vector<double>>;

Matrix Cholesky(const Matrix& A) {
	size_t n = A.size();
	const double EPS = 1e-12;

	if (A.empty()) {
		cout << "Greska: Matrica je prazna!" << endl;
		return {};
	}

	for (const auto& row : A) {
		if (row.size() != n) {
			cout << "Greska: Matrica nije kvadratna!" << endl;
			return {};
		}
	}


	for (size_t i = 0; i < n; i++) {
		for (size_t j = i + 1; j < n; j++) {
			if (fabs(A[i][j] - A[j][i]) > EPS) {
				cout << "Greska: Matrica nije simetricna!" << endl;
				return {};
			}
		}
	}


	Matrix L(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j <= i; j++) {
			double suma = 0.0;
#pragma omp parallel for reduction(+:suma)
			for (int k = 0; k < (int) j; k++)
				suma += L[i][k] * L[j][k];
			if (i == j) {
				double val = A[i][i] - suma;
				if (val <= 0) {
					cout << "Matrica nije pozitivno definitna!\n";
					exit(1);
				}
				L[i][j] = sqrt(val);
			}
			else {
				L[i][j] = (A[i][j] - suma) / L[j][j];
			}
		}
	}

	return L;
}
/*cout << "L = " << endl;
for (size_t i = 0; i < n; i++) {
	for (size_t j = 0; j < n; j++)
		cout << setw(10) << fixed << setprecision(4) << L[i][j] << " ";
	cout << endl;
}
*/
/*	cout << "\n Provjera: L * L^T = " << endl;
	Matrix check(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; i++){
		for (size_t j = 0; j < n; j++){
			for (size_t k = 0; k < n; k++)
				check[i][j] += L[i][k] * L[j][k];
		}
	}

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
			cout << setw(10) << fixed << setprecision(4) << check[i][j] << " ";
		cout << endl;
	}
	return L;
}

*/

Matrix generisiPozitivnoDefinitnu(size_t n) {
	Matrix A(n, vector<double>(n));

	for (size_t i = 0; i < n; i++) {
		for (size_t j = i; j < n; j++) {
			double val = (double)rand() / RAND_MAX;
			A[i][j] = A[j][i] = val;
		}
	}

	Matrix C(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++)
			for (size_t k = 0; k < n; k++)
				C[i][j] += A[i][k] * A[j][k];

	return C;
}

int main() {
	/*	cout << "Test 1: Ispravna matrica\n" << endl;
		Matrix A = { { 4, 12, -16 }, { 12, 37, -43 }, { -16, -43, 98 } };
		Cholesky(A);


		auto start = chrono::high_resolution_clock::now();
		auto L = Cholesky(A);
		auto end = chrono::high_resolution_clock::now();

		auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
		cout << "Vrijeme izvrsavanja: " << duration.count() << " ms" << endl;

	*/
	srand(time(0));

	vector<int> velicine = { 500, 1000, 2000, 3000, 5000, 6000 };

	for (int n : velicine) {
		cout << "\n Test za matricu " << n << "x" << n << endl;
		Matrix A = generisiPozitivnoDefinitnu(n);

		auto start = chrono::high_resolution_clock::now();
		Matrix L = Cholesky(A);
		auto end = chrono::high_resolution_clock::now();

		auto trajanje = chrono::duration_cast<chrono::milliseconds>(end - start).count();
		cout << "Vrijeme izvrsavanja: " << trajanje << " ms" << endl;
	}

	/*
	cout << "\n Test 2: Prazna matrica \n" << endl;
	Matrix B = {};
	Cholesky(B);

	cout << "\n Test 3: Matrica nije kvadratna \n " << endl;
	Matrix C = { {4, 12}, {-16, 12}, {37, -43} };
	Cholesky (C);

	cout << "\n Test 4: Matrica nije simetricna \n " << endl;
	Matrix D = { { 1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
	Cholesky(D);

	cout << "\n Test 5: Matrica nije pozitivno definitna \n" << endl;
	Matrix E = { {1, 0}, {0, -1} };
	Cholesky(E);

	*/
	//cout << "Broj niti: " << omp_get_max_threads() << endl;
	return 0;
}