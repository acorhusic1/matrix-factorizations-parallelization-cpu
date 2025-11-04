#include <iostream>
#include <vector>
#include <cmath> 
#include <iomanip>

using namespace std; 
using Matrix = vector<vector<double>>;

Matrix Cholesky(const Matrix& A)
{
	size_t n = A.size();

	for (size_t i = 0; i < n; i++) {
		for (size_t j = i + 1; j < n; j++) {
			if (fabs(A[i][j] - A[j][i]) > 1e-9) {
				cout << "Matrica nije simetri?na!" << endl;
				exit(1);
			}
		}
	}

	Matrix L(n, vector<double>(n, 0.0)); 
	for (size_t i = 0; i < n; i++) { 
		for (size_t j = 0; j <= i; j++) {
			double suma = 0.0;
			for (size_t k = 0; k < j; k++)
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
	cout << "L = " << endl;
	for (size_t i = 0; i < n; i++) 
	{ 
		for (size_t j = 0; j < n; j++)
			cout << setw(10) << fixed << setprecision(4) << L[i][j] << " ";
		cout << endl;
	}

	cout << "\n Provjera: L * L^T = " << endl;
	Matrix check(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{ 
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

int main() {
	Matrix A = { { 4, 12, -16 }, { 12, 37, -43 }, { -16, -43, 98 } };
	Cholesky(A); 
	return 0; 
}