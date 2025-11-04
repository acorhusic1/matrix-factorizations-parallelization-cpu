// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "QR_naivna.h"

int main() {
    try {
        std::cout << "--- Testovi klasa Vector i Matrix ---" << std::endl;
        Vector v1{ 1, 2, 3 };
        Matrix m1{ {1, 2, 3}, {4, 5, 6}, {7, 8, 10} };
        std::cout << "Originalna matrica m1:" << std::endl;
        m1.Print();
        Matrix m1_inv = Inverse(m1);
        std::cout << "Inverzna matrica m1:" << std::endl;
        m1_inv.Print();
        Matrix jedinicna = m1 * m1_inv;
        std::cout << "Proizvod m1 * m1_inv (trebala bi biti jedinicna):" << std::endl;
        jedinicna.Chop(1e-10); // "Ocisti" male greske
        jedinicna.Print();

        Matrix m_div = jedinicna / m1;
        std::cout << "Rezultat jedinicna / m1 (trebao bi biti inverz od m1):" << std::endl;
        m_div.Print();

        Matrix singularna{ {1, 2, 3}, {2, 4, 6}, {5, 6, 7} };
        std::cout << "Determinanta singularne matrice: " << Det(singularna) << std::endl;
        try {
            Inverse(singularna);
        }
        catch (const std::domain_error& e) {
            std::cout << "Ocekivan izuzetak za inverz singularne: " << e.what() << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "Neocekivan izuzetak u osnovnim testovima: " << e.what() << std::endl;
    }

    std::cout << "\n--- Test klase QRDecomposer ---" << std::endl;
    try {
        Matrix A{ {0, 3, 2}, {4, 6, 1}, {3, 1, 7} }; // Matrica sa a[0][0]=0
        std::cout << "Originalna matrica A:" << std::endl;
        A.Print();

        QRDecomposer qr(A);

        Matrix Q = qr.GetQ();
        Matrix R = qr.GetR();

        std::cout << "Matrica Q:" << std::endl;
        Q.Print();
        std::cout << "Matrica R:" << std::endl;
        R.Print();

        Matrix A_reconstructed = Q * R;
        std::cout << "Rekonstruisana matrica Q * R:" << std::endl;
        A_reconstructed.Print();

        if (A.EqualTo(A_reconstructed, 1e-9)) {
            std::cout << "Test USPJESAN: Q * R je jednako A." << std::endl;
        }
        else {
            std::cout << "Test NEUSPJESAN: Q * R nije jednako A." << std::endl;
        }

        // Test rjesavanja sistema Ax = b
        Vector b{ 10, 11, 20 };
        Vector x = qr.Solve(b);
        std::cout << "\nRjesavamo sistem Ax=b za b = ";
        b.Print(' ');
        std::cout << "\nRjesenje x je: ";
        x.Print(' ');
        std::cout << "\nProvjera A*x: ";
        (A * x).Print(' ');
        std::cout << std::endl;
        if (b.EqualTo(A * x, 1e-9)) {
            std::cout << "Test USPJESAN: Rjesenje sistema je tacno." << std::endl;
        }
        else {
            std::cout << "Test NEUSPJESAN: Rjesenje sistema nije tacno." << std::endl;
        }


        // Test sa singularnom matricom
        try {
            Matrix singularna{ {1, 2, 3}, {2, 4, 6}, {7, 8, 1} };
            QRDecomposer test(singularna);
        }
        catch (const std::domain_error& e) {
            std::cout << "\nOVO JE OCEKIVAN IZUZETAK za singularnu matricu! Detalji: " << e.what() << std::endl;
        }

        // Test sa neispravnim formatom
        try {
            Matrix neispravna{ {1, 2, 3}, {4, 5, 6} }; // redova < kolona
            QRDecomposer test(neispravna);
        }
        catch (const std::domain_error& e) {
            std::cout << "OVO JE OCEKIVAN IZUZETAK za neispravan format! Detalji: " << e.what() << std::endl;
        }

        // Test MulQWith i MulQTWith
        Vector v{ 1, 2, 3 };
        std::cout << "\nTest MulQTWith(v): ";
        qr.MulQTWith(v).Print(' ');
        std::cout << std::endl;


    }
    catch (const std::domain_error& e) {
        std::cout << "Neocekivan domain_error izuzetak u QR testovima: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "Neocekivan izuzetak u QR testovima: " << e.what() << std::endl;
    }

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
