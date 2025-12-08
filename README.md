# Matrix Factorizations and Parallelization

Ovaj projekat demonstrira **LU faktorizaciju matrica** korištenjem različitih pristupa u **C++**:  

- **Naivna implementacija**  
- **Optimizovana implementacija**  
- **Blokovska implementacija** (dvije verzije)  
- **Blokovska verzija sa OpenMP paralelizacijom**
- **Blokovska verzija sa SIMD**  
- **Blokovska verzija sa OpenMP + SIMD**  
- **Opcionalno: LU faktorizacija korištenjem LAPACK/OpenBLAS biblioteka**  

Glavna svrha je prikazati efikasne metode faktorizacije velikih matrica i mjeriti vrijeme izvođenja različitih implementacija.

---

## Opis koda

U `LU.cpp` implementirane su funkcije:  

- `LU_naivna` – jednostavna, direktna implementacija Doolittle LU faktorizacije.  
- `LU_optimizovana` – optimizovana verzija sa manjim brojem ponavljanja i direktnim pristupom memoriji.  
- `LU_blokovska_V1` i `LU_blokovska_V2` – blokovske verzije, koje koriste **Schur update** i blokiranje za bolje iskorištavanje cache memorije.  
- `LU_blokovska_V1_omp` – paralelizovana verzija blokovske implementacije koristeći **OpenMP**.
- `LU_blokovska_V1_omp` – verzija blokovske implementacije koristeći **SIMD**.  
- `LU_blokovska_V1_omp` – paralelizovana verzija blokovske implementacije koristeći **OpenMP** + **SIMD**.  

- `checkLU` – funkcija koja provjerava ispravnost faktorizacije (`A = L*U`).  

Primjeri optimizacija uključuju:  
- Loop unrolling  
- Blocked matrix operations za poboljšanje performansi cache-a  
- OpenMP paralelizaciju za velike matrice  

U `main()` se generiše nasumična matrica, a svaka implementacija se izvršava i mjeri njeno vrijeme izvođenja.

---

## Tehnologije

Za pokretanje projekta potrebno je:

- **GCC / g++** (C++17 ili noviji)  
- **OpenMP** (uključena u moderni GCC)  
- **LAPACK/OpenBLAS** biblioteke ako želite koristiti LAPACK verziju  
- Operativni sistem: Windows / Linux / macOS

---

## Kompajliranje i pokretanje

### 1. Naivna i optimizovana implementacija (OpenMP)

Kompajliranje jednim komandama:

```bash
g++ -O3 -fopenmp -march=native -mavx2 -ffast-math -funroll-loops LU.cpp -o LU.exe
```

---

#### Ako želite koristiti LAPACK/OpenBLAS implementaciju:

```bash
g++ -O3 -fopenmp -march=native -mavx2 -ffast-math -funroll-loops LU_LAPACK.cpp     -I<putanja_do_include> -L<putanja_do_lib> -lopenblas -llapack -o LU_LAPACK.exe
```
