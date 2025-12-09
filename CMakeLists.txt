cmake_minimum_required(VERSION 3.10)

project(CholeskyBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------
# 1. OPENMP
# -----------------------------------------------------------
find_package(OpenMP REQUIRED)

# -----------------------------------------------------------
# 2. EIGEN (OVO JE KLJUČNO)
# -----------------------------------------------------------
# Na osnovu tvoje slike, putanja mora voditi do foldera
# u kojem se nalazi onaj "Eigen" folder sa slike.
# PAŽNJA: Koristimo obicne kose crte (/) umesto Windows (\)

include_directories("C:/Users/PC/Downloads/eigen-5.0.0/eigen-5.0.0")

# -----------------------------------------------------------
# 3. ZASTAVICE (FLAGS)
# -----------------------------------------------------------
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast /openmp)
else()
    # Ovo je za tvoj MinGW (GCC) koji vidim u logovima
    add_compile_options(-O3 -mavx2 -mfma -march=native)
endif()

# -----------------------------------------------------------
# 4. DEFINISANJE PROGRAMA
# -----------------------------------------------------------

# Program 1
add_executable(program_omp_simd OMP_SIMD_.cpp)
target_link_libraries(program_omp_simd PRIVATE OpenMP::OpenMP_CXX)

# Program 2
add_executable(program_blokovska Blokovska_implementacija2.cpp)

# Program 3
add_executable(program_parallel Cholesky_blocked2_parallel.cpp)
target_link_libraries(program_parallel PRIVATE OpenMP::OpenMP_CXX)

# Program 4
add_executable(program_simd_v2 simd_v2.cpp)