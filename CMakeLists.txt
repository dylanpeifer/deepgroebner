cmake_minimum_required(VERSION 3.16.3)
project(deepgroebner)
set(CMAKE_CXX_STANDARD 17)

set(INSTALL_GTEST OFF)
add_subdirectory(googletest)

add_executable(cyclic5 benchmarks/cyclic5.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(cyclic5 PUBLIC deepgroebner)

add_executable(cyclic6 benchmarks/cyclic6.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(cyclic6 PUBLIC deepgroebner)

add_executable(cyclic7 benchmarks/cyclic7.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(cyclic7 PUBLIC deepgroebner)

add_executable(3-20-10-uniform benchmarks/3-20-10-uniform.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(3-20-10-uniform PUBLIC deepgroebner)

add_executable(test_polynomials tests/test_polynomials.cpp
  deepgroebner/polynomials.cpp)
target_include_directories(test_polynomials PUBLIC deepgroebner)
target_link_libraries(test_polynomials gtest_main)

add_executable(test_buchberger tests/test_buchberger.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp deepgroebner/ideals.cpp)
target_include_directories(test_buchberger PUBLIC deepgroebner)
target_link_libraries(test_buchberger gtest_main)
