cmake_minimum_required(VERSION 3.16.3)
project(deepgroebner)
set(CMAKE_CXX_STANDARD 17)

set(INSTALL_GTEST OFF)
add_subdirectory(googletest)


add_executable(cyclic scripts/cyclic.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(cyclic PUBLIC deepgroebner)

add_executable(random_episodes scripts/random_episodes.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(random_episodes PUBLIC deepgroebner)

add_executable(make_strat scripts/make_strat.cpp deepgroebner/ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp)
target_include_directories(make_strat PUBLIC deepgroebner)

add_executable(test_polynomials tests/test_polynomials.cpp
  deepgroebner/polynomials.cpp)
target_include_directories(test_polynomials PUBLIC deepgroebner)
target_link_libraries(test_polynomials gtest_main)

add_executable(test_ideals tests/test_ideals.cpp
  deepgroebner/polynomials.cpp deepgroebner/ideals.cpp)
target_include_directories(test_ideals PUBLIC deepgroebner)
target_link_libraries(test_ideals gtest_main)

add_executable(test_buchberger tests/test_buchberger.cpp
  deepgroebner/polynomials.cpp deepgroebner/buchberger.cpp deepgroebner/ideals.cpp)
target_include_directories(test_buchberger PUBLIC deepgroebner)
target_link_libraries(test_buchberger gtest_main)
