# distutils: language = c++
# distutils: sources = deepgroebner/polynomials.cpp
# distutils: extra_compile_args = -std=c++17

from buchberger cimport BuchbergerEnv


cdef class CBuchbergerEnv:
    cdef BuchbergerEnv c_env

    def __cinit__(self, i):
        self.c_env = BuchbergerEnv(i)

    def reset(self):
        x = self.c_env.reset()
        return x

    def step(self, action):
        i, j = action
        x = self.c_env.step(i, j)
        return x
