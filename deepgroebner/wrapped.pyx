# distutils: language = c++
# distutils: sources = deepgroebner/polynomials.cpp deepgroebner/ideals.cpp
# distutils: extra_compile_args = -std=c++17

import numpy as np

from buchberger cimport LeadMonomialsEnv


cdef class CLeadMonomialsEnv:
    cdef LeadMonomialsEnv c_env
    cdef int k, n

    def __cinit__(self, ideal_dist='3-20-10-weighted', elimination='gebauermoeller', rewards='additions', k=1):
        self.c_env = LeadMonomialsEnv()
        self.k = k
        self.n = 3

    def reset(self):
        self.c_env.reset()
        state = np.array(self.c_env.state, dtype=np.int32).reshape(-1, 2 * self.k * self.n)
        return state

    def step(self, action):
        reward = self.c_env.step(action)
        state = np.array(self.c_env.state, dtype=np.int32).reshape(-1, 2 * self.k * self.n)
        return state, reward, state.shape[0] == 0, {}

    def copy(self):
        copy = CLeadMonomialsEnv()
        copy.c_env = LeadMonomialsEnv(self.c_env)
        copy.k = self.k
        copy.n = self.n
        return copy

    def seed(self, seed=None):
        if seed is not None:
            self.c_env.seed(seed)
