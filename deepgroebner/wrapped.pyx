# distutils: language = c++
# distutils: sources = deepgroebner/polynomials.cpp deepgroebner/ideals.cpp
# distutils: extra_compile_args = -std=c++17
# cython: language_level = 3

import numpy as np

from .buchberger cimport LeadMonomialsEnv


cdef class CLeadMonomialsEnv:
    cdef LeadMonomialsEnv c_env

    def __cinit__(self, ideal_dist='3-20-10-weighted', elimination='gebauermoeller',
                  rewards='additions', sort_input=False, sort_reducers=True, k=1):
        self.c_env = LeadMonomialsEnv(ideal_dist.encode(), sort_input, sort_reducers, k)

    def reset(self):
        self.c_env.reset()
        state = np.array(self.c_env.state, dtype=np.int32).reshape(-1, self.c_env.cols)
        return state

    def step(self, action):
        reward = self.c_env.step(action)
        state = np.array(self.c_env.state, dtype=np.int32).reshape(-1, self.c_env.cols)
        return state, reward, state.shape[0] == 0, {}

    def seed(self, seed=None):
        if seed is not None:
            self.c_env.seed(seed)

    def value(self, gamma=0.99):
        return self.c_env.value(gamma)
