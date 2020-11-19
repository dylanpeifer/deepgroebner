# distutils: language = c++
# distutils: sources = deepgroebner/polynomials.cpp
# distutils: extra_compile_args = -std=c++17

import numpy as np

from buchberger cimport LeadMonomialsEnv


cdef class CLeadMonomialsEnv:
    cdef LeadMonomialsEnv c_env

    def __cinit__(self):
        self.c_env = LeadMonomialsEnv()

    def reset(self):
        self.c_env.reset()
        state = np.array(self.c_env.state)
        return state

    def step(self, action):
        reward = self.c_env.step(action)
        state = np.array(self.c_env.state)
        return state, reward, state.shape[0] == 0, {}

    def copy(self):
        copy = CLeadMonomialsEnv()
        copy.c_env = LeadMonomialsEnv(self.c_env)
        return copy

    def seed(self, seed=None):
        if seed is not None:
            self.c_env.seed(seed)
