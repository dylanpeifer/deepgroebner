from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "buchberger.cpp":
    pass

cdef extern from "buchberger.h":
    cdef cppclass LeadMonomialsEnv:
        LeadMonomialsEnv() except +
        LeadMonomialsEnv(string, bint, bint, int) except +
        void reset()
        double step(int)
        void seed(int)

        vector[int] state
        int cols
