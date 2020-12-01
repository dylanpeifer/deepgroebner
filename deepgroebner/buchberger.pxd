from libcpp.vector cimport vector

cdef extern from "buchberger.cpp":
    pass

cdef extern from "buchberger.h":
    cdef cppclass LeadMonomialsEnv:
        LeadMonomialsEnv() except +
        LeadMonomialsEnv(const LeadMonomialsEnv&)
        void reset()
        float step(int)
        void seed(int)

        vector[int] state
