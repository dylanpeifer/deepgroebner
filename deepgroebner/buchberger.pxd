from libcpp.utility cimport pair

cdef extern from "buchberger.cpp":
    pass

cdef extern from "buchberger.h":
    cdef cppclass LeadMonomialsEnv:
        LeadMonomialsEnv() except +
        LeadMonomialsEnv(const LeadMonomialsEnv&)
        void reset()
        float step(int)
        void seed(int)

        int[12] state
