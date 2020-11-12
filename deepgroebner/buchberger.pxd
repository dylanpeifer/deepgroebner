from libcpp.utility cimport pair

cdef extern from "buchberger.cpp":
    pass

cdef extern from "buchberger.h":
    cdef cppclass BuchbergerEnv:
        BuchbergerEnv() except +
        BuchbergerEnv(int) except +
        int reset()
        int step(int, int)
