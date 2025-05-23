// cppimport
#include <pybind11/pybind11.h>

#include <ptdalgorithms.h>


namespace py = pybind11;

int square(int x) {
    return x * x;
}

PYBIND11_MODULE(somecode, m) {
    m.def("square", &square);
}
/*
<%
setup_pybind11(cfg)
%>
*/