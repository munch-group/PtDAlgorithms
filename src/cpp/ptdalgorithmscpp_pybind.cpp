#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

// FIXME: Had to:
// cd ~/miniconda3/envs/phasetype/include
// ln -s eigen3/Eigen
// #include <eigen3/Eigen/Core>
#include <Eigen/Core>

#include "ptdalgorithmscpp.h"

#include <deque>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using std::deque;
using std::vector;
using std::tuple;
using std::deque;
using std::endl;


///////////////////////////////////////////////////////
// Jax interface
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include <cassert>
#include <iostream>
#include <iomanip>

extern "C" {

  // JAX custom call signature with scalar operands
  __attribute__((visibility("default")))
  void _pmf_jax_ffi_prim(void* out_ptr, void* in_ptrs);

  // JAX custom call signature for jax_graph_method_pmf
  void _pmf_jax_ffi_prim(void* out_ptr, void* in_ptrs) {

      void** buffers = reinterpret_cast<void**>(in_ptrs);
      ptdalgorithms::Graph* graph = reinterpret_cast<ptdalgorithms::Graph*>(buffers[0]);
      int64_t* times = reinterpret_cast<int64_t*>(buffers[1]);
      int64_t* n_ptr = reinterpret_cast<int64_t*>(buffers[2]);

      double* output = reinterpret_cast<double*>(out_ptr);      
      
      // Extract dimensions from scalar operands
      int64_t n = *n_ptr;

      for (int64_t idx = 0; idx < n; ++idx) {
        int64_t k = times[idx];
        output[idx] = graph->dph_pmf(k);
      }
  }

  // XLA custom call registration
  void register_jax_graph_method_pmf() {
      // This would normally register with XLA, but for simplicity we'll rely on 
      // the Python side custom call mechanism
  }

}

///////////////////////////////////////////////////////

using namespace pybind11::literals; // to bring in the `_a` literal

static void set_c_seed() {
  py::object random = py::module_::import("random");//.attr("randint");
  py::object obj = random.attr("randint")(0, 1000000);
  unsigned int i = (unsigned int) obj.cast<int>();
  srand(i);
  // py::print(i);
}


static int fac(int n) {
    if (n == 0) {
        return 1;
    }

    return n * fac(n - 1);
}


template<pybind11::return_value_policy Policy = pybind11::return_value_policy::reference_internal, typename Iterator, typename Sentinel, typename ValueType = typename pybind11::detail::iterator_access<Iterator>::result_type, typename ...Extra>
pybind11::typing::Iterator<ValueType> make_iterator(Iterator first, Sentinel last, Extra&&... extra);

/* Bind MatrixXd (or some other Eigen type) to Python */
// typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXd dMatrix;

typedef dMatrix::Scalar dScalar;
//  constexpr bool rowMajor = dMatrix::Flags & Eigen::RowMajorBit;

/* Bind MatrixXd (or some other Eigen type) to Python */
// typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXi iMatrix;

typedef iMatrix::Scalar iScalar;
constexpr bool rowMajor = iMatrix::Flags & Eigen::RowMajorBit;


struct matrix_representation {
    iMatrix states;
    dMatrix SIM;
    std::vector<double> IPV;
    std::vector<int> indices;
};

matrix_representation* _graph_as_matrix(ptdalgorithms::Graph graph) {

    ::ptd_phase_type_distribution *dist = ::ptd_graph_as_phase_type_distribution(graph.c_graph());

    int rows = dist->length;
    int cols = dist->length;
    dMatrix SIM = dMatrix(rows, cols);
    std::vector<double> IPV(dist->length);

    for (size_t i = 0; i < dist->length; ++i) {
        IPV[i] = dist->initial_probability_vector[i];

        for (size_t j = 0; j < dist->length; ++j) {
            SIM(i, j) = dist->sub_intensity_matrix[i][j];
        }
    }

    size_t state_length = graph.state_length();

    rows = dist->length;
    cols = state_length;
    iMatrix states = iMatrix(rows, cols);

    for (size_t i = 0; i < dist->length; i++) {
        for (size_t j = 0; j < state_length; j++) {
            states(i, j) = dist->vertices[i]->state[j];
        }
    }

    std::vector<int> indices(dist->length);
    for (size_t i = 0; i < dist->length; i++) {
        indices[i] = dist->vertices[i]->index + 1;
    }

    struct matrix_representation *matrix_rep;
    // auto *as_matrix = new matrix_representation(); 
    matrix_rep->states = states;
    matrix_rep->SIM = SIM;
    matrix_rep->IPV = IPV;
    matrix_rep->indices = indices;

    ::ptd_phase_type_distribution_destroy(dist);
    return matrix_rep;
}


class MatrixRepresentation {
    private:

    public:
        iMatrix states;
        dMatrix sim;
        std::vector<double> ipv;
        std::vector<int> indices;

        MatrixRepresentation(ptdalgorithms::Graph graph) {
            struct matrix_representation *rep = _graph_as_matrix(graph);
            this->states = rep->states;
            this->sim = rep->SIM;
            this->ipv = rep->IPV;
            this->indices = rep->indices;
        }

        // // pybind11 factory function
        // static MatrixRepresentation init_factory(ptdalgorithms::Graph graph) {
        //     return MatrixRepresentation(graph); 
        // }

        ~MatrixRepresentation() {
        }
};


iMatrix _states(ptdalgorithms::Graph &graph) {

      std::vector<ptdalgorithms::Vertex> ver = graph.vertices();

      int rows = ver.size();
      int cols = graph.state_length();
      iMatrix states = iMatrix(rows, cols);

      for (size_t i = 0; i < ver.size(); i++) {
          for (size_t j = 0; j < graph.state_length(); j++) {
              states(i, j) = ver[i].state()[j];
          }
      }

      return states;
  }

  
  // std::vector<double> _sample(ptdalgorithms::Graph graph, int n, std::vector<double> rewards) {

  //     if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
  //         char message[1024];

  //         snprintf(
  //                 message,
  //                 1024,
  //                 "Failed: Rewards must match the number of vertices. Expected %i, got %i",
  //                 (int) graph.c_graph()->vertices_length,
  //                 (int) rewards.size()
  //         );

  //         throw std::runtime_error(
  //                 message
  //         );
  //     }
  //     std::vector<double> res(n);

  //     set_c_seed();

  //     for (int i = 0; i < n; i++) {
  //         if (rewards.empty()) {
  //             res[i] = (double) (graph.random_sample());
  //         } else {
  //             res[i] = (double) (graph.random_sample(rewards));
  //         }
  //     }

  //     return res;

  //   }


// Utility function for use in both moments dph_expectation and dph_variance lambda functions
std::vector<double> _moments(ptdalgorithms::Graph &graph, int power, const std::vector<double> &rewards = vector<double>()) {

      if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
          char message[1024];

          snprintf(
                  message,
                  1024,
                  "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                  (int) graph.c_graph()->vertices_length,
                  (int) rewards.size()
          );

          throw std::runtime_error(
                  message
          );
      }

      if (power <= 0) {
          char message[1024];

          snprintf(
                  message,
                  1024,
                  "Failed: power must be a strictly positive integer. Got %i",
                  power
          );

          throw std::runtime_error(
                  message
          );
      }

      std::vector<double> res(power);
      std::vector<double> rewards2 = graph.expected_waiting_time(rewards);
      std::vector<double> rewards3(rewards2.size());
      res[0] = rewards2[0];

      std::vector<double> rw = rewards;

      // if (!rewards.empty()) {
      //     rw = as<std::vector<double> >(rewards);
      // }

      for (int i = 1; i < power; i++) {
          if (!rewards.empty()) {
              for (int j = 0; j < (int) rewards2.size(); j++) {
                  rewards3[j] = rewards2[j] * rw[j];
              }
          } else {
              rewards3 = rewards2;
          }

          rewards2 = graph.expected_waiting_time(rewards3);
          res[i] = fac(i + 1) * rewards2[0];
      }

      return res;

  }
  
// // Vectorize this
    
// py::array_t<double> _expectation(ptdalgorithms::Graph &graph, py::iterable_t<py::array_t<double> >() rewards) {


//     for (auto v : x)
//     std::cout << " " << v.to_string();
//   }


//   py::array_t<double> _expectation(ptdalgorithms::Graph &graph, py::array_t<double> rewards) {

//     py::buffer_info reward_buf = rewards.request();
//     if (reward_buf.ndim != 1)
//       throw std::runtime_error("Number of dimensions must be one");

//     /* No pointer is passed, so NumPy will allocate the buffer */
//     auto result = py::array_t<double>(reward_buf.size);

//     py::buffer_info result_buf = result.request();

//     double *reward_ptr = static_cast<double *>(reward_buf.ptr);
//     double *result_ptr = static_cast<double *>(result_buf.ptr);

//     for (size_t idx = 0; idx < reward_buf.shape[0]; idx++) {

//       // std::vector<double> _vector(reward_ptr, reward_ptr + reward_buf[idx].shape[0]);
//       std::vector<double> _vector(reward_ptr, reward_ptr + reward_buf.shape[0]);

//       result_ptr[idx] = _moments(graph, 1, _vector)[0];
//     }

//     return result;
// }


double _expectation(
  ptdalgorithms::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {

  return _moments(graph, 1, rewards)[0];
}



double _variance(
  ptdalgorithms::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {

    std::vector<double> exp = graph.expected_waiting_time(rewards);
    std::vector<double> second;

    if (rewards.empty()) {
        second = graph.expected_waiting_time(exp);
    } else {
        std::vector<double> new_rewards(exp.size());
        std::vector<double> rw = rewards;

        for (int i = 0; i < (int) exp.size(); i++) {
            new_rewards[i] = exp[i] * rw[i];
        }

        second = graph.expected_waiting_time(new_rewards);
    }

    return (2 * second[0] - exp[0] * exp[0]);    

}

double _covariance(ptdalgorithms::Graph &graph, 
  const std::vector<double> &rewards1 = vector<double>(),
  const std::vector<double> &rewards2 = vector<double>()) {

    std::vector<double> exp1 = graph.expected_waiting_time(rewards1);
    std::vector<double> exp2 = graph.expected_waiting_time(rewards2);


    std::vector<double> new_rewards(exp1.size());


    for (int i = 0; i < exp1.size(); i++) {
      new_rewards[i] = exp1[i] * rewards2[i];
    }

    std::vector<double> second1 = graph.expected_waiting_time(new_rewards);


    for (int i = 0; i < exp1.size(); i++) {
        new_rewards[i] = exp2[i] * rewards1[i];
    }

    std::vector<double> second2 = graph.expected_waiting_time(new_rewards);

    return (second1[0] + second2[0] - exp1[0] * exp2[0]);    

}

double _expectation_discrete(
  ptdalgorithms::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {
    return _moments(graph, 1, rewards)[0];
}

double _variance_discrete(
  ptdalgorithms::Graph &graph, 
  const std::vector<double> &rewards = vector<double>()) {

    if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
      char message[1024];

      snprintf(
              message,
              1024,
              "Failed: Rewards must match the number of vertices. Expected %i, got %i",
              (int) graph.c_graph()->vertices_length,
              (int) rewards.size()
      );

      throw std::runtime_error(
              message
      );
  }
  if (rewards.empty()) {
      std::vector<double> m = _moments(graph, 2);

      return m[1] - 2*m[0];
  } else {
      // std::vector<double> rw = as<std::vector<double> >(rewards);
      std::vector<double> sq_rewards(rewards.size());

      for (int i = 0; i < (int)rewards.size(); i++) {
          sq_rewards[i] = rewards[i] * rewards[i];
      }

      std::vector<double> momentsr = _moments(graph, 2, rewards);
      std::vector<double> momentsrr = _moments(graph, 1, sq_rewards);

      return momentsr[1] - momentsr[0] * momentsr[0] - momentsrr[0];
    }

}

double _covariance_discrete(ptdalgorithms::Graph &graph, 
  const std::vector<double> &rewards1 = vector<double>(),
  const std::vector<double> &rewards2 = vector<double>()) {

    std::vector<double> rw1(rewards1);
    std::vector<double> rw2(rewards2);
    std::vector<double> sq_rewards(rw1.size());

    for (int i = 0; i < (int)rw1.size(); i++) {
        sq_rewards[i] = rw1[i] * rw2[i];
    }

    std::vector<double> rw1to2(rw1.size());
    std::vector<double> rw2to1(rw2.size());
    std::vector<double> exp1 = graph.expected_waiting_time(rewards1);
    std::vector<double> exp2 = graph.expected_waiting_time(rewards2);

    for (int i = 0; i < (int)rw1.size(); i++) {
        rw1to2[i] = exp1[i] * rw2[i];
        rw2to1[i] = exp2[i] * rw1[i];
    }

    return graph.expected_waiting_time(rw1to2)[0] +
              _moments(graph, 1, sq_rewards)[0] - 
              _moments(graph, 1, rewards1)[0] *
              _moments(graph, 1, rewards2)[0];

}


// ptdalgorithms::Graph build_state_space_callback_dicts(
//   const std::function<std::vector<py::dict> (std::vector<int> &state)> &callback, std::vector<int> &initial_state) {

//       ptdalgorithms::Graph *graph = new ptdalgorithms::Graph(initial_state.size());

//       ptdalgorithms::Vertex init = graph->find_or_create_vertex(initial_state);

//         graph->starting_vertex().add_edge(init, 1);

//         int index = 1;
//         while (index < graph->vertices_length()) {

//           ptdalgorithms::Vertex this_vertex = graph->vertex_at(index);
//           std::vector<int> this_state = graph->vertex_at(index).state();

//           std::vector<py::dict> children = callback(this_state);
//               for (auto child : children) {
//                 std::vector<int> child_state = child["state"].cast<std::vector<int> >();
//                 long double weight = child["weight"].cast<long double>();
//                 ptdalgorithms::Vertex child_vertex = graph->find_or_create_vertex(child_state);
//                 if (child.size() == 3) {
//                   std::vector<double> edge_params = child["edge_params"].cast<std::vector<double> >();
//                   this_vertex.add_edge_parameterized(child_vertex, weight, edge_params);
//                 } else {
//                   this_vertex.add_edge(child_vertex, weight);
//                 }
//               }
//               ++index;
//             }
//       return *graph;
//   }

  ptdalgorithms::Graph build_state_space_callback_tuples(
    
      // const std::function< std::vector<const std::tuple<const py::array_t<int>, long double> > (const py::array_t<int> &state)> &callback) { 
      const std::function<std::vector<py::object> (const py::array_t<int> &state)> &callback) { 

      ptdalgorithms::Graph *graph = nullptr;

      // IPV from callback with no state argument
      std::vector<py::object> children = callback(py::array_t<int>());

      for (const auto child : children) {

        std::tuple<py::array_t<int>, long double> tup = child.cast<std::tuple<py::array_t<int>, long double> >();

        py::array_t<int> a = std::get<0>(tup);
        py::buffer_info buf = a.request();
        if (buf.ndim != 1)
          throw std::runtime_error("Number of dimensions must be one");
        int *ptr = static_cast<int *>(buf.ptr);
        std::vector<int> child_state(ptr, ptr + buf.shape[0]);

        long double weight = std::get<1>(tup);

        if (!graph) {
          graph = new ptdalgorithms::Graph(child_state.size());
        }
        ptdalgorithms::Vertex child_vertex = graph->find_or_create_vertex(child_state);
        graph->starting_vertex().add_edge(child_vertex, weight);
      }

        int index = 1;
        while (index < graph->vertices_length()) {

          ptdalgorithms::Vertex this_vertex = graph->vertex_at(index);

          auto a = new std::vector<int>(graph->vertex_at(index).state());
          auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
          py::array_t<int> this_state = py::array(a->size(), a->data(), capsule);

          std::vector<py::object> children = callback(this_state);

          for (auto child : children) {

            std::tuple<py::array_t<int>, long double> tup = child.cast<std::tuple<py::array_t<int>, long double> >();
            py::array_t<int> a = std::get<0>(tup);
            py::buffer_info buf = a.request();
            if (buf.ndim != 1)
              throw std::runtime_error("Number of dimensions must be one");
            int *ptr = static_cast<int *>(buf.ptr);
            std::vector<int> child_state(ptr, ptr + buf.shape[0]);
            long double weight = std::get<1>(tup);

            ptdalgorithms::Vertex child_vertex = graph->find_or_create_vertex(child_state);
            // if (child.size() == 3) {
            //   std::vector<double> edge_params = child[2].cast<std::vector<double> >();
            //   this_vertex.add_edge_parameterized(child_vertex, weight, edge_params);
            // } else {
              this_vertex.add_edge(child_vertex, weight);
            // }
          }
          ++index;
        }
      return *graph;
  }
        
  // ptdalgorithms::Graph build_state_space_callback_tuples(
  //   // const std::function<std::vector<const py::tuple> (std::vector<int> &state)> &callback, std::vector<int> &initial_state) {
  //   const std::function<std::vector<const py::tuple> (py::array_t<int> &state)> &callback, std::vector<int> &initial_state) {

  //     ptdalgorithms::Graph *graph = new ptdalgorithms::Graph(initial_state.size());

  //     ptdalgorithms::Vertex init = graph->find_or_create_vertex(initial_state);

  //       graph->starting_vertex().add_edge(init, 1);

  //       int index = 1;
  //       while (index < graph->vertices_length()) {

  //         ptdalgorithms::Vertex this_vertex = graph->vertex_at(index);

          
  //         // std::vector<int> this_state = graph->vertex_at(index).state();

  //         auto a = new std::vector<int>(graph->vertex_at(index).state());
  //         auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
  //         py::array_t<int> this_state = py::array(a->size(), a->data(), capsule);


  //         std::vector<const py::tuple> children = callback(this_state);

  //             for (auto child : children) {
  //               std::vector<int> child_state = child[0].cast<std::vector<int> >();
  //               long double weight = child[1].cast<long double>();
  //               ptdalgorithms::Vertex child_vertex = graph->find_or_create_vertex(child_state);
  //               if (child.size() == 3) {
  //                 std::vector<double> edge_params = child[2].cast<std::vector<double> >();
  //                 this_vertex.add_edge_parameterized(child_vertex, weight, edge_params);
  //               } else {
  //                 this_vertex.add_edge(child_vertex, weight);
  //               }
  //             }
  //             ++index;
  //           }
  //     return *graph;
  // }
    
  
PYBIND11_MODULE(ptdalgorithmscpp_pybind, m) {

  ///////////////////////////////////////////////////////
  // for jax interface
  m.def("jax_graph_method_pmf", []() {}); // No-op: only used for symbol registration

  ///////////////////////////////////////////////////////


  m.doc() = "These are the docs";

  py::class_<iMatrix>(m, "iMatrix", py::buffer_protocol())

    .def(py::init([](py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<iScalar>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
            info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(iScalar),
            info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(iScalar));

        auto map = Eigen::Map<iMatrix, 0, Strides>(
            static_cast<iScalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        return iMatrix(map);
    }))

    .def_buffer([](iMatrix &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                                /* Pointer to buffer */
        sizeof(iScalar),                          /* Size of one scalar */
        py::format_descriptor<iScalar>::format(), /* Python struct-style format descriptor */
        2,                                       /* Number of dimensions */
        { m.rows(), m.cols() },                  /* Buffer dimensions */
        { sizeof(iScalar) * (rowMajor ? m.cols() : 1),
          sizeof(iScalar) * (rowMajor ? 1 : m.rows()) }
                                                 /* Strides (in bytes) for each index */
    );
   })
  ;

  py::class_<dMatrix>(m, "dMatrix", py::buffer_protocol())

    .def(py::init([](py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some basic validation checks ... */
        if (info.format != py::format_descriptor<dScalar>::format())
            throw std::runtime_error("Incompatible format: expected a double array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
            info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(dScalar),
            info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(dScalar));

        auto map = Eigen::Map<dMatrix, 0, Strides>(
            static_cast<dScalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        return dMatrix(map);
    }))

    .def_buffer([](dMatrix &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),                                /* Pointer to buffer */
        sizeof(dScalar),                          /* Size of one scalar */
        py::format_descriptor<dScalar>::format(), /* Python struct-style format descriptor */
        2,                                       /* Number of dimensions */
        { m.rows(), m.cols() },                  /* Buffer dimensions */
        { sizeof(dScalar) * (rowMajor ? m.cols() : 1),
          sizeof(dScalar) * (rowMajor ? 1 : m.rows()) }
                                                 /* Strides (in bytes) for each index */
    );
   })
  ;

    
  py::class_<MatrixRepresentation>(m, "MatrixRepresentation", R"delim(
      Matrix representation of phase-type distribution
      )delim")
      
    // .def(py::init(&MatrixRepresentation::init_factory))
      
    .def(py::init<const MatrixRepresentation>(), py::arg("graph"), R"delim(

      )delim")

    .def_readwrite("states", &MatrixRepresentation::states, R"delim(

      )delim")
      
    .def_readwrite("sim", &MatrixRepresentation::sim, R"delim(

      )delim")
    .def_readwrite("ipv", &MatrixRepresentation::ipv, R"delim(

      )delim")
    .def_readwrite("indices", &MatrixRepresentation::indices, R"delim(

      )delim")
  ;            

    
  py::class_<ptdalgorithms::Graph>(m, "Graph")

    .def(py::init<int>(), py::arg("state_length"))


      // .def("__iter__",
      //   [](ptdalgorithms::Graph &g) {
      //       return make_iterator(g.begin(), g.end());
      //   }, py::return_value_policy::reference_internal, R"delim(
  
      //   )delim")
  

    .def(py::init<struct ::ptd_graph* >(), py::arg("ptd_graph"))

    .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"))
      
    .def(py::init<const ptdalgorithms::Graph>(), py::arg("o"))

    .def(py::init<struct ::ptd_graph*, struct ::ptd_avl_tree* >(), py::arg("ptd_graph"), py::arg("ptd_avl_tree"))
    
    // .def(py::init(&build_state_space_callback_dicts), 
    //   py::arg("callback_dicts"), py::arg("initial_state"))

    // .def(py::init(&build_state_space_callback_tuples),
    //       py::arg("callback_tuples"), py::arg("initial_state"))



    ///////////////////////////////////////////////////////
    // for jax interface
    .def("pointer", [](ptdalgorithms::Graph* self) -> uintptr_t {
        return reinterpret_cast<uintptr_t>(self);
    })
    ///////////////////////////////////////////////////////


    .def(py::init(&build_state_space_callback_tuples),
      py::arg("callback_tuples"))

    .def("create_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      
      Warning: the function [ptdalgorithms::find_or_create_vertex()] should be preferred. 
      This function will *not* update the lookup tree, so [ptdalgorithms::find_vertex()] will *not* return it.
      Creates a vertex matching `state`. Creates the vertex and adds it to the graph object. 

      Parameters
      ----------
      state : ArrayLike
          An integer sequence defining the state represented by the new vertex.

      Returns
      -------
      Vertex
          The newly inserted vertex in the graph.
      )delim")


    .def("find_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Finds a vertex matching the `state` parameter.

      Parameters
      ----------
      state : ArrayLike
          An integer sequence defining the state represented by the new vertex.

      Returns
      -------
      Vertex
          The found vertex in the graph or None.
      )delim")
      
    .def("vertex_exists", static_cast<bool (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::vertex_exists), py::arg("state"), 
      py::return_value_policy::reference_internal, R"delim(




      )delim")
      
    .def("find_or_create_vertex", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_or_create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Finds a vertex matching the `state` parameter. If no such vertex exists, it creates the vertex and adds it to the graph object instead.

      Parameters
      ----------
      state : ArrayLike
          An integer sequence defining the state represented by the new vertex.

      Returns
      -------
      Vertex
          The newly found or inserted vertex in the graph.

      Examples
      --------
      ```
      graph = Graph(4)
      graph.find_or_create_vertex([1,2,1,0])
      ````
      )delim")
      
    // .def("find_or_create_vertex",
    //   [](ptdalgorithms::Graph &graph, std::vector<double> state) {
    //     std::vector<int> int_vec(state.begin(), state.end());
    //     return graph.find_or_create_vertex(int_vec);
    //   }, R"delim(

    //   )delim")

    .def("focv", static_cast<ptdalgorithms::Vertex (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::find_or_create_vertex), py::arg("state"), 
      py::return_value_policy::copy, R"delim(
      Alias for find_or_create_vertex
      )delim")      

    .def("starting_vertex", &ptdalgorithms::Graph::starting_vertex, 
      py::return_value_policy::copy, R"delim(
      Returns the special starting vertex of the graph. The starting vertex is always added at graph creation and always has index 0.

      Returns
      -------
      Vertex
          The starting vertex.
      )delim")
      
    .def("vertices", &ptdalgorithms::Graph::vertices, 
      py::return_value_policy::copy, R"delim(
      Returns all vertices that have been added to the graph from either calling `find_or_create_vertex` or `create_vertex`. 
      The first vertex in the list is *always* the starting vertex.

      Returns
      -------
      List
          A list of all vertices in the graph.

      Examples
      --------
      graph.create_graph(4)
      vertex_a = find_or_create_vertex([1,2,1,0])
      vertex_b = find_or_create_vertex([2,0,1,0])
      graph.vertices()[0] == graph.starting_vertex()
      graph.vertices()[1] == graph.vertex_at(1)
      graph.vertices_length() == 3
      )delim")      

    .def("vertex_at", &ptdalgorithms::Graph::vertex_at, py::arg("index"), 
      py::return_value_policy::copy, R"delim(
      Returns a vertex at a particular index. This method is much faster than `Graph.vertices()[i]`.

      Parameters
      ----------
      phase_type_graph : SEXP
          A reference to the graph created by [ptdalgorithms::create_graph()].
      index : int
          The index of the vertex to find.

      Returns
      -------
      List
          The vertex at index `index` in the graph.
        )delim")
      
    .def("vertex_at",[](ptdalgorithms::Graph &graph, double index) {
      return graph.vertex_at((int) index);
  
    })

    .def("vertices_length", &ptdalgorithms::Graph::vertices_length, 
      py::return_value_policy::reference_internal, R"delim(
      Returns the number of vertices in the graph. This method is much faster than `len(Graph.vertices())`.

      Returns
      -------
      int
          The number of vertices.
      )delim")

    .def("states", &_states,
      py::return_value_policy::copy, R"delim(
      Returns a matrix where each row is the state of the vertex at that index.

      Returns
      -------
      int
         A matrix of size [ptdalgorithms::vertices_length()] where the rows match the state of the vertex at that index
      )delim")

    .def("__repr__",
      [](ptdalgorithms::Graph &g) {
          return "<Graph (" + std::to_string(g.vertices_length()) + " vertices)>";
      }, py::return_value_policy::move, 
      R"delim(

      )delim")

    .def("update_parameterized_weights", &ptdalgorithms::Graph::update_weights_parameterized, 
      py::arg("rewards"), 
      R"delim(
    Updates all parameterized edges of the graph by given scalars. Given a vector of scalars, 
    computes a new weight of the parameterized edges in the graph by a simple inner product of
    the edge state vector and the scalar vector.

    Parameters
    ----------
    scalars : ArrayLike
        A numeric vector of multiplies for the edge states.

    Examples
    --------
    graph = Graph(4)
    v1 = graph.find_or_create_vertex([1, 2, 1, 0])
    v2 = graph.find_or_create_vertex([2, 0, 1, 0])
    graph.starting_vertex().add_edge(v1, 5)
    v1.add_edge(v2, 0, [5,2])
    graph.starting_vertex().edges()[0].weight()5
    v1.edges()[0].weight() # => 0
    graph.update_weights_parameterized([9,7])
    graph.starting_vertex().edges()[0]].weight() # => 5
    v1.edges()[0].weight() # => 59
      )delim")

      
      // .def("moments", 
      //   [](ptdalgorithms::Graph &graph, int power, std::vector<double> &rewards) {
        
          // return _moments(graph, power, rewards);
      
    // }, 
    .def("moments", &_moments,
      py::arg("power"), py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, 
      R"delim(
      Computes the first `power` moments of the phase-type distribution. This function invokes 
      `Graph.expected_waiting_times()` consecutively to find the first moments, given by the `power` argument.

      Parameters
      ----------
      power : int
          The number of moments to compute.
      rewards : ArrayLike, optional
          Rewards to apply to the phase-type distribution.

      Returns
      -------
      Array
          Array of the first `power` moments. The first entry is the first moment (mean).

      Examples
      --------
      >>> graph = Graph(4)
      >>> v1 = graph.create_vertex([1,2,3,4])
      >>> v2 = graph.create_vertex([4,0,3,3])
      >>> a = graph.create_vertex([0,0,0,0])
      >>> graph.starting_vertex().add_edge(v1, 1)
      >>> v1.add_edge(v2, 4)
      >>> v2.add_edge(a, 10)
      >>> graph.moments(3)
      (0.350000 0.097500 0.025375)
      >>> ptdalgorithms::moments(graph, 3, [0,2,1,0])
      (0.600 0.160 0.041)
   )delim")
    

    .def("expectation", &_expectation,
      py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, 
      R"delim(

      Computes the expectation (mean) of the phase-type distribution.

    This function invokes `ptdalgorithms::expected_waiting_times()` and takes the first entry (from starting vertex).

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards : NumericVector, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    double
        The expectation of the distribution.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::expectation(graph) # => 0.35
    >>> ptdalgorithms::expectation(graph, c(0,2,1,0)) # => 0.6
    >>> ph <- ptdalgorithms::graph_as_matrix(graph)
    >>> # This is a much faster version of
    >>> ph$IPV %*% solve(-ph$SIM) %*% rep(1, length(ph$IPV)) # => 0.35
    >>> ph$IPV %*% solve(-ph$SIM) %*% diag(c(2,1)) %*% rep(1, length(ph$IPV)) # => 0.35
      )delim")      

    .def("variance", &_variance,
        py::arg("rewards")=std::vector<double>(), 
        py::return_value_policy::move, 
        R"delim(
    Computes the variance of the phase-type distribution.

    This function invokes `ptdalgorithms::expected_waiting_times()` twice to find the first and second moment.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards : NumericVector, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    double
        The variance of the distribution.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::variance(graph) # => 0.0725
    >>> ptdalgorithms::variance(graph, c(0,2,1,0)) # => 0.26
    >>> ph <- ptdalgorithms::graph_as_matrix(graph)
    >>> # This is a much faster version of
    >>> 2*ph$IPV%*%solve(-ph$SIM)%*%solve(-ph$SIM) %*% rep(1, length(ph$IPV)) - ph$IPV%*%solve(-ph$SIM) %*% rep(1, length(ph$IPV)) %*% ph$IPV%*%solve(-ph$SIM) %*% rep(1, length(ph$IPV)) # => 0.0725
    >>> 2*ph$IPV%*%solve(-ph$SIM)%*%diag(c(2,1))%*%solve(-ph$SIM)%*%diag(c(2,1)) %*% rep(1, length(ph$IPV)) - ph$IPV%*%solve(-ph$SIM)%*%diag(c(2,1)) %*% rep(1, length(ph$IPV)) %*% ph$IPV%*%solve(-ph$SIM)%*%diag(c(2,1)) %*% rep(1, length(ph$IPV)) # => 0.26
      )delim")    

      .def("covariance", &_covariance,
        py::arg("rewards1")=std::vector<double>(), 
        py::arg("rewards2")=std::vector<double>(), 
        py::return_value_policy::move, 
        R"delim(
    Computes the covariance of the phase-type distribution.

    This function invokes `ptdalgorithms::expected_waiting_times()` twice to find the first and second moments for two sets of rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards1 : NumericVector
        The first set of rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.
    rewards2 : NumericVector
        The second set of rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    double
        The covariance of the distribution.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::covariance(graph, c(0,2,1,0), c(1,0,2,1)) # => 0.15
    >>> ph <- ptdalgorithms::graph_as_matrix(graph)
    >>> # This is a much faster version of
    >>> ph$IPV %*% solve(-ph$SIM) %*% diag(c(2,1)) %*% solve(-ph$SIM) %*% diag(c(1,2)) %*% rep(1, length(ph$IPV)) - (ph$IPV %*% solve(-ph$SIM) %*% diag(c(2,1)) %*% rep(1, length(ph$IPV))) * (ph$IPV %*% solve(-ph$SIM) %*% diag(c(1,2)) %*% rep(1, length(ph$IPV))) # => 0.15
         )delim")   


    .def("covariance_discrete", &_covariance_discrete,
          py::arg("rewards1")=std::vector<double>(), 
          py::arg("rewards2")=std::vector<double>(), 
          py::return_value_policy::move, 
          R"delim(
    Computes the covariance of the discrete phase-type distribution.

    This function invokes `ptdalgorithms::dph_expected_waiting_times()` twice to find the first and second moments for two sets of rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards1 : NumericVector
        The first set of rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.
    rewards2 : NumericVector
        The second set of rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    double
        The covariance of the distribution.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::covariance_discrete(graph, c(0,2,1,0), c(1,0,2,1)) # => 0.15
    >>> ph <- ptdalgorithms::graph_as_matrix(graph)
    >>> # This is a much faster version of
    >>> ph$IPV %*% solve(-ph$SIM) %*% diag(c(2,1)) %*% solve(-ph$SIM) %*% diag(c(1,2)) %*% rep(1, length(ph$IPV)) - (ph$IPV %*% solve(-ph$SIM) %*% diag(c(2,1)) %*% rep(1, length(ph$IPV))) * (ph$IPV %*% solve(-ph$SIM) %*% diag(c(1,2)) %*% rep(1, length(ph$IPV))) # => 0.15
         )delim")   
      


    .def("expected_waiting_time", &ptdalgorithms::Graph::expected_waiting_time, py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, R"delim(
    Computes the expected waiting time of the phase-type distribution.

    This function computes the expected waiting time for the given rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards : NumericVector, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    NumericVector
        A numeric vector of the expected waiting times.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::expected_waiting_time(graph) # => [0.35, 0.1, 0.05]
    >>> ptdalgorithms::expected_waiting_time(graph, c(0,2,1,0)) # => [0.6, 0.2, 0.1]
      )delim")
      
    .def("expected_residence_time", &ptdalgorithms::Graph::expected_residence_time, py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, R"delim(
Computes the expected residence time of the phase-type distribution.

    This function computes the expected residence time for the given rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards : NumericVector, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    NumericVector
        A numeric vector of the expected residence times.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::expected_residence_time(graph) # => [0.35, 0.1, 0.05]
    >>> ptdalgorithms::expected_residence_time(graph, c(0,2,1,0)) # => [0.6, 0.2, 0.1]
      )delim")
      
      

    .def("sample",
      [](ptdalgorithms::Graph &graph, int n, std::vector<double>rewards) {


        if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
            char message[1024];

            snprintf(
                    message,
                    1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) graph.c_graph()->vertices_length,
                    (int) rewards.size()
            );

            throw std::runtime_error(
                    message
            );
        }
        std::vector<double> res(n);

        set_c_seed();

        for (int i = 0; i < n; i++) {
            if (rewards.empty()) {
                res[i] = (double) (graph.random_sample());
            } else {
                res[i] = (double) (graph.random_sample(rewards));
            }
        }

        return res;

      }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the phase-type distribution.

    This function generates samples from the phase-type distribution, optionally using a set of rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : NumericVector, optional
        Optional rewards, which should be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    NumericVector
        A numeric vector of samples.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::sample(graph, 5) # => [0.35, 0.1, 0.05, 0.2, 0.15]
    >>> ptdalgorithms::sample(graph, 5, c(0,2,1,0)) # => [0.6, 0.2, 0.1, 0.4, 0.3]
    )delim")


    .def("sample_discrete",
      [](ptdalgorithms::Graph &graph, int n, std::vector<double>rewards) {


        if (!rewards.empty() && (int) rewards.size() != (int) graph.c_graph()->vertices_length) {
            char message[1024];

            snprintf(
                    message,
                    1024,
                    "Failed: Rewards must match the number of vertices. Expected %i, got %i",
                    (int) graph.c_graph()->vertices_length,
                    (int) rewards.size()
            );

            throw std::runtime_error(
                    message
            );
        }
        std::vector<double> res(n);

        set_c_seed();

        for (int i = 0; i < n; i++) {
            if (rewards.empty()) {
                res[i] = (double) (graph.dph_random_sample_c(NULL));
            } else {
                res[i] = (double) (graph.dph_random_sample_c(&rewards[0]));
            }
        }

        return res;

      }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the discrete phase-type distribution.

    This function generates samples from the discrete phase-type distribution, optionally using a set of rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : NumericVector, optional
        Optional rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    NumericVector
        A numeric vector of samples.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::sample_discrete(graph, 5) # => [0.35, 0.1, 0.05, 0.2, 0.15]
    >>> ptdalgorithms::sample_discrete(graph, 5, c(0,2,1,0)) # => [0.6, 0.2, 0.1, 0.4, 0.3]
    )delim")

    ///////////////////////////////////////////

    .def("sample_multivariate",
      [](ptdalgorithms::Graph &graph, int n, dMatrix rewards) -> dMatrix  {

        if ((int) rewards.rows() != (int) graph.c_graph()->vertices_length) {
            char message[1024];
    
            snprintf(
                    message,
                    1024,
                    "Failed: Rewards rows must match the number of vertices. Expected %i, got %i",
                    (int) graph.c_graph()->vertices_length,
                    (int) rewards.rows()
            );
    
            throw std::runtime_error(
                    message
            );
        }
    
        double *vrewards = (double *) calloc(rewards.rows() * rewards.cols(), sizeof(double));
    
        size_t index = 0;
    
        for (int i = 0; i < rewards.rows(); i++) {
            for (int j = 0; j < rewards.cols(); j++) {
                vrewards[index] = rewards(i, j);
                index++;
            }
        }
    
        set_c_seed();

        dMatrix mat_res = dMatrix(rewards.cols(), n);
    
        for (int i = 0; i < n; i++) {
            long double *res = ptd_mph_random_sample(graph.c_graph(), vrewards, (size_t) rewards.cols());
    
            for (int j = 0; j < rewards.cols(); j++) {
                mat_res(j, i) = res[j];
            }
    
            free(res);
        }
    
        free(vrewards);
    
        return mat_res;

      }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the multivariate phase-type distribution.

    This function generates samples from the multivariate phase-type distribution, using a set of rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : dMatrix
        A matrix of rewards, which should be applied to the phase-type distribution. The number of rows must match the number of vertices.

    Returns
    -------
    dMatrix
        A matrix of samples.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> rewards <- matrix(c(1,2,3,4,5,6,7,8), nrow=4, ncol=2)
    >>> ptdalgorithms::sample_multivariate(graph, 5, rewards)
    )delim")

    
    ///////////////////////////////////////////


    .def("sample_multivariate_discrete",
      [](ptdalgorithms::Graph &graph, int n, dMatrix rewards) -> dMatrix  {

        if ((int) rewards.rows() != (int) graph.c_graph()->vertices_length) {
          char message[1024];
  
          snprintf(
                  message,
                  1024,
                  "Failed: Rewards rows must match the number of vertices. Expected %i, got %i",
                  (int) graph.c_graph()->vertices_length,
                  (int) rewards.rows()
          );
  
          throw std::runtime_error(
                  message
          );
      }

    
    
        double *vrewards = (double *) calloc(rewards.rows() * rewards.cols(), sizeof(double));
    
        size_t index = 0;
    
        for (int i = 0; i < rewards.rows(); i++) {
            for (int j = 0; j < rewards.cols(); j++) {
                vrewards[index] = rewards(i, j);
                index++;
            }
        }
    
        set_c_seed();
    
        dMatrix mat_res = dMatrix(rewards.cols(), n);
    
        for (int i = 0; i < n; i++) {
            long double *res = ptd_mdph_random_sample(graph.c_graph(), vrewards, (size_t) rewards.cols());
    
            for (int j = 0; j < rewards.cols(); j++) {
                mat_res(j, i) = res[j];
            }
    
            free(res);
        }
    
        free(vrewards);
    
        return mat_res;

    }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=std::vector<double>(), R"delim(
    Samples from the multivariate discrete phase-type distribution.

    This function generates samples from the multivariate discrete phase-type distribution, using a set of rewards.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    n : int, optional
        The number of samples to generate. Default is 1.
    rewards : dMatrix
        A matrix of rewards, which should be applied to the discrete phase-type distribution. The number of rows must match the number of vertices.

    Returns
    -------
    dMatrix
        A matrix of samples.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> rewards <- matrix(c(1,2,3,4,5,6,7,8), nrow=4, ncol=2)
    >>> ptdalgorithms::sample_multivariate_discrete(graph, 5, rewards)
    )delim")







    // .def("sample_multivariate", static_cast<std::vector<long double> (ptdalgorithms::Graph::*)(std::vector<double>, size_t)>(&ptdalgorithms::Graph::mph_random_sample), py::arg("rewards"), py::arg("vertex_rewards_length"), 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")


    // .def("sample_multivariate_discrete", static_cast<std::vector<long double> (ptdalgorithms::Graph::*)(std::vector<double>, size_t)>(&ptdalgorithms::Graph::mdph_random_sample), py::arg("rewards"), py::arg("vertex_rewards_length"), 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")

    
  //  .def("sample_multivariate",
  //     [](ptdalgorithms::Graph &graph, int n, std::vector<double> rewards) {

  //       py::print(py::str("not implemented"));

  //     }, py::return_value_policy::move, py::arg("n")=1, py::arg("rewards")=dMatrix(), R"delim(

  //   )delim")

      

      
    .def("random_sample_stop_vertex", &ptdalgorithms::Graph::random_sample_stop_vertex, py::arg("time"), 
      py::return_value_policy::copy, R"delim(
    Samples a stopping vertex from the phase-type distribution given a stopping time.

    This function generates a sample of the stopping vertex from the phase-type distribution given a stopping time.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    time : double
        The stopping time.

    Returns
    -------
    Vertex
        The stopping vertex.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::random_sample_stop_vertex(graph, 0.5) # => Vertex at stopping time 0.5
      )delim")
      
    .def("random_sample_discrete_stop_vertex", &ptdalgorithms::Graph::dph_random_sample_stop_vertex, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Samples a stopping vertex from the discrete phase-type distribution given a number of jumps.

    This function generates a sample of the stopping vertex from the discrete phase-type distribution given a number of jumps.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    jumps : int
        The number of jumps.

    Returns
    -------
    Vertex
        The stopping vertex.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::random_sample_discrete_stop_vertex(graph, 3) # => Vertex at 3 jumps
      )delim")
      
    .def("state_length", &ptdalgorithms::Graph::state_length, 
      py::return_value_policy::copy, R"delim(
    Returns the length of the state vector used to represent and reference a state in the graph.

    This function returns the length of the integer vector used to represent and reference a state in the graph.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.

    Returns
    -------
    int
        The length of the state vector.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::state_length(graph) # => 4
      )delim")
      
    .def("is_acyclic", &ptdalgorithms::Graph::is_acyclic, 
      py::return_value_policy::copy, R"delim(
    Checks if the graph is acyclic.

    This function checks if the graph is acyclic, meaning it does not contain any cycles.

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.

    Returns
    -------
    bool
        True if the graph is acyclic, False otherwise.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::is_acyclic(graph) # => True or False
      )delim")
      
    .def("validate", &ptdalgorithms::Graph::validate, R"delim(
    Validates the graph structure.

    This function checks the integrity and consistency of the graph structure.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::validate(graph)
      )delim")
      
    // .def("expectation_dag", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<double>)>(&ptdalgorithms::Graph::expectation_dag), py::arg("rewards"), 
    //   py::return_value_policy::reference_internal, R"delim(
    // Computes the expectation of the directed acyclic graph (DAG) representation of the phase-type distribution.

    // This function computes the expectation of the phase-type distribution when represented as a directed acyclic graph (DAG).

    // Parameters
    // ----------
    // rewards : NumericVector
    //     A numeric vector of rewards to be applied to the phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    // Returns
    // -------
    // Graph
    //     A graph object representing the expectation of the DAG.

    // Examples
    // --------
    // >>> graph <- ptdalgorithms::create_graph(4)
    // >>> rewards <- c(1.0, 2.0, 3.0, 4.0)
    // >>> dag_expectation <- ptdalgorithms::expectation_dag(graph, rewards)
    //   )delim")
      
    .def("reward_transform", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<double>)>(&ptdalgorithms::Graph::reward_transform), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(
    Transforms the graph using the given rewards.

    This function transforms the graph by applying the given rewards to the edges.

    Parameters
    ----------
    rewards : NumericVector
        A numeric vector of rewards to be applied to the edges of the graph. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    Graph
        A graph object with the transformed rewards.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> rewards <- c(1.0, 2.0, 3.0, 4.0)
    >>> transformed_graph <- ptdalgorithms::reward_transform(graph, rewards)
      )delim")
      
    .def("reward_transform_discrete", static_cast<ptdalgorithms::Graph (ptdalgorithms::Graph::*)(std::vector<int>)>(&ptdalgorithms::Graph::dph_reward_transform), py::arg("rewards"), 
      py::return_value_policy::reference_internal, R"delim(
    Transforms the discrete phase-type distribution graph using the given rewards.

    This function transforms the graph by applying the given rewards to the edges in the discrete phase-type distribution.

    Parameters
    ----------
    rewards : NumericVector
        A numeric vector of rewards to be applied to the edges of the graph. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    Graph
        A graph object with the transformed rewards.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> rewards <- c(1, 2, 3, 4)
    >>> transformed_graph <- ptdalgorithms::reward_transform_discrete(graph, rewards)
      )delim")
      
    .def("normalize", &ptdalgorithms::Graph::normalize, 
      py::return_value_policy::reference_internal, R"delim(
    Normalizes the graph.

    This function normalizes the graph by ensuring that the sum of the weights of the outgoing edges from each vertex is equal to 1.

    Parameters
    ----------
    None

    Returns
    -------
    Graph
        The normalized graph.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(1,2,3,4)), 0.5)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(4,0,3,3)), 0.5)
    >>> normalized_graph <- ptdalgorithms::normalize(graph)
      )delim")
      
    .def("normalize_discrete", &ptdalgorithms::Graph::dph_normalize, 
      py::return_value_policy::reference_internal, R"delim(
    Normalizes the discrete phase-type distribution graph.

    This function normalizes the graph by ensuring that the sum of the weights of the outgoing edges from each vertex is equal to 1 in the discrete phase-type distribution.

    Parameters
    ----------
    None

    Returns
    -------
    Graph
        The normalized graph.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(1,2,3,4)), 0.5)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(4,0,3,3)), 0.5)
    >>> normalized_graph <- ptdalgorithms::normalize_discrete(graph)
      )delim")
      
    .def("notify_change", &ptdalgorithms::Graph::notify_change, R"delim(
    Notifies the graph of a change.

    This function should be called whenever the graph structure is modified to ensure internal consistency.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(1,2,3,4)), 0.5)
    >>> ptdalgorithms::notify_change(graph)
      )delim")
      
    .def("defect", &ptdalgorithms::Graph::defect, 
      py::return_value_policy::copy, R"delim(
    Computes the defect of the graph.

    The defect is the probability that the process does not reach an absorbing state.

    Parameters
    ----------
    None

    Returns
    -------
    double
        The defect of the graph.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(1,2,3,4)), 0.5)
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), ptdalgorithms::create_vertex(graph, c(4,0,3,3)), 0.5)
    >>> defect_value <- ptdalgorithms::defect(graph)
      )delim")
      
    .def("clone", &ptdalgorithms::Graph::clone, 
      py::return_value_policy::reference_internal, R"delim(
    Creates a copy of the graph.

    This function creates a deep copy of the graph, including all vertices and edges.

    Parameters
    ----------
    None

    Returns
    -------
    Graph
        A copy of the graph.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> graph_copy <- ptdalgorithms::clone(graph)
      )delim")
      


    .def("distribution_context",
      [](ptdalgorithms::Graph &graph, int granularity) {
        return new ptdalgorithms::ProbabilityDistributionContext(graph, granularity);
      }, 
      
      py::arg("granularity")=0, py::return_value_policy::move, 
      
      R"delim(
    Creates a probability distribution context for the graph.

    This function creates a context for computing the probability distribution of the graph.

    Parameters
    ----------
    granularity : int, optional
        The granularity of the distribution context. Default is 0.

    Returns
    -------
    ProbabilityDistributionContext
        A context for computing the probability distribution of the graph.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> context <- ptdalgorithms::distribution_context(graph, 10)
     )delim")      
      


     .def("distribution_context_discrete",
      [](ptdalgorithms::Graph &graph) {
        return new ptdalgorithms::DPHProbabilityDistributionContext(graph);
      }, 
      
      py::return_value_policy::move, 
      
      R"delim(
    Creates a discrete probability distribution context for the graph.

    This function creates a context for computing the discrete probability distribution of the graph.

    Parameters
    ----------
    None

    Returns
    -------
    DPHProbabilityDistributionContext
        A context for computing the discrete probability distribution of the graph.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> context <- ptdalgorithms::distribution_context_discrete(graph)
     )delim")      
      


    .def("pdf",
         py::vectorize(&ptdalgorithms::Graph::pdf), py::arg("time"), py::arg("granularity") = 0,
         py::return_value_policy::copy, R"delim(
    Computes the probability density function (PDF) of the phase-type distribution at a given time.

    This function computes the PDF of the phase-type distribution at a specified time.

    Parameters
    ----------
    time : double
        The time at which to evaluate the PDF.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    double
        The value of the PDF at the specified time.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::pdf(graph, 1.0) # => PDF value at time 1.0
    >>> ptdalgorithms::pdf(graph, 1.0, 10) # => PDF value at time 1.0 with granularity 10
      )delim")
      
    .def("cdf",
         py::vectorize(&ptdalgorithms::Graph::cdf), py::arg("time"), py::arg("granularity") = 0,
         py::return_value_policy::copy, R"delim(
    Computes the cumulative distribution function (CDF) of the phase-type distribution at a given time.

    This function computes the CDF of the phase-type distribution at a specified time.

    Parameters
    ----------
    time : double
        The time at which to evaluate the CDF.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    double
        The value of the CDF at the specified time.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::cdf(graph, 1.0) # => CDF value at time 1.0
    >>> ptdalgorithms::cdf(graph, 1.0, 10) # => CDF value at time 1.0 with granularity 10
      )delim")
      
    .def("pmf_discrete", py::vectorize(&ptdalgorithms::Graph::dph_pmf), py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Probability mass function of the discrete phase-type distribution.

    Returns the density (probability mass function) at a specific number of jumps.

    Parameters
    ----------
    x : IntegerVector
        Vector of the number of jumps (discrete time).
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.

    Returns
    -------
    NumericVector
        A numeric vector of the density.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::ddph(c(1, 2, 3), graph) # => density values at jumps 1, 2, and 3
      )delim")
      
    .def("cdf_discrete", py::vectorize(&ptdalgorithms::Graph::dph_cdf), py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Cumulative distribution function of the discrete phase-type distribution.

    Returns the cumulative distribution function (CDF) at a specific number of jumps.

    Parameters
    ----------
    q : IntegerVector
        Vector of the quantiles (jumps, discrete time).
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.

    Returns
    -------
    NumericVector
        A numeric vector of the distribution function.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::pdph(c(1, 2, 3), graph) # => CDF values at jumps 1, 2, and 3
      )delim")

    .def("stop_probability", &ptdalgorithms::Graph::stop_probability, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(
    Computes the stopping probability of the phase-type distribution at a given time.

    This function computes the probability that the process has stopped (reached an absorbing state) by a specified time.

    Parameters
    ----------
    time : double
        The time at which to evaluate the stopping probability.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    double
        The stopping probability at the specified time.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::stop_probability(graph, 1.0) # => Stopping probability at time 1.0
    >>> ptdalgorithms::stop_probability(graph, 1.0, 10) # => Stopping probability at time 1.0 with granularity 10
      )delim")

    .def("accumulated_visiting_time", &ptdalgorithms::Graph::accumulated_visiting_time, py::arg("time"), py::arg("granularity") = 0, 
      py::return_value_policy::copy, R"delim(
    Computes the accumulated visiting time of the phase-type distribution at a given time.

    This function computes the accumulated visiting time of the phase-type distribution at a specified time.

    Parameters
    ----------
    time : double
        The time at which to evaluate the accumulated visiting time.
    granularity : int, optional
        The granularity of the computation. Default is 0.

    Returns
    -------
    double
        The accumulated visiting time at the specified time.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> ptdalgorithms::accumulated_visiting_time(graph, 1.0) # => Accumulated visiting time at time 1.0
    >>> ptdalgorithms::accumulated_visiting_time(graph, 1.0, 10) # => Accumulated visiting time at time 1.0 with granularity 10
      )delim")

    // .def("stop_probability", 
    //      py::vectorize(&ptdalgorithms::Graph::stop_probability), py::arg("time"), py::arg("granularity") = 0,
    //      py::return_value_policy::copy, R"delim(


    //   )delim")

    // .def("accumulated_visiting_time", 
    //      py::vectorize(&ptdalgorithms::Graph::accumulated_visiting_time), py::arg("time"), py::arg("granularity") = 0,
    //      py::return_value_policy::copy, R"delim(


    //   )delim")

    
    .def("expectation_discrete", &_expectation_discrete,
      py::arg("rewards")=std::vector<double>(), 
      py::return_value_policy::move, 
      R"delim(
    Computes the expectation (mean) of the discrete phase-type distribution.

    This function computes the expectation of the discrete phase-type distribution given a set of rewards.

    Parameters
    ----------
    rewards : std::vector<double>
        A vector of rewards to be applied to the discrete phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    double
        The expectation of the discrete phase-type distribution.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> rewards <- c(1.0, 2.0, 3.0, 4.0)
    >>> ptdalgorithms::expectation_discrete(graph, rewards) # => Expectation value
    )delim")


      .def("variance_discrete", &_variance_discrete,
        py::arg("rewards")=std::vector<double>(), 
        py::return_value_policy::move, 
        R"delim(    
    Computes the variance of the discrete phase-type distribution.

    This function computes the variance of the discrete phase-type distribution given a set of rewards.

    Parameters
    ----------
    rewards : std::vector<double>
        A vector of rewards to be applied to the discrete phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.

    Returns
    -------
    double
        The variance of the discrete phase-type distribution.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> rewards <- c(1.0, 2.0, 3.0, 4.0)
    >>> ptdalgorithms::variance_discrete(graph, rewards) # => Variance value
    )delim")


    .def("stop_probability_discrete", &ptdalgorithms::Graph::dph_stop_probability, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Computes the probability of the Markov Chain of the discrete phase-type distribution standing at each vertex after a given number of jumps.

    This function computes the probability of the Markov Chain of the discrete phase-type distribution standing at each vertex after a specified number of jumps.

    Parameters
    ----------
    jumps : int
        The number of jumps (discrete time).

    Returns
    -------
    NumericVector
        A numeric vector of the stop probabilities for each vertex.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 0.5)
    >>> ptdalgorithms::add_edge(v1, v2, 0.8)
    >>> ptdalgorithms::add_edge(v2, a, 0.5)
    >>> ptdalgorithms::dph_stop_probability(graph, 3) # => Stop probabilities after 3 jumps
      )delim")

    .def("accumulated_visits_discrete", &ptdalgorithms::Graph::dph_accumulated_visits, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Computes the number of visits of the Markov Chain of the discrete phase-type distribution at each vertex after a given number of jumps.

    This function computes the number of visits of the Markov Chain of the discrete phase-type distribution at each vertex after a specified number of jumps.

    Parameters
    ----------
    jumps : int
        The number of jumps (discrete time).

    Returns
    -------
    NumericVector
        A numeric vector of the accumulated visits for each vertex.

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 0.5)
    >>> ptdalgorithms::add_edge(v1, v2, 0.8)
    >>> ptdalgorithms::add_edge(v2, a, 0.5)
    >>> ptdalgorithms::dph_accumulated_visits(graph, 3) # => Accumulated visits after 3 jumps
      )delim")

    .def("expected_visits_discrete", &ptdalgorithms::Graph::dph_expected_visits, py::arg("jumps"), 
      py::return_value_policy::copy, R"delim(
    Computes the expected jumps (or accumulated rewards) until absorption.
    This function can be used to compute the moments of a discrete phase-type distribution very fast and without much
    memory usage compared with the traditional matrix-based equations.
    The function takes in non-integers as rewards, but to be a *strictly* valid rewarded discrete phase-type distribution these should be integers.
    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by `ptdalgorithms::create_graph()`.
    rewards : Nullable<NumericVector>, optional
        Optional rewards, which should be applied to the discrete phase-type distribution. Must have length equal to `ptdalgorithms::vertices_length()`.
    Returns
    -------
    NumericVector
        A numeric vector where entry `i` is the expected rewarded jumps starting at vertex `i`.
    See Also
    --------
    ptdalgorithms::moments
    ptdalgorithms::expectation
    ptdalgorithms::variance
    ptdalgorithms::covariance
    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> rewards <- c(1.0, 2.0, 3.0, 4.0)
    >>> ptdalgorithms::dph_expected_visits(graph, rewards) # => Expected visits value
      )delim")
      
    .def("as_matrices",
      [](ptdalgorithms::Graph &graph) {
              return MatrixRepresentation(graph);
      }, py::return_value_policy::copy, R"delim(
    Converts the graph-based phase-type distribution into a traditional sub-intensity matrix and initial probability vector.

    Used to convert to the traditional matrix-based formulation. Has three entries: `$SIM` the sub-intensity matrix, `$IPV` the initial probability vector, `$states` the state of each vertex. Does *not* have the same order as [ptdalgorithms::vertices()]. The indices returned are 1-based, like the input to [ptdalgorithms::vertex_at()].

    Parameters
    ----------
    phase_type_graph : SEXP
        A reference to the graph created by [ptdalgorithms::create_graph()].

    Returns
    -------
    List
        A list of the sub-intensity matrix, states, and initial probability vector, and graph indices matching the matrix (1-indexed).

    Examples
    --------
    >>> graph <- ptdalgorithms::create_graph(4)
    >>> v2 <- ptdalgorithms::create_vertex(graph, c(4,0,3,3))
    >>> v1 <- ptdalgorithms::create_vertex(graph, c(1,2,3,4))
    >>> a <- ptdalgorithms::create_vertex(graph, c(0,0,0,0))
    >>> ptdalgorithms::add_edge(ptdalgorithms::starting_vertex(graph), v1, 1)
    >>> ptdalgorithms::add_edge(v1, v2, 4)
    >>> ptdalgorithms::add_edge(v2, a, 10)
    >>> ptdalgorithms::graph_as_matrix(graph)
    >>> # $`states`
    >>> #         [,1] [,2] [,3] [,4]
    >>> #   [1,]    1    2    3    4
    >>> #   [2,]    4    0    3    3
    >>> # $SIM
    >>> #         [,1]  [,2]
    >>> #   [1,]   -4     4
    >>> #   [2,]    0   -10
    >>> # $IPV
    >>> #   [1] 1 0
    >>> # $indices
    >>> #   [1] 3 2
      )delim")
    ;


    // Converts the matrix-based representation into a phase-type graph.

    // Sometimes the user might want to use the fast graph algorithms, but have some state-space given as a matrix. Therefore we can construct a graph from a matrix. If desired, a discrete phase-type distribution should just have no self-loop given. Note that the function `graph_as_matrix` may reorder the vertices to make the graph represented as strongly connected components in an acyclic manner.

    // Parameters
    // ----------
    // IPV : NumericVector
    //     The initial probability vector (alpha).
    // SIM : NumericMatrix
    //     The sub-intensity matrix (S).
    // rewards : NumericMatrix, optional
    //     The state/rewards of each of the vertices.

    // Returns
    // -------
    // SEXP
    //     A graph object.

    // Examples
    // --------
    // >>> g <- matrix_as_graph(
    // >>>     c(0.5,0.3, 0),
    // >>>     matrix(c(-3, 0, 0, 2, -4, 1, 0, 1,-3), ncol=3),
    // >>>     matrix(c(1,4,5,9,2,7), ncol=2)
    // >>> )
    // >>> graph_as_matrix(g)

  py::class_<ptdalgorithms::Vertex>(m, "Vertex", R"delim(

      )delim")

    .def(py::init(&ptdalgorithms::Vertex::init_factory), R"delim(

      )delim")
      
    .def("add_edge", &ptdalgorithms::Vertex::add_edge, py::arg("to"), py::arg("weight"), R"delim(
    Adds an edge between two vertices in the graph.

    The graph represents transitions between states as a weighted direction edge between two vertices.

    Parameters
    ----------
    phase_type_vertex_from : SEXP
        The vertex that transitions from.
    phase_type_vertex_to : SEXP
        The vertex that transitions to.
    weight : double
        The weight of the edge, i.e. the transition rate.
    parameterized_edge_state : NumericVector, optional
        Associate a numeric vector to an edge, for faster computations of moments when weights are changed.

    Examples
    --------
    >>> graph <- create_graph(4)
    >>> vertex_a <- find_or_create_vertex(graph, c(1,2,1,0))
    >>> vertex_b <- find_or_create_vertex(graph, c(2,0,1,0))
    >>> add_edge(vertex_a, vertex_b, 1.5)
      )delim")

    .def("ae", &ptdalgorithms::Vertex::add_edge, py::arg("to"), py::arg("weight"), R"delim(
      Alias for add_edge
      )delim")

    .def("__repr__",
      [](ptdalgorithms::Vertex &v) {

        std::ostringstream s;
        s << "(";
        std::vector<int> state = v.state();
        for (auto i(state.begin()); i != state.end(); i++) {
            if (state.begin() != i) s << ",";
            s << *i;
        }
        s << ")";
        return s.str();
      }, R"delim(

      )delim")

    .def("index",
        [](ptdalgorithms::Vertex &v) {
          int idx = v.vertex->index; // why is index not already an int?
          return  idx;
        }, py::return_value_policy::copy, R"delim(
  
        )delim")

    .def("add_edge_parameterized", &ptdalgorithms::Vertex::add_edge_parameterized, py::arg("to"), py::arg("weight"), py::arg("edge_state"), R"delim(
      Adds an edge between two vertices in the graph.
      The graph represents transitions between states as a weighted directed edge between two vertices.
      Parameters
      ----------
      phase_type_vertex_from : SEXP
          The vertex that transitions from.
      phase_type_vertex_to : SEXP
          The vertex that transitions to.
      weight : double
          The weight of the edge, i.e., the transition rate.
      parameterized_edge_state : NumericVector, optional
          Associate a numeric vector to an edge, for faster computations of moments when weights are changed.
      See Also
      --------
      ptdalgorithms::expected_waiting_time
      ptdalgorithms::moments
      ptdalgorithms::variance
      ptdalgorithms::covariance
      ptdalgorithms::graph_update_weights_parameterized
      Examples
      --------
      >>> graph <- create_graph(4)
      >>> vertex_a <- find_or_create_vertex(graph, c(1,2,1,0))
      >>> vertex_b <- find_or_create_vertex(graph, c(2,0,1,0))
      >>> add_edge(vertex_a, vertex_b, 1.5)
      )delim")

    .def("state",
        [](ptdalgorithms::Vertex &v) {
          // to make it return np.array instead of list without copying data
          auto a = new std::vector<int>(v.state());
          auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<int>*>(a); });
          return py::array(a->size(), a->data(), capsule);
        }, py::return_value_policy::copy, 
        R"delim(

      )delim")

    // .def("state", &ptdalgorithms::Vertex::state, 
    //   py::return_value_policy::reference_internal, R"delim(

    //   )delim")
      
    .def("edges", &ptdalgorithms::Vertex::edges, 
      py::return_value_policy::reference_internal, R"delim(
    Returns the out-going edges of a vertex.

    Returns a list of edges added by [ptdalgorithms::add_edge()].

    Parameters
    ----------
    phase_type_vertex : SEXP
        The vertex to find the edges for.

    Returns
    -------
    List
        A list of out-going edges.
      )delim")
      
    .def(py::self == py::self)
    .def("__assign__", [](ptdalgorithms::Vertex &v, const ptdalgorithms::Vertex &o) {
          return v = o;
    }, py::is_operator(), 
    py::return_value_policy::move, R"delim(

      )delim")
      
    // .def("c_vertex", &ptdalgorithms::Vertex::c_vertex, R"delim(

    //   )delim")
      
    .def("rate", &ptdalgorithms::Vertex::rate, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    ;

  py::class_<ptdalgorithms::Edge>(m, "Edge", R"delim(

      )delim")
      
      .def("__repr__",
        [](ptdalgorithms::Edge &e) {
          std::ostringstream s;
          s << "" << e.weight() << "-(";
          std::vector<int> state = e.to().state();
          for (auto i(state.begin()); i != state.end(); i++) {
              if (state.begin() != i) s << ",";
              s << *i;
          }
          s << ")";
          return s.str();
        }, R"delim(
  
        )delim")

    .def(py::init(&ptdalgorithms::Edge::init_factory), R"delim(

      )delim")
      
    .def("to", &ptdalgorithms::Edge::to, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("weight", &ptdalgorithms::Edge::weight, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("update_to", &ptdalgorithms::Edge::update_to, R"delim(

      )delim")
      
    .def("update_weight", &ptdalgorithms::Edge::update_weight, R"delim(

      )delim")

    .def("__assign__", [](ptdalgorithms::Edge &e, const ptdalgorithms::Edge &o) {
          return e = o;
    }, py::is_operator(), R"delim(

      )delim")
      
    ;

  py::class_<ptdalgorithms::ParameterizedEdge>(m, "ParameterizedEdge", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::ParameterizedEdge::init_factory), R"delim(

      )delim")
      
    .def("to", &ptdalgorithms::ParameterizedEdge::to, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("weight", &ptdalgorithms::ParameterizedEdge::weight, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    // .def("edge_state", 
    //   [](ptdalgorithms::ParameterizedEdge &edge) {
    //     auto a = edge.state;
    //     auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<double>*>(a); });
    //     return py::array(a->size(), a->data(), capsule);
    //   }, R"delim(

    // )delim")  
    .def("edge_state", &ptdalgorithms::ParameterizedEdge::edge_state, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def("__assign__", [](ptdalgorithms::ParameterizedEdge &e, const ptdalgorithms::ParameterizedEdge &o) {
          return e = o;
    }, py::is_operator(), py::return_value_policy::move, R"delim(

      )delim")
      
    ;

  py::class_<ptdalgorithms::PhaseTypeDistribution>(m, "PhaseTypeDistribution", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::PhaseTypeDistribution::init_factory), R"delim(

      )delim")
      
    // .def("c_distribution", &ptdalgorithms::PhaseTypeDistribution::c_distribution, R"delim(

    //   )delim")
      
    .def_readwrite("length", &ptdalgorithms::PhaseTypeDistribution::length, 
      py::return_value_policy::reference_internal, R"delim(

      )delim")
      
    .def_readwrite("vertices", &ptdalgorithms::PhaseTypeDistribution::vertices,
       py::return_value_policy::reference_internal, R"delim(

      )delim")      
    ;


  py::class_<ptdalgorithms::AnyProbabilityDistributionContext>(m, "AnyProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init<>(), R"delim(

      )delim")
      
    .def("is_discrete", &ptdalgorithms::AnyProbabilityDistributionContext::is_discrete, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("step", &ptdalgorithms::AnyProbabilityDistributionContext::step, 
      
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("pmf", &ptdalgorithms::AnyProbabilityDistributionContext::pmf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("pdf", &ptdalgorithms::AnyProbabilityDistributionContext::pdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("cdf", &ptdalgorithms::AnyProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("time", &ptdalgorithms::AnyProbabilityDistributionContext::time, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("jumps", &ptdalgorithms::AnyProbabilityDistributionContext::jumps, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("stop_probability", &ptdalgorithms::AnyProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    .def("accumulated_visits", &ptdalgorithms::AnyProbabilityDistributionContext::accumulated_visits, 
      py::return_value_policy::copy, R"delim(

      )delim")
      
    // .def("accumulated_visiting_time", &ptdalgorithms::AnyProbabilityDistributionContext::accumulated_visiting_time, 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")

    .def("accumulated_visiting_time",
      [](ptdalgorithms::AnyProbabilityDistributionContext &context) {
        // to make it return np.array instead of list without copying data
        auto a = new std::vector<long double>(context.accumulated_visiting_time());
        auto capsule = py::capsule(a, [](void *a) { delete reinterpret_cast<std::vector<long double>*>(a); });
        return py::array(a->size(), a->data(), capsule);
      }, py::return_value_policy::copy, 
      R"delim(

      )delim")
  
    ;


  py::class_<ptdalgorithms::ProbabilityDistributionContext>(m, "ProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::ProbabilityDistributionContext::init_factory),     
      py::return_value_policy::reference_internal, R"delim(

        )delim")
        


      
    // .def("__enter__",
    //   [](ptdalgorithms::ProbabilityDistributionContext &ctx) {

    //     // reset context

    //     return ctx;

    //   }, py::return_value_policy::move, R"delim(

    //   )delim")


    // .def("__exit__",
    //   [](ptdalgorithms::ProbabilityDistributionContext &ctx) {


    //     // reset context

    //   }, R"delim(

    //   )delim")
      



    .def("step", &ptdalgorithms::ProbabilityDistributionContext::step, 
      py::return_value_policy::copy, R"delim(
      Performs one jump in the probability distribution context for the discrete phase-type distribution.
      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::distribution_context()`.
      See Also
      --------
      ptdalgorithms::distribution_context
      )delim")
      
    .def("pdf", &ptdalgorithms::ProbabilityDistributionContext::pdf, 
      py::return_value_policy::copy, R"delim(
      Returns the PDF for the current probability distribution context for the phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::distribution_context()`.

      See Also
      --------
      ptdalgorithms::distribution_context

      Returns
      -------
      List
          A list containing the PDF, PMF, CDF, and time for the current probability distribution context.

      )delim")
      
    .def("cdf", &ptdalgorithms::ProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(
      Returns the CDF for the current probability distribution context for the phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::distribution_context()`.

      See Also
      --------
      ptdalgorithms::distribution_context

      Returns
      -------
      List
          A list containing the PDF, PMF, CDF, and time for the current probability distribution context.


      )delim")
      
    .def("time", &ptdalgorithms::ProbabilityDistributionContext::time, 
      py::return_value_policy::copy, R"delim(
      Returns the time for the current probability distribution context for the phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::distribution_context()`.

      See Also
      --------
      ptdalgorithms::distribution_context

      Returns
      -------
      List
          A list containing the PDF, PMF, CDF, and time for the current probability distribution context.


      )delim");
      

//       distribution_context_stop_probability
// //' Returns the stop probability for the current probability distribution context for the phase-type distribution.
// //' 
// //' @description
// //' This allows the user to step through the distribution, computing e.g. the
// //' time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
// //' *mutates* the context
// //' 
// //' @seealso [ptdalgorithms::distribution_context()]
// //' 
// //' @param probability_distribution_context The context created by [ptdalgorithms::distribution_context()]
// //' 
// // [[Rcpp::export]]


// NumericVector distribution_context_accumulated_visiting_time
// //' Returns the accumulated visiting time (integral of stop probability) for the current probability distribution context for the phase-type distribution.
// //' 
// //' @description
// //' This allows the user to step through the distribution, computing e.g. the
// //' time-inhomogeneous distribution function or the expectation of a multivariate phase-type distribution.
// //' *mutates* the context
// //' 
// //' @seealso [ptdalgorithms::distribution_context()]
// //' 
// //' @param probability_distribution_context The context created by [ptdalgorithms::distribution_context()]
// //' 
// // [[Rcpp::export]]
      




    // .def("stop_probability", &ptdalgorithms::ProbabilityDistributionContext::stop_probability, 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")
      
    // .def("accumulated_visiting_time", &ptdalgorithms::ProbabilityDistributionContext::accumulated_visiting_time, 
    //   py::return_value_policy::copy, R"delim(

    //   )delim")
    ;


  py::class_<ptdalgorithms::DPHProbabilityDistributionContext>(m, "DPHProbabilityDistributionContext", R"delim(

      )delim")
      
    .def(py::init(&ptdalgorithms::DPHProbabilityDistributionContext::init_factory), R"delim(

      )delim")
      
    .def("step", &ptdalgorithms::DPHProbabilityDistributionContext::step, 
      py::return_value_policy::copy, R"delim(
//' Performs one jump in a probability distribution context for the discrete phase-type distribution.
//' 
//' @description
//' This allows the user to step through the distribution, computing e.g. the
//' time-inhomogeneous distribution function or the expectation of a multivariate discrete phase-type distribution.
//' *mutates* the context
//' 
//' @seealso [ptdalgorithms::dph_distribution_context()]
//' 
//' @param probability_distribution_context The context created by [ptdalgorithms::dph_distribution_context()]
//' 
// [[Rcpp::export]]
      )delim")
      
    .def("pmf", &ptdalgorithms::DPHProbabilityDistributionContext::pmf, 
      py::return_value_policy::copy, R"delim(
      Returns the PMF for the current probability distribution context for the discrete phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a discrete multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::dph_distribution_context()`.

      See Also
      --------
      ptdalgorithms::dph_distribution_context

      Returns
      -------
      List
           A list containing the PMF, CDF, and jumps for the current probability distribution context.
      )delim")
      
    .def("cdf", &ptdalgorithms::DPHProbabilityDistributionContext::cdf, 
      py::return_value_policy::copy, R"delim(
      Returns the CDF for the current probability distribution context for the discrete phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a discrete multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::dph_distribution_context()`.

      See Also
      --------
      ptdalgorithms::dph_distribution_context

      Returns
      -------
      List
           A list containing the PMF, CDF, and jumps for the current probability distribution context.
      )delim")
      
    .def("jumps", &ptdalgorithms::DPHProbabilityDistributionContext::jumps, 
      py::return_value_policy::copy, R"delim(
      Returns the jumps for the current probability distribution context for the discrete phase-type distribution.

      This allows the user to step through the distribution, computing e.g. the time-inhomogeneous distribution function or the expectation of a discrete multivariate phase-type distribution.
      *Mutates* the context.

      Parameters
      ----------
      probability_distribution_context : SEXP
          The context created by `ptdalgorithms::dph_distribution_context()`.

      See Also
      --------
      ptdalgorithms::dph_distribution_context

      Returns
      -------
      List
           A list containing the PMF, CDF, and jumps for the current probability distribution context.
      )delim")
      
    .def("stop_probability", &ptdalgorithms::DPHProbabilityDistributionContext::stop_probability, 
      py::return_value_policy::copy, R"delim(
//' Returns the stop probability for the current probability distribution context for the discrete phase-type distribution.
//' 
//' @description
//' This allows the user to step through the distribution, computing e.g. the
//' time-inhomogeneous distribution function or the expectation of a multivariate discrete phase-type distribution.
//' *mutates* the context
//' 
//' @seealso [ptdalgorithms::dph_distribution_context()]
//' 
//' @param probability_distribution_context The context created by [ptdalgorithms::dph_distribution_context()]
//' 
// [[Rcpp::export]]
      )delim")
      
    .def("accumulated_visits", &ptdalgorithms::DPHProbabilityDistributionContext::accumulated_visits, 
      py::return_value_policy::copy, R"delim(
//' Returns the accumulated visits for the current probability distribution context for the discrete phase-type distribution.
//' 
//' @description
//' This allows the user to step through the distribution, computing e.g. the
//' time-inhomogeneous distribution function or the expectation of a multivariate discrete phase-type distribution.
//' *mutates* the context
//' 
//' @seealso [ptdalgorithms::dph_distribution_context()]
//' 
//' @param probability_distribution_context The context created by [ptdalgorithms::dph_distribution_context()]
//' 
// [[Rcpp::export]]
      )delim")
    ;
}