#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ik_7dof_F2O_OLR5.h"


double ik_arm(double z_alpha, double y_beta, double x_gamma, double p_x,
                      double p_y, double p_z, double bet, const double
                      cur_theta[7], double arm_LRflag)
                      {
                        
                        ik_7dof_F2O_OLR5(z_alpha, y_beta,  x_gamma,  p_x,\
                      p_y,  p_z,  bet, cur_theta, arm_LRflag, double outTh[8]);
                      
                      return outTh;
                      }

namespace py = pybind11;
PYBIND11_MODULE(ik_module, m) {
    m.def("ik_arm", &ik_arm, "Inverse Kinematics function",
          py::arg("z_alpha"), py::arg("y_beta"), py::arg("x_gamma"),
          py::arg("p_x"), py::arg("p_y"), py::arg("p_z"), py::arg("bet"),
          py::arg("cur_theta"), py::arg("arm_LRflag"));
}
