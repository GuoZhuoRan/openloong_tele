/*
 * File: ik_7dof_F2O_OLR5.h
 *
 * MATLAB Coder version            : 3.1
 * C/C++ source code generated on  : 13-Jun-2024 13:46:18
 */

#ifndef IK_7DOF_F2O_OLR5_H
#define IK_7DOF_F2O_OLR5_H

/* Include Files */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "rt_defines.h"
#include "rt_nonfinite.h"
#include "rtwtypes.h"
#include "ik_7dof_F2O_OLR5_types.h"

/* Function Declarations */
extern void ik_7dof_F2O_OLR5(double z_alpha, double y_beta, double x_gamma,
  double p_x, double p_y, double p_z, double bet, const double cur_theta[7],
  double arm_LRflag, double outTh[8]);

#endif

/*
 * File trailer for ik_7dof_F2O_OLR5.h
 *
 * [EOF]
 */
