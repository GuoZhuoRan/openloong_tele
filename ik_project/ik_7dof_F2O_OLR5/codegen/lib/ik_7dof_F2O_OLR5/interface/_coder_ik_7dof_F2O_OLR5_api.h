/*
 * File: _coder_ik_7dof_F2O_OLR5_api.h
 *
 * MATLAB Coder version            : 3.1
 * C/C++ source code generated on  : 13-Jun-2024 13:46:18
 */

#ifndef _CODER_IK_7DOF_F2O_OLR5_API_H
#define _CODER_IK_7DOF_F2O_OLR5_API_H

/* Include Files */
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"
#include <stddef.h>
#include <stdlib.h>
#include "_coder_ik_7dof_F2O_OLR5_api.h"

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void ik_7dof_F2O_OLR5(real_T z_alpha, real_T y_beta, real_T x_gamma,
  real_T p_x, real_T p_y, real_T p_z, real_T bet, real_T cur_theta[7], real_T
  arm_LRflag, real_T outTh[8]);
extern void ik_7dof_F2O_OLR5_api(const mxArray *prhs[9], const mxArray *plhs[1]);
extern void ik_7dof_F2O_OLR5_atexit(void);
extern void ik_7dof_F2O_OLR5_initialize(void);
extern void ik_7dof_F2O_OLR5_terminate(void);
extern void ik_7dof_F2O_OLR5_xil_terminate(void);

#endif

/*
 * File trailer for _coder_ik_7dof_F2O_OLR5_api.h
 *
 * [EOF]
 */
