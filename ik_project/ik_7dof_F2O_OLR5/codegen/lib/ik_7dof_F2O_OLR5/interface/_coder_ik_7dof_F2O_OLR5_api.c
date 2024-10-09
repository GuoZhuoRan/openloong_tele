/*
 * File: _coder_ik_7dof_F2O_OLR5_api.c
 *
 * MATLAB Coder version            : 3.1
 * C/C++ source code generated on  : 13-Jun-2024 13:46:18
 */

/* Include Files */
#include "tmwtypes.h"
#include "_coder_ik_7dof_F2O_OLR5_api.h"
#include "_coder_ik_7dof_F2O_OLR5_mex.h"

/* Variable Definitions */
emlrtCTX emlrtRootTLSGlobal = NULL;
emlrtContext emlrtContextGlobal = { true, false, 131434U, NULL,
  "ik_7dof_F2O_OLR5", NULL, false, { 2045744189U, 2170104910U, 2743257031U,
    4284093946U }, NULL };

/* Function Declarations */
static real_T b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId);
static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray
  *cur_theta, const char_T *identifier))[7];
static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[7];
static real_T e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId);
static real_T emlrt_marshallIn(const emlrtStack *sp, const mxArray *z_alpha,
  const char_T *identifier);
static const mxArray *emlrt_marshallOut(const real_T u[8]);
static real_T (*f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[7];

/* Function Definitions */

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *u
 *                const emlrtMsgIdentifier *parentId
 * Return Type  : real_T
 */
static real_T b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId)
{
  real_T y;
  y = e_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *cur_theta
 *                const char_T *identifier
 * Return Type  : real_T (*)[7]
 */
static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray
  *cur_theta, const char_T *identifier))[7]
{
  real_T (*y)[7];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = d_emlrt_marshallIn(sp, emlrtAlias(cur_theta), &thisId);
  emlrtDestroyArray(&cur_theta);
  return y;
}
/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *u
 *                const emlrtMsgIdentifier *parentId
 * Return Type  : real_T (*)[7]
 */
  static real_T (*d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
  const emlrtMsgIdentifier *parentId))[7]
{
  real_T (*y)[7];
  y = f_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *src
 *                const emlrtMsgIdentifier *msgId
 * Return Type  : real_T
 */
static real_T e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src, const
  emlrtMsgIdentifier *msgId)
{
  real_T ret;
  static const int32_T dims = 0;
  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 0U, &dims);
  ret = *(real_T *)mxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *z_alpha
 *                const char_T *identifier
 * Return Type  : real_T
 */
static real_T emlrt_marshallIn(const emlrtStack *sp, const mxArray *z_alpha,
  const char_T *identifier)
{
  real_T y;
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(sp, emlrtAlias(z_alpha), &thisId);
  emlrtDestroyArray(&z_alpha);
  return y;
}

/*
 * Arguments    : const real_T u[8]
 * Return Type  : const mxArray *
 */
static const mxArray *emlrt_marshallOut(const real_T u[8])
{
  const mxArray *y;
  const mxArray *m0;
  static const int32_T iv0[2] = { 0, 0 };

  static const int32_T iv1[2] = { 1, 8 };

  y = NULL;
  m0 = emlrtCreateNumericArray(2, iv0, mxDOUBLE_CLASS, mxREAL);
  mxSetData((mxArray *)m0, (void *)u);
  emlrtSetDimensions((mxArray *)m0, iv1, 2);
  emlrtAssign(&y, m0);
  return y;
}

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *src
 *                const emlrtMsgIdentifier *msgId
 * Return Type  : real_T (*)[7]
 */
static real_T (*f_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[7]
{
  real_T (*ret)[7];
  static const int32_T dims[2] = { 1, 7 };

  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 2U, dims);
  ret = (real_T (*)[7])mxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}
/*
 * Arguments    : const mxArray *prhs[9]
 *                const mxArray *plhs[1]
 * Return Type  : void
 */
  void ik_7dof_F2O_OLR5_api(const mxArray *prhs[9], const mxArray *plhs[1])
{
  real_T (*outTh)[8];
  real_T z_alpha;
  real_T y_beta;
  real_T x_gamma;
  real_T p_x;
  real_T p_y;
  real_T p_z;
  real_T bet;
  real_T (*cur_theta)[7];
  real_T arm_LRflag;
  emlrtStack st = { NULL, NULL, NULL };

  st.tls = emlrtRootTLSGlobal;
  outTh = (real_T (*)[8])mxMalloc(sizeof(real_T [8]));
  prhs[7] = emlrtProtectR2012b(prhs[7], 7, false, -1);

  /* Marshall function inputs */
  z_alpha = emlrt_marshallIn(&st, emlrtAliasP(prhs[0]), "z_alpha");
  y_beta = emlrt_marshallIn(&st, emlrtAliasP(prhs[1]), "y_beta");
  x_gamma = emlrt_marshallIn(&st, emlrtAliasP(prhs[2]), "x_gamma");
  p_x = emlrt_marshallIn(&st, emlrtAliasP(prhs[3]), "p_x");
  p_y = emlrt_marshallIn(&st, emlrtAliasP(prhs[4]), "p_y");
  p_z = emlrt_marshallIn(&st, emlrtAliasP(prhs[5]), "p_z");
  bet = emlrt_marshallIn(&st, emlrtAliasP(prhs[6]), "bet");
  cur_theta = c_emlrt_marshallIn(&st, emlrtAlias(prhs[7]), "cur_theta");
  arm_LRflag = emlrt_marshallIn(&st, emlrtAliasP(prhs[8]), "arm_LRflag");

  /* Invoke the target function */
  ik_7dof_F2O_OLR5(z_alpha, y_beta, x_gamma, p_x, p_y, p_z, bet, *cur_theta,
                   arm_LRflag, *outTh);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(*outTh);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void ik_7dof_F2O_OLR5_atexit(void)
{
  emlrtStack st = { NULL, NULL, NULL };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  ik_7dof_F2O_OLR5_xil_terminate();
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void ik_7dof_F2O_OLR5_initialize(void)
{
  emlrtStack st = { NULL, NULL, NULL };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, 0);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void ik_7dof_F2O_OLR5_terminate(void)
{
  emlrtStack st = { NULL, NULL, NULL };

  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/*
 * File trailer for _coder_ik_7dof_F2O_OLR5_api.c
 *
 * [EOF]
 */
