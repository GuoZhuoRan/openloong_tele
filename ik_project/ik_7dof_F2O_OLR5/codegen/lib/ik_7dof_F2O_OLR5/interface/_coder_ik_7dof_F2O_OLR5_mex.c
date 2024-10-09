/*
 * File: _coder_ik_7dof_F2O_OLR5_mex.c
 *
 * MATLAB Coder version            : 3.1
 * C/C++ source code generated on  : 13-Jun-2024 13:46:18
 */

/* Include Files */
#include "_coder_ik_7dof_F2O_OLR5_api.h"
#include "_coder_ik_7dof_F2O_OLR5_mex.h"

/* Function Declarations */
static void ik_7dof_F2O_OLR5_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T
  nrhs, const mxArray *prhs[9]);

/* Function Definitions */

/*
 * Arguments    : int32_T nlhs
 *                const mxArray *plhs[1]
 *                int32_T nrhs
 *                const mxArray *prhs[9]
 * Return Type  : void
 */
static void ik_7dof_F2O_OLR5_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T
  nrhs, const mxArray *prhs[9])
{
  int32_T n;
  const mxArray *inputs[9];
  const mxArray *outputs[1];
  int32_T b_nlhs;
  emlrtStack st = { NULL, NULL, NULL };

  st.tls = emlrtRootTLSGlobal;

  /* Check for proper number of arguments. */
  if (nrhs != 9) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 9, 4,
                        16, "ik_7dof_F2O_OLR5");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 16,
                        "ik_7dof_F2O_OLR5");
  }

  /* Temporary copy for mex inputs. */
  for (n = 0; n < nrhs; n++) {
    inputs[n] = prhs[n];
  }

  /* Call the function. */
  ik_7dof_F2O_OLR5_api(inputs, outputs);

  /* Copy over outputs to the caller. */
  if (nlhs < 1) {
    b_nlhs = 1;
  } else {
    b_nlhs = nlhs;
  }

  emlrtReturnArrays(b_nlhs, plhs, outputs);

  /* Module termination. */
  ik_7dof_F2O_OLR5_terminate();
}

/*
 * Arguments    : int32_T nlhs
 *                const mxArray * const plhs[]
 *                int32_T nrhs
 *                const mxArray * const prhs[]
 * Return Type  : void
 */
void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  mexAtExit(ik_7dof_F2O_OLR5_atexit);

  /* Initialize the memory manager. */
  /* Module initialization. */
  ik_7dof_F2O_OLR5_initialize();

  /* Dispatch the entry-point. */
  ik_7dof_F2O_OLR5_mexFunction(nlhs, plhs, nrhs, prhs);
}

/*
 * Arguments    : void
 * Return Type  : emlrtCTX
 */
emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/*
 * File trailer for _coder_ik_7dof_F2O_OLR5_mex.c
 *
 * [EOF]
 */
