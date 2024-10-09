/*
 * File: main.c
 *
 * MATLAB Coder version            : 3.1
 * C/C++ source code generated on  : 13-Jun-2024 13:46:18
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/
/* Include Files */
#include "rt_nonfinite.h"
#include "ik_7dof_F2O_OLR5.h"
#include "main.h"
#include "ik_7dof_F2O_OLR5_terminate.h"
#include "ik_7dof_F2O_OLR5_initialize.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

/* Function Declarations */
static void argInit_1x7_real_T(double result[7]);
static double argInit_real_T(void);
static void main_ik_7dof_F2O_OLR5(void);

/* Function Definitions */

/*
 * Arguments    : double result[7]
 * Return Type  : void
 */
static void argInit_1x7_real_T(double result[7])
{
  int idx1;

  /* Loop over the array to initialize each element. */
  for (idx1 = 0; idx1 < 7; idx1++) {
    /* Set the value of the array element.
       Change this value to the value that the application requires. */
    result[idx1] = argInit_real_T();
  }
}

/*
 * Arguments    : void
 * Return Type  : double
 */
static double argInit_real_T(void)
{
  return 0.0;
}

/*
 * Arguments    : void
 * Return Type  : void
 */
static void main_ik_7dof_F2O_OLR5(void)
{
  double dv5[7];
  double outTh[8];

  /* Initialize function 'ik_7dof_F2O_OLR5' input arguments. */
  /* Initialize function input argument 'cur_theta'. */
  /* Call the entry-point 'ik_7dof_F2O_OLR5'. */
  argInit_1x7_real_T(dv5);
  ik_7dof_F2O_OLR5(argInit_real_T(), argInit_real_T(), argInit_real_T(),
                   argInit_real_T(), argInit_real_T(), argInit_real_T(),
                   argInit_real_T(), dv5, argInit_real_T(), outTh);
}

/*
 * Arguments    : int argc
 *                const char * const argv[]
 * Return Type  : int
 */
int main(int argc, const char * const argv[])
{
  (void)argc;
  (void)argv;

  /* Initialize the application.
     You do not need to do this more than one time. */
  ik_7dof_F2O_OLR5_initialize();

  /* Invoke the entry-point functions.
     You can call entry-point functions multiple times. */
  main_ik_7dof_F2O_OLR5();

  /* Terminate the application.
     You do not need to do this more than one time. */
  ik_7dof_F2O_OLR5_terminate();
  return 0;
}

PYBIND11_MODULE(ik_arm, m) {
    m.def("initialize", &ik_7dof_F2O_OLR5_initialize, "Initialize the application");
    m.def("run", &main_ik_7dof_F2O_OLR5, "Run the main function");
    m.def("terminate", &ik_7dof_F2O_OLR5_terminate, "Terminate the application");
}

