cmake_minimum_required(VERSION 3.10)
project(ik_arm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Add the source files
add_library(ik_arm SHARED ik_7dof_F2O_OLR5.c rt_nonfinite.c rtGetNaN.c norm.c rtGetInf.c)

