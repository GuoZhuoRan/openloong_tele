import ctypes
import time
# lib=ctypes.cdll.LoadLibrary('/home/zjk/OpenLoong_TeleVision/ik_project/ik_7dof_F2O_OLR5/codegen/lib/ik_7dof_F2O_OLR5/build/libik_arm.so'
#                             )

lib=ctypes.CDLL('/home/zjk/OpenLoong_TeleVision/ik_project/ik_7dof_F2O_OLR5/codegen/lib/ik_7dof_F2O_OLR5/build/libik_arm.so'
                            )

# Define the argument and return types for the C function
lib.ik_7dof_F2O_OLR5.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.POINTER(ctypes.c_double),
                                 ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
lib.ik_7dof_F2O_OLR5.restype = None

def ik_arm(z_alpha, y_beta, x_gamma, p_x, p_y, p_z, bet, cur_theta, arm_LRflag):
    # Prepare the input and output arrays
    cur_theta_array = (ctypes.c_double * 7)(*cur_theta)
    outTh = (ctypes.c_double * 8)()

    # Call the function
    lib.ik_7dof_F2O_OLR5(z_alpha, y_beta, x_gamma, p_x, p_y, p_z, bet, cur_theta_array, arm_LRflag, outTh)

    # Return the results as a list
    return [outTh[i] for i in range(8)]


# Example usage
cur_theta = [0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
time1=time.time()
result = ik_arm(0.0, 1.5708, 0.0, 0.0 ,100, 500.0, 0.0, cur_theta, 1.0)
t2=time.time()
t=t2-time1
print(t)
print(result)
