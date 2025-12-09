import numpy as np
from src import *
from casadi import *

kinematic_casadi = casadi.Function.load("/home/cusadi/src/casadi_functions/anymal_example_LF_FOOT_jacobian.casadi")

x0 = np.array([0.1, 0.2, 0.3, 0.3390053, 0.620543, 0.6205462, -0.3390048,
                                    1.5708, 1.5708, 0.71,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0])
u0 = np.array([1.0, 0., 0., 0., 0., 0.,
                                    0.1, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0])

J_casadi = kinematic_casadi(x0)
print("Casadi Jacobian:\n", J_casadi)