import numpy as np
from src import *
from casadi import *

kinematic_casadi_world = casadi.Function.load("/home/cusadi/src/casadi_functions/go1_com_jacobian_world.casadi")

x0 = np.array([0.9, 0.9, 0.9, 0.708, 0., 0., 0.708,
                                    0.9, 0.8, -0.9,
                                    0.9, -0.9, 0.9,
                                    0, 0, 0,
                                    0, 0, 0])
u0 = np.array([1.0, 0., 0., 0., 0., 0.,
                                    0.1, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0])

J_casadi = kinematic_casadi_world(x0)
print("Casadi Jacobian:\n", J_casadi)