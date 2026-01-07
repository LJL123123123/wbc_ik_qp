import numpy as np
from src import *
from casadi import *
import torch

J_casadi_world = casadi.Function.load("/home/ReLUQP-py/wbc_ik_qp/src/casadi_functions/go1_com_jacobian_world.casadi")

ori_casadi_world = casadi.Function.load("/home/ReLUQP-py/wbc_ik_qp/src/casadi_functions/go1_com_attitude.casadi")
ori_cusadi = CusadiFunction(ori_casadi_world, 1, "go1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

x0 = torch.tensor([0.9, 0.9, 0.9,  0.214, 0.572, -0.170,   0.773,
                                    0.9, 0.8, -0.9,
                                    0.9, -0.9, 0.9,
                                    0, 0, 0,
                                    0, 0, 0],device=device, dtype=dtype)
u0 = torch.tensor([0., 0., 0.0, 0., 0., 0.,
                                        0., 0., 0.0,
                                        0., 0., 0.0,
                                        0., 0., 0.0,
                                        0., 0., 0.0],device=device, dtype=dtype)

ori_cusadi.evaluate((x0, u0))
ori = ori_cusadi.getDenseOutput(0)

# J_casadi = J_casadi_world(x0)
# print("Casadi Jacobian:\n", J_casadi)

ori_casadi = ori_casadi_world(x0.cpu().numpy())
print("ori\n", ori)
print("ori_casadi\n", ori_casadi)