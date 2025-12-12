from wbc_ik_qp.ik import Model_Cusadi
from wbc_ik_qp.Centroidal import CentroidalModelInfoSimple
import torch

info = CentroidalModelInfoSimple(generalizedCoordinatesNum=19, actuatedDofNum=18, numThreeDofContacts=4)
m = Model_Cusadi(info, device=torch.device('cpu'), dtype=torch.float64)  # 先用 CPU 运行
q = torch.zeros((19,), dtype=torch.float64)  # CPU tensor
v = torch.zeros((18,), dtype=torch.float64)
# convert to numpy and call evaluate manually if needed (depending on Model_Cusadi impl)
print("calling getPosition with cpu tensors -> numpy conversion")
pos = m.getPosition(q, v, 'com')
print("pos:", pos)