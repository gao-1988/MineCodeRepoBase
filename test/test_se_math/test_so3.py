import torch
from se_math import so3

# mats=torch.rand((2,1,1))
# print(mats)
# traces=so3.btrace(mats)
# print(traces)

rotation_mat = torch.rand((2, 3, 3))
rotation_mat_inv=so3.inverse(rotation_mat)
print(rotation_mat)
print(rotation_mat_inv)