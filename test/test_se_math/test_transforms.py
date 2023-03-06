import numpy as np
import torch
import se_math.transforms as transforms
from utils.visual import *

points1 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
points2 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:, 0:3]
points1 = torch.from_numpy(points1).unsqueeze(0)
points2 = torch.from_numpy(points2).unsqueeze(0)
points_all = torch.concat([points1, points2])
unit=transforms.uniform_2_sphere(10240)
show(unit)
