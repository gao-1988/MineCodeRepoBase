import torch
from utils.visual import *
from se_math import se3
import numpy as np

if __name__ == '__main__':
    G = [[0.98695725, -0.11640979, -0.11119429, -0.10477918],
         [0.10738844, 0.990666, -0.08395614, 0.28473622],
         [0.11992971, 0.07092016, 0.99024606, 0.16098942],
         [0, 0, 0, 1, ]]
    G = np.array(G)
    G2 = G.copy()
    G2[0:3, 0:3] += 0.3
    G = torch.from_numpy(G).unsqueeze(0)
    G2 = torch.from_numpy(G2).unsqueeze(0)
    # [2, 4, 4] [batch, 4, 4]
    G_all = torch.concat([G, G2])
    #
    # twist_vectors = se3.vec(G_all)
    # G_all_transfrom_twist=se3.exp(twist_vectors)
    # # should equal to twist_vector
    # twist_vectors_transform_G=se3.log(G_all_transfrom_twist)

    # points1 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
    # points2 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:, 0:3]
    # points1 = torch.from_numpy(points1).unsqueeze(0)
    # points2 = torch.from_numpy(points2).unsqueeze(0)
    # points_all = torch.concat([points1, points2])
    # points_all = points_all.transpose(1, 2)
    # points_all_transformed = se3.transform(G_all, points_all)
    # points_all=points_all.transpose(1, 2)
    # points_all_transformed=points_all_transformed.transpose(1, 2)
    # show(points_all[0], points_all[1], points_all_transformed[0], points_all_transformed[1])
