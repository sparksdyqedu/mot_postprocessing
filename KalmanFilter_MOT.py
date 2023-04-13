import torch
import torch.nn as nn
import time
from Linear_KF import KalmanFilter
from Extended_data import N_T
from Linear_sysmdl import SystemModel
import numpy as np
from trajectory_dataloader import TrajectoryDataset
from torch.utils import data
import os

r2 = torch.tensor([10, 1., 0.1, 1e-2, 1e-3])
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)

# Number of Training Examples
N_E = 1

# Number of Cross Validation Examples
N_CV = 1

m1_0 = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.float)
m2_0 = 0 * 0 * torch.eye(8)


def KFTest(SysModel, val_dataloader):
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.empty(N_T)

    start = time.time()
    KF = KalmanFilter(SysModel)
    KF.InitSequence(m1_0, m2_0)

    for index, (test_input, test_target) in enumerate(val_dataloader):
        # print(cv_input.shape)
        if test_input.shape[2] == 0:
            continue

        KF.GenerateSequence(test_input[0, :, :], test_input[0, :, :].shape[1])

        MSE_KF_linear_arr[index] = loss_fn(KF.x, test_target[0, :, :]).item()
        # MSE_KF_linear_arr[j] = loss_fn(test_input[j, :, :], test_target[j, :, :]).item()
    end = time.time()
    t = end - start

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_dB_std = torch.std(MSE_KF_linear_arr, unbiased=True)
    MSE_KF_dB_std = 10 * torch.log10(MSE_KF_dB_std)

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - MSE STD:", MSE_KF_dB_std, "[dB]")
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg]


if __name__ == '__main__':
    ndim = 4
    dt = 1
    _motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float)
    for i in range(ndim):
        _motion_mat[i, ndim + i] = dt

    _update_mat = np.eye(ndim, 2 * ndim, dtype=np.float)

    sys_model = SystemModel(F=torch.from_numpy(_motion_mat), q=q2[0], H=torch.from_numpy(_update_mat), r=r2[0], T=1,
                            T_test=1)
    sys_model.InitSequence(m1_0, m2_0)

    seqs = os.listdir("/Users/sparksd/Desktop/研究生/导师研究方向/dataset/MOT17/train")
    seqs = [item for item in seqs if item[:3] == "MOT"]

    val_datasets = TrajectoryDataset(seqs[3:4], train=False)
    val_dataloader = data.DataLoader(val_datasets)

    [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, val_dataloader)



