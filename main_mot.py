from Pipeline_KF_MOT import Pipeline_KF
from datetime import datetime
import torch
import numpy as np
from KalmanNet_nn import KalmanNetNN
from Linear_sysmdl import SystemModel
from trajectory_dataloader import TrajectoryDataset
from torch.utils import data
import os

r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = torch.mul(v,r2)


# Number of Training Examples
N_E = 1

# Number of Cross Validation Examples
N_CV = 1



m1_0 = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=torch.float)
m2_0 = 0 * 0 * torch.eye(8)

if __name__ == '__main__':
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow

    ndim = 4
    dt = 1
    _motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float)
    for i in range(ndim):
        _motion_mat[i, ndim + i] = dt

    _update_mat = np.eye(ndim, 2 * ndim, dtype=np.float)

    # T 是轨迹的长度
    sys_model = SystemModel(F=torch.from_numpy(_motion_mat), q=q2[0], H=torch.from_numpy(_update_mat), r=r2[0], T=1, T_test=1)
    sys_model.InitSequence(m1_0, m2_0)

    KNet_Pipeline = Pipeline_KF(strTime, "KNet", "KNet_MOT.pt")
    KNet_Pipeline.setssModel(sys_model)
    KNet_model = KalmanNetNN()
    KNet_model.Build(sys_model)
    KNet_Pipeline.setModel(KNet_model)
    KNet_Pipeline.setTrainingParams(n_Epochs=500, n_Batch=30, learningRate=1 * 1E-4, weightDecay=0.99)

    seqs = os.listdir("/Users/sparksd/Desktop/研究生/导师研究方向/dataset/MOT17/train")
    seqs = [item for item in seqs if item[:3] == "MOT"]
    train_datasets = TrajectoryDataset(seqs[3:4])
    train_dataloader = data.DataLoader(train_datasets)

    val_datasets = TrajectoryDataset(seqs[3:4], train=False)
    val_dataloader = data.DataLoader(val_datasets)

    for index, (train_input, train_target) in enumerate(train_dataloader):

        KNet_Pipeline.NNTrain(N_E, train_dataloader, N_CV, val_dataloader)
        # KNet_Pipeline.NNTest(val_dataloader)
        # [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(1, test_input, test_target)