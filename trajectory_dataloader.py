from torch.utils import data
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from cython_bbox import bbox_overlaps as bbox_ious
import lap


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


np.set_printoptions(suppress=True)


class TrajectoryDataset(data.Dataset):

    def __init__(self, seqs, train=True):
        self.tracklets_labels = {}
        self.seqs = seqs
        self.tracklets_lens = []

        for seq in seqs:
            detections_file = '/Users/sparksd/deeplearningProject/bytetrack/yolox_x_mix_det/bbox_result/{}.txt'.format(
                seq)
            gt_file = '/Users/sparksd/Desktop/研究生/导师研究方向/dataset/MOT17/train/{}/gt/gt.txt'.format(seq)
            detections = np.loadtxt(detections_file, delimiter=',', dtype=np.float)
            gt_tracking = np.loadtxt(gt_file, delimiter=',', dtype=np.float)
            gt_tracking[:, 4] += gt_tracking[:, 2]
            gt_tracking[:, 5] += gt_tracking[:, 3]

            tracklets_label = {}
            start = int(min(detections[:, 0]))
            end = int(max(detections[:, 0])) + 1
            if train:
                end = int(end / 2)
            else:
                start = start + int((end - start) / 2)
            for frame in range(start, end):
                # for frame in range(int(min(detections[:, 0])), 2):
                ious = bbox_ious(np.ascontiguousarray(gt_tracking[gt_tracking[:, 0] == frame][:, 2:6], dtype=np.float),
                                 np.ascontiguousarray(detections[detections[:, 0] == frame][:, 2:6], dtype=np.float))
                cost_matrix = 1 - ious
                matches, u_track, u_detection = linear_assignment(cost_matrix, 0.9)
                matches_map = {}
                for match in matches:
                    matches_map[match[0]] = match[1]
                track_ids = [int(item[1]) for item in gt_tracking[gt_tracking[:, 0] == frame]]
                one_frame_gt = gt_tracking[gt_tracking[:, 0] == frame]
                one_frame_det = detections[detections[:, 0] == frame]
                cnt = 0
                for trackid in track_ids:
                    if trackid not in tracklets_label.keys():
                        tracklets_label[trackid] = {}

                    gt_box = one_frame_gt[one_frame_gt[:, 1] == trackid][0][2:6].tolist()
                    if len(tracklets_label[trackid]) == 0:
                        gt_box.extend([0, 0, 0, 0])
                    else:
                        keys = list(tracklets_label[trackid].keys())
                        gt_box_pre = tracklets_label[trackid][keys[-1]][0].tolist()
                        gt_box.extend([gt_box[0] - gt_box_pre[0], gt_box[1] - gt_box_pre[1], gt_box[2] - gt_box_pre[2],
                                       gt_box[3] - gt_box_pre[3]])
                    gt_box = np.array(gt_box)
                    tracklets_label[trackid][frame] = [gt_box]

                    if cnt in matches_map.keys():
                        tracklets_label[trackid][frame].append(one_frame_det[matches_map[cnt]][2:6])
                    else:
                        num, dim = 1, 2
                        np.random.seed(0)
                        center_offset = np.random.randn(num, dim) * 5
                        wh_offset = np.random.randn(num, dim) * 10
                        tracklets_label[trackid][frame].append(np.array(
                            [gt_box[0] + center_offset[0][0], gt_box[1] + center_offset[0][1],
                             gt_box[2] + wh_offset[0][0],
                             gt_box[3] + wh_offset[0][1]]))

                    cnt += 1

            self.tracklets_labels[seq] = tracklets_label
            if len(self.tracklets_lens) == 0:
                self.tracklets_lens.append(len(tracklets_label))
            else:
                self.tracklets_lens.append(len(tracklets_label) + self.tracklets_lens[-1])

    def __getitem__(self, index):
        # print("get index = {}".format(index))
        ind = -1
        for i in range(len(self.tracklets_lens)):
            if index < self.tracklets_lens[i]:
                ind = i
                break

        X_train = np.array([[0, 0, 0, 0]])
        X_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])

        track_ids = list(self.tracklets_labels[self.seqs[ind]].keys())
        trackid = track_ids[index - self.tracklets_lens[ind]]

        for key in self.tracklets_labels[self.seqs[ind]][trackid].keys():
            X_train = np.concatenate((X_train, np.array([self.tracklets_labels[self.seqs[ind]][trackid][key][1]])),
                                     axis=0)
            X_label = np.concatenate((X_label, np.array([self.tracklets_labels[self.seqs[ind]][trackid][key][0]])),
                                     axis=0)

        X_train = X_train[1:]
        X_label = X_label[1:]

        # print(X_train.shape)
        # print(X_label.shape)
        assert X_label.shape[1] == 8
        assert X_train.shape[0] == X_label.shape[0]

        train_lo = np.random.randint(0, max(int(X_train.shape[0] / 2), 1))
        len_train = np.random.randint(5, 20)

        # print(X_label[9:49].shape)

        train_hi = min(train_lo + len_train, X_train.shape[0])

        if train_lo >= train_hi:
            train_lo = train_hi - 1

        # 262
        train_lo = 9
        train_hi = 15

        X_train = X_train[train_lo: train_hi]
        X_label = X_label[train_lo: train_hi]

        # print("train_lo = {}, len_train = {}".format(train_lo, len_train))
        # print("X_label = {}".format(X_label))

        X_train = X_train.T
        X_label = X_label.T

        # print("X_train.shape = {}".format(X_train.shape))
        # print("X_label.shape = {}".format(X_label.shape))

        assert X_train.shape[1] == X_label.shape[1]

        # def feature_normalize(data_tmp):
        #     mu = np.mean(data_tmp, axis=0)
        #     std = np.std(data_tmp, axis=0)
        #     return (data_tmp - mu) / std, mu, std
        #
        # X_train, mu, std = feature_normalize(X_train)
        # X_label = (X_label - mu) / std

        return X_train, X_label

    def __len__(self):
        # len_seqs = 0
        # for seq in self.seqs:
        #     len_seqs += len(self.tracklets_labels[seq])
        # return len_seqs
        return 1


if __name__ == '__main__':
    seqs = os.listdir("/home/yuqing/dataset/MOT17/train")
    seqs = [item for item in seqs if item[:3] == "MOT"]
    train_datasets = TrajectoryDataset(seqs[:5])
    train_dataloader = data.DataLoader(train_datasets)

    val_datasets = TrajectoryDataset(seqs[:5])
    val_dataloader = data.DataLoader(val_datasets)
    print(val_datasets.__len__())

    for index, (X_train, X_label) in enumerate(val_dataloader):
        print(X_train.shape, X_label.shape)
        # pass
