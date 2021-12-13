import networkx as nx
import numpy as np
import random
import torch
import torch.utils.data
import os
from sklearn.model_selection import StratifiedKFold

path = os.getcwd()


def rotated_pd(norm_PD, angle): # @ return roated PD array
    rotate_angle = np.array([np.cos(np.pi / angle), np.sin(np.pi / angle), -np.sin(np.pi / angle), np.cos(np.pi / angle)]).reshape((2, 2))
    rotated_pd = np.einsum('pd, dm -> pm', norm_PD, rotate_angle)
    return rotated_pd


def separate_data(PDs_list, ratio = 0.8):
    norm_pds_list = PDs_list

    labels_ = np.load('labels.npz', allow_pickle = True)['arr_0']
    indices = np.where(labels_ == 1)[0].tolist() + np.where(labels_ == 4)[0].tolist() + np.where(labels_ == 5)[0].tolist()
    _labels_drop_ = labels_[indices]
    labels = np.zeros(shape=(_labels_drop_.shape[0],))
    for i in range(_labels_drop_.shape[0]):
        if _labels_drop_[i] == 1: # flat
            labels[i] = 0
        elif _labels_drop_[i] == 4: # rock-above
            labels[i] = 1
        elif _labels_drop_[i] == 5: # rock-below
            labels[i] = 2

    image = np.load('clipped_pattern_images.npz', allow_pickle=True)['arr_0'][indices,:,:]

    indices = np.random.permutation(labels.shape[0])
    train_idx, test_idx = indices[:int(np.round(labels.shape[0] * ratio))], indices[int(np.round(labels.shape[0] * ratio)):]

    # image
    train_image = image[train_idx, :, :, :]
    test_image = image[test_idx, :, :, :]

    # PDs group
    # different angles
    train_pds_list = [np.array([np.abs(rotated_pd(norm_pds_list[0][i], 6)) for i in train_idx]), np.array([np.abs(rotated_pd(norm_pds_list[0][i], 4)) for i in train_idx]),
                      np.array([np.abs(rotated_pd(norm_pds_list[0][i], 3)) for i in train_idx]), np.array([np.abs(rotated_pd(norm_pds_list[0][i], 2)) for i in train_idx])]

    test_pds_list = [np.array([np.abs(rotated_pd(norm_pds_list[0][i], 6)) for i in test_idx]), np.array([np.abs(rotated_pd(norm_pds_list[0][i], 4)) for i in test_idx]),
                     np.array([np.abs(rotated_pd(norm_pds_list[0][i], 3)) for i in test_idx]), np.array([np.abs(rotated_pd(norm_pds_list[0][i], 2)) for i in test_idx])]

    train_y = [labels[i] for i in train_idx]
    test_y = [labels[i] for i in test_idx]

    return train_pds_list, train_image, test_pds_list, test_image, train_y, test_y


def data_loader(X_r, image, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    if cuda == True:
        X_r = [torch.cuda.FloatTensor(X_r[i]) for i in range(len(X_r))]
        image = torch.cuda.FloatTensor(image)
        Y = torch.cuda.LongTensor(Y)
    else:
        X_r = [torch.FloatTensor(X_r[i]) for i in range(len(X_r))]
        image = torch.FloatTensor(image)
        Y = torch.LongTensor(Y)

    data = torch.utils.data.TensorDataset(X_r[0], X_r[1], X_r[2], X_r[3], image, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, PDs_list):
    train_pds_list, train_image, test_pds_list, test_image, train_y, test_y = separate_data(PDs_list, args.ratio)
    train_dataloader  = data_loader(train_pds_list, train_image, train_y, args.batch_size)
    test_dataloader = data_loader(test_pds_list, test_image, test_y, batch_size= 32)
    return train_dataloader, test_dataloader
