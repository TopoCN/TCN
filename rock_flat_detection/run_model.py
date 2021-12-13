import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cnn_pers_attention_layers import PeriodicAttensionTopoLayer, GraphModel
from torch.utils.data import DataLoader
import copy
from utility import separate_data, get_dataloader

path = os.getcwd()

loss_func = nn.CrossEntropyLoss()
log_interval = 10
dry_run = False
def train(model, train_set, optimizer, epoch, device):
    model.train()
    train_loader = train_set
    for batch_idx, (data_pd_1, data_pd_2, data_pd_3, data_pd_4, image, target) in enumerate(train_loader):
        data_pd_1, data_pd_2, data_pd_3, data_pd_4, image, target = data_pd_1.to(device), data_pd_2.to(device), data_pd_3.to(device), \
                                                                    data_pd_4.to(device), image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model([data_pd_1, data_pd_2, data_pd_3, data_pd_4], image)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 10:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, test_set, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_pd_1, data_pd_2, data_pd_3, data_pd_4, image, target in test_set:
            data_pd_1, data_pd_2, data_pd_3, data_pd_4, image, target = data_pd_1.to(device), data_pd_2.to(device), \
                                                                        data_pd_3.to(device), data_pd_4.to(device), image.to(device), target.to(device)
            output = model([data_pd_1, data_pd_2, data_pd_3, data_pd_4], image)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_set.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_set.dataset),
        100. * correct / 640))

    return 100. * correct / 640

DEVICE = 'cuda:0'

def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for classification task')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 3)')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='indices of GPUs')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 20)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.95,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--hidden_dim_0', type=int, default=1,
                        help='number of hidden units (default: 8)')
    parser.add_argument('--hidden_dim_1', type=int, default=32,
                        help='number of hidden units (default: 32)')
    parser.add_argument('--hidden_dim_2', type=int, default=64,
                        help='number of hidden units (default: 256)')
    parser.add_argument('--final_dropout', type=float, default=0.1,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--ratio', type=int, default=0.8,
                        help='training and testing split ratio')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='number of topological features')
    parser.add_argument('--nu', type=float, default=0.1,
                        help='nu parameter')
    parser.add_argument('--num_class', type=int, default=3,
                        help='number of classes')
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'

    labels = np.load('labels.npz', allow_pickle=True)['arr_0']
    indices = np.where(labels == 1)[0].tolist() + np.where(labels == 4)[0].tolist() + np.where(labels == 5)[0].tolist()
    PDs = np.load('input_pattern_image_PDs.npz', allow_pickle = True)['arr_0'][indices,:,:]

    PDs_list = [PDs]

    # model
    model = GraphModel(nu= args.nu, n_elements= args.num_pairs, point_dimension = 2, rotate_num_dgms = 4, hks_num_dgms = 0,
                       dim_intermediate= args.hidden_dim_0, dim_out= args.hidden_dim_1, final_dropout= args.final_dropout, num_class= args.num_class)
    model = model.to(args.device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    train_set, test_set = get_dataloader(args, PDs_list)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):

        train(model, train_set, optimizer, epoch, args.device)
        acc = test(model, test_set, args.device)
        if acc > best_acc:
            best_acc = acc
    print(best_acc)

if __name__ == '__main__':
    main()
