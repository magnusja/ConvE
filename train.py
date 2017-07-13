
import pickle
import argparse
import torch.nn as nn
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KnowledgeGraphDataset
from model import ConvE


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def train(epoch, data, conv_e, loss_fn, optimizer, batch_size):
    train_set = DataLoader(KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index),
                           batch_size=batch_size, num_workers=4, shuffle=True)

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0

    conv_e.train(True)
    y_onehot = torch.FloatTensor(batch_size, len(data.e_to_index))
    for x, y in progress_bar:
        s, r, o = x
        s, r, p = Variable(s).cuda(), Variable(r).cuda(), Variable(o).cuda()

        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)

        conv_e.zero_grad()
        output = conv_e(s, r)
        loss = loss_fn(output, Variable(y_onehot).cuda())
        loss.backward()
        optimizer.step()

        if moving_loss == 0:
            moving_loss = loss.data[0]
        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1

        progress_bar.set_description('Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.data[0], moving_loss))


def main():

    parser = argparse.ArgumentParser(description='Train ConvE with PyTorch.')
    parser.add_argument('train_path', action='store', type=str)
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=128)
    parser.add_argument('--epochs', action='store', type=str, dest='epochs', default=30)

    args = parser.parse_args()

    with open(args.train_path, 'rb') as f:
        data = AttributeDict(pickle.load(f))

    conv_e = ConvE(num_e=len(data.e_to_index), num_r=len(data.r_to_index)).cuda()
    loss = nn.BCELoss()
    optimizer = optim.Adam(conv_e.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        train(epoch, data, conv_e, loss, optimizer, args.batch_size)


if __name__ == '__main__':
    main()
