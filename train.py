import os
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
    train_set = DataLoader(
        KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index),
        batch_size=batch_size, num_workers=4, shuffle=True)

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0

    conv_e.train(True)
    y_onehot = torch.LongTensor(batch_size, len(data.e_to_index))
    for x, y in progress_bar:
        s, r, o = x
        s, r = Variable(s).cuda(), Variable(r).cuda()

        if s.size()[0] != batch_size:
            y_onehot = torch.LongTensor(s.size()[0], len(data.e_to_index))

        y_onehot.zero_()
        y_onehot = y_onehot.scatter_(1, o.view(-1, 1), y.view(-1, 1))

        conv_e.zero_grad()
        output = conv_e(s, r)
        loss = loss_fn(output, Variable(y_onehot, requires_grad=False).float().cuda())
        loss.backward()
        optimizer.step()

        if moving_loss == 0:
            moving_loss = loss.data[0]
        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1

        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.data[0], moving_loss))


def valid(data, conv_e, batch_size):
    dataset = KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index)
    valid_set = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    conv_e.train(False)
    correct_count = 0
    for x, y in tqdm(iter(valid_set)):
        s, r, o = x
        s, r = Variable(s).cuda(), Variable(r).cuda()

        output = conv_e(s, r)
        values, indices = output.data.max(1)

        correct = indices == o.cuda()
        correct_count += correct.sum()

    print('Avg Acc: {:.5f}'.format(correct_count /
                                   len(dataset)))


def main():
    parser = argparse.ArgumentParser(description='Train ConvE with PyTorch.')
    parser.add_argument('train_path', action='store', type=str)
    parser.add_argument('valid_path', action='store', type=str)
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=128)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=30)

    args = parser.parse_args()

    os.makedirs('checkpoint/', exist_ok=True)
    with open(args.train_path, 'rb') as f:
        train_data = AttributeDict(pickle.load(f))
    with open(args.valid_path, 'rb') as f:
        valid_data = AttributeDict(pickle.load(f))

    # always use training data dictionaries
    valid_data.e_to_index = train_data.e_to_index
    valid_data.index_to_e = train_data.index_to_e
    valid_data.r_to_index = train_data.r_to_index
    valid_data.index_to_r = train_data.index_to_r

    conv_e = ConvE(num_e=len(train_data.e_to_index), num_r=len(train_data.r_to_index)).cuda()
    loss = nn.BCELoss()
    optimizer = optim.Adam(conv_e.parameters(), lr=0.0001)

    for epoch in range(args.epochs):
        train(epoch, train_data, conv_e, loss, optimizer, args.batch_size)
        valid(valid_data, conv_e, args.batch_size)
        valid(train_data, conv_e, args.batch_size)

        with open('checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(conv_e, f)


if __name__ == '__main__':
    main()
