import os
import pickle
import argparse

import torch.nn as nn
import torch

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KnowledgeGraphDataset, collate_data, collate_valid
from model import ConvE
from util import AttributeDict


class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def train(epoch, data, conv_e, criterion, optimizer, batch_size):
    train_set = DataLoader(
        KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index),
        collate_fn=collate_data, batch_size=batch_size, num_workers=4, shuffle=True)

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0

    conv_e.train(True)
    y_onehot = torch.LongTensor(batch_size, len(data.e_to_index))
    for s, r, os in progress_bar:
        s, r = Variable(s).cuda(), Variable(r).cuda()

        if s.size()[0] != batch_size:
            y_onehot = torch.LongTensor(s.size()[0], len(data.e_to_index))

        y_onehot.zero_()
        y_onehot = y_onehot.scatter_(1, os, 1)
        y_smooth = (1 - 0.1) * y_onehot.float() + 0.1 / len(data.e_to_index)

        targets = Variable(y_onehot.float(), requires_grad=False).cuda()

        output = conv_e(s, r)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        conv_e.zero_grad()

        if moving_loss == 0:
            moving_loss = loss.data[0]
        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1

        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.data[0], moving_loss))


def valid(data, conv_e, batch_size):
    def mrr(ranks):
        return torch.mean(1 / ranks)
    dataset = KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index)
    valid_set = DataLoader(dataset, collate_fn=collate_valid, batch_size=batch_size, num_workers=4, shuffle=True)

    conv_e.train(False)
    running_mrr = .0
    for s, r, os in tqdm(iter(valid_set)):
        s, r = Variable(s).cuda(), Variable(r).cuda()
        ranks = list()
        output = conv_e.test(s, r)
        for i in range(min(batch_size, s.size()[0])):
            _, top_indices = output[i].topk(output.size()[1])
            for o in os[i]:
                _, rank = (top_indices == o).max(dim=0)
                ranks.append(rank.data[0] + 1)

        running_mrr = (running_mrr + mrr(torch.FloatTensor(ranks))) / 2

    print()
    print('MRR: {:.10f}'.format(running_mrr))


def main():
    parser = argparse.ArgumentParser(description='Train ConvE with PyTorch.')
    parser.add_argument('train_path', action='store', type=str)
    parser.add_argument('valid_path', action='store', type=str)
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=256)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=90)

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
    criterion = StableBCELoss()
    optimizer = optim.Adam(conv_e.parameters(), lr=0.003)

    for epoch in range(args.epochs):
        train(epoch, train_data, conv_e, criterion, optimizer, args.batch_size)
        valid(train_data, conv_e, args.batch_size)
        valid(valid_data, conv_e, args.batch_size)

        with open('checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(conv_e, f)


if __name__ == '__main__':
    main()
