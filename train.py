import os
import pickle
import argparse

import logging
import torch.nn as nn
import torch

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import KnowledgeGraphDataset, collate_train, collate_valid
from model import ConvE
from util import AttributeDict

logger = logging.getLogger(__file__)


class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def train(epoch, data, conv_e, criterion, optimizer, args):
    train_set = DataLoader(
        KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index),
        collate_fn=collate_train, batch_size=args.batch_size, num_workers=4, shuffle=True)

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0

    conv_e.train(True)
    y_multihot = torch.LongTensor(args.batch_size, len(data.e_to_index))
    for s, r, os in progress_bar:
        s, r = Variable(s).cuda(), Variable(r).cuda()

        if s.size()[0] != args.batch_size:
            y_multihot = torch.LongTensor(s.size()[0], len(data.e_to_index))

        y_multihot.zero_()
        y_multihot = y_multihot.scatter_(1, os, 1)
        y_smooth = (1 - args.label_smooth) * y_multihot.float() + args.label_smooth / len(data.e_to_index)

        targets = Variable(y_smooth, requires_grad=False).cuda()

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

    logger.info('Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.data[0], moving_loss))


def valid(data, conv_e, batch_size):
    dataset = KnowledgeGraphDataset(data.x, data.y, e_to_index=data.e_to_index, r_to_index=data.r_to_index)
    valid_set = DataLoader(dataset, collate_fn=collate_valid, batch_size=batch_size, num_workers=4, shuffle=True)

    conv_e.train(False)
    ranks = list()
    for s, r, os in tqdm(iter(valid_set)):
        s, r = Variable(s).cuda(), Variable(r).cuda()
        output = conv_e.test(s, r)

        for i in range(min(batch_size, s.size()[0])):
            _, top_indices = output[i].topk(output.size()[1])
            for o in os[i]:
                _, rank = (top_indices == o).max(dim=0)
                ranks.append(rank.data[0] + 1)

    ranks_t = torch.FloatTensor(ranks)
    mr = ranks_t.mean()
    mrr = (1 / ranks_t).mean()

    logger.info('MR: {:.3f}, MRR: {:.10f}'.format(mr, mrr))


def parse_args():
    parser = argparse.ArgumentParser(description='Train ConvE with PyTorch.')
    parser.add_argument('train_path', action='store', type=str,
                        help='Path to training .pkl produced by preprocess.py')
    parser.add_argument('valid_path', action='store', type=str,
                        help='Path to valid/test .pkl produced by preprocess.py')
    parser.add_argument('--name', action='store', type=str, default='',
                        help='name of the model, used to create a subfolder to save checkpoints')
    parser.add_argument('--batch-size', action='store', type=int, dest='batch_size', default=256)
    parser.add_argument('--epochs', action='store', type=int, dest='epochs', default=90)
    parser.add_argument('--label-smooth', action='store', type=float, dest='label_smooth', default=.1)
    parser.add_argument('--log-file', action='store', type=str)

    return parser.parse_args()


def setup_logger(args):
    log_file = args.log_file
    if args.log_file is None:
        if args.name == '':
            log_file = 'train.log'
        else:
            log_file = args.name + '.log'

    print('Logging to: ' + log_file)

    logging.basicConfig(filename=log_file, level=logging.INFO)


def main():

    args = parse_args()
    setup_logger(args)

    checkpoint_path = 'checkpoint-{}'.format(args.name)
    os.makedirs(checkpoint_path, exist_ok=True)
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

    for epoch in trange(args.epochs):
        train(epoch, train_data, conv_e, criterion, optimizer, args)
        valid(train_data, conv_e, args.batch_size)
        valid(valid_data, conv_e, args.batch_size)

        with open('{}/checkpoint_{}.model'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(conv_e, f)


if __name__ == '__main__':
    main()
