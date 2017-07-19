
import sys
import os
import numpy as np
import csv
import pickle
import argparse

# s = subject
# r = relation
# o = object
from util import AttributeDict


def read_data(file_path):
    s_dict = dict()
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for s, r, o in csv_reader:
            try:
                s_dict[s][r].append(o)
            except KeyError:
                s_dict[s] = dict()
                s_dict[s][r] = [o]

    return s_dict


def create_dataset(s_dict):
    x, y = list(), list()
    e_to_index, index_to_e, r_to_index, index_to_r = dict(), dict(), dict(), dict()
    for s, ro in s_dict.items():
        try:
            _ = e_to_index[s]
        except KeyError:
            index = len(e_to_index)
            e_to_index[s] = index
            index_to_e[index] = s

        for r, os in ro.items():
            try:
                _ = r_to_index[r]
            except KeyError:
                index = len(r_to_index)
                r_to_index[r] = index
                index_to_r[index] = r

            for o in os:
                # sometimes an entity only occurs as an object
                try:
                    _ = e_to_index[o]
                except KeyError:
                    index = len(e_to_index)
                    e_to_index[o] = index
                    index_to_e[index] = o

            x.append((s, r))
            y.append(os)

    return x, y, e_to_index, index_to_e, r_to_index, index_to_r


def preprocess_train(file_path):
    s_dict = read_data(file_path)
    x, y, e_to_index, index_to_e, r_to_index, index_to_r = create_dataset(s_dict)

    data = {
        'x': x,
        'y': y,
        'e_to_index': e_to_index,
        'index_to_e': index_to_e,
        'r_to_index': r_to_index,
        'index_to_r': index_to_r
    }

    print('#entities: ', len(e_to_index))
    print('#relations: ', len(r_to_index))

    for i in range(np.minimum(len(x), 200)):
        print(x[i], y[i])
        choice = np.random.choice(len(e_to_index))
        assert choice == e_to_index[index_to_e[choice]]
        choice = np.random.choice(len(r_to_index))
        assert choice == r_to_index[index_to_r[choice]]

    save_file_path = os.path.splitext(file_path)[0] + '.pkl'
    pickle.dump(data, open(save_file_path, 'wb'))


def preprocess_valid(train_path, valid_path):
    x, y = list(), list()
    with open(train_path, 'rb') as f:
        train_data = AttributeDict(pickle.load(f))

    s_dict = read_data(valid_path)
    for s, ro in s_dict.items():
        try:
            _ = train_data.e_to_index[s]
        except KeyError:
            continue

        for r, objects in ro.items():
            try:
                _ = train_data.r_to_index[r]
            except KeyError:
                continue

            filtered_objects = list()

            for o in objects:
                # sometimes an entity only occurs as an object
                try:
                    _ = train_data.e_to_index[o]
                    filtered_objects.append(o)
                except KeyError:
                    continue

            x.append((s, r))
            y.append(filtered_objects)

    data = {
        'x': x,
        'y': y,
    }

    save_file_path = os.path.splitext(valid_path)[0] + '.pkl'
    pickle.dump(data, open(save_file_path, 'wb'))


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess knowledge graph csv train/valid (test) data.')
    sub_parsers = parser.add_subparsers(help='mode', dest='mode')
    sub_parsers.required = True
    train_parser = sub_parsers.add_parser('train', help='Preprocess a training set')
    valid_parser = sub_parsers.add_parser('valid', help='Preprocess a valid or test set')
    train_parser.add_argument('train_path', action='store', type=str)

    valid_parser.add_argument('train_path', action='store', type=str)
    valid_parser.add_argument('valid_path', action='store', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == 'train':
        preprocess_train(args.train_path)
    else:
        preprocess_valid(args.train_path, args.valid_path)


if __name__ == '__main__':
    main()
