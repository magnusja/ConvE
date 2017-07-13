
import sys
import os
import numpy as np
import csv
import pickle

# s = subject
# r = relation
# o = object


def read_data(file_path):
    s_dict = dict()
    r_dict = dict()
    s_test_dict = dict()
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for s, r, o in csv_reader:
            try:
                s_dict[s].append((r, o))
                s_test_dict[s][(r, o)] = True
            except KeyError:
                s_dict[s] = [(r, o)]
                s_test_dict[s] = {(r, o): True}

            try:
                r_dict[r].append(o)
            except KeyError:
                r_dict[r] = [o]

    return s_dict, r_dict, s_test_dict


def find_negative_sample(s, r, o, s_test_dict, r_dict):
    # subject and relation stay, find a negative object
    potential_os = r_dict[r]
    while True:
        choice = np.random.choice(len(potential_os))
        potential_o = potential_os[choice]

        try:
            s_test_dict[s][(r, potential_o)]
        except KeyError:
            break

    return s, r, potential_o


def pre_process(s_dict, r_dict, s_test_dict):
    x, y = list(), list()
    e_to_index, index_to_e, r_to_index, index_to_r = dict(), dict(), dict(), dict()
    for s, ro in s_dict.items():
        try:
            _ = e_to_index[s]
        except KeyError:
            index = len(e_to_index)
            e_to_index[s] = index
            index_to_e[index] = s

        for r, o in ro:
            try:
                _ = r_to_index[r]
            except KeyError:
                index = len(r_to_index)
                r_to_index[r] = index
                index_to_r[index] = r
            try:
                _ = e_to_index[o]
            except KeyError:
                index = len(e_to_index)
                e_to_index[o] = index
                index_to_e[index] = o

            # add positive sample
            x.append((s, r, o))
            y.append(1)
            # add negative sample
            x.append(find_negative_sample(s, r, o, s_test_dict, r_dict))
            y.append(0)

    return x, y, e_to_index, index_to_e, r_to_index, index_to_r


def main():
    file_path = sys.argv[1]
    s_dict, r_dict, s_test_dict = read_data(file_path)
    x, y, e_to_index, index_to_e, r_to_index, index_to_r = pre_process(s_dict, r_dict, s_test_dict)

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

    for i in range(50):
        print(x[i], y[i])
        choice = np.random.choice(len(e_to_index))
        assert choice == e_to_index[index_to_e[choice]]

    save_file_path = os.path.splitext(file_path)[0] + '.pkl'
    pickle.dump(data, open(save_file_path, 'wb'))

if __name__ == '__main__':
    main()
