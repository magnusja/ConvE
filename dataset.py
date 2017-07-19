from itertools import repeat

import torch
from torch.utils.data import Dataset


class KnowledgeGraphDataset(Dataset):
    def __init__(self, x, y, e_to_index, r_to_index):
        self.x = x
        self.y = y
        self.e_to_index = e_to_index
        self.r_to_index = r_to_index

        assert len(x) == len(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        s, r = self.x[item]
        os = self.y[item]
        indices = [self.e_to_index[o] for o in os]
        return self.e_to_index[s], self.r_to_index[r], indices


def collate_data(batch):
    max_len = max(map(lambda x: len(x[2]), batch))

    # each object index list must have same length (to use torch.scatter_), therefore we pad with the first index
    for _, _, indices in batch:
        indices.extend(repeat(indices[0], max_len - len(indices)))

    s, o, i = zip(*batch)
    return torch.LongTensor(s), torch.LongTensor(o), torch.LongTensor(i)


def collate_valid(batch):
    s, o, i = zip(*batch)
    return torch.LongTensor(s), torch.LongTensor(o), list(i)
