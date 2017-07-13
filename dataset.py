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
        s, r, o = self.x[item]
        return (self.e_to_index[s], self.r_to_index[r], self.e_to_index[o]), self.y[item]
