import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def rm_none(self):
        i = 0
        while i < len(self.data):
            if self.data[i] == None:
                self.data.pop(i)
                self.labels = torch.vstack((self.labels[:i], self.labels[i+1:]))
            i += 1

    def batch_padding(self, batch_size):
        current = 0
        to_return = []

        while current + batch_size < len(self.data):
            batch = self.data[current:current+batch_size]

            max_len_in_batch = max([x.shape[0] for x in batch])

            for i in range(len(batch)):
                tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
                to_return.append(tmp)

            current += batch_size

        batch = self.data[current:]
        max_len_in_batch = max([x.shape[0] for x in batch])

        for i in range(len(batch)):
            tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
            to_return.append(tmp)

        self.data = to_return

    def __len__(self):
        return len(self.data)