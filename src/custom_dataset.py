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
            else:
                i += 1

    def batch_padding(self, batch_size):
        current = 0
        to_return = []

        while current + batch_size < len(self.data):
            batch = self.data[current:current+batch_size]
            max_len_in_batch = max([len(x) for x in batch])
            
            for i in range(len(batch)):
                for _ in range(max_len_in_batch - len(batch[i])):
                    batch[i].append([('', ' ', torch.tensor([0., 0., 0., 0.])), 
                                     ('', ' '),
                                     ('', ' ', torch.tensor([0., 0., 0., 0.]))])
                
            current += batch_size

        batch = self.data[current:]
        max_len_in_batch = max([len(x) for x in batch])
        
        for i in range(len(batch)):
            for _ in range(max_len_in_batch - len(batch[i])):
                batch[i].append([(' ', 'PAD', torch.tensor([0., 0., 0., 0.])), 
                                    (' ', ' '),
                                    (' ', 'PAD', torch.tensor([0., 0., 0., 0.]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label