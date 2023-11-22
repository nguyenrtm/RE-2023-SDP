from one_hot import OneHotEncoder
from process_pkl import load_pkl
from typing import Union, Iterable
import spacy
import torch

class WordEmbedding:
    def __init__(self, path='cache/w2v/w2v.pkl'):
        self.word_dct = load_pkl(path)

    def get_word_vector(self, word):
        if word not in self.word_dct.keys():
            return torch.zeros((1, 200))
        return torch.tensor(self.word_dct[word])
 

class EdgeEmbedding:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_lg")
        self.labels = list(self.nlp.get_pipe("parser").labels)
        self.one_hot_encoder = OneHotEncoder([self.labels])

    def one_hot(self, keys: Union[str, Iterable]):
        return self.one_hot_encoder.one_hot(keys)
    
    def get_direction(self, direction):
        assert direction in ['forward', 'reverse'], "direction should be forward or reverse"
        if direction == 'forward':
            return torch.tensor([1])
        elif direction == 'reverse':
            return torch.tensor([-1])
        
    def edge_to_tensor(self, edge):
        direction, dep = edge[1][0], edge[1][1]
        return torch.cat((self.get_direction(direction), self.one_hot(dep)))
    
    def path_with_dep_to_tensor(self, path_with_dep):
        to_return = torch.empty((0, 49))
        
        for edge in path_with_dep:
            tensor_tmp = self.edge_to_tensor(edge)
            to_return = torch.vstack((to_return, tensor_tmp))
        
        return to_return