from spacy_features import SpacyFeatures
from one_hot import OneHotEncoder
import torch
import torch.nn.functional as F


class SentenceFeatureBuilder:
    def __init__(self, 
                 word_embedding_instance, 
                 padding_size: int = 200,
                 crop_in_between: int = 0):
        self.preprocesser = SpacyFeatures()
        self.we = word_embedding_instance 
        self.padding_size = padding_size
        self.crop_in_between = crop_in_between
        self.pos_labels = self.preprocesser.nlp.get_pipe("tagger").labels
        self.pos_ohe = OneHotEncoder([self.pos_labels])

    def build_word_embedding(self, row):
        token_lst = self.get_tokens(row['text'])
        for tok in token_lst:
            try:
                word_embedding_tok = self.we.get_word_vector(tok[0].lower())
            except:
                word_embedding_tok = torch.zeros((1, 200))
            if tok[2] == 0:
                word_embedding = word_embedding_tok
            else:
                word_embedding = torch.vstack((word_embedding, word_embedding_tok))
        
        return word_embedding
        
    def get_position_embedding_given_ent(self, 
                                         ent_start: int, 
                                         ent_end: int, 
                                         text_length: int):
        '''
        Given entity index, get position embedding of sentence
        '''
        lst = []
        count_bef = ent_start
        count_in = ent_end - ent_start
        count_aft = text_length - ent_end

        for i in range(count_bef, 0, -1):
            lst.append(-i)
        
        for i in range(count_in + 1):
            lst.append(0)

        for i in range(1, count_aft + 1):
            lst.append(i)
        return lst

    def build_position_embedding(self, row):
        token_lst = self.get_tokens(row['text'])
        ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx = self.return_idx(row)
        text_length = token_lst[-1][2]
        pos_ent1 = torch.tensor(self.get_position_embedding_given_ent(ent1_start_idx, ent1_end_idx, text_length)).view(-1, 1)
        pos_ent2 = torch.tensor(self.get_position_embedding_given_ent(ent2_start_idx, ent2_end_idx, text_length)).view(-1, 1)
        pos_ent1 = pos_ent1 / (len(token_lst) / 2.)
        pos_ent2 = pos_ent2 / (len(token_lst) / 2.)
        zero_ent1 = torch.zeros((len(token_lst), 1))
        zero_ent1[ent1_start_idx] = 1.
        zero_ent1[ent1_end_idx] = 1.
        zero_ent2 = torch.zeros((len(token_lst), 1))
        zero_ent2[ent2_start_idx] = 1.
        zero_ent2[ent2_end_idx] = 1.
        to_return = torch.hstack((pos_ent1, pos_ent2, zero_ent1, zero_ent2))
        return to_return
    
    def get_tokens(self, text: str):
        return self.preprocesser.wordTokenizer(text)

    def return_idx(self, row):
        token_lst = self.get_tokens(row['text'])
        ent1_start_idx = self.find_idx_start_given_offset(token_lst, offset=row['ent1_start'])
        ent1_end_idx = self.find_idx_end_given_offset(token_lst, offset=row['ent1_end'])
        ent2_start_idx = self.find_idx_start_given_offset(token_lst, offset=row['ent2_start'])
        ent2_end_idx = self.find_idx_end_given_offset(token_lst, offset=row['ent2_end'])
        
        return ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx
                
    def find_idx_start_given_offset(self,
                                    token_lst, 
                                    offset):
        for i in range(len(token_lst) - 1):
            if offset >= token_lst[i][1] and offset < token_lst[i+1][1]:
                return token_lst[i][2]
        
        return len(token_lst) - 1
            
    def find_idx_end_given_offset(self, 
                                  token_lst, 
                                  offset):
        for i in range(len(token_lst) - 1):
            if offset >= token_lst[i][1] and offset - 1 < token_lst[i+1][1]:
                return token_lst[i][2]
            
        return len(token_lst) - 1
    
    def build_pos_embedding(self, text: str):
        pos_lst = self.preprocesser.tag(text)
        for i in range(len(pos_lst)):
            if pos_lst[i] not in self.pos_labels:
                pos_ohe = torch.zeros((len(self.pos_labels)))
            else:
                pos_ohe = self.pos_ohe.one_hot(pos_lst[i])
            if i == 0:
                pos_embedding = pos_ohe
            else:
                pos_embedding = torch.vstack((pos_embedding, pos_ohe))
        return pos_embedding
    
    def padding(self, embedding):
        return F.pad(embedding, (0, 0, 0, self.padding_size - embedding.shape[0]), mode="constant")

    def build_embedding(self, 
                        row):
        pos = self.build_pos_embedding(row['text'])
        position = self.build_position_embedding(row)
        word = self.build_word_embedding(row)
        embedding = torch.hstack((word, pos, position))
        if self.crop_in_between > -1:
            ent1_start_idx = self.find_idx_start_given_offset(row['text'], offset=row['ent1_start'])
            ent2_end_idx = self.find_idx_end_given_offset(row['text'], offset=row['ent2_end'])

            if ent1_start_idx == ent2_end_idx:
                print(f"Warning: 2 entities in the same tokens in sentence {row['sent_id']}")
            if ent1_start_idx < ent2_end_idx:
                embedding = embedding[max(0, ent1_start_idx - self.crop_in_between):min(ent2_end_idx + 1 + self.crop_in_between, len(embedding) - 1)]
            elif ent1_start_idx > ent2_end_idx:
                embedding = embedding[max(0, ent2_end_idx - self.crop_in_between):min(ent1_start_idx + 1 + self.crop_in_between, len(embedding) - 1)]
                
        if self.padding_size > -1:
            embedding = self.padding(embedding)
        return embedding
    
    def build_embedding_for_df(self, df):
        from tqdm import tqdm
        for i in tqdm(range(len(df))):
            embedding = self.build_embedding(df.iloc[i])
            if i == 0:
                embeddings = embedding.unsqueeze(dim=0)
            else:
                embeddings = torch.vstack((embeddings, embedding.unsqueeze(dim=0)))
        
        return embeddings
    
    def build_label_for_df(self, df):
        for i in range(len(df)):
            label = torch.tensor([df.iloc[i]['label']]).type(torch.FloatTensor)
            if i == 0:
                labels = label
            else:
                labels = torch.vstack((labels, label))
        
        return labels