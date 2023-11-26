import torch
from tqdm import tqdm

class RowToEmbedding:
    def __init__(self, sfb, ee, dp):
        self.sfb = sfb
        self.ee = ee
        self.dp = dp

    def row_to_embedding(self, row):
        path = self.dp.get_sdp_with_dep(text=row['text'],
                                       source_i=self.sfb.return_idx(row)[0],
                                       target_i=self.sfb.return_idx(row)[3])
        
        edge_one_hot = self.ee.path_with_dep_to_tensor(path)
        
        word_embedding_dictionary = self.sfb.build_embedding(row)

        for i in range(len(path)):
            ent1_embedding = word_embedding_dictionary[str(path[i][0])]
            ent2_embedding = word_embedding_dictionary[str(path[i][2])]
            edge_embedding = edge_one_hot[i]
            embedding = torch.cat((ent1_embedding, edge_embedding, ent2_embedding))
            
            if i == 0:
                to_return = embedding.unsqueeze(0)
            else:
                to_return = torch.vstack((to_return, embedding))

        return to_return
    
    def row_to_embedding_df(self, df):
        to_return = list()
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            try: 
                embedding = self.row_to_embedding(row)
                to_return.append(embedding)
            except:
                to_return.append(None)
            
        return to_return