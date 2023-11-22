import pandas as pd

class IntraSentenceDataCreator:
    def __init__(self):
        pass

    def get_data(self, abstract, entity, relation, preprocesser):
        from tqdm import tqdm
    
        abstract_out = []
        entity_out = []
        relation_out = []
        pair_out = []
        sentences_out = []
        
        for index in tqdm(abstract):
            
            sent_abstract_split = []
            sent_abstract = []
            sent_entity = []
            sent_relation = []
            sent_pair = []

            text = abstract[index]['t'] + " " + abstract[index]['a']

            # Add abstract of splitted sentences
            sent_abstract_split = preprocesser.sentTokenizer(abstract[index]['a'])
            sent_abstract_split.insert(0, abstract[index]['t'])
            sentences_out.append(sent_abstract_split)

            for i in range(len(sent_abstract_split)):
                tmp = (index + "_" + str(i), sent_abstract_split[i])
                sent_abstract.append(tmp)
                
            # Add entities of splitted sentences
            sent_length = []
            length_counter = 0
            sent_pos = 0

            for i in sent_abstract:
                if sent_pos == 0:
                    length_counter += len(i[1])
                    sent_length.append(length_counter)
                    sent_pos += 1
                else:
                    length_counter += len(i[1]) + 1
                    sent_length.append(length_counter)
                    sent_pos += 1
                    
            for i in entity[index]:
                sent_pos = 0
                place = i[1]
                while place > sent_length[sent_pos]:
                    sent_pos += 1
                tmp = list(i)
                tmp[0] = index + "_" + str(sent_pos)
                if sent_pos > 0:
                    tmp[1] -= sent_length[sent_pos - 1] + 1
                    tmp[2] -= sent_length[sent_pos - 1] + 1
                i = tuple(tmp)
                sent_entity.append(i)
            
            # Add relations of splitted sentences
            for r in relation[index]:
                for e1 in sent_entity:
                    if e1[5] == r[2]:
                        for e2 in sent_entity:
                            if (e2[5] == r[3]) and (e2[0] == e1[0]):
                                tmp = list(r)
                                tmp[0] = e1[0]
                                tmp = tuple(tmp)
                                sent_relation.append(tmp)

            # Add chemical-disease pairs and labels to document data
            for e1 in sent_entity:
                if e1[4] == 'Chemical':
                    for e2 in sent_entity:
                        if e1[0] == e2[0] and e2[4] == 'Disease':
                            tmp = (e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], 
                                e2[1], e2[2], e2[3], e2[4], e2[5], 0)
                            sent_pair.append(tmp)

            for i in range(len(sent_pair)): 
                for r in sent_relation:
                    if sent_pair[i][5] == r[2] and sent_pair[i][10] == r[3]:
                        sent_pair[i] = list(sent_pair[i])
                        sent_pair[i][11] = 1
                        sent_pair[i] = tuple(sent_pair[i])
            
            # Add data of document to dataset
            abstract_out.append(sent_abstract)
            entity_out.append(sent_entity)
            relation_out.append(sent_relation)
            pair_out.append(sent_pair)
        
        return abstract_out, entity_out, relation_out, pair_out
    
    def find_sent_given_id(self, intra_abstract, sent_id):
        '''
        Find sentence given sentence ID
        '''
        for text in intra_abstract:
            for sent in text:
                if sent[0] == sent_id:
                    return sent[1]

    def convert_to_df(self, intra_abstract, intra_pair):
        '''
        Given information about intra-sentence entities and relations, 
        return dataframe to display intra-sentence relation labels
        '''

        df = pd.DataFrame(columns=['sent_id', 'text', 'ent1_start', 'ent1_end', 
                                   'ent1_name', 'ent1_type', 'ent1_id', 'ent2_start', 
                                   'ent2_end', 'ent2_name', 'ent2_type', 'ent2_id', 'label'])
        
        for text in intra_pair:
            for sent in text:
                sent_id = sent[0]
                sent_tmp = self.find_sent_given_id(intra_abstract, sent_id)
                row = list(sent)
                row = [row[0]] + [sent_tmp] + row[1:]
                df.loc[len(df.index)] = row
        
        return df