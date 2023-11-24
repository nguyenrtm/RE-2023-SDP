import networkx as nx
from tqdm import tqdm

class DependencyParser:
    def __init__(self, nlp):
        self.nlp = nlp

    def get_dependency_graph(self, text):
        doc = self.nlp(text)
        edges = list()
        for token in doc:
            for child in token.children:
                edges.append((token.i, child.i))
                
        graph = nx.Graph(edges)

        return graph
    
    def get_sdp(self,
                text,
                source_i,
                target_i):
        
        graph = self.get_dependency_graph(text)
        source = source_i
        target = target_i

        return nx.shortest_path(graph, source=source_i, target=target_i)
    
    def get_edge_dep(self, text, source_i, target_i):
        doc = self.nlp(text)
        direction = None

        for token in doc:
            if token.i == source_i:
                if token.head.i == target_i:
                    direction = 'reverse'
                    return direction, token.dep_
                else:
                    for child in token.children:
                        if child.i == target_i:
                            direction = 'forward'
                            return direction, child.dep_
        
        return None
    
    def get_sdp_with_dep(self,
                         text,
                         source_i,
                         target_i):
        path_with_dep = list()
        
        path = self.get_sdp(text,
                            source_i,
                            target_i)
        
        for i in range(len(path) - 1):
            edge_dep_tmp = self.get_edge_dep(text, path[i], path[i+1])
            path_with_dep.append((path[i], edge_dep_tmp, path[i+1]))
        
        return path_with_dep
    
    def get_sdp_with_dep_df(self, df, sfb):
        sdp_with_dep = list()
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            tmp = self.get_sdp_with_dep(text=row['text'],
                                        source_i=sfb.return_idx(row)[0],
                                        target_i=sfb.return_idx(row)[3])

            sdp_with_dep.append(tmp)

        df['sdp'] = sdp_with_dep
        return df