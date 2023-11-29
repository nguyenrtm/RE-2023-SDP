import networkx as nx

class DependencyParser:
    def __init__(self, nlp):
        self.nlp = nlp

    def get_dependency_graph(self, text):
        doc = self.nlp(text)
        edges = list()
        for token in doc:
            for child in token.children:
                edges.append(((token.lower_, token.i),
                              (child.lower_, child.i)))
        print(edges)
        graph = nx.Graph(edges)

        return graph

    def get_sdp(self,
                text,
                source_text,
                source_i,
                target_text,
                target_i):

        graph = self.get_dependency_graph(text)
        source = (source_text, source_i)
        target = (target_text, target_i)

        return nx.shortest_path(graph, source=source, target=target)

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
                         source_text,
                         source_i,
                         target_text,
                         target_i):
        path_with_dep = list()

        path = self.get_sdp(text,
                            source_text,
                            source_i,
                            target_text,
                            target_i)

        for i in range(len(path) - 1):
            edge_dep_tmp = self.get_edge_dep(text, path[i][1], path[i+1][1])
            path_with_dep.append((path[i], edge_dep_tmp, path[i+1]))

        return path_with_dep