from math import inf
import numpy as np

class Similarity:
    def __init__(self, graph, sim_type, file):
        self.type = sim_type
        self.graph = graph
        self.file = file
        self.emb = None
        self.min = inf
        self.max = -inf
        self.similarity = {}

    def get_similarity_range(self):
        return self.min, self.max

    def get_similarity(self):
        return self.similarity

    def init_edge_similarity(self):
        self.emb = self.file.get_data()
        for edge in self.graph.edges():
            v1, v2 = self.normalize(edge[0]), self.normalize(edge[1])
            sim = self.calculate_similarity(v1, v2)
            self.min = min(self.min, sim)
            self.max = max(self.max, sim)
            edgeTuple = tuple(sorted(edge))
            self.similarity[edgeTuple] = sim

    def normalize(self, node):
        v = self.emb[node]
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def calculate_similarity(self, v1, v2):
        sim = 0
        if self.type == 'cos':
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif self.type == 'rbf':
            distance = np.linalg.norm(v1 - v2)
            sim = np.exp((-1.0 / 2.0) * np.power(distance, 2.0))
        return sim

