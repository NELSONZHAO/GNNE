# coding: utf-8
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class DeepWalk(object):
    def __init__(self, graph: nx.Graph, r=5, walk_len=10, embed_dim=8, workers=4):
        self.graph = graph
        self.model = None
        self.params = dict()
        self.params['round'] = r
        self.params['walk_len'] = walk_len
        self.params['embed_dim'] = embed_dim
        self.params['workers'] = workers

    def _random_walk(self, source_node):
        walk = [source_node]
        while len(walk) < self.params['walk_len']:
            v = walk[-1]

            nbrs = list(self.graph.neighbors(v))
            if len(nbrs) > 0:
                nbr = np.random.choice(nbrs)
                walk.append(nbr)
            else:
                break

        return walk

    @staticmethod
    def _make_nodes_generator(r, nodes):
        for _ in range(r):
            for n in nodes:
                yield n

    def deep_walk(self):
        workers = self.params['workers']
        r = self.params['round']
        nodes = self.graph.nodes

        g = self._make_nodes_generator(r, nodes)
        walks = []

        with ThreadPoolExecutor(workers) as executor:
            for walk in executor.map(self._random_walk, g):
                walks.append(walk)

        return walks

    def train(self, batch_size=64, epochs=100):
        walks = self.deep_walk()
        walks = [[str(w) for w in walk] for walk in walks]

        model = Word2Vec(
            sentences=walks,
            size=self.params['embed_dim'],
            window=5,
            sg=1,
            hs=1,
            batch_words=batch_size,
            iter=epochs,
            workers=8
        )

        self.model = model.wv
        return model.wv

    def get_embeddings(self, g=True):
        def embedding():
            for node in self.graph.nodes:
                yield (node, self.get_node_embeddings(node))
        if not g:
            return [(v, self.get_node_embeddings(v)) for v in self.graph.nodes]
        else:
            return embedding()

    def get_node_embeddings(self, v):
        if not self.model:
            raise ValueError("Not train model")

        return self.model.get_vector(str(v))

    def get_most_similar_node(self, v, topn=10):
        return self.model.most_similar(str(v), topn=topn)

    def plot_node_embeddings(self):
        annotate = []
        x = []
        y = []
        for n, e in self.get_embeddings():
            annotate.append(n)
            x.append(e[0])
            y.append(e[1])
        plt.scatter(x, y)
        for i in range(len(annotate)):
            plt.annotate(annotate[i], xy=(x[i], y[i]), xytext=(x[i] + 0.01, y[i] + 0.01))
        plt.show()