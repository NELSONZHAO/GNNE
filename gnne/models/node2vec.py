# coding: utf-8
import networkx as nx
import json
import logging
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from gensim.models import Word2Vec
from utils.sampling_methods import AliasSampling


class Node2Vec(object):
    def __init__(self, graph: nx.Graph, p=1.0, q=0.5, r=5, walk_len=10, embed_dim=8, workers=4):
        """
        Node2Vec
        https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf
        :param graph: The object of networkx.Graph
        :param p: The return parameter used in model
        :param q: The in-out parameter used in model
        :param r: The iteration of all the nodes in graph
        :param walk_len: The length of a random walk
        :param embed_dim: The embedding dimensions of the node
        :param workers: The maximum number of threads that can be used to execute the given calls
        """
        self.graph = graph
        self.edge_alias_table = {}
        self.node_alias_table = {}
        self.model = None
        self.params = dict()
        self.params['p'] = p
        self.params['q'] = q
        self.params['round'] = r
        self.params['walk_len'] = walk_len
        self.params['embed_dim'] = embed_dim
        self.params['workers'] = workers

    def _get_edge_alias(self, t, v) -> AliasSampling:
        """
        Get alias table for t->v hop
        :param t: previous node
        :param v: current node
        :return: alias table
        """
        p = self.params['p']
        q = self.params['q']

        pi = []
        labels = []
        for nbr in self.graph.neighbors(v):
            if nbr == t:
                pi.append(self.graph.get_edge_data(nbr, v)['weight'] / p)
            elif self.graph.has_edge(t, nbr):
                pi.append(self.graph.get_edge_data(nbr, v)['weight'])
            else:
                pi.append(self.graph.get_edge_data(nbr, v)['weight'] / q)

            labels.append(nbr)

        alias_sampling = AliasSampling()
        alias_sampling.fit(pi, labels, False)

        return alias_sampling

    def _get_node_alias(self, v) -> AliasSampling:
        """
        Get alias table for v
        :param v: current node
        :return: alias table
        """
        pi = []
        labels = []
        for nbr in self.graph.neighbors(v):
            pi.append(self.graph.get_edge_data(v, nbr)['weight'])
            labels.append(nbr)

        alias_sampling = AliasSampling()
        alias_sampling.fit(pi, labels, False)

        return alias_sampling

    def _make_alias_table(self):
        for v in self.graph.nodes:
            self.node_alias_table[v] = self._get_node_alias(v)

        for t, v in self.graph.edges:
            self.edge_alias_table[(t, v)] = self._get_edge_alias(t, v)
            self.edge_alias_table[(v, t)] = self._get_edge_alias(v, t)

    def _random_walk(self, source_node):
        walk = [source_node]
        while len(walk) < self.params['walk_len']:
            v = walk[-1]

            if len(walk) == 1:
                s = self.node_alias_table[v].sample()
            else:
                t = walk[-2]
                s = self.edge_alias_table[(t, v)].sample()

            walk.append(s)

        return walk

    @staticmethod
    def _make_nodes_generator(r, nodes):
        for _ in range(r):
            for n in nodes:
                yield n

    def node2vec_random_walk(self):
        if not (self.node_alias_table and self.edge_alias_table):
            self._make_alias_table()

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
        self._make_alias_table()
        walks = self.node2vec_random_walk()
        walks = [[str(w) for w in walk] for walk in walks]

        model = Word2Vec(
            sentences=walks,
            size=self.params['embed_dim'],
            window=5,
            sg=1,
            hs=0,
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
            raise ValueError("Not train model.")

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

    def load_model(self, model_path):
        with open(model_path, "r") as f:
            model = json.load(f)
        return model

    def save_model(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.model, f)
        logging.info("model saved.")