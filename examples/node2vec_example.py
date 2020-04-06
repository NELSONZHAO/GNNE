# coding: utf-8
import networkx as nx
from models.node2vec import Node2Vec

with open("../datasets/example_data.txt", "r") as f:
    data = f.read()
    data = [[int(v) for v in line.split()] + [1] for line in data.split("\n")]

graph = nx.Graph()
graph.add_weighted_edges_from(data)

node2vec = Node2Vec(graph, r=5, p=1, q=0.1, walk_len=10, embed_dim=2)
node2vec.train(batch_size=64, epochs=300)
print(node2vec.get_node_embeddings(0))
print(node2vec.get_most_similar_node(0))
print(node2vec.get_embeddings(g=False))
node2vec.plot_node_embeddings()
