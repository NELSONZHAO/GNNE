# GNNE: Graph Neural Network Embeddings

GNNE: Abbreviation of Graph Neural Network Embeddings.

This repo mainly implements the mainstream Graph Embedding algorithm, which can quickly and efficiently help researchers conduct model experiments.

# Installation

```pip install gnne```

# Example

```
import networkx as nx
from gnne.models.node2vec import Node2Vec

# Load graph dataset
with open("datasets/example_data.txt", "r") as f:
    data = f.read()
    data = [[int(v) for v in line.split()] + [1] for line in data.split("\n")]

# Initialize graph object
graph = nx.Graph()
graph.add_weighted_edges_from(data)

# Using Node2Vec
node2vec = Node2Vec(graph, r=5, p=1, q=0.1, walk_len=10, embed_dim=2)
node2vec.train(batch_size=64, epochs=300)
```