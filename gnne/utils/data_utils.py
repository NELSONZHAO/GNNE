# coding: utf-8
import networkx as nx


def load_graph(fname, g_type: nx.Graph, ):
    nx.read_edgelist(fname, create_using=g_type)

