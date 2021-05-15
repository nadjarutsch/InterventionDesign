import numpy as np 
from copy import copy, deepcopy
from random import shuffle
import importlib
import random
import string
import math
import sys
sys.path.append("../")

from causal_graphs.graph_utils import *
from causal_graphs.graph_visualization import *
from causal_graphs.variable_distributions import *
from causal_graphs.graph_definition import *


def graph_from_adjmatrix(variable_names, dist_func, adj_matrix):
    variables = []
    for v_idx, name in enumerate(variable_names):
        parents = np.where(adj_matrix[:,v_idx])[0]
        prob_dist = dist_func(input_names=[variable_names[p] for p in parents], name=name)
        var = CausalVariable(name=name, prob_dist=prob_dist)
        variables.append(var)

    graph = CausalDAG(variables, adj_matrix=adj_matrix)
    return graph


def graph_from_edges(variable_names, dist_func, edges):
    adj_matrix = edges_to_adj_matrix(edges, len(variable_names))
    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_random_graph(variable_names, dist_func, edge_prob, connected=False, max_parents=-1, **kwargs):
    shuffle(variable_names) # To have a random order
    num_vars = len(variable_names)

    adj_matrix = np.random.binomial(n=1, p=edge_prob, size=(num_vars, num_vars))

    # Make sure that adjacency matrix is half diagonal
    for v_idx in range(num_vars):
        adj_matrix[v_idx,:v_idx+1] = 0

    # Nodes that do not have any parents are connected
    for v_idx in range(num_vars):
        has_connection = (adj_matrix[v_idx,:].any() or adj_matrix[:,v_idx].any())
        if not has_connection:
            con_idx = np.random.randint(num_vars-1)
            if con_idx >= v_idx:
                con_idx += 1
                adj_matrix[v_idx,con_idx] = True
            else:
                adj_matrix[con_idx,v_idx] = True

    # Ensure that a node has less than N parents
    if max_parents > 0:
        for v_idx in range(adj_matrix.shape[0]):
            num_parents = adj_matrix[:,v_idx].sum()
            if num_parents > max_parents:
                indices = np.where(adj_matrix[:,v_idx] == 1)[0]
                indices = indices[np.random.permutation(indices.shape[0])[:num_parents-max_parents]]
                adj_matrix[indices,v_idx] = 0

    # Connect nodes to one fully connected graph
    if connected:
        visited_nodes, connected_nodes = [], [0]
        while len(visited_nodes) < num_vars:
            while len(connected_nodes) > 0:
                v_idx = connected_nodes.pop(0)
                children = np.where(adj_matrix[v_idx,:])[0].tolist()
                parents = np.where(adj_matrix[:,v_idx])[0].tolist()
                neighbours = children + parents 
                for n in neighbours:
                    if (n not in visited_nodes) and (n not in connected_nodes):
                        connected_nodes.append(n)
                if v_idx not in visited_nodes:
                    visited_nodes.append(v_idx)
            if len(visited_nodes) < num_vars:
                node1 = np.random.choice(np.array(visited_nodes))
                node2 = np.random.choice(np.array([i for i in range(num_vars) if i not in visited_nodes]))
                adj_matrix[min(node1, node2), max(node1, node2)] = True
                connected_nodes.append(node1)

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_chain(variable_names, dist_func, **kwargs):
    shuffle(variable_names) # To have a random order
    num_vars = len(variable_names)
    
    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    for v_idx in range(num_vars-1):
        adj_matrix[v_idx,v_idx+1] = True

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_bidiag(variable_names, dist_func, **kwargs):
    shuffle(variable_names)
    num_vars = len(variable_names)
    
    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    for v_idx in range(num_vars-1):
        adj_matrix[v_idx,v_idx+1] = True
        if v_idx < num_vars - 2:
            adj_matrix[v_idx,v_idx+2] = True

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_collider(variable_names, dist_func, **kwargs):
    shuffle(variable_names)
    num_vars = len(variable_names)
    
    adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
    adj_matrix[:-1,-1] = True 

    return graph_from_adjmatrix(variable_names, dist_func, adj_matrix)


def generate_jungle(variable_names, dist_func, num_levels=2, **kwargs):
    shuffle(variable_names)
    num_vars = len(variable_names)
    
    edges = []
    for i in range(num_vars):
        level = int(np.log2(i+1))
        idx = i + 1 - 2 ** level
        for l in range(1,num_levels+1):
            gl = (2**l) * idx + 2 ** (level + l) - 1
            edges += [[i,gl + j] for j in range(2**l)]
    edges = [e for e in edges if max(e)<num_vars]

    return graph_from_edges(variable_names, dist_func, edges)


def generate_full(variable_names, dist_func, **kwargs):
    return generate_random_graph(variable_names, dist_func, edge_prob=1.0)


def generate_regular_graph(variable_names, dist_func, num_neigh=10, **kwargs):
    shuffle(variable_names)
    num_vars = len(variable_names)
    num_neigh = min(num_neigh, num_vars-1)
    graphs = nx.random_graphs.random_regular_graph(num_neigh, num_vars)
    edges = np.array(graphs.edges())
    edges.sort(axis=-1)

    return graph_from_edges(variable_names, dist_func, edges)



def get_graph_func(name):
    if name == "chain":
        f = generate_chain
    elif name == "bidiag":
        f = generate_bidiag 
    elif name == "collider":
        f = generate_collider 
    elif name == "jungle":
        f = generate_jungle
    elif name == "full":
        f = generate_full
    elif name == "regular":
        f = generate_regular_graph
    elif name == "random":
        f = generate_random_graph
    elif name.startswith("random_max_"):
        max_parents = int(name.split("_")[-1])
        f = lambda *args, **kwargs: generate_random_graph(*args, max_parents=max_parents, **kwargs)
    else:
        f = generate_random_graph
    return f


def generate_categorical_graph(num_vars, 
                               min_categs, 
                               max_categs, 
                               inputs_independent=True, 
                               use_nn=False,
                               graph_func=generate_random_graph,
                               seed=-1,
                               **kwargs):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)

    if num_vars <= 26:
        variable_names = [n for i, n in zip(range(1, num_vars+1), string.ascii_uppercase)]
    else:
        variable_names = [r"$X_{%s}$" % i for i in range(1, num_vars+1)]
    var_num_categs = np.random.randint(min_categs, max_categs+1, size=(num_vars,))

    def dist_func(input_names, name):
        dist = get_random_categorical(input_names=input_names, 
                                      input_num_categs=[var_num_categs[variable_names.index(v_name)] for v_name in input_names],
                                      num_categs=var_num_categs[variable_names.index(name)],
                                      inputs_independent=inputs_independent,
                                      use_nn=use_nn)
        return dist

    return graph_func(variable_names, dist_func, **kwargs)


def generate_continuous_graph(num_vars, 
                              num_coeff=4,
                              graph_func=generate_random_graph,
                              seed=-1,
                              **kwargs):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
    
    if num_vars <= 26:
        variable_names = [n for i, n in zip(range(1, num_vars+1), string.ascii_uppercase)]
    else:
        variable_names = [r"$X_{%s}$" % i for i in range(1, num_vars+1)]

    def dist_func(input_names, name):
        dist = get_random_gaussian(input_names=input_names, num_coeff=num_coeff)
        return dist

    return graph_func(variable_names, dist_func, **kwargs)
