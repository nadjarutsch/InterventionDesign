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


class CausalVariable(object):

    
    def __init__(self, name, prob_dist):
        super().__init__()
        self.name = name
        self.prob_dist = prob_dist


    def sample(self, inputs, *args, **kwargs):
        return self.prob_dist.sample(inputs, *args, **kwargs)


    def get_prob(self, inputs, output, *args, **kwargs):
        return self.prob_dist.prob(inputs, output, *args, **kwargs)


    def __str__(self):
        return "CausalVariable " + self.name


    def get_state_dict(self):
        state_dict = {"name": self.name,
                      "prob_dist": self.prob_dist.get_state_dict()}
        state_dict["prob_dist"]["class_name"] = str(self.prob_dist.__class__.__name__)
        return state_dict 


    @staticmethod
    def load_from_state_dict(state_dict):
        module = importlib.import_module("causal_graphs.variable_distributions")
        prob_dist_class = getattr(module, state_dict["prob_dist"]["class_name"])
        prob_dist = prob_dist_class.load_from_state_dict(state_dict["prob_dist"])
        obj = CausalVariable(state_dict["name"], prob_dist)
        return obj



class CausalDAG(object):


    def __init__(self, variables, edges=None, adj_matrix=None):
        super().__init__()
        assert len(set([v.name for v in variables])) == len(variables), "Variables need to have unique names to distinguish them."
        edges, adj_matrix = edges_or_adj_matrix(edges, adj_matrix, len(variables))
        
        self.variables = variables
        self.edges = edges
        self.adj_matrix = adj_matrix
        self.node_relations = get_node_relations(self.adj_matrix)
        self.name_to_var = {v.name: v for v in variables}
        self._sort_variables()


    def _sort_variables(self):
        self.variables, self.edges, self.adj_matrix, _ = sort_graph_by_vars(self.variables, self.edges, self.adj_matrix)


    def sample(self, interventions=None, batch_size=1, as_array=False):
        if interventions is None:
            interventions = dict()

        var_vals = []
        for v_idx, var in enumerate(self.variables):
            parents = np.where(self.adj_matrix[:,v_idx])[0]
            parent_vals = {self.variables[i].name: var_vals[i] for i in parents}
            if interventions is None or (var.name not in interventions):
                sample = var.sample(parent_vals, batch_size=batch_size)
            elif isinstance(interventions[var.name], ProbDist):
                sample = interventions[var.name].sample(parent_vals, batch_size=batch_size)
            elif isinstance(var.prob_dist, DiscreteProbDist) and (interventions[var.name] == -1).any():
                sample = var.sample(parent_vals, batch_size=batch_size)
                sample = np.where(interventions[var.name] != -1, interventions[var.name], sample)
            else:
                sample = interventions[var.name]
            var_vals.append(sample)

        if not as_array:
            var_vals = {var.name: var_vals[v_idx] for v_idx, var in enumerate(self.variables)}
        elif not isinstance(var_vals[0], np.ndarray):
            var_vals = np.array(var_vals)
        else:
            var_vals = np.stack(var_vals, axis=1)
        return var_vals


    def get_intervened_graph(self, interventions):
        intervened_graph = deepcopy(self)
        for v_name in interventions:
            v_idx = [idx for idx, v in enumerate(intervened_graph.variables) if v.name == v_name][0]
            if isinstance(interventions[v_name], ProbDist):
                intervened_graph.variables[v_idx].prob_dist = interventions[v_name]
            else:
                intervened_graph.adj_matrix[:,v_idx] = False
                intervened_graph.variables[v_idx].prob_dist = ConstantDist(interventions[v_name])
        intervened_graph.edges = adj_matrix_to_edges(intervened_graph.adj_matrix)
        intervened_graph._sort_variables()
        return intervened_graph


    def __str__(self):
        s  = "CausalDAG with %i variables [%s]" % (len(self.variables), ",".join([v.name for v in self.variables]))
        s += " and %i edges%s\n" % (len(self.edges), ":" if len(self.edges) > 0 else "")
        for v_idx, v in enumerate(self.variables):
            children = np.where(self.adj_matrix[v_idx,:])[0]
            if len(children) > 0:
                s += "%s => %s" % (v.name, ",".join([self.variables[c].name for c in children])) + "\n"
        return s


    @property
    def num_vars(self):
        return len(self.variables)


    def get_state_dict(self):
        state_dict = {"edges": self.edges,
                      "variables": [v.get_state_dict() for v in self.variables]}
        return state_dict

    def save_to_file(self, filename):
        torch.save(self.get_state_dict(), filename)

    @staticmethod
    def load_from_state_dict(state_dict):
        edges = state_dict["edges"]
        variables = [CausalVariable.load_from_state_dict(v_dict) for v_dict in state_dict["variables"]]
        obj = CausalDAG(variables, edges)
        return obj

    @staticmethod
    def load_from_file(filename):
        state_dict = torch.load(filename)
        return CausalDAG.load_from_state_dict(state_dict)