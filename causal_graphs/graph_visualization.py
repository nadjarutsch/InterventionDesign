import networkx as nx  
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")

from causal_graphs.graph_utils import find_nodes_on_paths



def visualize_graph(graph, **kwargs):
	G = nx.DiGraph()
	G.add_nodes_from([v.name for v in graph.variables])
	edges = [[graph.variables[v_idx].name for v_idx in e] for e in graph.edges.tolist()]
	G.add_edges_from(edges)
	graph_to_image(G, **kwargs)


def visualize_local_graph(graph, nodes, **kwargs):
	G = nx.DiGraph()
	nodes_on_paths = find_nodes_on_paths(graph, nodes[0], nodes[1])
	nodes_on_paths = np.where(nodes_on_paths)[0].tolist()
	nodes += nodes_on_paths
	nodes = [graph.variables[n_idx].name if isinstance(n_idx, int) else n_idx for n_idx in nodes]
	edges = [[graph.variables[v_idx].name for v_idx in e] for e in graph.edges.tolist()]
	short_edges = [e for e in edges if any([n in nodes[:2] for n in e])]
	nodes = list(set([n for e in short_edges for n in e]))
	edges = [e for e in edges if all([n in nodes for n in e])]
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	graph_to_image(G, **kwargs)


def graph_to_image(G, filename=None, show_plot=False, layout="graphviz", **kwargs):
	fig = plt.figure(**kwargs)
	if layout == "graphviz":
		pos = graphviz_layout(G, prog="dot")
	elif layout == "circular":
		pos = nx.circular_layout(G)
	elif layout == "planar":
		pos = nx.planar_layout(G)
	nx.draw(G, pos,
			 arrows=True, 
			 with_labels=True, 
			 font_weight='bold',
			 node_color='lightgrey',
			 edgecolors='black',
			 node_size=600,
			 arrowstyle='-|>',
			 arrowsize=16)
	if filename is not None:
		plt.savefig(filename)
	if show_plot:
		plt.show()
	plt.close()