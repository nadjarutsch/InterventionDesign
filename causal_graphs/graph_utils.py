import numpy as np


def adj_matrix_to_edges(adj_matrix):
	edges = np.where(adj_matrix)
	edges = np.stack([edges[0], edges[1]], axis=1)
	return edges


def edges_to_adj_matrix(edges, num_vars):
	if not isinstance(edges, np.ndarray):
		edges = np.array(edges)
	adj_matrix = np.zeros((num_vars, num_vars), dtype=np.bool)
	if edges.shape[0] > 0:
		adj_matrix[edges[:,0], edges[:,1]] = True
	return adj_matrix


def edges_or_adj_matrix(edges, adj_matrix, num_vars):
	assert edges is not None or adj_matrix is not None, 'Either the edges or adjacency matrix must be provided for the DAG'
	if edges is None:
		edges = adj_matrix_to_edges(adj_matrix)
	elif not isinstance(edges, np.ndarray):
		edges = np.array(edges)
	if adj_matrix is None:
		adj_matrix = edges_to_adj_matrix(edges, num_vars)
	return edges, adj_matrix


def sort_graph_by_vars(variables, edges=None, adj_matrix=None):
	edges, adj_matrix = edges_or_adj_matrix(edges, adj_matrix, len(variables))
	matrix_copy = np.copy(adj_matrix)
	
	sorted_idxs = []
	def get_empty_nodes():
		return [i for i in np.where(~matrix_copy.any(axis=0))[0] if i not in sorted_idxs]
	empty_nodes = get_empty_nodes()
	while len(empty_nodes) > 0:
		node = empty_nodes.pop(0)
		sorted_idxs.append(node)
		matrix_copy[node,:] = False
		empty_nodes = get_empty_nodes()
	assert not matrix_copy.any(), "Sorting the graph failed because it is not a DAG!"

	variables = [variables[i] for i in sorted_idxs]
	adj_matrix = adj_matrix[sorted_idxs][:,sorted_idxs]

	num_vars = len(variables)
	edges = edges - num_vars # To have a better replacement
	for v_idx, n_idx in enumerate(sorted_idxs):
		edges[edges == (n_idx - num_vars)] = v_idx

	return variables, edges, adj_matrix, sorted_idxs


def get_node_relations(adj_matrix):
	# An element (i,j) being True represents that j is a parent of i
	ancestors = adj_matrix.T
	changed = True
	while changed:
		new_anc = np.logical_and(ancestors[...,None], ancestors[None]).any(axis=1)
		new_anc = np.logical_or(ancestors, new_anc)
		changed = not (new_anc == ancestors).all().item()
		ancestors = new_anc
	
	# Output: matrix with (i,j)
	# 		  = 1: j is an ancestor of i
	# 		  = -1: j is a descendant of i,
	#		  = 0: j and i are independent
	#		  = 2: j and i share a confounder
	ancestors = ancestors.astype(np.int32)
	descendant = ancestors.T
	node_relations = ancestors - descendant
	confounder = (node_relations == 0) * ((ancestors[None] * ancestors[:,None]).sum(axis=-1) > 0)
	node_relations += 2 * confounder
	node_relations[np.arange(node_relations.shape[0]), np.arange(node_relations.shape[1])] = 0

	return node_relations


def find_nodes_on_paths(graph, source_node, target_node, nodes_on_path=None):
	"""
	Find all nodes that are parts of paths from the source node to the target node.
	Simple, recursive algorithm: iterate for all children of the source node. 
	"""
	if nodes_on_path is None:
		nodes_on_path = np.zeros(graph.num_vars)
	if source_node == target_node:
		nodes_on_path[source_node] = 1
		return nodes_on_path
	elif nodes_on_path[source_node] == 1:
		return nodes_on_path
	elif nodes_on_path[source_node] == -1:
		return None 
	else:
		children = np.where(graph.adj_matrix[source_node])[0]
		for c in children:
			ret = find_nodes_on_paths(graph, c, target_node, nodes_on_path=nodes_on_path)
			if ret is not None:
				nodes_on_path[source_node] = 1

		if nodes_on_path[source_node] <= 0:
			nodes_on_path[source_node] = -1
			return None 
		else:
			return nodes_on_path

