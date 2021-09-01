import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as geom_nn
import networkx as nx


class MLP(nn.Module):
    def __init__(self, num_variables, n_hidden):
        super().__init__()
        num_variables = int(num_variables)
        nodes_in = [2 * num_variables * num_variables] + n_hidden
        nodes_out = n_hidden + [num_variables]
        
        self.layers = nn.ModuleList()
        for inputs, outputs in zip(nodes_in[:-1], nodes_out[:-1]):
            self.layers.append(nn.Linear(inputs, outputs)) 
            self.layers.append(nn.PReLU())     
        
        # add final linear layer
        self.layers.append(nn.Linear(nodes_in[-1], nodes_out[-1]))
        self.layers.append(nn.Softmax(dim=0))
    
    def forward(self, x):
        for layer in self.layers:     
            x = layer(x)
        return x
    
    def act(self, state):
        x = torch.cat((state[0].flatten(), state[1].flatten()), dim=-1)
        probs = self.forward(x)
        action = torch.multinomial(probs, 1)
        return int(action.item()), torch.log(probs[action])
   
    
    
class GAT(nn.Module):
    def __init__(self, num_variables, c_hidden=[9, 9, 9], n_heads=3, edge_dim=3, device='cuda:0'):
        super().__init__()
        self.num_variables = int(num_variables)
      #  nodes_in = c_hidden
       # nodes_out = c_hidden[1:] + [self.num_variables]
        
        self.layers = nn.ModuleList()

        self.layers.append(GATLayer(c_in=0, c_out=c_hidden[0], edge_dim=edge_dim, num_heads=n_heads))
        self.layers.append(nn.PReLU())
        
        for inputs, outputs in zip(c_hidden, c_hidden):
            self.layers.append(GATLayer(c_in=inputs, c_out=outputs, edge_dim=edge_dim, num_heads=n_heads))
            self.layers.append(nn.PReLU())
        
        self.layers.append(GATLayer(c_in=c_hidden[-1], c_out=1, edge_dim=edge_dim, num_heads=n_heads, concat_heads=False))
        self.softmax = nn.Softmax(dim=0)
        
        # fully connected graph
 #       graph = nx.complete_graph(num_variables)
  #      self.edge_index = torch_geometric.utils.from_networkx(graph).edge_index.to(device)
     #   self.edge_index = edge_index.type(torch.LongTensor)
        adj_matrix = torch.ones((num_variables, num_variables), device=device)
        self.adj_matrix = adj_matrix[None,:,:]
        
    def forward(self, node_feats, edge_feats, adj_matrix):
        for layer in self.layers:    
          #  print(node_feats.shape, edge_feats.shape)
            if isinstance(layer, GATLayer):
                node_feats = layer(node_feats, edge_feats, adj_matrix=adj_matrix)
            else:
                node_feats = layer(node_feats)
        node_feats = self.softmax(node_feats)
        return node_feats
    
    def act(self, state):
   #     node_feats = torch.zeros((1, self.num_variables), device=state[0].device) # no node features
        node_feats = None
        edge_feats = torch.stack([state[0], state[0].T, state[1]], dim=-1)
        edge_feats = edge_feats[None,:,:] # TODO: implement for batch_size > 1
        adj_matrix = self.adj_matrix
        probs = self.forward(node_feats, edge_feats, adj_matrix).squeeze() # TODO: support batch_size > 1
   #     print(probs.shape, probs)
        action = torch.multinomial(probs, 1)
        return int(action.item()), torch.log(probs[action])
    

    
    
class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, edge_dim, num_heads, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        if c_in != 0:
            self.node_projection = nn.Linear(c_in, c_out * num_heads)
            nn.init.xavier_uniform_(self.node_projection.weight.data, gain=1.414)
        self.edge_projection = nn.Linear(edge_dim, c_out * num_heads)
      #  self.projection = nn.Linear(c_in, c_out * num_heads)
        if c_in != 0:
            self.a = nn.Parameter(torch.Tensor(num_heads, 3 * c_out)) # One per head
        else:
            self.a = nn.Parameter(torch.Tensor(num_heads, c_out))    
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        
        nn.init.xavier_uniform_(self.edge_projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, edge_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = edge_feats.size(0), adj_matrix.size(-1)

        # Apply linear layer and sort nodes by head
        if node_feats != None:
            node_feats = self.node_projection(node_feats)
            node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
            node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        
        edge_feats = self.edge_projection(edge_feats)
        edge_feats = edge_feats.view(batch_size, num_nodes, num_nodes, self.num_heads, -1) # TODO: DOUBLECHECK

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        
        edge_feats_flat = edge_feats.view(batch_size * num_nodes * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]
        
        if node_feats != None:
            a_input = torch.cat([
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
                edge_feats_flat
            ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
        else:
            a_input = edge_feats_flat 
            
        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        values = node_feats + edge_feats if node_feats != None else edge_feats # TODO: DOUBLECHECK
        node_feats = torch.einsum('bijh,bijhc->bihc', attn_probs, values)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats
    