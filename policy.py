import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as geom_nn
import networkx as nx


class MLPolicy(nn.Module):
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
        x = torch.cat((state[0], state[1]), dim=-1)
        probs = self.forward(x)
        action = torch.multinomial(probs, 1)
        return int(action.item()), torch.log(probs[action])
   
    
    
class GAT(nn.Module):
    def __init__(self, num_variables, n_hidden=[], n_heads=1, edge_dim=3):
        super().__init__()
        self.num_variables = int(num_variables)
        nodes_in = [self.num_variables] + n_hidden
        nodes_out = n_hidden + [self.num_variables]
        
        self.layers = nn.ModuleList()
        for inputs, outputs in zip(nodes_in[:-1], nodes_out[:-1]):
            pass
        
        self.layers.append(geom_nn.TransformerConv(nodes_in[-1], 
                                                   nodes_out[-1], 
                                                   heads=n_heads, 
                                                   concat=False, 
                                                   edge_dim=3))
        self.softmax = nn.Softmax(dim=0)
        
        # fully connected graph
        graph = nx.complete_graph(num_variables)
        self.edge_index = torch_geometric.utils.from_networkx(graph).edge_index.to('cuda') 
        
    def forward(self, x, edge_features):
        for layer in self.layers:     
            x = layer(x, self.edge_index, edge_features)
        x = self.softmax(x)
        return x
    
    def act(self, state):
        x = torch.zeros(self.num_variables, device=state[0].device) # no node features
        edge_features = torch.stack([state[0].flatten(), state[0].T.flatten(), state[1].flatten()], dim=-1)
        probs = self.forward(x, edge_features)
        action = torch.multinomial(probs, 1)
        return int(action.item()), torch.log(probs[action])
    