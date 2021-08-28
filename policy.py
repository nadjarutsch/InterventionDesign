import torch
import torch.nn as nn


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
        probs = self.forward(state.flatten())
        action = torch.multinomial(probs, 1)
        return int(action.item()), torch.log(probs[action])