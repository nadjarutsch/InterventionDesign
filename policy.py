import torch
import torch.nn as nn


class MLPolicy(nn.Module):
    def __init__(self, num_variables):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_variables * num_variables, (num_variables * num_variables / 2))) 
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear((num_variables * num_variables / 2), num_variables)) 
        self.layers.append(nn.Softmax())
    
    def forward(self, x):
        for layer in self.layers:     
            x = layer(x)
        return x
    
    def act(self, state):
        probs = self.forward(state.flatten())
        action = torch.multinomial(probs, 1)
        return int(action.item()), torch.log(probs[action])