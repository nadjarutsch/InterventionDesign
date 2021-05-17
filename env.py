import gym
from gym import spaces
import numpy as np
import torch

from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func
from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.variable_distributions import _random_categ
from datasets import CategoricalData



class CausalEnv(gym.Env):
    """Environment based on a sampled Directed Acyclic Graph (DAG).
    
    Attributes:
        num_vars: Number of variables of the sampled DAG.
        min_categs: Minimum number of categories of each causal variable.
        max_categs: Maximum number of categories of each causal variable.
        graph_structure: Structure of the sampled DAG.
    """
    metadata = {'render.modes': ['human']}
  
    def __init__(self, 
                 num_vars: int, 
                 min_categs: int,
                 max_categs: int,
                 graph_structure: str = 'random'):
        """Inits an instance of the environment with the given attributes."""
        
        super(CausalEnv, self).__init__()
        
        # Sample an underlying DAG
        self.dag = generate_categorical_graph(num_vars=num_vars,
                                              min_categs=min_categs,
                                              max_categs=max_categs,
                                              connected=True,
                                              graph_func=get_graph_func(graph_structure),
                                              edge_prob=0.4,
                                              use_nn=True)
        
        # One action is an intervention on one node (sparse interventions)
        self.action_space = spaces.Discrete(num_vars)
             
        # Dummy observational space
        # TODO: Implement observational space for learning a policy, e.g. 
        # possible values (0-1) in adjacency matrix (box space)
        self.observation_space = spaces.Discrete(1)
        
    def step(self, action_idx, num_samples):
        
        # Perform intervention, sample from interventional SCM     
        var = self.dag.variables[action_idx]
        
        # Soft intervention => replace p(X_n) by uniform categorical
        # TODO: I believe this is still called a (stochastic) hard intervention?
        int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)
        value = np.random.multinomial(n=1, pvals=int_dist, size=(num_samples,))
        value = np.argmax(value, axis=-1) # One-hot to index
        intervention_dict = {var.name: value}
        
        int_data = self.dag.sample(interventions=intervention_dict, 
                                   batch_size=num_samples, 
                                   as_array=True)        
        int_data = torch.from_numpy(int_data) # TODO: provide as torch.utils.data.Dataset

        
        reward = -1 # incentivice agent to perform as few interventions as possible
        info = {} # TODO: do I want to use this?
        
        return int_data, reward, info
    
    def reset(self, n_samples: int) -> CategoricalData:
        dataset = CategoricalData(self.dag, dataset_size=n_samples)  
        return dataset
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def close(self):
        pass
