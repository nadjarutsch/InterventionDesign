import gym
from gym import spaces
import numpy as np
import torch

from causal_graphs.graph_generation import generate_categorical_graph
from causal_graphs.graph_visualization import visualize_graph
from datasets import CategoricalData



class CausalEnv(gym.Env):
    """Environment based on a sampled Directed Acyclic Graph (DAG).
    
    Attributes:
        num_vars: Number of variables of the sampled DAG.
        max_categs: (Maximum) number of categories of each causal variable.
        graph_structure: Structure of the sampled DAG.
    """
    metadata = {'render.modes': ['human']}
  
    def __init__(self, 
                 num_vars: int, 
                 max_categs: int,
                 graph_structure: str = 'random'):
        """Inits an instance of the environment with the given attributes."""
        
        super(CausalEnv, self).__init__()
        
        # Sample an underlying DAG
        # TODO: add graph_structure as parameter in generate_categorical_graph
        self.dag = generate_categorical_graph(num_vars=num_vars,
                                   min_categs=max_categs, # TODO: can this be changed?
                                   max_categs=max_categs,
                                   edge_prob=0.0,
                                   connected=True,
                                   seed=42)
        
        # One action is an intervention on one node (sparse interventions)
        self.action_space = spaces.Discrete(num_vars)
             
        # Dummy observational space
        # TODO: Implement observational space for learning a policy, e.g. 
        # possible values (0-1) in adjacency matrix (box space)
        self.observation_space = spaces.Discrete(1)
        
    def step(self, action):
        
        # Perform intervention, sample from interventional SCM     
#        var = self.dag.variables[action]
        
        # Soft intervention => replace p(X_n) by random categorical
#        int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)
#        value = np.random.multinomial(n=1, pvals=int_dist, size=(N_SAMPLES,))
#        value = np.argmax(value, axis=-1) # One-hot to index
#		 value[:] = 0
#        intervention_dict = {var.name: value}
        
        
#        int_sample = self.dag.sample(interventions=intervention_dict, 
#		                         batch_size=N_SAMPLES, 
#		                         as_array=True)
        
#        int_sample = torch.from_numpy(int_sample)

        
#        reward = -1 # incentivice agent to perform as few interventions as possible
#        info = {} # TODO: do I want to use this?
        
#        return int_sample, reward, info
        pass
    
    def reset(self, n_samples: int) -> CategoricalData:
        dataset = CategoricalData(self.dag, dataset_size=n_samples)  
        return dataset
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def close(self):
        pass
