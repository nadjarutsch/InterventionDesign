import gym
from gym import spaces
import numpy as np
import torch
from typing import Tuple

from causal_graphs.graph_generation import generate_categorical_graph, get_graph_func
#from causal_graphs.graph_visualization import visualize_graph
from causal_graphs.variable_distributions import _random_categ
from datasets import GraphData
from visualize import DrawGraph
from causal_graphs.graph_definition import CausalDAG



class CausalEnv(gym.Env):
    """Environment based on a sampled or given Directed Acyclic Graph (DAG).
    
    Attributes:
        num_vars: Number of variables of the sampled DAG.
        min_categs: Minimum number of categories of each causal variable.
        max_categs: Maximum number of categories of each causal variable.
        graph_structure: Structure of the sampled DAG.
        edge_prob: Initial edge likelihood.
        dag: DAG for data sampling; if given, no DAG will be generated.
    """
    metadata = {'render.modes': ['human']}
  
    def __init__(self, 
                 num_vars: int, 
                 min_categs: int,
                 max_categs: int,
                 graph_structure: str='random',
                 edge_prob: float=0.3,
                 dag: CausalDAG=None):
        """Inits an instance of the environment with the given attributes."""
        
        super(CausalEnv, self).__init__()
        
        # Sample an underlying DAG
        if dag is None:
            self.dag = generate_categorical_graph(num_vars=num_vars,
                                                  min_categs=min_categs,
                                                  max_categs=max_categs,
                                                  connected=True,
                                                  graph_func=get_graph_func(graph_structure),
                                                  edge_prob=edge_prob,
                                                  use_nn=True)
        else:
            self.dag = dag   
        
        # One action is an intervention on one node (sparse interventions)
        self.action_space = spaces.Discrete(num_vars)
             
        # Dummy observational space
        # TODO: Implement observational space for learning a policy, e.g. 
        # possible values (0-1) in adjacency matrix (box space)
        self.observation_space = spaces.Discrete(1)
        
        self.visualization = None
        
    def step(self, 
             action_idx: int, 
             num_samples: int) -> Tuple[GraphData, int, dict]:
        
        # Perform intervention, sample from interventional SCM     
        var = self.dag.variables[action_idx]
        
        # Hard intervention => replace p(X_n) by uniform categorical
        int_dist = _random_categ(size=(var.prob_dist.num_categs,), scale=0.0, axis=-1)
        value = np.random.multinomial(n=1, pvals=int_dist, size=(num_samples,))
        value = np.argmax(value, axis=-1) # One-hot to index
        intervention_dict = {var.name: value}
        
        int_data = self.dag.sample(interventions=intervention_dict, 
                                   batch_size=num_samples, 
                                   as_array=True)        
        int_data = GraphData(data=torch.from_numpy(int_data)) 

        
        reward = -1 * num_samples # incentivice agent to perform as few interventions as possible
        info = {}
        
        return int_data, reward, info
    
    def reset(self, n_samples: int) -> GraphData:
        dataset = GraphData(graph=self.dag, dataset_size=n_samples)  
        return dataset
    
    def render(self, gamma, theta, mode='human', title=None, close=False):
        # Render the environment to the screen
        
        if self.visualization == None:
            self.visualization = DrawGraph(len(self.dag.variables), gamma, theta)
    
   #     if self.current_step > LOOKBACK_WINDOW_SIZE:        
    #        self.visualization.render(self.current_step, gamma, theta, window_size=LOOKBACK_WINDOW_SIZE)
    
    def close(self):
        pass
