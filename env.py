import gym
from gym import spaces
import numpy as np

# number of samples drawn from observational SCM
N_OBS = 100

# number of samples drawn for one intervention
N_SAMPLES = 10

class CausalEnv(gym.Env):
    """Environment based on a given Structural Causal Model (SCM)
    
    Attributes:
        scm: Structural Causal Model
    """
    metadata = {'render.modes': ['human']}
  
    def __init__(self, scm):
        super(CausalEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Structural Causal Model
        self.scm = scm
        
        # One action is an intervention on one node (sparse interventions)
        self.action_space = spaces.Discrete(len(list(self.scm.assignment)))
             
        # Dummy observational space
        # TODO: Implement observational space for learning a policy
        self.observation_space = spaces.Discrete(1)
        
    def step(self, action):
        
        # Perform intervention, sample from interventional SCM     
        node_key = list(self.scm.assignment.keys())[action]
        scm_do = self.scm.do(node_key)
        # TODO: set values by sampling from the codomain (wait for implementation
        # of codomain in scm.py) // sample with or without replacement?
        samples = scm_do.sample(n_samples=N_SAMPLES, set_values={node_key : np.arange(N_SAMPLES)})
        
        reward = -1 # incentivice agent to perform as few interventions as possible
        info = {} # TODO: do I want to use this?
        
        return samples, reward, info
    
    def reset(self):
        # TODO: Extend when observational space is implemented
        
        obs_samples = self.scm.sample(n_samples=N_OBS) # TODO: implement in main?           
        return obs_samples
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
    
    def close(self):
        pass
    
    

