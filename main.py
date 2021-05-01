from env import CausalEnv
from scm import StructuralCausalModel
import numpy as np


# TODO: training loop


# test example
scm = StructuralCausalModel({
    "x1": lambda     n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
    "x2": lambda x1, n_samples: np.where(x1==0, 1, 0),
    "x3": lambda x2, n_samples: np.random.binomial(n=1,p=0.5,size=n_samples) + x2,
})

env = CausalEnv(scm)
obs_samples = env.reset()
print(obs_samples)

samples, reward, info = env.step(1)
print(samples)
