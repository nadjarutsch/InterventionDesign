import torch
import numpy as np 

CPU_ONLY = False

def get_device():
	return torch.device("cuda:0" if (not CPU_ONLY and torch.cuda.is_available()) else "cpu")