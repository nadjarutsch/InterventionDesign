import torch

class CategoricalData(torch.utils.data.Dataset):
    
    def __init__(self, graph, dataset_size):
        super().__init__()
        self.graph = graph
        self.var_names = [v.name for v in self.graph.variables]
        data = graph.sample(batch_size=dataset_size, as_array=True)
        self.data = torch.from_numpy(data).long()
        
    
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        return self.data[idx]