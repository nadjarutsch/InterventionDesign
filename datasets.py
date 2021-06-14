import torch

class GraphData(torch.utils.data.Dataset):
    
    def __init__(self, graph=None, dataset_size=None, data=None):
        super().__init__()
        if graph is not None:
            self.graph = graph
            self.var_names = [v.name for v in self.graph.variables]
            data = torch.from_numpy(graph.sample(batch_size=dataset_size, as_array=True)).long()
        self.data =  data
        
    
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        return self.data[idx]
    
