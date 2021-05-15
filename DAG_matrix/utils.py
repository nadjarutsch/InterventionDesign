import torch
import torch.nn.functional as F


def matrix_to_order(theta_matrix):
    log_thetas = F.logsigmoid(theta_matrix)
    order = []
    for i in range(theta_matrix.shape[0]):
        _, max_idx = log_thetas.min(dim=1).values.max(dim=0)
        order.append(max_idx)
        log_thetas[max_idx] = 0
        log_thetas[:,max_idx] = 0
        log_thetas[max_idx,max_idx] = -9e15
    return torch.stack(order, dim=0)


def update_stats(stats_dict, new_val):
    if len(stats_dict) == 0 or stats_dict["mean"] is None:
        stats_dict["mean"] = new_val.clone() 
        stats_dict["var"] = torch.zeros_like(new_val)
        stats_dict["max"] = torch.ones_like(new_val) * (-9e15)
        stats_dict["min"] = torch.ones_like(new_val) * (9e15)
        stats_dict["count"] = (new_val != 0.0).float()
    else:
        mean, var, count = stats_dict["mean"], stats_dict["var"], stats_dict["count"]
        count.add_(new_val != 0.0)
        delta = new_val - mean
        delta.mul_(new_val != 0.0)
        mean.add_(delta / count.clamp(min=1e-5))
        delta2 = new_val - mean
        var.add_(delta * delta2)
    stats_dict["min"] = torch.where(torch.logical_and(stats_dict["min"]>new_val, new_val != 0.0),
                                    new_val,
                                    stats_dict["min"])
    stats_dict["max"] = torch.where(torch.logical_and(stats_dict["max"]<new_val, new_val != 0.0),
                                    new_val,
                                    stats_dict["max"])


def get_final_stats(stats_dict):
    mean, var, count, min_v, max_v = stats_dict["mean"], stats_dict["var"], stats_dict["count"], \
                                     stats_dict["min"], stats_dict["max"]
    var = torch.where(count > 1, var / (count - 1).clamp(min=1), -torch.ones_like(var))
    t = torch.stack([mean, var, min_v, max_v], dim=-1)
    return t