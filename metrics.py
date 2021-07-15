import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle



class Logger(object):
    def __init__(self, args):
        """Inits an instance of Logger."""       
        super().__init__()
        date = datetime.today().strftime('%Y-%m-%d')
        suffix = args.graph_structure + "-" + args.heuristic + datetime.today().strftime('-%H-%M')
        self.writer = SummaryWriter("tb_logs/%s/%s" % (date, suffix))
        self.stop_count = 0
        self.max_epochs = args.epochs
    
    def before_training(self, adjmatrix_pred, dag):
        metrics = get_metrics(adjmatrix_pred, torch.from_numpy(dag.adj_matrix))
        dag.save_to_file('%s/dag.pt' % self.writer.log_dir)

        self.writer.add_scalar('SHD', metrics['SHD'], global_step=0)
        self.writer.add_scalar('Precision', metrics['precision'], global_step=0)
        self.writer.add_scalar('Recall', metrics['recall'], global_step=0)
        self.writer.add_scalar('False positives', metrics['FP'], global_step=0)
        self.writer.add_scalar('False negatives', metrics['FN'], global_step=0)
        
        self.shd = metrics['SHD'] # needed as stop criterion
    
    def on_epoch_end(self, adjmatrix_pred, adjmatrix_target, epoch):
        metrics = get_metrics(adjmatrix_pred, adjmatrix_target)
        chi = chi_square(self.stats, adjmatrix_pred.num_variables)
        
        self.writer.add_scalar('SHD', metrics['SHD'], global_step=epoch+1)
        self.writer.add_scalar('Chi-Square value vs. uniform', chi, global_step=epoch)
        self.writer.add_scalar('Precision', metrics['precision'], global_step=epoch+1)
        self.writer.add_scalar('Recall', metrics['recall'], global_step=epoch+1)
        self.writer.add_scalar('False positives', metrics['FP'], global_step=epoch+1)
        self.writer.add_scalar('False negatives', metrics['FN'], global_step=epoch+1)
        
        self.shd = metrics['SHD'] # needed as stop criterion
        
        if self.shd == 0:
            self.stop_count += 1
        else:
            self.stop_count = 0
            
        print('\nSHD: ', self.shd)
            
        if self.stop_count == 3 or epoch == self.max_epochs:
            self.writer.add_scalar('Intervention loops needed', epoch+1-self.stop_count)
            return 1
     
        return 0


def get_metrics(pred_adjmatrix, true_adjmatrix):
    true_adjmatrix = true_adjmatrix.to(dtype=torch.bool)
    binary_gamma = pred_adjmatrix.binary
    false_positives = torch.logical_and(binary_gamma, ~true_adjmatrix)
    false_negatives = torch.logical_and(~binary_gamma, true_adjmatrix)
    TP = torch.logical_and(binary_gamma, true_adjmatrix).float().sum().item()
    TN = torch.logical_and(~binary_gamma, ~true_adjmatrix).float().sum().item()
    FP = false_positives.float().sum().item()
    FN = false_negatives.float().sum().item()
    TN = TN - pred_adjmatrix.num_variables # Remove diagonal as it is not being predicted

    recall = TP / max(TP + FN, 1e-5)
    precision = TP / max(TP + FP, 1e-5)

    rev = torch.logical_and(binary_gamma, true_adjmatrix.T)
    num_revs = rev.float().sum().item()
    SHD = (false_positives + false_negatives + rev + rev.T).float().sum().item() - num_revs

    metrics = {
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "SHD": int(SHD),
        "reverse": int(num_revs),
        "recall": recall,
        "precision": precision
    }
    return metrics






def chi_square(int_stats, num_variables):
    """Chi-squared test value of distribution over intervention variables compared 
    to uniform distribution"""
    
    N = sum(int_stats.values())
    chi_square = 0
    unif_prob = 1 / num_variables

    for i in range(num_variables):
        chi_square += (int_stats[i] / N - unif_prob)**2 / unif_prob
        
    return chi_square * N