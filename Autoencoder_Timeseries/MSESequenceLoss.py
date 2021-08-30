import torch

def MSESequenceLoss(y_pred, y_gt):
    batch_dim = 0
    mse = ((y_pred - y_gt) ** 2).mean(batch_dim)
    mse_seq_avg = mse.mean()
    return mse_seq_avg