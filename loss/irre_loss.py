import torch

def cal_irre(x, y):
    m_x = x.view(x.shape[0], x.shape[1], -1)
    m_y = y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)
    return torch.bmm(m_x, m_y).mean()