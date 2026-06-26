import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class AnomalyDetectionGCN(nn.Module):
    """专门为异常检测优化的GCN"""
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, 
                 activation=F.leaky_relu, dropout=0.3, bias=False,
                 use_batch_norm=True):
        super(AnomalyDetectionGCN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = dropout
        
        # 输入层
        self.layers.append(GraphConv(in_feats, n_hidden, bias=bias, activation=activation))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        
        # 隐藏层
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, bias=bias, activation=activation))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(n_hidden))
        
        # 输出层（无激活函数）
        self.layers.append(GraphConv(n_hidden, out_feats, bias=bias, activation=None))
    
    def forward(self, g, features):
        h = features
        
        for i, layer in enumerate(self.layers):
            # 应用GCN层
            h = layer(g, h)
            
            # 如果不是输出层，应用BatchNorm和Dropout
            if i < len(self.layers) - 1:
                if self.batch_norms is not None:
                    h = self.batch_norms[i](h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h

# 推荐的异常检测配置
'''anomaly_gcn = AnomalyDetectionGCN(
    in_feats=100,     # 输入特征维度
    n_hidden=64,      # 隐藏层维度
    out_feats=16,     # 输出维度
    n_layers=2,       # 推荐2层
    activation=F.leaky_relu,  # LeakyReLU更适合异常检测
    dropout=0.3,      # 稍低的dropout
    bias=False,       # 不使用偏置
    use_batch_norm=True  # 使用BatchNorm
)'''