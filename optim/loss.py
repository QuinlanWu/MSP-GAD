import torch    
import numpy as np
import torch.nn.functional as F
from itertools import cycle
    
def loss_function_MD(nu,data_centers,outputs,radius,graph_masks, mask=None):
    dist,scores=anomaly_score_mc(data_centers,outputs,radius,None)

    multi_radius = False
    if radius.shape[0] == len(graph_masks): #  这里也可能会存在只有一个radius，但是仍可以作为多图处理
        rg = list(zip(radius, graph_masks))
        multi_radius = True
    elif radius.shape[0] == 1:
        rg = list(zip(cycle(radius.detach()), graph_masks))
    else:
        raise ValueError('number of radius must be same as graph_masks')

    sl_list = []
    for data_radius, graph_mask in rg:
        mymask = torch.zeros(outputs.shape[0], dtype=torch.bool, device=outputs.device)
        start_idx = graph_mask['start_idx']
        num_nodes = graph_mask['num_nodes']
        mymask[start_idx:start_idx + num_nodes] = True
        if mask is not None:
            mymask = mymask & mask

        if multi_radius:    # multi radius
            _,data_score=anomaly_score_mc(data_centers,outputs[mymask],data_radius,None)
        else:   # only one radius
            data_score = scores[mymask]

        sl = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(data_score), data_score))
        sl_list.append(sl)

    svdd_loss = torch.mean(torch.stack(sl_list))
    # svdd_loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    '''lambda_reg = 1.0
    cds = centers_distances(data_centers)
    # 创建掩码，排除对角线元素（自身到自身的距离）
    mask_cds = ~torch.eye(cds.size(0), dtype=torch.bool, device=cds.device)
    if mask_cds.sum() > 0:
        # 计算非对角线元素的平均距离
        center_distances = cds[mask_cds]
        center_distances = torch.clamp(center_distances, min=1e-16)  # 防止过小值
        center_separation_loss = torch.mean(center_distances)
        loss = svdd_loss # + lambda_reg*center_separation_loss
    else:
        loss = svdd_loss'''
    loss = svdd_loss
    # loss is necessary, dist and scores are need when only one radius(means radius==0)
    return loss,dist[mask],scores[mask] 


def loss_function(nu,data_centers,outputs,radius,graph_masks, mask=None):
    dist,scores=anomaly_score_mc(data_centers,outputs,radius,mask)
    svdd_loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    '''lambda_reg = 1.0
    cds = centers_distances(data_centers)
    # 创建掩码，排除对角线元素（自身到自身的距离）
    mask_cds = ~torch.eye(cds.size(0), dtype=torch.bool, device=cds.device)
    if mask_cds.sum() > 0:
        # 计算非对角线元素的平均距离
        center_distances = cds[mask_cds]
        center_distances = torch.clamp(center_distances, min=1e-16)  # 防止过小值
        center_separation_loss = torch.mean(center_distances)
        loss = svdd_loss # + lambda_reg*center_separation_loss
    else:
        loss = svdd_loss'''
    loss = svdd_loss
    return loss,dist,scores


def anomaly_score_mc(data_centers, outputs, radius, mask=None):
    if mask is None:
        actual_outputs = outputs
    else:
        # 对mask应用的情况
        actual_outputs = outputs[mask]
    # 计算到所有中心点的距离
    squared_dist = torch.sum((actual_outputs.unsqueeze(1) - data_centers.unsqueeze(0)) ** 2, dim=2)
    # euclidean_dist = torch.cdist(actual_outputs, data_centers, p=2)
    # 计算到所有中心点的平均距离
    dist = torch.mean(squared_dist, dim=1)
    scores = dist - radius ** 2
    return dist,scores


def anomaly_score(data_center,outputs,radius,mask= None):
    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    # c=data_center.repeat(outputs[mask].size()[0],1)
    # res=outputs[mask]-c
    # res=torch.mean(res, 1, keepdim=True)
    # dist=torch.diag(torch.mm(res,torch.transpose(res, 0, 1)))

    scores = dist - radius ** 2
    return dist,scores


def centers_distances(data_centers):
    """
    使用广播机制计算每个中心到其他中心的欧几里得距离
    
    参数:
    data_centers (torch.Tensor): 二维张量，每行代表一个中心节点的坐标
    
    返回:
    torch.Tensor: 距离矩阵，其中元素 (i, j) 表示中心 i 到中心 j 的距离
    """
    # 扩展维度以利用广播机制
    diff = data_centers.unsqueeze(1) - data_centers.unsqueeze(0)    # 差值
    # 计算欧几里得距离
    squared_diff = torch.sum(diff ** 2, dim=2)
    # 为了避免数值问题，将平方和限制在一个很小的正数以上
    squared_diff = torch.clamp(squared_diff, min=1e-32)
    distances = torch.sqrt(squared_diff)
    return distances


def init_centers(args, outputs, graph_masks, eps=0.001):
    """Initialize hypersphere centers c as the mean from an initial forward pass on the data."""
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)

    num_graph = len(graph_masks)
    if args.module == 'GIN':
        c = torch.zeros(num_graph, args.n_hidden*2, device=device) # gin模型输出embedding维度是2*n_hidden
    elif args.module == 'myGCN':
        c = torch.zeros(num_graph, int(args.n_hidden/2), device=device) # mygcn模型输出embedding维度是n_hidden/2
    else:
        c = torch.zeros(num_graph, args.n_hidden, device=device) # gcn,mygin模型输出维度是n_hidden

    for i,graph_mask in enumerate(graph_masks):
        c[i] = torch.sum(outputs[graph_mask['start_idx']:
                            graph_mask['start_idx']+graph_mask['num_nodes'],:], 
                            dim=0)/graph_mask['num_nodes']

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[i][(abs(c[i]) < eps) & (c[i] < 0)] = -eps
        c[i][(abs(c[i]) < eps) & (c[i] > 0)] = eps

    return c


def init_center(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=device)

    model.eval()
    with torch.no_grad():

        outputs= model(input_g,input_feat)

        # get the inputs of the batch

        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius_MD(dist: torch.Tensor, nu: float, graph_masks, mask=None):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    mr = []
    for graph_mask in graph_masks:
        mymask = torch.zeros(dist.shape[0], dtype=torch.bool, device=dist.device)
        start_idx = graph_mask['start_idx']
        num_nodes = graph_mask['num_nodes']
        mymask[start_idx:start_idx + num_nodes] = True
        if mask is not None:
            mymask = mymask & mask

        radius=torch.quantile(torch.sqrt(dist[mymask]), 1 - nu)
        mr.append(torch.tensor(radius, device=dist.device))
  
    return torch.stack(mr)


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = 0.0
        self.best_epoch = -1
        self.lowest_loss = None
        self.early_stop = False
        self.acc = 0.0
        self.auc = 0.0
        self.ap = 0.0

    def step(self, acc,auc,ap,loss, model,epoch,path):
        score = auc+ap
        cur_loss=loss
        if (self.best_score is None) or (self.lowest_loss is None):
        #if self.lowest_loss is None:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.acc = acc
            self.auc = auc
            self.ap = ap
            self.save_checkpoint(acc,auc,ap,loss,model,path)
        #elif cur_loss > self.lowest_loss:
        elif (score > self.best_score): #  and (cur_loss <= self.lowest_loss):
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.acc = acc
            self.auc = auc
            self.ap = ap
            self.save_checkpoint(acc,auc,ap,loss,model,path)
            self.counter = 0
        else:
            self.counter += 1
            # if self.counter >= 0.8*(self.patience):
            #     print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, acc,auc,ap,loss,model,path):
        '''Saves model when validation loss decrease.'''
        # print('model saved. loss={:.4f} AUC={:.4f}'. format(loss,acc))
        torch.save(model.state_dict(), path)