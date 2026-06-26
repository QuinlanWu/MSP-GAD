import scipy.io as sio
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset, load_data
from dgl.data.utils import save_graphs, load_graphs
import os
import sys

from scipy.sparse import csr_matrix, lil_matrix, block_diag, csr_matrix
import warnings


class CustomGraphDataset(DGLDataset):
    """自定义图数据集类，专为单个图数据设计"""
    
    def __init__(self, name, mat_file_path, file_names, force_reload=False, verbose=False):
        self.mat_file_path = mat_file_path
        self.file_names = file_names
        self._num_graphs = 1
        self._graph_masks = []
        super(CustomGraphDataset, self).__init__(name=name, force_reload=force_reload, verbose=verbose)
    
    def process(self):
        """处理原始数据并转换为DGL图"""
        if self.verbose:
            print(f"Loading data from {self.mat_file_path}")
        
        # 加载.mat文件(当仅给出路径时，需要把目录下的所有.mat文件都加载进来)
        mat_data = merge_graph_datasets(self.mat_file_path, self.file_names)
        # dict_keys(['adj_matrix', 'node_features', 'node_labels', 'num_nodes', 'num_edges', 'num_features', 'num_graphs', 'graph_masks'])
        if self.verbose:
            print(f"Available keys in .mat file: {list(mat_data.keys())}")
        
        self._num_graphs = mat_data['num_graphs']
        self._graph_masks = mat_data['graph_masks']
        
        # 提取邻接矩阵 (Network)
        if 'adj_matrix' in mat_data:
            adj_matrix = mat_data['adj_matrix']
            if self.verbose:
                print(f"Network matrix: {type(adj_matrix)}, shape: {adj_matrix.shape}")
        else:
            raise ValueError("Could not find 'adj_matrix' matrix in .mat file")
        
        # 提取节点特征 (Attributes)
        if 'node_features' in mat_data:
            features = mat_data['node_features']
            if hasattr(features, 'todense'):
                features = features.todense()
            if self.verbose:
                print(f"Attributes matrix: {type(features)}, shape: {features.shape}")
        else:
            raise ValueError("Could not find 'node_features' matrix in .mat file")
        
        # 提取节点标签 (Label)
        if 'node_labels' in mat_data:
            labels = mat_data['node_labels']
            # 确保标签是一维数组
            if labels.ndim > 1:
                labels = labels.flatten()
            if self.verbose:
                print(f"Label vector: {type(labels)}, shape: {labels.shape}, unique values: {np.unique(labels)}")
        else:
            raise ValueError("Could not find 'node_labels' matrix in .mat file")
        
        # 转换为DGL图
        src, dst = adj_matrix.nonzero()
        self._g = dgl.graph((src, dst))
        
        # 添加节点特征
        self._g.ndata['feat'] = torch.FloatTensor(features)
        
        # 添加节点标签
        self._g.ndata['label'] = torch.LongTensor(labels)
        '''
        # 添加训练/验证/测试掩码
        n_nodes = self._g.number_of_nodes()
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        
        # 随机排列节点索引
        indices = torch.randperm(n_nodes)
        
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train+n_val]] = True
        test_mask[indices[n_train+n_val:]] = True
        
        self._g.ndata['train_mask'] = train_mask
        self._g.ndata['val_mask'] = val_mask
        self._g.ndata['test_mask'] = test_mask
        
        if self.verbose:
            print(f"Processed graph with {n_nodes} nodes and {self._g.number_of_edges()} edges")
            print(f"Feature dimension: {features.shape[1]}")
            print(f"Number of classes: {len(torch.unique(self._g.ndata['label']))}")
            print(f"Training nodes: {train_mask.sum().item()}")
            print(f"Validation nodes: {val_mask.sum().item()}")
            print(f"Test nodes: {test_mask.sum().item()}")'''
    
    def __getitem__(self, idx):
        """获取第idx个图（只有一个图）"""
        if idx != 0:
            raise IndexError("This dataset contains only one graph")
        return self._g
    
    def __len__(self):
        """返回图中数量（只有一个图）"""
        return 1
    
    @property
    def labels(self):
        """返回标签向量"""
        return self._g.ndata['label']
    
    @property
    def features(self):
        """返回节点特征"""
        return self._g.ndata['feat']   

    @property
    def num_graphs(self):
        """返回合并图数量"""
        return self._num_graphs
    
    @property
    def graph_masks(self):
        """返回合并图的信息"""
        '''list of graph_mask:
        [{
            'file_name': mat_file,
            'start_idx': node_offset,
            'num_nodes': num_nodes
        }]
        '''
        return self._graph_masks

    @property
    def num_classes(self):
        """返回类别数量"""
        return len(torch.unique(self._g.ndata['label']))
    
    @property
    def num_features(self):
        """返回节点特征维度"""
        return self._g.ndata['feat'].shape[1]

    def save(self):
        """保存处理后的数据"""
        save_graphs(os.path.join(self.save_path, self.name + '_dgl.bin'), [self._g])
    
    def load(self):
        """加载处理后的数据"""
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl.bin'))
        self._g = graphs[0]
    
    def has_cache(self):
        """检查是否有已处理的数据缓存"""
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl.bin'))

def load_my_data(args, domain_choice):
    """仿照DGL的load_data函数"""
    print('Loading custom data...')
    # 加载您的自定义数据
    if domain_choice == 'source':
        file_names = args.source_domain
    elif domain_choice == 'target':
        file_names = args.target_domain
    else:
        file_names = None
    return CustomGraphDataset(
        args.source_domain[0],                    # 数据集名称
        args.data_path or 'path/to/your/data.mat',  # 数据路径
        file_names,                  # 源域数据集
        force_reload=args.force_reload,      # 是否强制重新加载
        verbose=args.verbose                 # 是否显示详细信息
    )


# TODO: 在合并图之间添加边，待修改
def merge_graph_datasets_with_connections(directory_path, add_inter_graph_edges=False, 
                                        connection_strategy='random', num_connections_per_graph=5):
    """
    读取并合并图数据集，可选择在图之间添加连接
    
    Parameters:
    directory_path (str): 包含图数据的目录
    add_inter_graph_edges (bool): 是否在不同图之间添加边
    connection_strategy (str): 连接策略 ('random', 'first_to_first')
    num_connections_per_graph (int): 每个图连接的节点数
    
    Returns:
    dict: 合并后的图数据
    """
    
    # 存储所有图的数据
    all_adj_matrices = []
    all_node_features = []
    all_node_labels = []
    node_offsets = [0]  # 记录每个图的节点起始位置
    
    # 获取所有.mat文件
    mat_files = [f for f in os.listdir(directory_path) 
                 if f.endswith('.mat') and os.path.isfile(os.path.join(directory_path, f))]
    
    # 读取所有图数据
    total_nodes = 0
    for mat_file in mat_files:
        file_path = os.path.join(directory_path, mat_file)
        try:
            data = sio.loadmat(file_path)
            
            # 提取图数据
            adj_matrix = data['Network']
            node_features = data['Attributes']
            node_labels = [int(nl) for nl in data['Label'][0]]

            if adj_matrix is None or node_features is None or node_labels is None:
                warnings.warn(f"文件 {mat_file} 缺少必要的图数据字段")
                continue
            
            all_adj_matrices.append(adj_matrix)
            all_node_features.append(node_features)
            all_node_labels.append(node_labels)
            
            # 记录节点偏移量
            total_nodes += node_features.shape[0]
            node_offsets.append(total_nodes)
            
        except Exception as e:
            print(f"读取文件 {mat_file} 时出错: {e}")
            continue
    
    if not all_adj_matrices:
        raise ValueError("没有成功读取到任何有效的图数据")
    
    # 创建大型邻接矩阵
    merged_adj = lil_matrix((total_nodes, total_nodes))
    current_node = 0
    
    # 填充每个图的内部连接
    for i, adj in enumerate(all_adj_matrices):
        adj_shape = adj.shape[0]
        if hasattr(adj, 'toarray'):  # 稀疏矩阵
            adj_dense = adj.toarray()
        else:
            adj_dense = np.array(adj)
        
        # 将子图的邻接矩阵放置在大矩阵的对角线上
        end_node = current_node + adj_shape
        merged_adj[current_node:end_node, current_node:end_node] = adj_dense
        current_node = end_node
    
    # 可选：在不同图之间添加连接
    if add_inter_graph_edges and len(all_adj_matrices) > 1:
        for i in range(len(all_adj_matrices) - 1):
            # 获取两个相邻图的节点范围
            start1, end1 = node_offsets[i], node_offsets[i+1]
            start2, end2 = node_offsets[i+1], node_offsets[i+2]
            
            if connection_strategy == 'random':
                # 随机选择节点进行连接
                nodes1 = np.random.choice(range(start1, end1), 
                                        min(num_connections_per_graph, end1-start1), 
                                        replace=False)
                nodes2 = np.random.choice(range(start2, end2), 
                                        min(num_connections_per_graph, end2-start2), 
                                        replace=False)
                
                for n1 in nodes1:
                    for n2 in nodes2:
                        merged_adj[n1, n2] = 1
                        merged_adj[n2, n1] = 1
                        
            elif connection_strategy == 'first_to_first':
                # 连接每个图的第一个节点
                merged_adj[start1, start2] = 1
                merged_adj[start2, start1] = 1
    
    # 合并节点特征
    merged_features = np.vstack(all_node_features)
    
    # 合并节点标签
    merged_labels = np.concatenate(all_node_labels, axis=0)
    
    # 转换为CSR格式以提高效率
    merged_adj_csr = merged_adj.tocsr()
    
    return {
        'adj_matrix': merged_adj_csr,
        'node_features': merged_features,
        'node_labels': merged_labels,
        'num_nodes': merged_features.shape[0],
        'num_edges': merged_adj_csr.nnz // 2,
        'num_features': merged_features.shape[1],
        'num_graphs': len(all_adj_matrices)
    }


def load_my_mat(directory_path, mat_file):
    
    file_path = os.path.join(directory_path, mat_file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None
    
    data = sio.loadmat(file_path)

    print(f"正在处理文件 {mat_file}...")
    '''print(data.keys())

    print(data['__header__']) # 输出mat文件的头部信息

    print(data['Network'].shape)  # 输出Network的形状
    print(type(data['Network']))  # 输出Network的数据类型  稀疏压缩矩阵<class 'scipy.sparse._csc.csc_matrix'>  indptr indices data
    print(data['Network'].nnz)

    print((data['Network'] != data['Network'].T).nnz == 0) # 检查Network是否为对称矩阵

    print(data['Label'].shape)  # 输出Label的形状
    print(sum(data['Label'].tolist()[0]))  # 输出Label的非零元素个数
    
    
    print(data['Attributes'].shape)  # 输出Attributes的形状
    print(data['Attributes'].tolist()[0])  # 输出Attributes的第一个元素
    '''
    # 提取图数据
    adj_matrix = data['Network']
    node_features = data['Attributes']
    node_labels = [int(nl) for nl in data['Label'][0]]

    return adj_matrix, node_features, node_labels


def merge_graph_datasets(directory_path, file_names):
    """
    读取目录下所有.mat文件并将图数据合并成一张大图
    
    假设每个.mat文件包含:
    - adj_matrix: 邻接矩阵
    - node_features: 节点特征矩阵
    - node_labels: 节点标签
    
    Parameters:
    directory_path (str): 包含图数据.mat文件的目录路径
    
    Returns:
    dict: 合并后的图数据 {adj_matrix, node_features, node_labels}
    """
    if isinstance(file_names, str):
        file_names = [file_names]
    elif not isinstance(file_names, list):
        exit("file_names参数必须是字符串或列表")
    
    # 存储所有图的数据
    all_adj_matrices = []
    all_node_features = []
    all_node_labels = []
    all_graph_masks = []  # 新增：存储每个图的节点掩码信息
    
    # 获取所有.mat文件
    mat_all = [f for f in os.listdir(directory_path) 
                 if f.endswith('.mat') and os.path.isfile(os.path.join(directory_path, f))]
    # ['elliptic_svd.mat', 'amazon_svd.mat', 'reddit_svd.mat', 'tolokers_svd.mat', 'Facebook_svd.mat']
    if file_names is None:
        mat_files = mat_all
    else:
        mat_files = [mat_file for mat_file in mat_all if mat_file in file_names]
        print("找到的文件: ", mat_files)
        if not mat_files or not len(mat_files) == len(file_names):
            raise ValueError(f"目录 {directory_path} 中没有找到域的对应.mat文件")
    
    # 记录节点索引的起始位置
    node_offset = 0

    # 读取所有.mat文件
    for mat_file in mat_files:
        try:
            adj_matrix, node_features, node_labels = load_my_mat(directory_path, mat_file)
            
            if adj_matrix is None or node_features is None or node_labels is None:
                warnings.warn(f"文件 {mat_file} 缺少必要的图数据字段，跳过该文件")
                continue

            # 计算当前图的节点数量
            num_nodes = adj_matrix.shape[0]

            # 创建节点掩码（记录当前图节点在合并图中的索引范围）
            graph_mask = {
                'file_name': mat_file,
                'start_idx': node_offset,
                'num_nodes': num_nodes
            }

            all_adj_matrices.append(adj_matrix)
            all_node_features.append(node_features)
            all_node_labels.append(node_labels)
            all_graph_masks.append(graph_mask)  # 添加节点掩码
            
            # 更新节点偏移量
            node_offset += num_nodes
            
        except Exception as e:
            print(f"读取文件 {mat_file} 时出错: {e}")
            continue
    
    if not all_adj_matrices:
        raise ValueError("没有成功读取到任何有效的图数据")
    
    # 合并成一张大图
    merged_graph = {}
    
    # 1. 合并邻接矩阵（使用块对角矩阵）
    if len(all_adj_matrices) == 1:
        merged_graph['adj_matrix'] = csr_matrix(all_adj_matrices[0])
    else:
        # 创建块对角矩阵，图之间没有连接
        merged_graph['adj_matrix'] = block_diag(all_adj_matrices, format='csc')
    
    # 2. 合并节点特征（垂直连接）
    try:
        if isinstance(all_node_features[0], np.ndarray):
            merged_graph['node_features'] = np.vstack(all_node_features)
        else:
            merged_graph['node_features'] = np.vstack([f.toarray() if hasattr(f, 'toarray') else f 
                                                      for f in all_node_features])
    except Exception as e:
        raise ValueError(f"合并节点特征时出错: {e}")
    
    # 3. 合并节点标签（垂直连接）
    try:
        if isinstance(all_node_labels[0], np.ndarray):
            merged_graph['node_labels'] = np.concatenate(all_node_labels, axis=0)
        else:
            merged_graph['node_labels'] = np.concatenate([l.toarray().flatten() if hasattr(l, 'toarray') 
                                                         else l.flatten() if hasattr(l, 'flatten') else l 
                                                         for l in all_node_labels], axis=0)
            
    except Exception as e:
        raise ValueError(f"合并节点标签时出错: {e}")
    
    # 添加统计信息
    merged_graph['num_nodes'] = merged_graph['node_features'].shape[0]
    merged_graph['num_edges'] = merged_graph['adj_matrix'].nnz # 部分图有向 // 2  # 无向图
    merged_graph['num_features'] = merged_graph['node_features'].shape[1]
    merged_graph['num_graphs'] = len(all_graph_masks)
    merged_graph['graph_masks'] = all_graph_masks
    
    # print('merged_graph.keys()')
    # print(merged_graph.keys())

    return merged_graph
