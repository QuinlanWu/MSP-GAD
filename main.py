import argparse
import torch
import logging
#import fire
from optim import MD_trainer, retrainer
from data_process import dataloader
from networks.init import init_model
import numpy as np
import torch
from dgl import random as dr
import os


def main(args):
    if args.seed!=-1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        dr.seed(args.seed)

    logger = logging.getLogger('MSP-GAD')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # print('model:',args.module)
    # print('seed:',args.seed)

    # 多域协同训练
    dataname = [sd.split('_svd.mat')[0] for sd in args.source_domain]
    source_name = '-'.join(dataname)
    # target_name = args.target_domain.split('_svd.mat')[0]
    code_edition = "MD"
    if args.dynamic_center:
        code_edition = "dynamicMD"
    save_path = os.path.join(args.save_dir, code_edition, 'usemodel_'+str(args.edition)+'_'+source_name)
    os.makedirs(save_path, exist_ok=True)
    args.save_path = save_path

    checkpoints_source=f'{save_path}/source_{source_name}_{args.module}+checkpoint.pt'
    # source_data=dataloader.loader(args, "source")
    source_data=[]
    source_domain = args.source_domain
    for source in source_domain:
        args.source_domain = [source]
        source_data.append(dataloader.loader(args, "source"))
    args.source_domain = source_domain

    '''visualize_source_2d(source_data,'t-SNE Visualization of raw source fat_'+str(args.edition))
    source_new = []
    test_model = init_model(args,8)
    test_model.load_state_dict(torch.load(checkpoints_source))
    for data in source_data:
        data_new = my_MD_trainer.feature_alignment(args,test_model,data, checkpoints_source)
        source_new.append(data_new)
    visualize_source_2d(source_new,'t-SNE Visualization of aligned source fat_'+str(args.edition))
    # visualize_source_3d_plotly(source_new,'t-SNE Visualization of aligned source fat')

    '''
    
    logfile = logging.FileHandler(f"{save_path}/source_{source_name}_{args.module}.log", 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    logger.info(f'source-domain:{args.source_domain}, target-domain:{args.target_domain},dynamic-center:{args.dynamic_center}')
    # 训练比例切换，预训练需要提高比例
    train_ratio = args.train_ratio
    args.train_ratio = 0.8
    source_model=init_model(args,8)
    source_model=MD_trainer.train(args,logger,source_data,source_model,checkpoints_source)
    logger.removeHandler(logfile)
    logfile.close()
    args.train_ratio = train_ratio

    # 预训练完成后，在目标域上训练
    args.early_stop = True
    target_domain = args.target_domain
    for i, domain in enumerate(target_domain):
        dataname = domain.split('_svd.mat')[0]
        args.target_domain = domain
        target_raw = dataloader.loader(args, "target")

        '''
        mymodel = args.module
        args.module = "GIN"
        train_mode = 'raw'
        checkpoints_raw=f'{save_path}/{train_mode}_{dataname}_{args.module}+bestcheckpoint.pt'
        logfile_raw = logging.FileHandler(f"{save_path}/{train_mode}_{dataname}_{args.module}.log", 'w')
        logfile_raw.setFormatter(fmt)
        logger.addHandler(logfile_raw)
        # 单个域原始数据集训练    即原始的deep-svdd
        raw_model = init_model(args,target_raw['input_dim'])
        return_v=retrainer.train(args,logger,target_raw,raw_model,checkpoints_raw, train_mode)
        logger.removeHandler(logfile_raw)
        logfile_raw.close()
        args.module = mymodel
        '''

        train_mode = 'model_target'
        checkpoints_target=f'{save_path}/{train_mode}_{dataname}_{args.module}+checkpoint.pt'
        logfile_new_model = logging.FileHandler(f"{save_path}/{train_mode}_{dataname}_{args.module}.log", 'w')
        logfile_new_model.setFormatter(fmt)
        logger.addHandler(logfile_new_model)
        # 
        target_model = init_model(args,target_raw['input_dim'])
        target_model.load_state_dict(torch.load(checkpoints_source))
        return_v=retrainer.train(args,logger,target_raw,target_model,checkpoints_target, train_mode)
        logger.removeHandler(logfile_new_model)
        logfile_new_model.close()

        train_mode = 'data_target'
        checkpoints_target=f'{save_path}/{train_mode}_{dataname}_{args.module}+checkpoint.pt'
        logfile_new_data = logging.FileHandler(f"{save_path}/{train_mode}_{dataname}_{args.module}.log", 'w')
        logfile_new_data.setFormatter(fmt)
        logger.addHandler(logfile_new_data)
        # 数据对齐后维度改变    使用协同训练模型输出的节点特征重新训练
        target_new = retrainer.feature_alignment(args,source_model,target_raw, checkpoints_source)  # 
        target_model = init_model(args,target_new['input_dim'])
        return_v=retrainer.train(args,logger,target_new,target_model,checkpoints_target, train_mode)
        logger.removeHandler(logfile_new_data)
        logfile_new_data.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCGNN')
    
    parser.add_argument('--source-domain', nargs='+', type=str, 
                   default=['Facebook_svd.mat', 'amazon_svd.mat' , 'elliptic_svd.mat'],# "tolokers_svd.mat"],
                   help='Dataset name')
    parser.add_argument('--target-domain', type=str, 
                        default=['Facebook_svd.mat', 'amazon_svd.mat', 't_finance_svd.mat'],
                        choices=['Facebook_svd.mat', 'amazon_svd.mat', 't_finance_svd.mat'],
                        help='Dataset name')
    parser.add_argument('--target-domainn', type=str, 
                        default=['Facebook_svd.mat', 'questions_svd.mat', 'reddit_svd.mat', 
                                 'AmazonUpu_svd.mat', 'amazon_svd.mat', 'yelp_svd.mat', 
                                 'Disney_svd.mat', 't_finance_svd.mat', 'elliptic_svd.mat', 'tolokers_svd.mat'],
                        choices=['Facebook_svd.mat', 'questions_svd.mat', 'reddit_svd.mat', 
                                 'AmazonUpu_svd.mat', 'amazon_svd.mat', 'yelp_svd.mat', 
                                 'Disney_svd.mat', 't_finance_svd.mat', 'elliptic_svd.mat', 'tolokers_svd.mat'],
                        help='Dataset name')
    parser.add_argument("--edition", type=int, default=0)
    parser.add_argument('--data-path', type=str, default='./dataset_GFM/',
                        help='Path to the .mat file for custom dataset')
    parser.add_argument('--save-dir', type=str, default='./dump/',
                        help='save model directory')
    parser.add_argument('--save-path', type=str, default='./dump/',
                        help='save model path, will be overwritten')
    parser.add_argument('--force-reload', action='store_true', default=False,
                        help='Force reload and reprocess the data')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose output')    
    parser.add_argument('--dynamic-center', action='store_true', default=False,
                        help='update center each epoch, may be divergent')
    parser.add_argument("--train-ratio", type=float, default=0.8,
            help="train ratio")
    parser.add_argument("--softmax-T", type=float, default=0.1,
            help="softmax temperature")
   

    parser.add_argument("--dropout", type=float, default=0.3,
            help="dropout probability")
    parser.add_argument("--nu", type=float, default=0.1,
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--seed", type=int, default=52,
            help="random seed, -1 means dont fix seed")
    parser.add_argument("--module", type=str, default='GCN',
            help="GCN/GAT/GIN/GraphSAGE/GAE")
    parser.add_argument('--n-worker', type=int,default=1,
            help='number of workers when dataloading')
    parser.add_argument('--batch-size', type=int,default=128,
            help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--normal-class", type=int, default=0,
            help="normal class")
    parser.add_argument("--n-epochs", type=int, default=301,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
            help="graph normalization (default=False)")
    parser.set_defaults(self_loop=False)
    parser.set_defaults(norm=False)
    
    args = parser.parse_args()
    if args.module=='GIN':
        args.lr=0.0001
    if args.module=='GCN':
        #args.self_loop=True
        args.norm=True
    if args.module=='GAT':
        args.self_loop=True
    if args.module=='GAE':
        args.lr=0.002
        args.dropout=0.
        args.weight_decay=0.

    args.normal_class=0

    print(args)
    main(args)
