import time
import numpy as np
import torch
import logging

from optim.loss import loss_function,init_centers,get_radius,EarlyStopping
from utils.evaluate import fixed_graph_evaluate
import torch.nn.functional as F

def train(args,logger,data,model,checkpoints_path, train_type):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    checkpoints_path=checkpoints_path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    
    # dataname = ['amazon','facebook', 'elliptic']
    dataname = [mask['file_name'].split('_svd.mat')[0] for mask in data['graph_masks']]
    train_name = '-'.join(dataname)

    logger.info(f'-------------------retrainer on target_domain-{train_name}-{train_type}--------------------------')

    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # 我们监测 Loss
        factor=0.1,      # 当触发时，LR 乘以 0.1
        patience=20,     # 20 个 Epoch Loss 没有改善就触发
        verbose=True     # 在控制台打印触发信息
    )
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    else:
        stopper = EarlyStopping(patience=args.n_epochs)
    # initialize data center

    input_feat=data['features']
    input_g=data['g']
    graph_masks=data['graph_masks']

    # data_center= init_center(args,input_g,input_feat, model)
    # radius=torch.tensor(0, device=device)# radius R initialized with 0 by default.
    # overrall_center = init_center(args,input_g,input_feat, model).unsqueeze(0)
    
    #创立矩阵以存储结果曲线
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)
    center_history = []# : list of numpy arrays, 形状为 [n_steps, n_centers, n_features]
    dur = []
    Update_center = 5
    
    logger.info('Start training')

    radius_zero = torch.tensor(0, device=device)
    radius = torch.tensor(0, device=device)

    model.eval()
    with torch.no_grad():
        outputs= model(input_g,input_feat)
        outputs_l2 = outputs# torch.nn.functional.normalize(outputs, p=2, dim=1, eps=1e-12)
        data_centers=init_data_centers= init_centers(args, outputs_l2, graph_masks)
    overrall_center = torch.mean(data_centers,dim=0).unsqueeze(0)
    # TODO V3 test done
    if train_type=='model_target':
        dist_list=[]
        logger.info("load and weight centers")
        init_data_centers = torch.load(f'{args.save_path}/init_data_centers.pth')
        for dc in init_data_centers:
            loss,_,_ = loss_function(args.nu, dc,outputs_l2,radius_zero,graph_masks, data['train_mask'])
            dist_list.append(loss)

        dist_tensor = torch.tensor(dist_list, device=device)
        # 使用Softmax计算权重 -dist_tensor 使得距离越小的，其值越大，权重也就越高。
        weights = F.softmax(-dist_tensor/args.softmax_T, dim=0) 
        weighted_centers = init_data_centers * weights.unsqueeze(1)
        overrall_center = torch.sum(weighted_centers, dim=0).unsqueeze(0)
    
    center_history.append(init_data_centers.detach().cpu().numpy())

    for epoch in range(args.n_epochs):
        if epoch % 2 == 0:
            t0 = time.time()
        # forward
        model.train()
        outputs= model(input_g,input_feat)
        outputs_l2 = outputs# torch.nn.functional.normalize(outputs, p=2, dim=1, eps=1e-12)
        ### 由于作者修改了gin，实际的output层和hidden层维度一样
        loss,dist,_=loss_function(args.nu, overrall_center,outputs_l2,radius_zero,graph_masks, data['train_mask'])
        radius=torch.tensor(get_radius(dist, 1.0), device=device)  # one class设置为1
        
        arr_loss[epoch]=loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 2 == 0:
            dur.append(time.time() - t0)
        
        data_centers= init_centers(args, outputs_l2.detach(), graph_masks)
        center_history.append(data_centers.detach().cpu().numpy())
        '''if args.dynamic_center and epoch % Update_center == 0:
            overrall_center = data_centers.detach()'''

        auc,ap,f1,acc,precision,recall,val_loss = fixed_graph_evaluate(args, model, overrall_center,data,radius,graph_masks, data['val_mask'])
        #保存验证集AUC
        arr_valauc[epoch]=auc

        if epoch % 2 == 0:
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                            val_loss.item()*100000, auc, ap, data['n_edges'] / np.mean(dur) / 1000))
            logger.info(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')

        
        if args.early_stop:
            if stopper.step(acc,auc,ap,val_loss.item(), model,epoch,checkpoints_path):
                break

    logger.info(f'train on {train_name}, {train_type}')

    # checkpoint保存最好的模型
    if args.early_stop:
        best_epoch = stopper.best_epoch
        # logger.info("stopper stored best epoch {:05d} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f}". format(best_epoch, stopper.lowest_loss,stopper.auc,stopper.ap))
        model.load_state_dict(torch.load(checkpoints_path))
    else:
        best_epoch = epoch
        torch.save(model.state_dict(),checkpoints_path)
    
    auc,ap,f1,acc,precision,recall,val_loss = fixed_graph_evaluate(args, model, overrall_center,data,radius,graph_masks, data['val_mask'])
    logger.info("Best epoch {:05d} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f}". format(best_epoch, val_loss.item()*100000,auc,ap))
    logger.info(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')

    t0 = time.time()
    auc,ap,f1,acc,precision,recall,test_loss = fixed_graph_evaluate(args,model, overrall_center,data,radius,graph_masks, data['test_mask'])
    test_dur = time.time() - t0

    #保存测试集AUC
    arr_testauc[epoch]=auc

    logger.info("Current epoch: {:d} Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(best_epoch, test_dur,auc,ap))
    logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    # logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    # logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    logger.info('\n')

    #np.savez('SAGE-2.npz',epoch=arr_epoch,loss=arr_loss,valauc=arr_valauc,testauc=arr_testauc)
    # plot_loss_curve(arr_loss[0:epoch])
    
    # visualize_centers_tsne_static(center_history)
    # visualize_centers(center_history)
    # plot_centers(np.array(center_history))

    ll = data['labels'].detach().cpu().numpy()
    model.eval()
    outputs= model(data['g'],data['features'])
    outputs_l2 = outputs# torch.nn.functional.normalize(outputs, p=2, dim=1, eps=1e-12)
    outputs_cut = outputs_l2.detach().cpu()
    outputs_normalized = (outputs_cut - outputs_cut.min()) / (outputs_cut.max() - outputs_cut.min() + 1e-16)
    ff = outputs_normalized.numpy()

    '''one_class_affinity('one-class-one-center_retrainer_'+train_name, dataname[0], input_g.adj(scipy_fmt='csr'), ff, ll, 
                      center_history[best_epoch], radius, logger,base_path=args.save_path)'''

    return model


def feature_alignment(args,model,data,pt_path): 
    torch.cuda.set_device(args.gpu)
    model.load_state_dict(torch.load(pt_path))
    model.eval()
    outputs= model(data['g'],data['features'])
    outputs_l2 = torch.nn.functional.normalize(outputs, p=2, dim=1, eps=1e-12)
    outputs = outputs_l2.detach().cpu()
    outputs_normalized = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-16)
    outputs_normalized = outputs_normalized.numpy()
    data['features'] = torch.FloatTensor(outputs_normalized).cuda()
    data['input_dim'] = data['features'].shape[1]
    return data

