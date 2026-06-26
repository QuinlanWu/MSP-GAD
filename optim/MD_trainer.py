import time
import numpy as np
import torch
import logging

from optim.loss import loss_function,init_centers,get_radius,EarlyStopping,anomaly_score_mc
from utils.evaluate import fixed_graph_evaluate
from utils.plot_culve import plot_loss_curve


def train(args,logger,data,model,checkpoints_path):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    logger.info(f'-------------------trainer on source_domain---------------------------')
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    logger.info(f"Using device: {device}")
    checkpoints_path=checkpoints_path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    

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

    '''input_feat=data['features']
    input_g=data['g']
    graph_masks=data['graph_masks']
    
    # dataname = ['amazon','facebook', 'elliptic']
    dataname = [mask['file_name'].split('_svd.mat')[0] for mask in graph_masks]
    train_name = '-'.join(dataname)

    data_MD = []
    for i, graph_mask in enumerate(graph_masks):
        mymask = torch.zeros(data['features'].shape[0], dtype=torch.bool, device=device)
        start_idx = graph_mask['start_idx']
        num_nodes = graph_mask['num_nodes']
        mymask[start_idx:start_idx + num_nodes] = True

        node_list = list(range(graph_mask['start_idx'],graph_mask['start_idx']+graph_mask['num_nodes']))

        label_cut = data['labels'][node_list]
        
        feature_cut = data['features'][node_list]


        sub_g = input_g.subgraph(node_list)
        ajd_matrix = sub_g.adj(scipy_fmt='csr')
        train_mask = data['train_mask'][node_list]
        test_mask = data['test_mask'][node_list]
        val_mask = data['val_mask'][node_list]

        datadict={'g':sub_g,'features':feature_cut,'labels':label_cut,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'input_dim':feature_cut.shape[1],'n_classes':data['n_classes'],'n_edges':sub_g.number_of_edges(),
        'graph_masks':[graph_mask]}
        data_MD.append(datadict)'''
    
    #创立矩阵以存储结果曲线
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)
    center_history = []# : list of numpy arrays, 形状为 [n_steps, n_centers, n_features]
    dur = []
    Update_center = 5
    
    logger.info('Start training')
    dc_list=[]
    model.eval()
    with torch.no_grad():
        for d in data:
            input_g = d['g']
            input_feat = d['features']
            graph_mask = d['graph_masks']
            outputs= model(input_g,input_feat)
            outputs_l2 = torch.nn.functional.normalize(outputs, p=2, dim=1, eps=1e-12)
            print(outputs_l2.shape)
            dc_list.append(init_centers(args, outputs_l2, graph_mask))
    data_centers=init_data_centers= torch.cat(dc_list, dim=0)
    overrall_center = torch.mean(data_centers,dim=0).unsqueeze(0)
    center_history.append(init_data_centers.detach().cpu().numpy())
    torch.save(init_data_centers, f'{args.save_path}/init_data_centers.pth')# 保存初始化数据集的初始中心

    radius_zero = torch.tensor(0, device=device)
    radiuss = torch.zeros(len(data),1, device=device)
    
    for epoch in range(201):
        model.train()
        if epoch % 10 == 0:
            t0 = time.time()
        # forward
        model.train()
        loss = []
        for i, d in enumerate(data):
            input_gi = d['g']
            input_feati = d['features']
            label_cuti = d['labels']
            train_maski = d['train_mask']
            test_maski = d['test_mask']
            val_maski = d['val_mask']
            graph_maski = d['graph_masks']
            centeri = init_data_centers[i]
            radiusi = radiuss[i]
            outputsi= model(input_gi,input_feati)
            outputsi_l2 = torch.nn.functional.normalize(outputsi, p=2, dim=1, eps=1e-12)
            ### 由于作者修改了gin，实际的output层和hidden层维度一样
            lossi,disti,_=loss_function(args.nu, centeri,outputsi_l2,radius_zero,graph_maski, train_maski)
            radiusi=torch.tensor(get_radius(disti, 1.0), device=device).detach()
            centeri =(torch.mean(outputsi_l2, dim=0)).detach()

            loss.append(lossi)
            radiuss[i] = radiusi
            data_centers[i] = centeri
        loss = torch.stack(loss).mean()
        arr_loss[epoch]=loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 10 == 0:
            dur.append(time.time() - t0)
        # TODO V4 test done
        if args.dynamic_center and epoch % Update_center == 0 and len(data) > 1:
            init_data_centers = data_centers

        center_history.append(data_centers.detach().cpu().numpy())
        # cd = centers_distances(data_centers).cpu().tolist()
        # logger.info(f'centers distance:{[cd[0][1],cd[0][2],cd[1][2]]}')

        results = []
        for i, d in enumerate(data):
            input_gi = d['g']
            input_feati = d['features']
            label_cuti = d['labels']
            train_maski = d['train_mask']
            test_maski = d['test_mask']
            val_maski = d['val_mask']
            graph_maski = d['graph_masks']
            centeri = init_data_centers[i]
            radiusi = radiuss[i]
            
            auc,ap,f1,acc,precision,recall,val_loss = fixed_graph_evaluate(args, model, centeri,d,radiusi,graph_maski, val_maski)
            results.append([auc, ap, f1, acc, precision, recall, np.float64(val_loss.detach().cpu().item())])

            '''if epoch % 10 == 0:
                logger.info(f"data {graph_maski[0]['file_name']}")
                logger.info("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                                val_loss.item()*100000, auc, ap, d['n_edges'] / np.mean(dur) / 1000))
                logger.info(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')'''
        
        # 转换为numpy数组并计算每列的平均值
        results_array = np.array(results)
        avg_auc, avg_ap, avg_f1, avg_acc, avg_precision, avg_recall, avg_val_loss = np.mean(results_array, axis=0)
        arr_valauc[epoch] = avg_auc
        if epoch % 10 == 0:
            logger.info("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f} | "
            "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                            avg_val_loss.item()*100000, avg_auc, avg_ap, d['n_edges'] / np.mean(dur) / 1000))
            logger.info(f'Val f1:{round(avg_f1,4)},acc:{round(avg_acc,4)},pre:{round(avg_precision,4)},recall:{round(avg_recall,4)}')
        if args.early_stop:
            if stopper.step(avg_acc,avg_auc,avg_ap,avg_val_loss.item(), model,epoch,checkpoints_path):
                # torch.save(model.state_dict(), checkpoints_path) stopper已经保存了
                break

    # checkpoint保存最好的模型
    if args.early_stop:
        best_epoch = stopper.best_epoch
        # logger.info("stopper best epoch {:05d} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f}". format(best_epoch, stopper.lowest_loss,stopper.auc,stopper.ap))
        model.load_state_dict(torch.load(checkpoints_path))
    else:
        best_epoch = epoch
        torch.save(model.state_dict(),checkpoints_path)
    
    if args.dynamic_center:
        torch.save(init_data_centers, f'{args.save_path}/final_data_centers.pth')# 保存数据集的最终中心
    
    results = []
    for i, d in enumerate(data):
        input_gi = d['g']
        input_feati = d['features']
        label_cuti = d['labels']
        train_maski = d['train_mask']
        test_maski = d['test_mask']
        val_maski = d['val_mask']
        graph_maski = d['graph_masks']
        centeri = init_data_centers[i]
        radiusi = radiuss[i]
        auc,ap,f1,acc,precision,recall,val_loss = fixed_graph_evaluate(args, model, centeri,d,radiusi,graph_maski, val_maski)
        results.append([auc, ap, f1, acc, precision, recall, np.float64(val_loss.detach().cpu().item())])
    results_array = np.array(results)
    avg_auc, avg_ap, avg_f1, avg_acc, avg_precision, avg_recall, avg_val_loss = np.mean(results_array, axis=0)
    logger.info("Best epoch {:05d} | Val Loss {:.4f} | Val AUROC {:.4f} | Val AUPRC {:.4f}". format(best_epoch, avg_val_loss.item()*100000,avg_auc,avg_ap))
    logger.info(f'Val f1:{round(avg_f1,4)},acc:{round(avg_acc,4)},pre:{round(avg_precision,4)},recall:{round(avg_recall,4)}')

    t0 = time.time()
    results = []
    for i, d in enumerate(data):
        input_gi = d['g']
        input_feati = d['features']
        label_cuti = d['labels']
        train_maski = d['train_mask']
        test_maski = d['test_mask']
        val_maski = d['val_mask']
        graph_maski = d['graph_masks']
        centeri = init_data_centers[i]
        radiusi = radiuss[i]
        auc,ap,f1,acc,precision,recall,test_loss = fixed_graph_evaluate(args, model, centeri,d,radiusi,[graph_maski], test_maski)
        results.append([auc, ap, f1, acc, precision, recall, np.float64(test_loss.detach().cpu().item())])
    results_array = np.array(results)
    avg_auc, avg_ap, avg_f1, avg_acc, avg_precision, avg_recall, avg_loss = np.mean(results_array, axis=0)
    test_dur = time.time() - t0
    #保存测试集AUC
    arr_testauc[epoch]=avg_auc
    logger.info("Current epoch: {:d} Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(best_epoch, test_dur,avg_auc,avg_ap))
    logger.info(f'Test f1:{round(avg_f1,4)},acc:{round(avg_acc,4)},pre:{round(avg_precision,4)},recall:{round(avg_recall,4)}')
    logger.info('\n')

    #np.savez('SAGE-2.npz',epoch=arr_epoch,loss=arr_loss,valauc=arr_valauc,testauc=arr_testauc)
    plot_loss_curve(arr_loss[0:epoch])

    # visualize_centers_tsne_static(center_history)
    # visualize_centers(center_history)
    # plot_centers(np.array(center_history))

    logger.info('----------source domain test----------')        
    # ['elliptic_svd.mat', 'amazon_svd.mat', 'reddit_svd.mat', 'tolokers_svd.mat', 'Facebook_svd.mat']0,1,4
    dataname = [d['graph_masks'][0]['file_name'].split('_svd.mat')[0] for d in data]
    train_name = '-'.join(dataname)
    model.eval()
    for i,d in enumerate(data):
        input_gi = d['g']
        input_feati = d['features']
        input_labeli = d['labels']
        train_maski = d['train_mask']
        test_maski = d['test_mask']
        val_maski = d['val_mask']
        graph_maski = d['graph_masks']
        centeri = init_data_centers[i]
        
        outputs= model(input_gi,input_feati)
        outputs_l2 = torch.nn.functional.normalize(outputs, p=2, dim=1, eps=1e-12)

        dist,scores=anomaly_score_mc(centeri,outputs_l2,radius_zero)
        radius=get_radius(dist, 1.0)
        logger.info(f"source domain {graph_maski[0]['file_name']} radius:{radius}")

        auc,ap,f1,acc,precision,recall,loss = fixed_graph_evaluate(args,model, centeri,d,radius,graph_maski,None)
        logger.info(f"source domain {graph_maski[0]['file_name']} Time {0.0:.4f} | onedomain AUROC {auc:.4f} | onedomain AUPRC {ap:.4f}")
        logger.info(f'source domain f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')

        '''one_class_affinity('one-class-one-center_'+train_name, dataname[i], 
                    input_gi.adj(scipy_fmt='csr'), 
                    outputs_l2.detach().cpu().numpy(), 
                    input_labeli.detach().cpu().numpy(), 
                    np.array([center_history[best_epoch][i]]), radius, logger,base_path=args.save_path)'''

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



