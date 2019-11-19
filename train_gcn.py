"""
    create by Qiang Zhang
    function: train gcn model
"""
import torch
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GcnModel_n
from utils import Eval_det
from utils import gallery_gcn, gallery_gcn_det,_compute_iou,Eval
from data import Traindata,getTestdata
import torch.nn as nn

from config import log_root,data_root,neighbor_num

def train_one_epoch(Loader,model,optimizer,margin,N=4):
    loss_all,all_pred,all_label = 0,[],[]
    for i,item in enumerate(tqdm(Loader)):
        probe1,probe,label,gallery = item
        data = gallery_gcn(probe,gallery,num=N)
        data,probe,label = torch.tensor(data).cuda(),\
             torch.tensor(probe).cuda(),torch.tensor(label).cuda()

        prob,gal,sim = model(data,probe)

        #target = (torch.tensor(label) * 2 - 1).float().cuda()
        target = (label.clone().detach() *2 - 1).float().cuda()

        try:
            loss = F.cosine_embedding_loss(prob, gal, target, margin)
        except:
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    return model

def test_one_epoch(epoch,dataset,model,logdir,best,N=4):
    aps = []
    accs = []
    aps_p = []
    accs_p = []
    topk = [1, 5, 10]
    recall_rates = np.load(os.path.join(data_root,'recall.npy'))

    for i, item in enumerate(tqdm(dataset)):
        probe, label, gallery_feat = item
        probe = probe.squeeze()

        gcn_data = gallery_gcn_det(probe, gallery_feat,num=N)

        gcn_data0 = gcn_data[:, 0, :].squeeze()
        sim0 = gcn_data0.dot(probe[0, :].squeeze()).ravel()

        gcn_data = torch.tensor(gcn_data).cuda()
        probe = torch.tensor(probe).cuda()

        prob, gal, sim = model(gcn_data, probe)
        pred = sim.cpu().data.numpy()
        pred_p = pred + sim0

        ap_p, acc_p = Eval(label, pred_p, topk)

        aps_p.append(ap_p*recall_rates[i])
        accs_p.append(acc_p)

    map = np.mean(aps_p)
    print('  mAP = {:.2%}'.format(map))
    accs = np.mean(accs_p, axis=0)
    for i, k in enumerate(topk):
        print('  top-{:2d} = {:.2%}'.format(k, accs[i]))

    #print('epoch:{}  acc:{:.4f}   map:{:.4f}'.format(epoch, accs[0], map))

    if accs[0] > best:
        best = accs[0]

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    np.save(os.path.join(logdir, 'log_{}'.format(epoch)), accs)
    logfile = os.path.join(logdir, 'log_gcnnet{}.txt'.format(epoch))
    writer = open(logfile, 'w')
    writer.write("company number:  " + str(N))
    writer.write('\n')
    writer.write('map:{:.4f}\n'.format(map))
    ids = topk
    for i, topi in enumerate(ids):
        writer.write('top_{}   acc:{:.4f}\n'.format(topi, accs[i]))
    writer.close()

    return best


def test_zero(epoch,dataset,logdir,N=4):
    recall_rates = np.load(os.path.join(data_root,'recall.npy'))
    aps = []
    accs = []
    topk = [1, 5, 10]
    for i, item in enumerate(tqdm(dataset)):
        probe, label, gallery_feat = item
        probe = probe.squeeze()

        gcn_data = gallery_gcn_det(probe, gallery_feat, num=N)

        gcn_data = gcn_data[:, 0, :].squeeze()

        sim = gcn_data.dot(probe[0,:].squeeze()).ravel()
        pred = sim

        ap, acc = Eval(label, pred, topk)

        aps.append(ap*recall_rates[i])
        accs.append(acc)

    print('  mAP = {:.2%}'.format(np.mean(aps)))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print('  top-{:2d} = {:.2%}'.format(k, accs[i]))

    return 0

def main(learning_rate=0.1,margin=0.4,N=4):
    model = GcnModel_n(neibor=N).cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    TrainLoader = Traindata(thresh=0.0,neibor=N)
    TestLoader = getTestdata(neibor=N)
    gcnnet_log = os.path.join(log_root,'gcnnet')
    if not os.path.exists(gcnnet_log):
        os.mkdir(gcnnet_log)
    num = len(os.listdir(gcnnet_log))
    nowTime = datetime.today().strftime("%y%m%d-%H%M%S")
    logdir = os.path.join(log_root,'gcnnet/{}-{}-{}'.format(num,nowTime,N))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    collect_path = os.path.join(log_root, 'collect_models')
    if not os.path.exists(collect_path):
        os.mkdir(collect_path)

    best = 0
    #test_zero(0, TestLoader, logdir, N=N)
    for epoch in range(20):
        if epoch==10:
            learning_rate = learning_rate / 2
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model.parameters()), lr=learning_rate)
        print("epoch:{}".format(epoch))

        model = train_one_epoch(TrainLoader,model,optimizer,margin,N=N)
        torch.save(model.state_dict(),os.path.join(logdir,'model_{}.pkl'.format(epoch)))
        torch.save(model.state_dict(),os.path.join(collect_path,'model_{}.pkl'.format(N)))
        best = test_one_epoch(epoch,TestLoader,model,logdir,best,N=N)

if __name__ == '__main__':
    main(N=neighbor_num)
