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
from utils import gallery_abc,gallery_gcn,gallery_gcn_det,_compute_iou,Eval
from data import Traindata,getTestdata

from config import log_root,data_root,neighbor_num

def train_one_epoch(Loader,model,optimizer,margin,N=4):
    loss_all,all_pred,all_label = 0,[],[]
    for i,item in enumerate(tqdm(Loader)):
        probe1,probe,label,gallery = item
        data = gallery_gcn(probe,gallery,num=N)
        data,probe,label = torch.tensor(data).cuda(),\
             torch.tensor(probe).cuda(),torch.tensor(label).cuda()

        prob,gal,sim = model(data,probe)

        pred = sim.cpu().data.numpy()
        pred = np.argsort(-pred)
        all_pred.append(pred)
        all_label.append(label.cpu().data.numpy())

        target = (torch.tensor(label) * 2 - 1).float().cuda()

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
    all_label,all_pred,all_pred0,all_det = [],[],[],[]
    num = 100
    det_all = np.load(os.path.join(data_root,'testdata/testdata_gt.npy'),encoding='latin1')
    for i, item in enumerate(tqdm(dataset)):
        probe, label, gallery_feat = item
        probe = probe.squeeze()

        gcn_data,det = gallery_gcn_det(probe, gallery_feat,num=N)
        galdet,gt = det_all[i]
        det_info,det = [],det[0]
        for j in range(len(label)):
            if label[j]==0:
                det_info.append(0)
            else:
                roi = gt[j]
                roi[2:] += roi[:2]
                iou = _compute_iou(galdet[j][det[j]],roi)
                if iou>0.5:
                    det_info.append(1)
                else:
                    det_info.append(0)

        gcn_data = torch.tensor(gcn_data[0:num]).cuda()
        probe = torch.tensor(probe).cuda()
        prob, gal, sim = model(gcn_data, probe)
        pred = sim.cpu().data.numpy()
        pred = np.argsort(-pred)

        all_pred.append(pred)
        all_label.append(label)

    all_pred = np.asarray(all_pred)
    all_label = np.asarray(all_label)

    eval, map = Eval(all_label, all_pred)
    print('epoch:{}  acc:{:.4f}   map:{:.4f}'.format(epoch, eval[0], map))

    if eval[0] > best:
        best = eval[0]

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    np.save(os.path.join(logdir, 'log_{}'.format(epoch)), eval)
    logfile = os.path.join(logdir, 'log_gcnnet{}.txt'.format(epoch))
    writer = open(logfile, 'w')
    writer.write("company number:  " + str(N))
    writer.write('\n')
    ids = [1, 2, 3, 4, 5, 10, 20]
    for id in ids:
        writer.write('top_{}   acc:{:.4f}\n'.format(id, eval[id - 1]))
    writer.close()

    logpic = os.path.join(logdir, 'picgcnnet{}.jpg'.format(epoch))
    plt.plot(eval, 'b')
    plt.savefig(logpic)
    plt.cla()

    return best


def main(learning_rate=0.1,margin=0.6,N=4):
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
    for epoch in range(20):
        if epoch==10:
            learning_rate = learning_rate / 2
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                   model.parameters()), lr=learning_rate)
        if epoch==15:
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
