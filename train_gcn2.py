"""
    created by Qiang Zhang
    function: select companies using attention into gcn model input
"""
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from model import GcnModel_n,Attention
from utils import gallery_gcn,padding,Eval
from data import Traindata,getTestdata

from config import log_root,neighbor_num

def Train_one_epoch(model,N=3):
    Gcn = GcnModel_n(neibor=N).cuda()
    file = Gcn.loadfile()
    Gcn.load_state_dict(torch.load(file))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    TrainLoader = Traindata(thresh=0.0,neibor=N)

    loss_all = 0
    all_pred = []
    all_label = []
    for i,item in enumerate(tqdm(TrainLoader)):
        probe1,probe,label,gallery = item
        data = gallery_gcn(probe, gallery, num=N)
        probe = torch.tensor(probe).cuda()
        gallery = padding(gallery)
        gallery = torch.tensor(gallery).float().cuda()

        feat = model(gallery,probe)

        prob, gal, sim = Gcn(feat, probe)
        pred = sim.cpu().data.numpy()
        pred = np.argsort(-pred)
        all_pred.append(pred)
        all_label.append(label)
        target = torch.tensor(data).cuda()

        try:
            loss = F.mse_loss(feat, target,size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            continue
        loss_all += loss.item()
    return model

def Test_one_epoch(epoch,model,N,logdir):
    Gcn = GcnModel_n(neibor=N).cuda()
    file = Gcn.loadfile()
    Gcn.load_state_dict(torch.load(file))
    TestLoader = getTestdata(neibor=N)

    all_pred = []
    all_label = []
    best = 0
    for i,item in enumerate(tqdm(TestLoader)):
        probe, label, gallery_feat = item
        probe = probe.squeeze()
        probe = torch.tensor(probe).cuda()
        gallery = padding(gallery_feat)
        gallery = torch.tensor(gallery).float().cuda()

        feat = model(gallery,probe)

        prob, gal, sim = Gcn(feat, probe)
        pred = sim.cpu().data.numpy()
        pred = np.argsort(-pred)
        all_pred.append(pred)
        all_label.append(label)

    all_pred = np.asarray(all_pred)
    all_label = np.asarray(all_label)

    eval,map = Eval(all_label, all_pred)
    print('epoch:{}  acc:{:.4f}   map:{:.4f}'.format(epoch, eval[0],map))

    if eval[0]>best:
        best = eval[0]

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    np.save(os.path.join(logdir,'log_{}'.format(epoch)),eval)
    logfile = os.path.join(logdir, 'log_gcnnet{}.txt'.format(epoch))
    writer = open(logfile, 'w')
    writer.write("company number:  "+str(N))
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

if __name__ == '__main__':
    model = Attention(neibor=neighbor_num).cuda()
    attention_log = os.path.join(log_root,'attention')
    if not os.path.exists(attention_log):
        os.mkdir(attention_log)
    num = len(os.listdir(attention_log))
    nowTime = datetime.today().strftime("%y%m%d-%H%M%S")
    logdir = os.path.join(attention_log,'{}-{}-{}'.format(num, nowTime, neighbor_num))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    for i in range(10):
        model = Train_one_epoch(model,N=neighbor_num)
        Test_one_epoch(epoch=i,model=model,N=neighbor_num,logdir=logdir)
