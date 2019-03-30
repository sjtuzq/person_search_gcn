"""
    create by Qiang Zhang
    function: gcn models
"""
import torch
import torch.nn as nn
import os

from config import log_root

class GcnModel_n(nn.Module):
    def __init__(self,train=True,neibor=4):
        super(GcnModel_n,self).__init__()
        self.train = train
        self.neibor = neibor

        a = torch.pow(torch.tensor(2. / self.neibor), 0.5)
        self.A = torch.zeros(self.neibor,self.neibor)
        self.A[0][0] = 1.
        for i in range(self.neibor-1):
            self.A[0][i+1] = a
            self.A[i+1][0] = 1./a
            self.A[i+1][i+1] = 1.
        self.A = self.A.unsqueeze(0).cuda()

        self.A.requires_grad = False
        self.W1 = nn.Linear(in_features=256+64,out_features=512)
        self.W1.bias.data.fill_(0.)
        self.W1.bias.requires_grad = False
        self.W2 = nn.Linear(in_features=512,out_features=1024)
        self.W2.bias.requires_grad = False
        self.W2.bias.data.fill_(0.)

        self.W3 = nn.Linear(in_features=512, out_features=64)
        self.W3.bias.requires_grad = False
        self.W3.bias.data.fill_(0.)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=1024,out_features=1024)

        if self.train:
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        self.featfc = nn.Linear(in_features=1024,out_features=512)


    def forward(self,gallery,probe):
        probe = probe.unsqueeze(0).expand(gallery.shape[0], probe.shape[0], probe.shape[1])
        feat = torch.cat((probe, gallery), 2)

        feat = torch.bmm(self.A.expand(feat.shape[0], self.neibor, self.neibor), feat)
        feat = self.relu(self.W3(feat))

        gal = torch.cat((gallery, feat), 2)
        gal = torch.bmm(self.A.expand(gal.shape[0], self.neibor, self.neibor), gal)
        gal = self.relu(self.W1(gal))
        gal = torch.bmm(self.A.expand(gal.shape[0],self.neibor,self.neibor),gal)
        gal = self.relu(self.W2(gal))
        gal = gal[:,0,:].squeeze()
        gal = self.fc(gal)

        probe = torch.cat((probe, feat), 2)
        prob = torch.bmm(self.A.expand(probe.shape[0],self.neibor,self.neibor),probe)
        prob = self.relu(self.W1(prob))
        prob = torch.bmm(self.A.expand(prob.shape[0],self.neibor,self.neibor),prob)
        prob = self.relu(self.W2(prob))
        prob = prob[:,0,:].squeeze()
        prob = self.fc(prob)

        return prob,gal,self.sim(prob,gal)


    def loadfile(self):
        collect_path = os.path.join(log_root,'collect_models')
        return os.path.join(collect_path,"model_{}.pkl".format(self.neibor))
