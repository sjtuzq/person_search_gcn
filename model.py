"""
    create by Yichao Yan
    function: gcn models
"""

import torch
import torch.nn as nn
import os
from torch.nn import functional as F

from config import log_root

class GcnModel_core(nn.Module):
    def __init__(self,train=True,neibor=4):
        super(GcnModel_core,self).__init__()
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

        '''
        self.W1 = nn.Linear(in_features=256, out_features=512)
        self.W1.bias.data.fill_(0.)
        self.W1.bias.requires_grad = False
        self.W2 = nn.Linear(in_features=512, out_features=1024)
        self.W2.bias.requires_grad = False
        self.W2.bias.data.fill_(0.)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=1024, out_features=1024)
        '''
        self.W1 = nn.Linear(in_features=256, out_features=512)
        self.W1.bias.data.fill_(0.)
        self.W1.bias.requires_grad = False
        self.W2 = nn.Linear(in_features=512, out_features=1024)
        self.W2.bias.requires_grad = False
        self.W2.bias.data.fill_(0.)
        self.W3 = nn.Linear(in_features=1024, out_features=2048)
        self.W3.bias.requires_grad = False
        self.W3.bias.data.fill_(0.)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=2048, out_features=2048)
        #self.fc = nn.Linear(in_features=1024, out_features=256)


        if self.train:
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self,gallery,probe):
        # return gallery[:, 0].squeeze(), probe[0, :].squeeze(), self.sim(gallery[:, 0].squeeze(), probe[0, :].unsqueeze(0))

        gal = torch.bmm(self.A.expand(gallery.shape[0], self.neibor, self.neibor), gallery)
        gal = self.relu(self.W1(gal))
        gal = torch.bmm(self.A.expand(gal.shape[0],self.neibor,self.neibor),gal)
        gal = self.relu(self.W2(gal))
        gal = torch.bmm(self.A.expand(gal.shape[0],self.neibor,self.neibor),gal)
        gal = self.relu(self.W3(gal))
        gal = gal[:, 0, :].squeeze()
        gal = self.fc(gal)

        probe = probe.unsqueeze(0).expand(gallery.shape[0], probe.shape[0], probe.shape[1])
        prob = torch.bmm(self.A.expand(probe.shape[0],self.neibor,self.neibor),probe)
        prob = self.relu(self.W1(prob))
        prob = torch.bmm(self.A.expand(prob.shape[0],self.neibor,self.neibor),prob)
        prob = self.relu(self.W2(prob))
        prob = torch.bmm(self.A.expand(prob.shape[0],self.neibor,self.neibor),prob)
        prob = self.relu(self.W3(prob))
        prob = prob[:, 0, :].squeeze()
        prob = self.fc(prob)

        return prob, gal, self.sim(prob, gal)

class GcnModel_n(nn.Module):
    def __init__(self, train=True, neibor=4):
        super(GcnModel_n, self).__init__()
        self.train = train
        self.neibor = neibor

        #self.GCN = nn.ModuleList([GcnModel_core(neibor=i+1) for i in range(neibor)])
        self.GCN = GcnModel_core(neibor=self.neibor)

        if self.train:
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, gallery, probe):
        prob,gal,_ = self.GCN(gallery,probe)
        return prob, gal, self.sim(prob, gal)
