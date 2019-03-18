"""
    create by Qiang Zhang
    function: gcn models
"""
import torch
import torch.nn as nn
import os

from config import log_root

# class abcNet(nn.Module):
#     def __init__(self):
#         super(abcNet,self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=3,out_features=10),
#             nn.ReLU(),
#             nn.Linear(in_features=10,out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32,out_features=1)
#         )
#
#     def forward(self,input):
#         return self.fc(input).squeeze()
#
# class RawGcn(nn.Module):
#     def __init__(self):
#         super(RawGcn,self).__init__()
#         a = torch.pow(torch.tensor(6.),-0.5)
#         self.A = torch.tensor([
#             [1/3.,a,a],
#             [a,0.5,0],
#             [a,0,0.5]]).unsqueeze(0).cuda()
#         self.A.requires_grad = False
#         self.W1 = nn.Linear(in_features=256,out_features=512)
#         self.W1.bias.data.fill_(0.)
#         self.W1.bias.requires_grad = False
#         self.W2 = nn.Linear(in_features=512,out_features=1024)
#         self.W2.bias.requires_grad = False
#         self.W2.bias.data.fill_(0.)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(1024,5532)
#
#     def forward(self,input):
#         input = torch.bmm(self.A.expand(input.shape[0],3,3),input)
#         input = self.relu(self.W1(input))
#         input = torch.bmm(self.A.expand(input.shape[0],3,3),input)
#         input = self.relu(self.W2(input))
#         outf = input[:,0,:]
#         out = self.fc(outf)
#         return F.log_softmax(out,dim=1),outf
#
# class NewGcn(nn.Module):
#     def __init__(self,train=True):
#         super(NewGcn,self).__init__()
#         self.train = train
#         a = torch.pow(torch.tensor(6.),-0.5)
#         self.A = torch.tensor([
#             [1/3.,a,a],
#             [a,0.5,0],
#             [a,0,0.5]]).unsqueeze(0).cuda()
#         self.A.requires_grad = False
#         self.W1 = nn.Linear(in_features=256,out_features=256)
#         self.W1.bias.data.fill_(0.)
#         self.W1.bias.requires_grad = False
#         self.W2 = nn.Linear(in_features=256,out_features=256)
#         self.W2.bias.requires_grad = False
#         self.W2.bias.data.fill_(0.)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(in_features=256,out_features=256)
#
#         self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#         self.sim0 = nn.CosineSimilarity(dim=0, eps=1e-6)
#
#     def forward(self,gallery,probe):
#         gal = torch.bmm(self.A.expand(gallery.shape[0],3,3),gallery)
#         gal = self.relu(self.W1(gal))
#         gal = torch.bmm(self.A.expand(gal.shape[0],3,3),gal)
#         gal = self.relu(self.W2(gal))
#         gal = gal[:,0,:].squeeze()
#         gal = self.fc(gal)
#
#         probe = probe.unsqueeze(0).expand(gallery.shape[0],probe.shape[0],probe.shape[1])
#         prob = torch.bmm(self.A.expand(probe.shape[0],3,3),probe)
#         prob = self.relu(self.W1(prob))
#         prob = torch.bmm(self.A.expand(prob.shape[0],3,3),prob)
#         prob = self.relu(self.W2(prob))
#         prob = prob[:,0,:].squeeze()
#         prob = self.fc(prob)
#
#         if self.train:
#             return self.sim(prob,gal)
#         return self.sim0(prob, gal)
#
# class GcnModel(nn.Module):
#     def __init__(self,train=True):
#         super(GcnModel,self).__init__()
#         self.train = train
#         a = torch.pow(torch.tensor(6.),-0.5)
#         self.A = torch.tensor([
#             [1/3.,a,a],
#             [a,0.5,0],
#             [a,0,0.5]]).unsqueeze(0).cuda()
#         self.A.requires_grad = False
#         self.W1 = nn.Linear(in_features=256,out_features=512)
#         self.W1.bias.data.fill_(0.)
#         self.W1.bias.requires_grad = False
#         self.W2 = nn.Linear(in_features=512,out_features=1024)
#         self.W2.bias.requires_grad = False
#         self.W2.bias.data.fill_(0.)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(in_features=1024,out_features=1024)
#         self.pool = nn.AvgPool1d(3)
#
#         self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#
#     def forward(self,gallery,probe):
#         gal = torch.bmm(self.A.expand(gallery.shape[0],3,3),gallery)
#         gal = self.relu(self.W1(gal))
#         gal = torch.bmm(self.A.expand(gal.shape[0],3,3),gal)
#         gal = self.relu(self.W2(gal))
#         gal = gal[:,0,:].squeeze()
#         gal = self.fc(gal)
#
#         probe = probe.unsqueeze(0).expand(gallery.shape[0],probe.shape[0],probe.shape[1])
#         prob = torch.bmm(self.A.expand(probe.shape[0],3,3),probe)
#         prob = self.relu(self.W1(prob))
#         prob = torch.bmm(self.A.expand(prob.shape[0],3,3),prob)
#         prob = self.relu(self.W2(prob))
#         # prob = prob[:,0,:].squeeze()
#         prob = prob.transpose(1,2)
#         prob = self.pool(prob).squeeze()
#         prob = self.fc(prob)
#
#         return prob,gal,self.sim(prob,gal)
#
# class GcnModel_4(nn.Module):
#     def __init__(self,train=True,neibor=4):
#         super(GcnModel_4,self).__init__()
#         self.train = train
#         self.neibor = neibor
#
#         if self.neibor==3:
#             a = torch.pow(torch.tensor(6.),0.5)
#             self.A = torch.tensor([
#                 [1.,a/3,a/3],
#                 [a/2,1.,0],
#                 [a/2,0,1.]]).unsqueeze(0).cuda()
#
#         if self.neibor==4:
#             a = torch.pow(torch.tensor(2.),0.5)
#             self.A = torch.tensor([
#                 [1.,a/2,a/2,a/2],
#                 [a,1.,0.,0.],
#                 [a,0.,1.,0.],
#                 [a,0.,0.,1.]
#             ]).unsqueeze(0).cuda()
#
#         self.A.requires_grad = False
#         self.W1 = nn.Linear(in_features=256,out_features=512)
#         self.W1.bias.data.fill_(0.)
#         self.W1.bias.requires_grad = False
#         self.W2 = nn.Linear(in_features=512,out_features=1024)
#         self.W2.bias.requires_grad = False
#         self.W2.bias.data.fill_(0.)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(in_features=1024,out_features=1024)
#         self.pool = nn.AvgPool1d(self.neibor)
#
#         self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
#
#
#     def forward(self,gallery,probe):
#         gal = torch.bmm(self.A.expand(gallery.shape[0],self.neibor,self.neibor),gallery)
#         gal = self.relu(self.W1(gal))
#         gal = torch.bmm(self.A.expand(gal.shape[0],self.neibor,self.neibor),gal)
#         gal = self.relu(self.W2(gal))
#         gal = gal[:,0,:].squeeze()
#         gal = self.fc(gal)
#
#         probe = probe.unsqueeze(0).expand(gallery.shape[0],probe.shape[0],probe.shape[1])
#         prob = torch.bmm(self.A.expand(probe.shape[0],self.neibor,self.neibor),probe)
#         prob = self.relu(self.W1(prob))
#         prob = torch.bmm(self.A.expand(prob.shape[0],self.neibor,self.neibor),prob)
#         prob = self.relu(self.W2(prob))
#         # prob = prob[:,0,:].squeeze()
#         prob = prob.transpose(1,2)
#         prob = self.pool(prob).squeeze()
#         prob = self.fc(prob)
#
#         return prob,gal,self.sim(prob,gal)

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
        self.W1 = nn.Linear(in_features=256,out_features=512)
        self.W1.bias.data.fill_(0.)
        self.W1.bias.requires_grad = False
        self.W2 = nn.Linear(in_features=512,out_features=1024)
        self.W2.bias.requires_grad = False
        self.W2.bias.data.fill_(0.)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=1024,out_features=1024)
        self.pool = nn.AvgPool1d(self.neibor)

        if self.train:
            self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        else:
            self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self,gallery,probe):
        gal = torch.bmm(self.A.expand(gallery.shape[0],self.neibor,self.neibor),gallery)
        gal = self.relu(self.W1(gal))
        gal = torch.bmm(self.A.expand(gal.shape[0],self.neibor,self.neibor),gal)
        gal = self.relu(self.W2(gal))
        gal = gal[:,0,:].squeeze()
        gal = self.fc(gal)

        probe = probe.unsqueeze(0).expand(gallery.shape[0],probe.shape[0],probe.shape[1])
        prob = torch.bmm(self.A.expand(probe.shape[0],self.neibor,self.neibor),probe)
        prob = self.relu(self.W1(prob))
        prob = torch.bmm(self.A.expand(prob.shape[0],self.neibor,self.neibor),prob)
        prob = self.relu(self.W2(prob))
        # prob = prob[:,0,:].squeeze()
        prob = prob.transpose(1,2)
        prob = self.pool(prob).squeeze()
        prob = self.fc(prob)

        return prob,gal,self.sim(prob,gal)

    def loadfile(self):
        collect_path = os.path.join(log_root,'collect_models')
        return os.path.join(collect_path,"model_{}.pkl".format(self.neibor))

class Attention(nn.Module):
    def __init__(self,neibor=3):
        super(Attention,self).__init__()
        self.neibor = neibor
        self.fc = nn.Sequential(nn.Linear(256,256),nn.ReLU())
        self.softmax = nn.Softmax(dim=1)

    def forward(self,gallery,probe):
        feat_ = []
        for i in range(self.neibor):
            compi = probe[i].unsqueeze(0).expand(gallery.shape[0], gallery.shape[1], probe.shape[1])
            compi = self.fc(compi)
            e0 = (gallery * compi).sum(2).unsqueeze(2)
            e0 = self.softmax(e0).transpose(1, 2)
            feati = torch.bmm(e0, gallery)
            feat_.append(feati)
        feat = torch.cat(feat_,1)
        return feat