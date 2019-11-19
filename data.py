"""
    create by Qiang Zhang
    function: provide datasets
"""
import os
import numpy as np
import random
import torch.utils.data as Data
from utils import readpkl,select_p,select_p_n

from config import data_root

class RawTrain_core(Data.Dataset):
    def __init__(self,thresh):
        self.traindata = Traindata(thresh)

    def __getitem__(self, id0):
        id = self.traindata.keys[id0]
        items = self.traindata.data[id]
        data = []
        for item in items:
            det = self.traindata.det[item[0]]
            feat = self.traindata.feat[item[0]]
            roi = item[1]
            feat1,feat3 = select_p(det,feat,roi)
            # data.append(feat3)
            data.append(np.asarray([feat1,feat1,feat1]))
        return id,np.asarray(data)

    def __len__(self):
        return self.traindata.__len__()

class RawTrain(Data.Dataset):
    def __init__(self,thresh):
        self.data_id = RawTrain_core(thresh)
        self.core()

    def core(self):
        data = []
        for id,item in enumerate(self.data_id):
            for person in item[1]:
                data.append([item[0],person])
        self.data = data

    def __getitem__(self, id):
        return self.data[id]

    def __len__(self):
        return len(self.data)

    @classmethod
    def getLoader(cls,thresh=0.5,batch_size=2):
        dataset = cls(thresh)
        return Data.DataLoader(
            dataset = dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

class Traindata(Data.Dataset):
    """
    feat1:only the probe person feature:1 x 256
    feat3:the probe person and his two partners:3 x 256
    label:true or false:1 x 256
    gallery_all:all of the gallery features:100 x n x 256
    """
    def __init__(self,thresh,neibor=4):
        self.neibor = neibor
        self.root = os.path.join(data_root,'predata/traindata')
        self.traindata = readpkl(self.root + "/psdb_train_gt_roidb.pkl")
        traindetections = readpkl(self.root + "/gallery_detections.pkl")
        trainfeatures = readpkl(self.root + "/gallery_features.pkl")
        self.filter(traindetections,trainfeatures['feat'],thresh)
        self.data = self.getIndex()
        self.keys = list(self.data.keys())
        self.sample = list(range(len(self.det)))

    def filter(self,traindetections,trainfeatures,thresh=0.0):
        self.det = []
        self.feat = []
        for item,item2 in zip(traindetections,trainfeatures):
            ids = item[:,4]>thresh
            item1 = item[ids]
            item2 = item2[ids,:,0,0]
            self.det.append(item1)
            self.feat.append(item2)

    def getIndex(self):
        personSet = {}
        for i in range(11204):
            ids = self.traindata[i + 1]['gt_pids']
            detection = self.det[i + 1]
            if len(detection)==0:
                continue
            for j,person in enumerate(ids):
                if person==-1:
                    continue
                try:
                    tmp = [i+1,detection[j]]
                except:
                    continue
                try:
                    personSet[person].append(tmp)
                except:
                    personSet[person] = [tmp]
        return personSet

    def getGallery(self,probe):
        slice = random.sample(self.sample,100)
        label = [0]*100
        gallery = []
        for i in range(100):
            tmp = self.feat[slice[i]]
            gallery.append(tmp)
        if len(probe)==1:
            return label,gallery
        for i,item in enumerate(probe[1:]):
            if item[0] in slice:
                id = slice.index(item[0])
                tmp = random.sample(self.sample,1)[0]
                gallery[id] = self.feat[tmp]
            gallery[i] = self.feat[item[0]]
            label[i] = 1
        if not sum(label) == sum(label[:sum(label)]):
            print("error")
        return label,gallery

    def getGalleryRandom(self,probe, randIdx):
        slice = random.sample(self.sample,100)
        label = [0]*100
        gallery = []
        idx = list(range(len(probe)))
        idx.remove(randIdx)
        for i in range(100):
            tmp = self.feat[slice[i]]
            gallery.append(tmp)
        if len(probe)==1:
            return label,gallery
        for i, ix in enumerate(idx):
            item = probe[ix]
            if item[0] in slice:
                id = slice.index(item[0])
                tmp = random.sample(self.sample,1)[0]
                gallery[id] = self.feat[tmp]
            gallery[i] = self.feat[item[0]]
            label[i] = 1
        if not sum(label) == sum(label[:sum(label)]):
            print("error")
        return label,gallery

    def __getitem__(self, id):
        id = self.keys[id]
        randIdx = random.randint(0, len(self.data[id])-1)
        item = self.data[id][randIdx]
        det = self.det[item[0]]
        feat = self.feat[item[0]]
        roi = item[1]
        feat1, featn = select_p_n(det, feat, roi, num=self.neibor)
        label, gallery_all = self.getGalleryRandom(self.data[id], randIdx)
        return feat1, featn, np.asarray(label), np.asarray(gallery_all)
        
        '''
        id = self.keys[id]
        item = self.data[id][0]
        det = self.det[item[0]]
        feat = self.feat[item[0]]
        roi = item[1]
        feat1,featn = select_p_n(det,feat,roi,num=self.neibor)
        label,gallery_all = self.getGallery(self.data[id])
        return feat1,featn,np.asarray(label),np.asarray(gallery_all)
        '''

    def __len__(self):
        return len(self.data)

def getTestdata(neibor=4):
    testFile = os.path.join(data_root,'testdata/testdata_featn_{}.npy'.format(neibor))
    testData = np.load(testFile,encoding='latin1',allow_pickle = True)
    return testData
