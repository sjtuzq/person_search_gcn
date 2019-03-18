"""
    create by Qiang Zhang
    function: provide some util apis
"""
import torch
import random
import pickle
import math
import numpy as np
import numpy.linalg as linagy

def readpkl(file):
    fr = open(file,'rb')
    data = pickle.load(fr,encoding='latin1')
    return data

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def _compute_dist(a, b):
    center0 = (a[0]+a[2])/2,(a[1]+a[3])/2
    center1 = (b[0]+b[2])/2,(b[1]+b[3])/2
    return (center0[0]-center1[0])**2+(center0[1]-center1[1])**2


def select_p(det,feat_p,roi):
    def _compute_comp(a, b,iou_thresh=0.0):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        iou = inter * 1.0 / union
        if iou>iou_thresh:
            return 0
        else:
            return a[4]
    detN = det.shape[0]
    if detN==1:
        feat1 = feat_p[0]
        feat3 = [feat_p[0],feat_p[0],feat_p[0]]
    elif detN==2:
        feat1 = feat_p[0]
        feat3 = [feat_p[0], feat_p[1], feat_p[1]]
    else:
        index = np.zeros(detN)
        for i in range(detN):
            index[i] = _compute_iou(det[i],roi)
            # index[i] = _compute_comp(det[i], roi)
        index = [[x, i] for i, x in enumerate(index)]
        index = sorted(index, key=lambda x: x[0], reverse=True)
        feat1 = feat_p[index[0][1]]
        feat3 = [feat_p[index[0][1]],feat_p[index[1][1]],feat_p[index[2][1]]]

        index = np.zeros(detN)
        for i in range(detN):
            # index[i] = _compute_iou(det[i],roi)
            index[i] = _compute_comp(det[i], roi)
        index = [[x, i] for i, x in enumerate(index)]
        index = sorted(index, key=lambda x: x[0], reverse=True)
        feat3 = [feat3[0], feat_p[index[1][1]], feat_p[index[2][1]]]
    return feat1,np.asarray(feat3)

def select_p_4(det,feat_p,roi):
    def _compute_comp(a, b,iou_thresh=0.0):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        iou = inter * 1.0 / union
        if iou>iou_thresh:
            return 0
        else:
            return a[4]
    detN = det.shape[0]
    if detN==1:
        feat1 = feat_p[0]
        feat4 = [feat_p[0],feat_p[0],feat_p[0],feat_p[0]]
    elif detN==2:
        feat1 = feat_p[0]
        feat4 = [feat_p[0], feat_p[1], feat_p[1],feat_p[1]]
    elif detN==3:
        feat1 = feat_p[0]
        feat4 = [feat_p[0], feat_p[1], feat_p[2],feat_p[2]]
    else:
        index = np.zeros(detN)
        for i in range(detN):
            index[i] = _compute_iou(det[i],roi)
        index = [[x, i] for i, x in enumerate(index)]
        index = sorted(index, key=lambda x: x[0], reverse=True)
        feat1 = feat_p[index[0][1]]
        feat4 = [feat_p[index[0][1]],feat_p[index[1][1]],feat_p[index[2][1]]]

        index = np.zeros(detN)
        for i in range(detN):
            index[i] = _compute_comp(det[i], roi)
        index = [[x, i] for i, x in enumerate(index)]
        index = sorted(index, key=lambda x: x[0], reverse=True)
        feat4 = [feat4[0], feat_p[index[1][1]], feat_p[index[2][1]],feat_p[index[3][1]]]
    return feat1,np.asarray(feat4)

def select_p_n(det,feat_p,roi,num=4):
    def _compute_comp(a, b,iou_thresh=0.0):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        iou = inter * 1.0 / union
        if iou>iou_thresh:
            return 0
        else:
            return a[4]
    detN = det.shape[0]
    if detN==1:
        feat1 = feat_p[0]
        featn = [feat_p[0]]*num
        return feat1,np.asarray(featn)
    index = np.zeros(detN)
    for i in range(detN):
        index[i] = _compute_iou(det[i],roi)
    index = [[x, i] for i, x in enumerate(index)]
    index = sorted(index, key=lambda x: x[0], reverse=True)
    feat1 = feat_p[index[0][1]]

    index = np.zeros(detN)
    for i in range(detN):
        index[i] = _compute_comp(det[i], roi)
    index = [[x, i] for i, x in enumerate(index)]
    index = sorted(index, key=lambda x: x[0], reverse=True)

    featn = [feat1]
    for i in range(num-1):
        if i>=detN-1:
            featn.append(feat_p[index[1][1]])
        else:
            featn.append(feat_p[index[i+1][1]])
    return feat1,np.asarray(featn)


def Eval(label,pred,num=100):
    # acc metric
    correct = np.zeros([pred.shape[0],num+1])
    correct0 = np.zeros(num+1)
    for i in range(label.shape[0]):
        for j in range(1,num+1):
            if min(pred[i][0:j])<=sum(label[i]):
                correct[i][j] += 1
    for j in range(1,num+1):
        correct0[j] = float(correct[:,j].sum())/label.shape[0]

    # map metric
    map,probe_num = 0,pred.shape[0]
    for i in range(probe_num):
        ap, num = 0, sum(label[i])
        id = np.where(pred[i] < num)[0] + 1
        for j, item in enumerate(id):
            ap += (j + 1) / item
        ap /= id.shape[0]
        map += ap
    map /= probe_num

    return correct0[1:],map

def Eval_det(label,pred,det,num=100):
    correct = np.zeros([pred.shape[0],num+1])
    correct0 = np.zeros(num+1)
    for i in range(label.shape[0]):
        ids = np.where(np.asarray(det[i])==1)[0]
        # for item in ids:
        #     id = np.where(pred[i]==item)
        #     correct[i][id] += 1
        for j in range(1, num + 1):
            if any([item in pred[i][0:j] for item in ids]):
                correct[i][j] += 1
        # for j in range(1, num + 1):
        #     if min(pred[i][0:j])<=sum(label[i]) and det[i][min(pred[i][0:j])]:
        #         correct[i][j] += 1
    for j in range(1,num+1):
        correct0[j] = float(correct[:,j].sum())/label.shape[0]
    return correct0[1:]

def shuffle0(data, label):
    ids = range(data.shape[0])
    random.shuffle(ids)
    ids = torch.tensor(ids).cuda()
    data = torch.index_select(data, 0, ids)
    label = torch.index_select(label, 0, ids)
    return data, label, ids.sort()[1]

def Sim2(feat1,feat2):
    sim = -np.linalg.norm(feat1 - feat2)
    return sim

def Sim1(feat1,feat2):
    sim = feat1.dot(feat2)/(linagy.norm(feat1)*linagy.norm(feat2))
    return sim

def gallery_abc(probe,gallery_feat,num=100):
    if len(gallery_feat) == 99:
        gallery_feat.append(gallery_feat[-1])
    gallery_feat = gallery_feat[:num]
    index1 = []
    idsave1 = []
    featuresave1 = []
    for item in gallery_feat:
        sim0 = -1
        nowid = 0
        nowfeature = item[0]
        for id,person in enumerate(item):
            sim = Sim1(probe[0],person)
            if sim>sim0:
                sim0 = sim
                nowid = id
                nowfeature = person
        index1.append(sim0)
        idsave1.append(nowid)
        featuresave1.append(nowfeature)
    index0 = np.asarray(index1)
    index0 = np.argsort(-index0)

    index2 = []
    idsave2 = []
    featuresave2 = []
    for item in gallery_feat:
        sim0 = -1
        nowid = 0
        nowfeature = item[0]
        for id,person in enumerate(item):
            if id==idsave1[id]:
                continue
            sim = Sim1(probe[1],person)
            # sim = np.random.rand(1)[0]
            if sim>sim0:
                sim0 = sim
                nowid = id
                nowfeature = person
        index2.append(sim0)
        idsave2.append(nowid)
        featuresave2.append(nowfeature)

    index3 = []
    idsave3 = []
    featuresave3 = []
    for item in gallery_feat:
        sim0 = -1
        nowid = 0
        nowfeature = item[0]
        for id,person in enumerate(item):
            if id == idsave1[id]:
                continue
            if id == idsave2[id]:
                continue
            sim = Sim1(probe[2],person)
            # sim = np.random.rand(1)[0]
            if sim>sim0:
                sim0 = sim
                nowid = id
                nowfeature = person
        index3.append(sim0)
        idsave3.append(nowid)
        featuresave3.append(nowfeature)

    abc_data = []
    for i in range(len(index1)):
        a = index1[i]
        b = index2[i]
        c = index3[i]
        abc_data.append(np.asarray([a,b,c]))
    abc_data = np.asarray(abc_data)

    return np.asarray(abc_data)

def gallery_gcn_det(probe,gallery_feat,num=3):
    if len(gallery_feat) == 99:
        gallery_feat.append(gallery_feat[-1])

    detsave = [[],[],[],[],[],[],[],[],[],[]]
    featuresave = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(num):
        for j,item in enumerate(gallery_feat):
            sim0 = -1
            id0 = 0
            try:
                nowfeature = item[0]
            except:
                continue
            for id,person in enumerate(item):
                sim = Sim1(probe[i],person)
                if sim>sim0:
                    sim0 = sim
                    id0 =id
                    nowfeature = person
            detsave[i].append(id0)
            featuresave[i].append(nowfeature)

    abc_data = []
    for i in range(len(gallery_feat)):
        feature = []
        for j in range(num):
            try:
                feature.append(featuresave[j][i])
            except:
                break
        else:
            abc_data.append(np.asarray(feature))

    if len(abc_data)==99:
        abc_data.append(abc_data[-1])
    abc_data = np.asarray(abc_data)
    return abc_data,detsave

def gallery_gcn(probe,gallery_feat,num=3):
    if len(gallery_feat) == 99:
        gallery_feat.append(gallery_feat[-1])

    featuresave = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(num):
        for j,item in enumerate(gallery_feat):
            sim0 = -1
            try:
                nowfeature = item[0]
            except:
                continue
            for id,person in enumerate(item):
                sim = Sim1(probe[i],person)
                if sim>sim0:
                    sim0 = sim
                    nowfeature = person
            featuresave[i].append(nowfeature)

    abc_data = []
    for i in range(len(gallery_feat)):
        feature = []
        for j in range(num):
            try:
                feature.append(featuresave[j][i])
            except:
                break
        else:
            abc_data.append(np.asarray(feature))

    if len(abc_data)==99:
        abc_data.append(abc_data[-1])
    abc_data = np.asarray(abc_data)
    return abc_data

def Rank0(probe,gallery_feat):
    """
    :param probe: probe feat:3 x 256
    :param gallery_feat: 100 x 256
    :return: baseline method ,   new method calculated by math formula
    """
    if len(gallery_feat) == 99:
        gallery_feat.append(gallery_feat[-1])
    if len(gallery_feat[0]) == 0:
        gallery_feat[0] = gallery_feat[-1]
    index1 = []
    idsave1 = []
    featuresave1 = []
    for item in gallery_feat:
        sim0 = -1
        nowid = 0
        nowfeature = item[0]
        for id,person in enumerate(item):
            sim = Sim2(probe[0],person)
            # sim = np.random.rand(1)[0]
            if sim>sim0:
                sim0 = sim
                nowid = id
                nowfeature = person
        index1.append(sim0)
        idsave1.append(nowid)
        featuresave1.append(nowfeature)
    index0 = np.asarray(index1)
    index0 = np.argsort(-index0)

    index2 = []
    idsave2 = []
    featuresave2 = []
    for item in gallery_feat:
        sim0 = -1
        nowid = 0
        nowfeature = item[0]
        for id,person in enumerate(item):
            if id==idsave1[id]:
                continue
            sim = Sim2(probe[1],person)
            # sim = np.random.rand(1)[0]
            if sim>sim0:
                sim0 = sim
                nowid = id
                nowfeature = person
        index2.append(sim0)
        idsave2.append(nowid)
        featuresave2.append(nowfeature)
    # index2 = np.asarray(index2)
    # index2 = np.argsort(-index2)

    index3 = []
    idsave3 = []
    featuresave3 = []
    for item in gallery_feat:
        sim0 = -1
        nowid = 0
        nowfeature = item[0]
        for id,person in enumerate(item):
            if id == idsave1[id]:
                continue
            if id == idsave2[id]:
                continue
            sim = Sim2(probe[2],person)
            # sim = np.random.rand(1)[0]
            if sim>sim0:
                sim0 = sim
                nowid = id
                nowfeature = person
        index3.append(sim0)
        idsave3.append(nowid)
        featuresave3.append(nowfeature)
    # index3 = np.asarray(index3)
    # index3 = np.argsort(-index3)

    index_all = []
    for i in range(len(index1)):
        a = index1[i]
        b = index2[i]
        c = index3[i]
        tmp = a*(math.exp(b*b)+0.3*math.exp(c*c)+2.5)
        index_all.append(tmp)

    index_all = np.asarray(index_all)
    index_all = np.argsort(-index_all)
    return index0+1,index_all+1

def padding(gallery):
    maxlen = 0
    for item in gallery:
        if item.shape[0]>maxlen:
            maxlen = item.shape[0]
    tmp = []
    for i,item in enumerate(gallery):
        gallery[i] = np.concatenate((item,np.zeros((maxlen-item.shape[0],256))),0)
        tmp.append(gallery[i])
    return np.asarray(tmp)

