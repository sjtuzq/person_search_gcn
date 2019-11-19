"""
    create by Qiang Zhang
    function: prepare dataset into gcn input format
"""
import os
import pickle
import os.path as osp
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from config import data_root,neighbor_num

_root_dir = os.path.join(data_root,"psdb/dataset")

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
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f,encoding='latin1')
    return data

def select_g(det,feat_g):
    # det = det[:,:4]
    detN = det.shape[0]
    personSet = []
    if detN==1:
        personSet.append([feat_g[0],feat_g[0],feat_g[0]])
    elif detN==2:
        personSet.append([feat_g[0],feat_g[0],feat_g[1]])
        personSet.append([feat_g[1],feat_g[1],feat_g[0]])
    else:
        for id in range(detN):
            index = np.zeros(detN)
            for j in range(detN):
                index[j] = _compute_dist(det[id],det[j])
            index = [[x, i] for i, x in enumerate(index)]
            index = sorted(index, key=lambda x: x[0], reverse=False)
            personSet.append([feat_g[id],feat_g[index[1][1]],feat_g[index[2][1]]])
    return np.asarray(personSet)

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

def getTestdata(det_thresh=0.5,neibor=4,is_gt=False):
    base = os.path.join(data_root,"predata/testdata")
    _image_index = unpickle(base + "/image_index.pkl")
    probes = unpickle(base + '/probes.pkl')
    gallery_det = unpickle(base + '/gallery_detections.pkl')
    gallery_feat = unpickle(base + '/gallery_features.pkl')['feat']
    probe_feat = unpickle(base + '/probe_features.pkl')['feat']
    fname = 'TestG{}'.format(100)
    protoc = loadmat(osp.join(data_root,'predata/{}.mat'.format(fname)))[fname].squeeze()

    name_to_det_feat = {}
    for name, det, feat in zip(_image_index,
                               gallery_det, gallery_feat):
        scores = det[:, 4].ravel()
        inds = np.where(scores >= det_thresh)[0]
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds])

    data = []
    for i in tqdm(range(len(probes))):
        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt, count_tp = 0, 0
        # Get L2-normalized feature vector
        feat_p = probe_feat[i].ravel()
        # Ignore the probe image
        probe_imname = str(protoc['Query'][i]['imname'][0, 0][0])
        probe_roi = protoc['Query'][i]['idlocate'][0, 0][0].astype(np.int32)
        probe_roi[2:] += probe_roi[:2]

        if probe_imname not in name_to_det_feat:
            featP3 = np.asarray([feat_p,feat_p,feat_p])
            featP4 = np.asarray([feat_p, feat_p, feat_p,feat_p])
            featPn = np.asarray([feat_p]*neibor)
            # print("probe not has company  (-)")
        else:
            detP,featP = name_to_det_feat[probe_imname]
            featP1,featP3 = select_p(detP,featP,probe_roi)
            featP1,featP4 = select_p_4(detP, featP, probe_roi)
            featP1,featPn = select_p_n(detP,featP,probe_roi,num=neibor)
            featPn[0, :, 0, 0] = feat_p

        #featPn = np.asarray([feat_p] * neibor)
        probe_gt = []
        tested = set([probe_imname])
        # 1. Go through the gallery samples defined by the protocol
        # item = []
        gallery_now = []
        #label = []
        label_true = []
        label_gt = []
        gallery_all = []
        det_all = []
        for item in protoc['Gallery'][i].squeeze():
            gallery_imname = str(item[0][0])
            # some contain the probe (gt not empty), some not
            gt = item[1][0].astype(np.int32)
            count_gt += (gt.size > 0)
            # compute distance between probe and gallery dets
            if gallery_imname not in name_to_det_feat: continue
            det, feat_g = name_to_det_feat[gallery_imname]
            # get L2-normalized feature matrix NxD
            # det, feat_g = name_to_det_feat['s7785.jpg']
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            #gallery_all.append(feat_g)
            det_all.append(det)
            # tmp = select_g(det,feat_g)
            # gallery_now.append(tmp)

            #-------------------------------
            sim = feat_g.dot(feat_p).ravel()
            # assign label for each det
            label = np.zeros(len(sim), dtype=np.int32)
            if gt.size > 0:
                w, h = gt[2], gt[3]
                gt[2:] += gt[:2]
                probe_gt.append({'img': str(gallery_imname),
                                 'roi': map(float, list(gt))})
                iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]
                feat_g = feat_g[inds]
                # only set the first matched det as true positive
                for j, roi in enumerate(det[:, :4]):
                    if _compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            #label_true.append(label)
            label_true += list(label)
            gallery_all.append(feat_g)

            #------------------------------------
            if gt.size>0:
                label_gt.append(gt)
                #label.append(1)
            else:
                label_gt.append(0)
                #label.append(0)
        if not is_gt:
            #item = [featPn,label,gallery_all]
            item = [featPn, label_true, gallery_all]
        else:
            item = [det_all,label_gt]
        data.append(item)
    return np.asarray(data)

if __name__ == '__main__':
    Nlist = [neighbor_num]
    test_dataset = os.path.join(data_root,'testdata')
    if not os.path.exists(test_dataset):
        os.makedirs(test_dataset)
    for N in Nlist:
        data = getTestdata(neibor=N,is_gt=False)
        np.save(os.path.join(test_dataset,'testdata_featn_{}'.format(N)),data)

    data = getTestdata(neibor=neighbor_num,is_gt=True)
    np.save(os.path.join(test_dataset,'testdata_gt'),data)
