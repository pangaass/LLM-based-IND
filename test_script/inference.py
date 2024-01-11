# -*- coding: utf-8 -*-
import json
from metric import compute_metric
import os
path = os.getcwd()

dirs = os.listdir(os.path.join(path,'eval_result'))
dirs = ['result-2024-1-5latest.json']
label_path = "dataset/val_data.json"
with open(os.path.join(path,label_path), 'r', encoding='utf-8') as f:
    label_dict = json.load(f)
data = []
for author in label_dict.keys():
    data.extend(label_dict[author]['normal_data'])
    data.extend(label_dict[author]['outliers'])

for dir in dirs:
    with open(os.path.join(path,'eval_result',dir), 'r', encoding='utf-8') as f:
        result = json.load(f)

    res_list = {}
    for i in result:
        [author,pub,pred,yes_prob,no_prob,label] = i
        if author not in res_list.keys():
            res_list[author] = {}
            res_list[author]['normal_data'] ={}
            res_list[author]['outliers'] ={}
        logit = yes_prob/(yes_prob+no_prob)
        if pred:
            res_list[author]['normal_data'][pub]= logit
        else:
            res_list[author]['outliers'][pub]= logit
    mean_AUC, mAP, acc, f1 = compute_metric(label_dict,res_list)
    print('checkpoint path:{}, AUC:{:.2f}, MAP:{:.2f}, ACC:{:.2f}, F1：{:.2f}'.format(dir, float(mean_AUC*100), float(mAP*100), float(acc*100), float(f1*100)))