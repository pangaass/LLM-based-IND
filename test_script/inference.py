# -*- coding: utf-8 -*-
import json
from metric import compute_metric
import os
path = os.getcwd()

dirs = os.listdir(os.path.join(path,'eval_result'))

label_path = "dataset/val_data.json"
with open(os.path.join(path,label_path), 'r', encoding='utf-8') as f:
    label_dict = json.load(f)


for dir in dirs:
    with open(os.path.join(path,'eval_result',dir), 'r', encoding='utf-8') as f:
        result = json.load(f)


    res_list = {}
    for i in result:
        [author,pub,pred,yes_prob,no_prob,label] = i
        if author not in res_list.keys():
            res_list[author] = {}
            res_list[author]['normal_data'] =[]
            res_list[author]['outliers'] =[]
        if pred:
            res_list[author]['normal_data'].append(pub)
        else:
            res_list[author]['outliers'].append(pub)
    acc, f1 = compute_metric(label_dict,res_list)
    print(dir, acc, f1)