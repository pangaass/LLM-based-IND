from sklearn import metrics
import numpy as np

def weighted_metric(pred:list , label: list) -> float:
    num_pred = [len(i) for i in pred]
    num_label = [len(i) for i in label]
    assert all(a==b for a, b in zip(num_pred, num_label))
    
    acc_pred = [metrics.accuracy_score(l,p) for l,p in zip(label,pred)]
    f1_pred = [metrics.f1_score(l,p) for l,p in zip(label,pred)]
    # abnormal = 0, normal = 1
    num0 = np.array([i.count(0) for i in label])
    weight = num0/np.array(num0.sum())
    weighted_acc = sum(weight * acc_pred)
    weight_f1 = sum(weight * f1_pred) 
    return weighted_acc,weight_f1

def compute_metric(ground_truth:dict, res: dict) -> float:
    
    res_list = []
    label_list = []
    
    for author,pubs in ground_truth.items():
        sub_res = res[author]
        keys = pubs['normal_data'] +pubs['outliers']
        label = [1]* len(pubs['normal_data'])+[0]* len(pubs['outliers'])
        
        pred = []
        for i in keys:
            if i in sub_res['normal_data'] and i not in sub_res['outliers']:
                pred.append(1)
            elif i in sub_res['outliers'] and i not in sub_res['normal_data']:
                pred.append(0)
            else:
                # 对于回复异常的文本，直接将其认定为outlier
                pred.append(0)

        res_list.append(pred)
        label_list.append(label)
    acc, f1 = weighted_metric(res_list, label_list)
    return acc, f1
