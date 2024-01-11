from sklearn import metrics
import numpy as np

def weighted_metric(pred_score:list , label: list) -> float:
    num_pred = [len(i) for i in pred]
    num_label = [len(i) for i in label]
    assert all(a==b for a, b in zip(num_pred, num_label))
    
    pred_label = [1 if score>=0.5 else 0  for i in pred_score for score in i]

    acc_pred = [metrics.accuracy_score(l,p) for l,p in zip(label,pred_label)]
    f1_pred = [metrics.f1_score(l,p) for l,p in zip(label,pred_label)]
    AP = [metrics.average_precision_score(l,p) for l,p in zip(label,pred_score)]
    AUC = [metrics.roc_auc_score(l,p) for l,p in zip(label,pred_score)]
    # abnormal = 0, normal = 1
    num0 = np.array([i.count(0) for i in label])
    weight = num0/np.array(num0.sum())
    mean_AUC = sum(weight * AUC)
    mAP = sum(weight * AP)
    weighted_acc = sum(weight * acc_pred)
    weighted_f1 = sum(weight * f1_pred) 
    return mean_AUC,mAP,weighted_acc,weighted_f1

def compute_metric(ground_truth:dict, res: dict) -> (float,float):
    
    res_list = []
    label_list = []
    
    for author,pubs in ground_truth.items():
        sub_res = res[author]
        keys = pubs['normal_data'] +pubs['outliers']
        res_keys = list(res[author]['normal_data'].keys()) + list(res[author]['outliers'].keys())
        assert set(keys) == set(res_keys)

        label = [1]* len(pubs['normal_data'])+[0]* len(pubs['outliers'])
        
        pred = []
        for i in keys:
            if i in sub_res['normal_data'].keys() and i not in sub_res['outliers'].keys():
                pred.append(sub_res['normal_data'][i])
            elif i in sub_res['outliers'].keys() and i not in sub_res['normal_data'].keys():
                pred.append(sub_res['outliers'][i])
            else:
                #报错
                raise Exception('缺少预测值')

        res_list.append(pred)
        label_list.append(label)
    mean_AUC,mAP,acc, f1 = weighted_metric(res_list, label_list)
    return mean_AUC,mAP,acc, f1
