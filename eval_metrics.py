import operator
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def evaluate_from_csv(csv_file):
    f = open(csv_file)
    lines = f.readlines()
    AU1,AU1_label = [],[]
    for l in lines:
        data = l.strip().split(" ")
        AU1.append(int(data[2]))
        AU1_label.append(int(data[14]))
    # print(AU1)
    # print(AU1_label)
    # print(f1_score(AU1,AU1_label))




# evaluate_from_csv("data/BP4D_test_res.csv")

def save_error(distances1, distances2, distances3,names,types):
    N = len(names)
    correct = []
    error = []
    error_types = []
    for i in range(N):
        if distances1[i]< distances2[i] and distances1[i] < distances3[i]:
            correct.append(names[i])
        elif distances2[i]<distances3[i] and distances2[i] < distances3[i]:
            error.append(names[i])
            error_types.append("NAP")
        elif distances3[i]<distances1[i] and distances3[i] < distances1[i]:
            error.append(names[i])
            error_types.append("PNA")
    f = open("FEC_cases.txt","w")
    for i in range(len(error)):
        f.write(names[i] + " " + error_types[i] + " " + types[i] + "\n")
    f.close()





def triplet_prediction_accuracy(distances1,distances2,distances3,N):
    # distances1: anc and pos
    # distances2: anc and neg
    # distances3: pos and neg


    c1 = distances2-distances1
    c2 = distances3-distances1
    n =0
    for i in range(len(c1)):
        if c1[i] > 0 and c2[i] > 0:
            n+=1
    acc = n/N
    return acc



def triplet_prediction_accuracy_ensemble(distances1,distances2,distances3):
    c1 = distances2 - distances1
    c2 = distances3 - distances1
    res = []
    for i in range(len(c1)):
        if c1[i] > 0 and c2[i] > 0:
            res.append(1)
        else:
            res.append(0)
    return res




def triplet_prediction_accuracy_by_class(distances1,distances2,distances3,types,N):
    # distances1: anc and pos
    # distances2: anc and neg
    # distances3: pos and neg


    c1 = distances2-distances1
    c2 = distances3-distances1
    n =0
    s1,s2,s3,N1,N2,N3=0,0,0,0,0,0
    for i in range(len(c1)):
        if types[i] == "ONE_CLASS_TRIPLET":
            N1 += 1
        elif types[i] == "TWO_CLASS_TRIPLET":
            N2 += 1
        elif types[i] == "THREE_CLASS_TRIPLET":
            N3 += 1
        if c1[i] > 0 and c2[i] > 0:
            n+=1
            if types[i]=="ONE_CLASS_TRIPLET":
                s1+=1
            elif types[i]=="TWO_CLASS_TRIPLET":
                s2+=1
            elif types[i]=="THREE_CLASS_TRIPLET":
                s3+=1
    acc1 = s1/N1
    acc2 = s2/N2
    acc3 = s3/N3
    return acc1,acc2,acc3


def one_hot_transfer(label,class_num=7):
    return np.eye(class_num)[label]




def metric_for_Exp(gt,pred,class_num=7):
    # compute_acc
    acc = accuracy_score(gt,pred)
    # compute_F1
    gt = one_hot_transfer(gt,class_num)
    pred = one_hot_transfer(pred,class_num)
    F1 = []
    for i in range(class_num):
        gt_ = gt[:,i]
        pred_ = pred[:,i]
        F1.append(f1_score(gt_.flatten(), pred_))
    F1_mean = np.mean(F1)
    return F1_mean,acc,F1


def metric_for_AU(gt,pred,class_num=12,type="Aff2"):
    #compute_F1,acc
    F1 = []
    gt = np.array(gt)
    pred = np.array(pred)
    if type=="Aff2":
        index = [i for i in range(12)]
    elif type =="bp4d":
        index = [0,1,2,3,4,5,6,7,8,9,12,13]
    elif type =="disfa":
        index = [0,1,2,3,6,10,11,14]
    
    cate_acc = np.sum((np.array(pred[:,index]>0,dtype=np.float))==gt[:,index])/(gt.shape[0]*len(index))
    # print(pred.shape)
    print(pred[:,index].shape)
    print(index)
    for t in index:
        gt_ = gt[:, t]
        pred_ = pred[:, t]
        new_pred = ((pred_ >= 0.5) * 1).flatten()
        F1.append(f1_score(gt_.flatten(), new_pred))

    F1_mean = np.mean(F1)
    print(F1)

    #compute total acc
    counts = gt.shape[0]
    accs = 0
    for i in range(counts):
        pred_label = ((pred[i,:] >= 0.5) * 1).flatten()
        gg = gt[i].flatten()
        j = 0
        for k in index:
            if int(gg[k]) == int(pred_label[k]):
                    j+=1
        if j==12:
            accs+=1

    acc = 1.0*accs/counts

    return F1_mean,acc,F1,cate_acc


def metric_for_AU_mlce(gt,pred,class_num=12,type="Aff2"):
    #compute_F1,acc
    F1 = []
    gt = np.array(gt)
    pred = np.array(pred)

    if type=="Aff2":
        index = [i for i in range(12)]
    elif type =="bp4d":
        index = [0,1,2,3,4,5,6,7,8,9,12,13]
    elif type =="disfa":
        index = [0,1,2,3,6,10,11,14]

    cate_acc = np.sum((np.array(pred[:,index]>0,dtype=np.float))==gt[:,index])/(gt.shape[0]*len(index))
    # print(pred.shape)
    for t in index:
        gt_ = gt[:, t]
        pred_ = pred[:, t]
        new_pred = ((pred_ >= 0.) * 1).flatten()
        F1.append(f1_score(gt_.flatten(), new_pred))

    F1_mean = np.mean(F1)

    #compute total acc
    counts = gt.shape[0]
    accs = 0
    for i in range(counts):
        pred_label = ((pred[i,:] >= 0.) * 1).flatten()
        gg = gt[i].flatten()
        j = 0
        for k in index:
            if int(gg[k]) == int(pred_label[k]):
                    j+=1
        if j==12:
            accs+=1

    acc = 1.0*accs/counts

    return F1_mean,acc,F1,cate_acc

# def CCC_score(x, y):
#     vx = x - np.mean(np.hstack(x))
#     vy = y - np.mean(np.hstack(y))
#     rho = np.sum(np.hstack(vx * vy)) / (np.sqrt(np.sum(np.hstack(vx**2))) * np.sqrt(np.sum(np.hstack(vy**2))))
#     x_m = np.mean(np.hstack(x))
#     y_m = np.mean(np.hstack(y))
#     x_s = np.std(np.hstack(x))
#     y_s = np.std(np.hstack(y))
#     ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
#     return ccc


def CCC_score(x, y):
    x = np.array(x)
    y = np.array(y)
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc


def PCC(x,y):
    x = np.array(x)
    y = np.array(y)
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    pcc = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    return pcc

def rmse(x,y):
    x = np.array(x)
    y = np.array(y)
    N = x.shape[0]
    rm = np.linalg.norm(x-y) / np.sqrt(N)
    return rm

def SAGR(x,y):
    x = np.array(x)
    y = np.array(y)
    N = x.shape[0]
    signx = np.sign(x)
    signy = np.sign(y)
    equ_count = np.sum((signx == signy))
    sagr = equ_count/N
    return sagr

def metric_for_VA(gt_V,gt_A,pred_V,pred_A):
    ccc_V,ccc_A = CCC_score(gt_V,pred_V),CCC_score(gt_A,pred_A)
    return ccc_V,ccc_A

def metric_for_VA_full(gt_V,gt_A,pred_V,pred_A):
    ccc_V,ccc_A = CCC_score(gt_V,pred_V),CCC_score(gt_A,pred_A)
    pcc_V,pcc_A = PCC(gt_V,pred_V),PCC(gt_A,pred_A)
    rmse_V,rmse_A = rmse(gt_V,pred_V),rmse(gt_A,pred_A)
    sagr_V,sagr_A = SAGR(gt_V,pred_V),SAGR(gt_A,pred_A)
    return ccc_V,ccc_A,pcc_V,pcc_A,rmse_V,rmse_A,sagr_V,sagr_A




def f1_score_max_for_AU_one_class(gt, pred, thresh,type=0):


    gt = gt[:,type]
    pred = pred[:,type]
    P = []
    R = []
    ACC = []
    F1 = []
    for i in thresh:
        new_pred = ((pred >= i) * 1).flatten()
        # if i==0.5:
        #     print("class type",type)
        #     print(f1_score(gt.flatten(), new_pred))
        P.append(precision_score(gt.flatten(), new_pred))
        R.append(recall_score(gt.flatten(), new_pred))
        ACC.append(accuracy_score(gt.flatten(), new_pred))
        F1.append(f1_score(gt.flatten(), new_pred))


    # P = np.array(P).flatten()
    # R = np.array(R).flatten()
    # F1 = 2 * P * R / (P + R)
    F1_MAX = max(F1)
    if F1_MAX < 0 or math.isnan(F1_MAX):
        F1_MAX = 0
        F1_THRESH = 0
        accuracy = 0
    else:
        idx_thresh = np.argmax(F1)
        F1_THRESH = thresh[idx_thresh]
        accuracy = ACC[idx_thresh]


    return F1,F1_MAX,F1_THRESH,accuracy



def f1_score_max(gt, pred, thresh,c=12):
    F1_s = []
    F1_t = []
    ACC = []
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    new_pred = ((pred >= 0.5) * 1)[:,1]
    for i in range(c):
        F1, F1_MAX, F1_THRESH,acc = f1_score_max_for_one_class(gt,pred,thresh,i)
        F1_s.append(F1_MAX)
        F1_t.append(F1_THRESH)
        ACC.append(acc)
    return F1_s,F1_t,ACC



# pred = np.array([[0.95,0.7,0.4,0.1,0.32,0.9],[0,0.1,0.23,0.87,0.13,0.54]])
# gt = np.array([[1,1,0,1,0,0],[0,0,0,0,0,0]])
# F1, F1_MAX, F1_THRESH = f1_score_max(gt,pred,[0.5 for i in range(6)])




def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, distances,
        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 30, 0.001)
    val, val_std, far = calculate_val(thresholds, distances,
        labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def calculate_roc(thresholds, distances, labels, nrof_folds=10):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

# def plot_roc(fpr,tpr,figure_name="roc.png"):
#     import matplotlib.pyplot as plt
#     plt.switch_backend('Agg')
#
#     from sklearn.metrics import roc_curve, auc
#     roc_auc = auc(fpr, tpr)
#     fig = plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='red',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     fig.savefig(figure_name, dpi=fig.dpi)

# dis1 =np.array( [1,2,1.2,4.1,8])
# dis2 = np.array([2,3,4.2,5,1])
# print(triplet_prediction_accuracy(dis1,dis2,5))