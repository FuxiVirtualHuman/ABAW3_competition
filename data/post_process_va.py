import glob
import os

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import savgol_filter
from copy import deepcopy
from sklearn.metrics import f1_score, confusion_matrix, precision_score
# import seaborn as sns
from tqdm import tqdm
def cnt_continuous(array):
    continuous = np.ones_like(array)
    windows = []
    for i in range(1, len(array)):
        if array[i] == array[i-1]:
            continuous[i] = continuous[i-1]+1
        else:
            continuous[i] = 1
            windows.append(continuous[i-1])
    return windows

def filter_outlier(array,  thresh=5, window=10, classes=np.arange(0,8)):
    cnt = 1
    prev = array[0]
    revise_value = classes
    # revise_value = [1,2,3,4,5,6]
    for i in range(1, len(array)):
        if array[i] == array[i-1]:
            cnt += 1
            continue
        else:
            if cnt <= thresh and array[i-1] in revise_value:
                array_nearby = array[max(i-window,0):min(i+thresh+window, len(array))]
                value = Counter(array_nearby).most_common(1)
                array[i-cnt:i] = [value[0][0]] * cnt
            else:
                cnt = 1
    return array

def filter_find_step(array, thresh=10, window=20):
    # value = Counter(array[:thresh]).most_common(1)[0][0]
    # cnt = 0
    new_array = np.zeros_like(array)
    # new_array[:thresh] = value
    for i in range(window, len(array)-window):
        clip = array[max(0, i-window):min(len(array),i+window)]
        stat = Counter(clip).most_common(1)
        if stat[0][1] > thresh:
            index = np.array(np.where(clip == stat[0][0]))
            start = i - window
            new_array[start+np.min(index):start+np.max(index)+1]=[stat[0][0]]*((np.max(index)-np.min(index))+1)
        else:
            pass
    return new_array.tolist()

def plot_confusion_matrix(matrix, title='Confusion matrix', cmap=plt.cm.gray_r):
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=False, ax=ax)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()

def revise_results(pred, prob, out, thresh=0.9):
    # 概率<0.9且pred为0的index
    prob = np.array(prob)
    out = np.array(out)
    index_below_prob = np.where(np.array(results['prob'])<0.9 )
    print('revising prediction results')
    for i in tqdm(list(index_below_prob)[0]):
        if prob[i] < thresh  and pred[i] == 0:
            new_label = np.argmax(out[i,1:])+1
            if out[i,new_label]>0.5:
                pred[i] = new_label
    return pred

def filter_most_common(array,window=25):
    count = Counter(array)
    candidate = []
    for label, n in count.most_common():
        if label in [0, 7]:
            continue
        if not candidate:
            candidate.append([label, n])
        else:
            if n > candidate[-1][1] * 0.3:
                candidate.append([label, n])
    candidate_label = [c[0] for c in candidate]+[0,7]
    index = [i for i in range(len(array)) if array[i] not in candidate_label]
    for i in index:
        clip = array[max(0, i-window): min(i+window, len(array))]
        new_label = Counter(clip).most_common(1)[0][0]
        array[i] = new_label


    return array

def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    # --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    # 0.97678916827853024
    # """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

def smooth_va():
    results_file = 'pred_va_folder0.pkl'
    validation_file = 'random_VA_validation_0.csv'
    print('reading result ')
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    print('reading img list')
    pd_data = pd.read_csv(validation_file)
    data = pd_data.to_dict("list")
    imgs_path = data['img']
    video_names = [ip.split('/')[0] for ip in imgs_path]
    count_videoname = Counter(video_names)
    count_videoname_sort = sorted(list(count_videoname.keys()), key=lambda x:video_names.index(x))

    cnt = 0
    pred_smooth_all = []



    pred_v = np.array(results['V'])
    pred_a = np.array(results['A'])

    with open('V_val.pkl', 'rb') as f:
        pred_v = pickle.load(f)['V']
    with open('A_val.pkl', 'rb') as f:
        pred_a = pickle.load(f)['A']

    label_v = np.array(data['V'])
    label_a = np.array(data['A'])

    smooth_pred = savgol_filter(pred_a,15, 3)
    ccc = concordance_correlation_coefficient(label_a, pred_a)
    ccc_smooth = concordance_correlation_coefficient(label_a, smooth_pred)
    for video in count_videoname_sort[:]:
        n_frames = count_videoname[video]
        pred = pred_v[cnt:cnt+n_frames]
        label = label_v[cnt:cnt+n_frames]
        fig,ax = plt.subplots(1,2)
        # smooth 1
        # pred_smooth = filter_outlier(deepcopy(pred), thresh=30, window=60, classes=np.arange(1, 7))  # 10 20
        # pred_smooth = filter_outlier(deepcopy(pred_smooth), thresh=20, window=40, classes=[0,7])  # 10 20
        #
        # pred_smooth_all += pred_smooth
        x = np.arange(0, n_frames)

        #
        # plt.scatter(x, pred,s=1, alpha=0.5)
        # plt.scatter(x, pred_smooth,s=1, alpha=0.5)
        # for i in range(12):
            # ax[i//4][i%4].scatter(x, np.array(label)[:,i],s=1, alpha=0.5)
            # ax[i//4][i%4].scatter(x, (np.array(pred)[:,i]>0).astype(np.int),s=1, alpha=0.5)
        # ax[0].set_xlim([-1, 1])
        ax[0].set_ylim([-1, 1])
        # ax[1].set_xlim([-1, 1])
        ax[1].set_ylim([-1, 1])

        ax[0].scatter(x, np.array(pred),s=1, alpha=0.5)
        ax[1].scatter(x, np.array(label),s=1, alpha=0.5)
        plt.show()
        plt.close()
        plt.clf()
        cnt += n_frames

    f1 = f1_score(results['label'], results['pred'], average=None)
    f1_smooth = f1_score(results['label'], np.array(pred_smooth_all), average=None)
    print('******f1*******')
    print(f1)
    print('mean:', np.mean(f1))
    print('******smooth f1*******')
    print(f1_smooth)
    print('mean:', np.mean(f1_smooth))
    print(1)

def merge():
    print('reading result ')
    results_file = 'pred_va_folder0.pkl'
    validation_file = 'random_VA_validation_0.csv' # 53.8
    pred_v = []
    pred_a = []
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    pred_v.append(np.array(results['V']))
    pred_a.append(np.array(results['A']))

    with open('pred_va_folder0_1.pkl', 'rb') as f:
        results = pickle.load(f)
    pred_v.append(np.array(results['V']))
    pred_a.append(np.array(results['A']))


    # result 2
    with open('V_val.pkl', 'rb') as f:
        data_v = pickle.load(f)['V']
    with open('A_val.pkl', 'rb') as f:
        data_a = pickle.load(f)['A']
    pred_v.append(np.array(data_v))
    pred_a.append(np.array(data_a))

    pred_v = np.stack(pred_v, axis=0)
    pred_a =np.stack(pred_a, axis=0)

    print('reading img list')
    pd_data = pd.read_csv(validation_file)
    data = pd_data.to_dict("list")
    imgs_path = data['img']
    video_names = [ip.split('/')[0] for ip in imgs_path]
    count_videoname = Counter(video_names)
    count_videoname_sort = sorted(list(count_videoname.keys()), key=lambda x:video_names.index(x))

    multi_train_file = 'ABAW3_new2_MTL_training_for_AU_fold0.csv'
    pd_data_multi = pd.read_csv(multi_train_file)
    data_multi = pd_data_multi.to_dict("list")
    imgs_path_multi = data_multi['img']
    video_names_multi = [ip.split('/')[0] for ip in imgs_path_multi]
    count_videoname_multi = Counter(video_names_multi)
    count_videoname_multi_sort = sorted(list(count_videoname_multi.keys()), key=lambda x:video_names_multi.index(x))


    label_v = np.array(data['V'])
    label_a = np.array(data['A'])

    print('*************** V ***************')
    for i in range(len(pred_v)):
        ccc = concordance_correlation_coefficient(label_v, pred_v[i])
        print(f'result {i}: {ccc:.06f}')
    ccc_avg = concordance_correlation_coefficient(label_v, np.mean(pred_v, axis=0))
    print(f'Avg: {ccc_avg:.06f}')
    cnt = 0

    pred_not_seen = []
    pred_seen = []
    label_not_seen = []
    label_seen = []
    pred_v_avg = np.mean(pred_v, axis=0)
    for video in count_videoname_sort:
        n_frames = count_videoname[video]
        pred = pred_v_avg[cnt:cnt+n_frames]
        label = label_v[cnt:cnt+n_frames]
        if video in count_videoname_multi_sort:
            pred_seen.append(pred)
            label_seen.append(label)
        else:
            pred_not_seen.append(pred)
            label_not_seen.append(label)
        cnt+=n_frames

    print('*************** A ***************')
    for i in range(len(pred_v)):
        ccc = concordance_correlation_coefficient(label_a, pred_a[i])
        print(f'result {i}: {ccc:.06f}')
    ccc_avg = concordance_correlation_coefficient(label_a, np.mean(pred_a, axis=0))
    ccc_seen = concordance_correlation_coefficient(np.concatenate(label_seen,0), np.concatenate(pred_seen,0))
    ccc_not_seen = concordance_correlation_coefficient(np.concatenate(label_not_seen,0), np.concatenate(pred_not_seen,0))
    print(f'Avg: {ccc_avg:.06f}; seen: {ccc_seen:.06f}; not seen :{ccc_not_seen:.06f}')


def merge_va():
    # 基于A，根据相关性分组。
    # 0,7,8
    # 1,2,4,6,9
    # 3,5,7

    # 基于V， 根据相关性分组。
    # 0
    # 1,2,4,6,9
    # 3,5,7
    # 8

    print('reading result ')
    root = '/project/ABAW2_dataset/original_dataset/submission/single_task'
    save_file = os.path.join(root, 'VA_avg_all.csv')
    result_folders = glob.glob(root+'/Eff_MULTIfor_VA*')
    # result_files = [os.path.join(rf, 'VA_pred_results.csv') for rf in result_folders]
    test_file = '../annos/ABAW3_VA_test.csv' # 53.8
    pred_v = []
    pred_a = []
    for fold in result_folders:
        file = glob.glob(fold+'/*.csv')[0]
        results = pd.read_csv(file)
        pred_v.append(np.array(results['V']))
        pred_a.append(np.array(results['A']))

    pred_v = np.clip(np.stack(pred_v, axis=0),-1,1)
    pred_a = np.clip(np.stack(pred_a, axis=0),-1,1)

    # check results
    for i in range(len(pred_v)):
        ccc = []
        for j in range(len(pred_v)):
            ccc.append(concordance_correlation_coefficient(pred_v[i], pred_v[j]))

        ccc_out = [','.join(f'{c:.4f}' for c in ccc)]
        print(f'{i},', ccc_out, np.mean(ccc))

    print('reading img list')
    pd_data = pd.read_csv(test_file)
    pred_v_avg = np.mean(pred_v, axis=0)
    pred_a_avg = np.mean(pred_a, axis=0)
    pd_data['V'] = pred_v_avg
    pd_data['A'] = pred_a_avg

    # pd_data.to_csv(save_file, index=False)


def merge_au():

    print('reading result ')
    root = '/project/ABAW2_dataset/original_dataset/submission/single_task'
    save_file_avg = os.path.join(root, 'AU_avg_all.csv')
    save_file_vote = os.path.join(root, 'AU_vote_all.csv')

    result_folders = glob.glob(root+'/Eff_MULTIfor_AU*')
    # result_files = [os.path.join(rf, 'VA_pred_results.csv') for rf in result_folders]
    test_file_all = '../annos/ABAW3_AU_test.csv' # 53.8
    test_file = '../annos/ABAW3_new_AU_test1.csv'
    pred = []
    for fold in result_folders:
        file = glob.glob(fold+'/*.csv')[0]
        # with open(file, 'rb') as f:
        #     results = pickle.load(f)
        results = pd.read_csv(file)
        # print(results.keys())
        pred.append(np.array([list(map(float,p.strip('[ ]').split())) for p in results['pred']]))

    pred = np.stack(pred, axis=0)

    # check results
    for i in range(len(pred)):
        ccc = []
        for j in range(len(pred)):
            ccc_k = 0
            for k in range(12):
                ccc_k+=concordance_correlation_coefficient(pred[i,:,k], pred[j,:,k])

            ccc.append(ccc_k/12.)
        ccc_out = [','.join(f'{c:.4f}' for c in ccc)]
        print(f'{i},', ccc_out, np.mean(ccc))
    return
    pd_data = pd.read_csv(test_file_all)

    print(f'reading img list: {len(pd_data)}')
    # avg
    pred_avg = np.mean(pred, axis=0)
    pred_avg = (pred_avg>0).astype(np.int)

    print(pred_avg)
    pd_data['AU'] = pred_avg.tolist()
    pd_data.to_csv(save_file_avg, index=False)

    # vote
    pred_vote = (pred>0).astype(np.int)
    pred_vote = pred_vote.reshape(5,-1).transpose(1,0)
    pred_vote = [Counter(pred_vote[i]).most_common(1)[0][0] for i in tqdm(range(len(pred_vote)))]
    pred_vote = np.array(pred_vote).reshape(-1,12)



    print(pred_vote)
    pd_data['AU'] = pred_vote.tolist()
    pd_data.to_csv(save_file_vote, index=False)


def reformat_au_results():
    print('reading result ')
    root = '/project/ABAW2_dataset/original_dataset/submission/single_task'
    # save_file_csv = os.path.join(root, 'AU_avg_all.csv')
    # save_file_vote = os.path.join(root, 'AU_vote_all.csv')

    result_folders = glob.glob(root + '/Eff_MULTIfor_AU*')
    # result_files = [os.path.join(rf, 'VA_pred_results.csv') for rf in result_folders]
    test_file = '../annos/ABAW3_AU_test.csv'  # 53.8
    test_file_all = '../annos/ABAW3_new_AU_test1.csv'
    pd_data = pd.read_csv(test_file)
    data_test = pd_data
    data = pd_data.to_dict("list")
    imgs_path = data['img']
    video_names = [ip.split('/')[0] for ip in imgs_path]
    count_videoname = Counter(video_names)
    count_videoname_sort = sorted(list(count_videoname.keys()), key=lambda x:video_names.index(x))

    pd_data = pd.read_csv(test_file_all)
    data = pd_data.to_dict("list")
    imgs_path = data['img']
    video_names = [ip.split('/')[0] for ip in imgs_path]
    count_videoname = Counter(video_names)
    count_videoname_sort_all = sorted(list(count_videoname.keys()), key=lambda x: video_names.index(x))

    frame_index = {}
    cnt = 0

    for vname in count_videoname_sort_all:
        frame_index[vname] = [cnt, cnt + count_videoname[vname]]
        cnt += count_videoname[vname]

    for fold in result_folders:
        file = glob.glob(fold + '/*.pkl')[0]
        with open(file, 'rb') as f:
            results = pickle.load(f)
        print(results.keys())
        pred = []
        for vname in count_videoname_sort:
            pred.append(results['pred'][frame_index[vname][0]:frame_index[vname][1]])
        pred = np.concatenate(pred, axis=0)
        label  = (pred>0).astype(np.int)
        data_test['label'] = [l for l in label]
        data_test['pred'] = [p for p in pred]
        data_test.to_csv(file.replace('.pkl', '.csv'), index=False)


def merge_multi():
    print('reading result ')
    root = '/project/ABAW2_dataset/original_dataset/submission/multi_task'

    result_folders = glob.glob(root + '/*fold*')
    test_file = '../annos/ABAW3_MTL_test.csv'  # 53.8
    pd_data = pd.read_csv(test_file)
    data_test = pd_data
    data = pd_data.to_dict("list")
    imgs_path = data['img']
    video_names = [ip.split('/')[0] for ip in imgs_path]
    count_videoname = Counter(video_names)
    count_videoname_sort = sorted(list(count_videoname.keys()), key=lambda x:video_names.index(x))

    # VA
    V = []
    A = []
    for fold in result_folders:
        file = os.path.join(fold,'VA_pred.csv')
        pd_data = pd.read_csv(file)
        V.append(np.array(pd_data['V']))
        A.append(np.array(pd_data['A']))

    res_v = np.clip(np.mean(np.stack(V),axis=0),-1,1)
    res_a = np.clip(np.mean(np.stack(V),axis=0), -1,1)
    data_test['V'] = res_v
    data_test['A'] = res_a

    # au
    au = []
    for fold in result_folders:
        file = os.path.join(fold, 'AU_pred.csv')
        pd_data = pd.read_csv(file)
        au.append(np.array([list(map(float,p.strip('[ ]').split())) for p in pd_data['AU_label']]))
    # avg
    pred_avg = np.mean(au, axis=0)
    pred_avg = (pred_avg>0).astype(np.int)

    data_test['AU'] = pred_avg.tolist()

    # vote
    # pred_vote = (au>0).astype(np.int)
    # pred_vote = pred_vote.reshape(5,-1).transpose(1,0)
    # pred_vote = [Counter(pred_vote[i]).most_common(1)[0][0] for i in tqdm(range(len(pred_vote)))]
    # pred_vote = np.array(pred_vote).reshape(-1,12)

    # EXP
    exp = []
    for fold in result_folders:
        file = os.path.join(fold, 'EXP_pred.csv')
        pd_data = pd.read_csv(file)
        label = pd_data['exp_label']
        exp.append(label)

    # TODO： vote
    # exp_vote = None
    # data_test['EXP'] = exp_vote

    data_test.to_csv(os.path.join(root, 'avg_all.csv'))
    # data_test.to_csv(os.path.join(root, 'avg_all.csv'))

    return


if __name__ == '__main__':
    # smooth_va()
    # merge()
    # merge_va()
    # merge_au()
    merge_multi()
    # reformat_au_results()