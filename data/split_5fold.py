import csv
import os.path
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm
random.seed(0)
def split_5fold_and_val(task,fold_n=5, root='../annos'):

    print('Task: ', task)
    root = root
    pd_data1 = pd.read_csv(os.path.join(root, f'ABAW3_new_{task}_training.csv'))
    pd_data2 = pd.read_csv(os.path.join(root, f'ABAW3_new_{task}_validation.csv'))

    pd_data = pd.concat([pd_data1, pd_data2])
    pd_img = pd_data.to_dict()['img']
    video_counter = Counter([pi.split("/")[0] for pi in pd_img.values()])
    video_names = list(video_counter.keys())
    random.shuffle(video_names)
    video_names_val = video_names[:int(len(video_names)*0.1)]
    video_names_fold = video_names[int(len(video_names)*0.1):]
    n_lenth = len(video_names_fold) / 5.

    pd_data_test = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in video_names_val])
    pd_data_test.to_csv(os.path.join(root,f'random_{task.upper()}_test.csv'), index=False)
    print('Test set saved!')
    print('Processing 5 fold training/val set ....')
    for i in tqdm(range(fold_n)):
        names_train = video_names_fold[int(i*n_lenth):int((i+1)*n_lenth)]
        names_test = video_names_fold[:int(i*n_lenth)] + video_names_fold[int((i+1)*n_lenth):]
        pd_data_train = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in names_train])
        pd_data_val = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in names_test])
        pd_data_val.to_csv(os.path.join(root, f'random_{task.upper()}_train_{i}.csv'), index=False)
        pd_data_train.to_csv(os.path.join(root, f'random_{task.upper()}_validation_{i}.csv'), index=False)
    print('done!')
    return

def split_5fold_and_val_cls(task,fold_n=5, root='../annos'):

    print('Task: ', task)
    root = root
    pd_data1 = pd.read_csv(os.path.join(root, f'ABAW3_new_{task}_training.csv'))
    pd_data2 = pd.read_csv(os.path.join(root, f'ABAW3_new_{task}_validation.csv'))

    pd_data = pd.concat([pd_data1, pd_data2])
    pd_img = pd_data.to_dict()['img']
    video_counter = Counter([pi.split("/")[0] for pi in pd_img.values()])
    video_names = list(video_counter.keys())
    video_names_8 = [[],[],[],[],[],[],[],[],]
    print('get labels for videos')
    for video_name in tqdm(video_names):
        data_clip = pd_data[pd_data['img'].str.contains(video_name)]
        n_frame = len(data_clip)
        label_cnt = Counter(list(data_clip['Expression']))
        label = label_cnt.most_common(1)[0][0]
        for l in label_cnt.most_common():
            if l[0] not in [0,7]:
                label = l[0]
        video_names_8[label].append(video_name)
    # random.shuffle(video_names)
    video_names_val = [vname_cls[:round(len(vname_cls)*0.1)] for vname_cls in video_names_8]
    video_names_fold = [vname_cls[round(len(vname_cls)*0.1):] for vname_cls in video_names_8]
    n_lenth = [len(vfold) / 5. for vfold in video_names_fold ]

    video_names_val = [item for sublist in video_names_val for item in sublist]
    pd_data_test = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in video_names_val])
    pd_data_test.to_csv(os.path.join(root,f'random_{task.upper()}_cls_test.csv'), index=False)
    print('Test set saved!')
    print('Processing 5 fold training/val set ....')
    for i in tqdm(range(fold_n)):
        names_train = [vname[round(i*len(vname)/5.):round((i+1)*len(vname)/5.)] for vname in video_names_fold]
        names_train = [item for sublist in names_train for item in sublist]
        names_test  = [vname[:round(i*len(vname)/5.)] for vname in video_names_fold] + [vname[round((i+1)*len(vname)/5.):] for vname in video_names_fold]
        names_test = [item for sublist in names_test for item in sublist]
        # names_test = video_names_fold[:round(i*n_lenth)] + video_names_fold[int((i+1)*n_lenth):]
        pd_data_train = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in names_train])
        pd_data_val = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in names_test])
        pd_data_val.to_csv(os.path.join(root, f'random_{task.upper()}_train_cls_{i}.csv'), index=False)
        pd_data_train.to_csv(os.path.join(root, f'random_{task.upper()}_validation_cls_{i}.csv'), index=False)
    print('done!')
    return


if __name__ == '__main__':
    # split_5fold_and_val('va', fold_n=5, root='../annos')

    split_5fold_and_val_cls('exp')