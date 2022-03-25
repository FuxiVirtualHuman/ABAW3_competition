import os
import json
import glob
import numpy as np
from tqdm import tqdm

def process_3dmm_178():
    root = '/project/ABAW2_dataset/original_dataset/crop_face_3dmm_coff'
    save_root = '/project/ABAW2_dataset/original_dataset/crop_face_3dmm_coff_processed_178'
    os.makedirs(save_root, exist_ok=True)
    json_files = glob.glob((os.path.join(root+'/*.json')))
    for i, json_file in enumerate(json_files):
        print(f'{i}/{len(json_files)} : ', json_file)
        save_file = os.path.join(save_root,json_file.split('/')[-1].replace('.json','.txt'))
        if os.path.exists(save_file):
            continue
        with open(json_file,'r') as f:
            data = json.load(f)
        features_all = []
        n = 0
        try:
            total = max(map(int,data.keys()))
        except:
            print('file error:', json_file)
            continue
        for i in tqdm(range(int(total))):
            frame_n = f'{i}:05d'
            if frame_n in data:
                n += 1
                features_all.append(data[frame_n]['detail']+data[frame_n]['exp'])
            else:
                clip = list(data.values())[max(0, n-2):min(n+2, total)]
                feature_clip = [d['detail'][0] + d['exp'][0] for d in clip]
                feature = np.mean(np.array(feature_clip), axis=0)
                features_all.append(feature)
        np.savetxt(save_file,np.array(features_all))
    print(1)

if __name__ == '__main__':
    # file = '/project/ABAW2_dataset/original_dataset/crop_face_3dmm_coff_processed_chazhi/video13_aligned.txt'
    # emb = np.loadtxt(file, delimiter=',') #10567
    process_3dmm_178()
    print(1)
