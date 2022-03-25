import os
import cv2
import pandas as pd
import torch.utils.data as data
from PIL import Image
import torch
from PIL import ImageFile
import numpy as np
import torchvision.transforms.transforms as transforms
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ABAW2_multitask_data(data.dataset.Dataset):
    """
    Args:
        # this data type for images with three kinds of annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_AU = self.data['AU']
        self.labels_V = self.data['V']
        self.labels_A = self.data['A']
        self.labels_Exp = self.data['EXP']
        self.img_path = img_path
        #self.embs = self.data["emb"]

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path,anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        label_exp = int(self.labels_Exp[index])


        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        return anc_img,label_au_,label_V,label_A,label_exp,anc_list



class ABAW2_multitask_data2(data.dataset.Dataset):
    """
    Args:
        # this data type for images with specific annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, Exp_csv_file,VA_csv_file,AU_csv_file,img_path,Exp_VA_transform=None,AU_transform=None):
        self.Exp_VA_transform = Exp_VA_transform
        self.AU_transform = AU_transform
        self.Exp_pd_data = pd.read_csv(Exp_csv_file)
        self.AU_pd_data = pd.read_csv(AU_csv_file)
        self.VA_pd_data = pd.read_csv(VA_csv_file)
        self.Exp_data = self.Exp_pd_data.to_dict("list")
        self.VA_data = self.VA_pd_data.to_dict("list")
        self.AU_data = self.AU_pd_data.to_dict("list")
        self.Exp_imgs = self.Exp_data['img']
        self.AU_imgs = self.AU_data['img']
        self.VA_imgs = self.VA_data['img']
        self.labels_AU = self.AU_data['AU']
        self.labels_V = self.VA_data['V']
        self.labels_A = self.VA_data['A']
        self.labels_Exp = self.Exp_data['Expression']
        self.img_path = img_path
        #self.embs = self.data["emb"]

    def __len__(self):
        return max(len(self.Exp_data["img"]),len(self.VA_data["img"]),len(self.AU_data["img"]))

    def __getitem__(self, index):
        va_index,au_index,exp_index = index,index,index
        if index >= len(self.AU_data["img"]):
            au_index  = index - len(self.AU_data["img"])

        if index >= len(self.Exp_data["img"]) and index< 2*len(self.Exp_data["img"]):
            exp_index = index - len(self.Exp_data["img"])

        elif index >= 2*len(self.Exp_data["img"]) and index< 3*len(self.Exp_data["img"]):
            exp_index = index - 2*len(self.Exp_data["img"])

        elif index >= 3*len(self.Exp_data["img"]) and index< 4*len(self.Exp_data["img"]):
            exp_index = index - 3*len(self.Exp_data["img"])

        elif index >= 4*len(self.Exp_data["img"]) and index< 5*len(self.Exp_data["img"]):
            exp_index = index - 4*len(self.Exp_data["img"])

        if index >= len(self.VA_data["img"]) and index < 2 * len(self.VA_data["img"]):
            va_index = index - len(self.VA_data["img"])

        elif index >= 2 * len(self.VA_data["img"]) and index < 3 * len(self.VA_data["img"]):
            va_index = index - 2 * len(self.VA_data["img"])

        elif index >= 3 * len(self.VA_data["img"]) and index < 4 * len(self.VA_data["img"]):
            va_index = index - 3 * len(self.VA_data["img"])

        elif index >= 4 * len(self.VA_data["img"]) and index < 5 * len(self.VA_data["img"]):
            va_index = index - 4 * len(self.VA_data["img"])

        # exp_data
        exp_name_list = self.Exp_imgs[exp_index]
        exp_img = Image.open(os.path.join(self.img_path,exp_name_list))
        if exp_img.getbands()[0] != 'R':
            exp_img = exp_img.convert('RGB')
        label_exp = int(self.labels_Exp[exp_index])

        #AU_data
        au_name_list = self.AU_imgs[au_index]
        au_img = Image.open(os.path.join(self.img_path, au_name_list))
        if au_img.getbands()[0] != 'R':
            au_img = au_img.convert('RGB')
        label_au = self.labels_AU[au_index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0
        label_au_ = torch.tensor(label_au_)

        #VA_data
        va_name_list = self.VA_imgs[va_index]
        va_img = Image.open(os.path.join(self.img_path, va_name_list))
        if va_img.getbands()[0] != 'R':
            va_img = va_img.convert('RGB')
        label_V = float(self.labels_V[va_index])
        label_A = float(self.labels_A[va_index])

        if self.AU_transform is not None and self.Exp_VA_transform is not None:
            exp_img = self.Exp_VA_transform(exp_img)
            va_img = self.Exp_VA_transform(va_img)
            au_img = self.AU_transform(au_img)

        return au_img,va_img,exp_img,label_au_,label_V,label_A,label_exp



class ABAW2_multitask_data3(data.dataset.Dataset):
    """
    Args:
        # this data type for images with specific annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, Exp_csv_file,VA_csv_file,AU_csv_file,img_path1,img_path2,img_path3,Exp_VA_transform=None,AU_transform=None):
        self.Exp_VA_transform = Exp_VA_transform
        self.AU_transform = AU_transform
        self.Exp_pd_data = pd.read_csv(Exp_csv_file)
        self.AU_pd_data = pd.read_csv(AU_csv_file)
        self.VA_pd_data = pd.read_csv(VA_csv_file)
        self.Exp_data = self.Exp_pd_data.to_dict("list")
        self.VA_data = self.VA_pd_data.to_dict("list")
        self.AU_data = self.AU_pd_data.to_dict("list")
        self.AU_types = self.AU_data['type']
        self.Exp_imgs = self.Exp_data['img']
        self.AU_imgs = self.AU_data['img']
        self.VA_imgs = self.VA_data['img']
        self.labels_AU = self.AU_data['AU']
        self.labels_V = self.VA_data['V']
        self.labels_A = self.VA_data['A']
        self.labels_Exp = self.Exp_data['Expression']
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.img_path3 = img_path3
        #self.embs = self.data["emb"]

    def __len__(self):
        return max(len(self.Exp_data["img"]),len(self.VA_data["img"]),len(self.AU_data["img"]))

    def __getitem__(self, index):
        va_index,au_index,exp_index = index,index,index
        if index >= len(self.AU_data["img"]):
            au_index  = index - len(self.AU_data["img"])

        if index >= len(self.Exp_data["img"]) and index< 2*len(self.Exp_data["img"]):
            exp_index = index - len(self.Exp_data["img"])

        elif index >= 2*len(self.Exp_data["img"]) and index< 3*len(self.Exp_data["img"]):
            exp_index = index - 2*len(self.Exp_data["img"])

        elif index >= 3*len(self.Exp_data["img"]) and index< 4*len(self.Exp_data["img"]):
            exp_index = index - 3*len(self.Exp_data["img"])

        elif index >= 4*len(self.Exp_data["img"]) and index< 5*len(self.Exp_data["img"]):
            exp_index = index - 4*len(self.Exp_data["img"])

        if index >= len(self.VA_data["img"]) and index < 2 * len(self.VA_data["img"]):
            va_index = index - len(self.VA_data["img"])

        elif index >= 2 * len(self.VA_data["img"]) and index < 3 * len(self.VA_data["img"]):
            va_index = index - 2 * len(self.VA_data["img"])

        elif index >= 3 * len(self.VA_data["img"]) and index < 4 * len(self.VA_data["img"]):
            va_index = index - 3 * len(self.VA_data["img"])

        elif index >= 4 * len(self.VA_data["img"]) and index < 5 * len(self.VA_data["img"]):
            va_index = index - 4 * len(self.VA_data["img"])

        # exp_data
        exp_name_list = self.Exp_imgs[exp_index]
        exp_img = Image.open(os.path.join(self.img_path1,exp_name_list))
        if exp_img.getbands()[0] != 'R':
            exp_img = exp_img.convert('RGB')
        label_exp = int(self.labels_Exp[exp_index])

        #AU_data
        au_name_list = self.AU_imgs[au_index]
        au_type = self.AU_types[au_index]
        if au_type=="Aff2":
            au_img = Image.open(os.path.join(self.img_path1, au_name_list))
        elif au_type=="bp4d":
            au_img = Image.open(os.path.join(self.img_path2, au_name_list))
        elif au_type=="disfa":
            au_img = Image.open(os.path.join(self.img_path3, au_name_list))
        if au_img.getbands()[0] != 'R':
            au_img = au_img.convert('RGB')
        label_au = self.labels_AU[au_index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0
        label_au_ = torch.tensor(label_au_)

        #VA_data
        va_name_list = self.VA_imgs[va_index]
        va_img = Image.open(os.path.join(self.img_path1, va_name_list))
        if va_img.getbands()[0] != 'R':
            va_img = va_img.convert('RGB')
        label_V = float(self.labels_V[va_index])
        label_A = float(self.labels_A[va_index])

        if self.AU_transform is not None and self.Exp_VA_transform is not None:
            exp_img = self.Exp_VA_transform(exp_img)
            va_img = self.Exp_VA_transform(va_img)
            au_img = self.AU_transform(au_img)

        return au_img,va_img,exp_img,label_au_,label_V,label_A,label_exp,au_type



class ABAW2_multitask_embedding_data(data.dataset.Dataset):
    """
    Args:
        # this data type for images with three kinds of annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_AU = self.data['AU']
        self.labels_V = self.data['V']
        self.labels_A = self.data['A']
        self.labels_Exp = self.data['EXP']
        self.emb_before2, self.emb_before1, self.emb_current, self.emb_after1, self.emb_after2 = \
            self.data['emb_before2'], \
            self.data['emb_before1'], self.data['emb_current'], self.data['emb_after1'], self.data[
                'emb_after2']
        self.img_path = img_path
        #self.embs = self.data["emb"]
    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb
    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path , anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        label_exp = int(self.labels_Exp[index])

        emb1, emb2, emb3, emb4, emb5 = self.emb_before2[index], self.emb_before1[
            index], self.emb_current[index], self.emb_after1[index], self.emb_after2[index]
        emb1, emb2, emb3, emb4, emb5 = self.process_emb(emb1), self.process_emb(
            emb2), self.process_emb(emb3), self.process_emb(emb4), self.process_emb(emb5)
        embs = np.concatenate((emb1, emb2, emb3, emb4, emb5), axis=0)

        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        return anc_img,label_au_,label_V,label_A,label_exp,embs,anc_list





class ABAW3_multitask_data(data.dataset.Dataset):
    def __init__(self, csv_file,img_path, emb_dict = None, mfcc_dict = None, word_dict = None,coff_dict=None, mfcc_seq = 15, word_seq=15, emb_seq=15,coff_seq=15,transform=None):
        self.transform = transform
        self.csv_file = pd.read_csv(csv_file)

        self.imgs = self.csv_file["img"]
        self.labels_AU = self.csv_file["AU"]
        self.labels_Exp = self.csv_file["EXP"]
        self.labels_V = self.csv_file["V"]
        self.labels_A = self.csv_file["A"]
        self.img_path = img_path
        self.emb_dict = emb_dict
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.coff_dict = coff_dict
        self.coff_seq = coff_seq

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(os.path.join(self.img_path,img_name))
        img = img.convert('RGB')
        if self.transform is not None:
           img = self.transform(img)

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0
        label_au_ = torch.tensor(label_au_)

        label_exp = int(self.labels_Exp[index])

        feas = []
        if self.coff_dict!=None:
            v_id = img_name.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(img_name.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.coff_seq
            frame_end = frame+self.coff_seq

            S,B = coff_arr.shape
            frame = min(S-1,frame)

            coff_fea = torch.zeros(2*self.coff_seq+1,50)
            total_len = 2*self.coff_seq+1
            coff = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    coff_fea[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(coff_fea)

        if self.emb_dict!=None:
            v_id = img_name.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(img_name.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.emb_seq
            frame_end = frame+self.emb_seq

            S,B = emb_arr.shape
            frame = min(S-1,frame)
            #print(S,B)
            emb_fea = torch.zeros(2*self.emb_seq+1,16)
            #print(mfcc_arr.shape)
            total_len = 2*self.emb_seq+1
            embs = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    # print(f,frame)
                    #print(emb_arr[f,:])
                    emb_fea[index,:] = emb_arr[f,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(emb_fea)

        if self.mfcc_dict !=None:
            v_id = img_name.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(img_name.split("/")[1].split(".")[0]) - 1
            #mfcc_file = os.path.join(self.mfcc_path,v_id)
            #mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            #mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame-self.mfcc_seq
            frame_end = frame+self.mfcc_seq
            s,f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2*self.mfcc_seq+1)
            #print(mfcc_arr.shape)
            total_len = 2*self.mfcc_seq+1
            if frame_start<0:
                if f-1<frame_end:
                    delta = frame_end-(f-1)
                    mfcc_fea[:,-frame_start:(total_len-delta)] = mfcc_arr[:,0:frame_end+1]
                else:
                    mfcc_fea[:,-frame_start:total_len] = mfcc_arr[:,0:frame_end+1]
            elif frame_end>=f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:,:total_len-(frame_end-f+1)] = mfcc_arr[:,frame_start:]
            else:
                mfcc_fea = mfcc_arr[:,frame_start:frame_end+1]
            mfcc_fea = mfcc_fea.permute(1,0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            #return anc_img,label_V,label_A,mfcc_fea,anc_list


        if self.word_dict !=None:
            v_id = img_name.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(img_name.split("/")[1].split(".")[0])-1
            words = []
            frames_start = int(frame)-self.word_seq
            frames_end = int(frame)+self.word_seq
            #print(range(frames_start,frames_end))
            for f in range(frames_start,frames_end+1):

                if f<0:
                    words.append("-1")
                elif f>=len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            #print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)
        return img,label_V,label_A,label_au_,label_exp,feas,img_name




class ABAW3_multitask_data_nofull(data.dataset.Dataset):
    def __init__(self, csv_file,img_path, emb_dict = None, mfcc_dict = None, word_dict = None,coff_dict=None, mfcc_seq = 15, word_seq=15, emb_seq=15,coff_seq=15,transform=None):
        self.transform = transform
        self.csv_file = pd.read_csv(csv_file,dtype={'img': "string","AU":"string","EXP":"string","V":"string","A":"string"})

        self.imgs = self.csv_file["img"]
        self.labels_AU = self.csv_file["AU"]
        self.labels_Exp = self.csv_file["EXP"]
        self.labels_V = self.csv_file["V"]
        self.labels_A = self.csv_file["A"]
        self.img_path = img_path
        self.emb_dict = emb_dict
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.coff_dict = coff_dict
        self.coff_seq = coff_seq

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = Image.open(os.path.join(self.img_path,img_name))
        img = img.convert('RGB')
        if self.transform is not None:
           img = self.transform(img)
        has_au = 1
        has_exp =1
        has_VA = 1

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        if label_V==-5:
            has_VA = 0
            label_V= 0.0
            label_A= 0.0


        label_au = self.labels_AU[index]
        if label_au=="-5" or label_au==-5:
            label_au_ = [-5.0 for i in range(12)]
            has_au = 0
        else:
            label_au = label_au.split(" ")
            label_au_ = [0.0 for i in range(len(label_au))]
            for i in range(len(label_au)):
                if label_au[i]!='0':
                    label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)

        label_exp = int(self.labels_Exp[index])
        if label_exp == -5:
            has_exp = 0
            label_exp = -5

        feas = []
        if self.coff_dict!=None:
            v_id = img_name.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(img_name.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.coff_seq
            frame_end = frame+self.coff_seq

            S,B = coff_arr.shape
            frame = min(S-1,frame)

            coff_fea = torch.zeros(2*self.coff_seq+1,50)
            total_len = 2*self.coff_seq+1
            coff = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    coff_fea[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(coff_fea)

        if self.emb_dict!=None:
            v_id = img_name.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(img_name.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.emb_seq
            frame_end = frame+self.emb_seq

            S,B = emb_arr.shape
            frame = min(S-1,frame)
            #print(S,B)
            emb_fea = torch.zeros(2*self.emb_seq+1,16)
            #print(mfcc_arr.shape)
            total_len = 2*self.emb_seq+1
            embs = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    # print(f,frame)
                    #print(emb_arr[f,:])
                    emb_fea[index,:] = emb_arr[f,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(emb_fea)

        if self.mfcc_dict !=None:
            v_id = img_name.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(img_name.split("/")[1].split(".")[0]) - 1
            #mfcc_file = os.path.join(self.mfcc_path,v_id)
            #mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            #mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame-self.mfcc_seq
            frame_end = frame+self.mfcc_seq
            s,f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2*self.mfcc_seq+1)
            #print(mfcc_arr.shape)
            total_len = 2*self.mfcc_seq+1
            if frame_start<0:
                if f-1<frame_end:
                    delta = frame_end-(f-1)
                    mfcc_fea[:,-frame_start:(total_len-delta)] = mfcc_arr[:,0:frame_end+1]
                else:
                    mfcc_fea[:,-frame_start:total_len] = mfcc_arr[:,0:frame_end+1]
            elif frame_end>=f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:,:total_len-(frame_end-f+1)] = mfcc_arr[:,frame_start:]
            else:
                mfcc_fea = mfcc_arr[:,frame_start:frame_end+1]
            mfcc_fea = mfcc_fea.permute(1,0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            #return anc_img,label_V,label_A,mfcc_fea,anc_list


        if self.word_dict !=None:
            v_id = img_name.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(img_name.split("/")[1].split(".")[0])-1
            words = []
            frames_start = int(frame)-self.word_seq
            frames_end = int(frame)+self.word_seq
            #print(range(frames_start,frames_end))
            for f in range(frames_start,frames_end+1):

                if f<0:
                    words.append("-1")
                elif f>=len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            #print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)
        return img,label_V,label_A,label_au_,label_exp,feas,img_name,has_au,has_exp,has_VA


class ABAW2_multitask_embedding_data2(data.dataset.Dataset):
    """
    Args:
        # this data type for images with specific annotations
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, Exp_csv_file,VA_csv_file,AU_csv_file,img_path,transform=None):
        self.transform = transform
        self.Exp_pd_data = pd.read_csv(Exp_csv_file)
        self.AU_pd_data = pd.read_csv(AU_csv_file)
        self.VA_pd_data = pd.read_csv(VA_csv_file)
        self.Exp_data = self.Exp_pd_data.to_dict("list")
        self.VA_data = self.VA_pd_data.to_dict("list")
        self.AU_data = self.AU_pd_data.to_dict("list")
        self.Exp_imgs = self.Exp_data['img']
        self.AU_imgs = self.AU_data['img']
        self.VA_imgs = self.VA_data['img']
        self.labels_AU = self.AU_data['AU']
        self.labels_V = self.VA_data['V']
        self.labels_A = self.VA_data['A']
        self.labels_Exp = self.Exp_data['Expression']

        self.VA_emb_before2,self.VA_emb_before1,self.VA_emb_current,self.VA_emb_after1,self.VA_emb_after2 = self.VA_data['emb_before2'],\
                                                                                                            self.VA_data['emb_before1'],self.VA_data['emb_current'],self.VA_data['emb_after1'],self.VA_data['emb_after2']
        self.AU_emb_before2, self.AU_emb_before1, self.AU_emb_current, self.AU_emb_after1, self.AU_emb_after2 = \
        self.AU_data['emb_before2'], \
        self.AU_data['emb_before1'], self.AU_data['emb_current'], self.AU_data['emb_after1'], self.AU_data['emb_after2']

        self.Exp_emb_before2, self.Exp_emb_before1, self.Exp_emb_current, self.Exp_emb_after1, self.Exp_emb_after2 = \
            self.Exp_data['emb_before2'], \
            self.Exp_data['emb_before1'], self.Exp_data['emb_current'], self.Exp_data['emb_after1'], self.Exp_data[
                'emb_after2']

        self.img_path = img_path
        #self.embs = self.data["emb"]

    def __len__(self):
        return max(len(self.Exp_data["img"]),len(self.VA_data["img"]),len(self.AU_data["img"]))
    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb

    def __getitem__(self, index):
        va_index,au_index,exp_index = index,index,index
        if index >= len(self.AU_data["img"]):
            au_index  = index -  len(self.AU_data["img"])

        if index >= len(self.Exp_data["img"]) and index< 2*len(self.Exp_data["img"]):
            exp_index = index - len(self.Exp_data["img"])

        elif index >= 2*len(self.Exp_data["img"]) and index< 3*len(self.Exp_data["img"]):
            exp_index = index - 2*len(self.Exp_data["img"])

        elif index >= 3*len(self.Exp_data["img"]) and index< 4*len(self.Exp_data["img"]):
            exp_index = index - 3*len(self.Exp_data["img"])

        elif index >= 4*len(self.Exp_data["img"]) and index< 5*len(self.Exp_data["img"]):
            exp_index = index - 4*len(self.Exp_data["img"])

        # exp_data
        exp_name_list = self.Exp_imgs[exp_index]
        #exp_name_list = exp_name_list.split("_aligned")[0]+"/" + exp_name_list
        exp_img = Image.open(os.path.join(self.img_path,exp_name_list))
        if exp_img.getbands()[0] != 'R':
            exp_img = exp_img.convert('RGB')
        label_exp = int(self.labels_Exp[exp_index])
        exp_emb1,exp_emb2,exp_emb3,exp_emb4,exp_emb5 = self.Exp_emb_before2[exp_index],self.Exp_emb_before1[exp_index],self.Exp_emb_current[exp_index],self.Exp_emb_after1[exp_index],self.Exp_emb_after2[exp_index]
        exp_emb1, exp_emb2, exp_emb3, exp_emb4, exp_emb5 = self.process_emb(exp_emb1),self.process_emb(exp_emb2),self.process_emb(exp_emb3),self.process_emb(exp_emb4),self.process_emb(exp_emb5)
        exp_embs = np.concatenate((exp_emb1,exp_emb2,exp_emb3,exp_emb4,exp_emb5),axis=0)

        #AU_data
        au_name_list = self.AU_imgs[au_index]
        #au_name_list = au_name_list.split("_aligned")[0] + "/" + au_name_list
        au_img = Image.open(os.path.join(self.img_path, au_name_list))
        if au_img.getbands()[0] != 'R':
            au_img = au_img.convert('RGB')
        label_au = self.labels_AU[au_index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0
        label_au_ = torch.tensor(label_au_)
        AU_emb1, AU_emb2, AU_emb3, AU_emb4, AU_emb5 = self.AU_emb_before2[au_index], self.AU_emb_before1[
            au_index], self.AU_emb_current[au_index], self.AU_emb_after1[au_index], self.AU_emb_after2[au_index]
        AU_emb1, AU_emb2, AU_emb3, AU_emb4, AU_emb5 = self.process_emb(AU_emb1), self.process_emb(
            AU_emb2), self.process_emb(AU_emb3), self.process_emb(AU_emb4), self.process_emb(AU_emb5)
        au_embs = np.concatenate((AU_emb1, AU_emb2, AU_emb3, AU_emb4, AU_emb5), axis=0)
        #VA_data
        va_name_list = self.VA_imgs[va_index]
        #va_name_list = va_name_list.split("_aligned")[0] + "/" + va_name_list
        va_img = Image.open(os.path.join(self.img_path, va_name_list))
        if va_img.getbands()[0] != 'R':
            va_img = va_img.convert('RGB')
        label_V = float(self.labels_V[va_index])
        label_A = float(self.labels_A[va_index])
        VA_emb1, VA_emb2, VA_emb3, VA_emb4, VA_emb5 = self.VA_emb_before2[va_index], self.VA_emb_before1[
            va_index], self.VA_emb_current[va_index], self.VA_emb_after1[va_index], self.VA_emb_after2[va_index]
        VA_emb1, VA_emb2, VA_emb3, VA_emb4, VA_emb5 = self.process_emb(VA_emb1), self.process_emb(
            VA_emb2), self.process_emb(VA_emb3), self.process_emb(VA_emb4), self.process_emb(VA_emb5)
        va_embs = np.concatenate((VA_emb1, VA_emb2, VA_emb3, VA_emb4, VA_emb5), axis=0)
        if self.transform is not None:
            exp_img = self.transform(exp_img)
            va_img = self.transform(va_img)
            au_img = self.transform(au_img)

        return au_img,va_img,exp_img,label_au_,label_V,label_A,label_exp,au_embs,va_embs,exp_embs


class ABAW2_VA_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,emb_dict=None,mfcc_dict = None,word_dict=None,coff_dict=None, mfcc_seq = 15, word_seq=15, emb_seq=15,coff_seq=15,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_V = self.data['V']
        self.labels_A = self.data['A']
        self.img_path = img_path
        self.emb_dict = emb_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.coff_dict = coff_dict
        self.coff_seq = coff_seq


    def process_emb(self,emb):
        emb = emb[1:-1].replace(" ","").split(",")
        emb =np.array([float(e) for e in emb])
        emb = emb[np.newaxis,:]
        return emb

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path, anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        feas = []
        if self.coff_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.coff_seq
            frame_end = frame+self.coff_seq

            S,B = coff_arr.shape
            frame = min(S-1,frame)

            coff_fea = torch.zeros(2*self.coff_seq+1,50)
            total_len = 2*self.coff_seq+1
            coff = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    coff_fea[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(coff_fea)

        if self.emb_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.emb_seq
            frame_end = frame+self.emb_seq

            S,B = emb_arr.shape
            frame = min(S-1,frame)
            #print(S,B)
            emb_fea = torch.zeros(2*self.emb_seq+1,16)
            #print(mfcc_arr.shape)
            total_len = 2*self.emb_seq+1
            embs = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    emb_fea[index,:] = emb_arr[f,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(emb_fea)

        if self.mfcc_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            #mfcc_file = os.path.join(self.mfcc_path,v_id)
            #mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            #mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame-self.mfcc_seq
            frame_end = frame+self.mfcc_seq
            s,f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2*self.mfcc_seq+1)
            #print(mfcc_arr.shape)
            total_len = 2*self.mfcc_seq+1
            if frame_start<0:
                if f-1<frame_end:
                    delta = frame_end-(f-1)
                    mfcc_fea[:,-frame_start:(total_len-delta)] = mfcc_arr[:,0:frame_end+1]
                else:
                    mfcc_fea[:,-frame_start:total_len] = mfcc_arr[:,0:frame_end+1]
            elif frame_end>=f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:,:total_len-(frame_end-f+1)] = mfcc_arr[:,frame_start:]
            else:
                mfcc_fea = mfcc_arr[:,frame_start:frame_end+1]
            mfcc_fea = mfcc_fea.permute(1,0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            #return anc_img,label_V,label_A,mfcc_fea,anc_list


        if self.word_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(anc_list.split("/")[1].split(".")[0])-1
            words = []
            frames_start = int(frame)-self.word_seq
            frames_end = int(frame)+self.word_seq
            #print(range(frames_start,frames_end))
            for f in range(frames_start,frames_end+1):

                if f<0:
                    words.append("-1")
                elif f>=len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            #print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)

        return anc_img,label_V,label_A,feas,anc_list





class ABAW2_AU_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,emb_dict=None,mfcc_dict = None,word_dict=None,coff_dict=None, mfcc_seq = 15, word_seq=15, emb_seq=15,coff_seq=15,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_AU = self.data['AU']
        #self.AU_types = self.data["type"]
        self.img_path = img_path

        self.emb_dict = emb_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.coff_dict = coff_dict
        self.coff_seq = coff_seq

    def __len__(self):
        return len(self.data["img"])



    def __getitem__(self, index):
        anc_list = self.imgs[index]
        #au_type = self.AU_types[index]
        au_type = "Aff3"

        #print(anc_list)
        anc_img = Image.open(os.path.join(self.img_path, anc_list))

        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
        #t = self.AU_types[index]
        label_au = self.labels_AU[index].split(" ")
        label_au_ = [0.0 for i in range(len(label_au))]

        for i in range(len(label_au)):
            if label_au[i]!='0':
                label_au_[i] = 1.0

        label_au_ = torch.tensor(label_au_)
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        feas = []
        if self.coff_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.coff_seq
            frame_end = frame+self.coff_seq

            S,B = coff_arr.shape
            frame = min(S-1,frame)

            coff_fea = torch.zeros(2*self.coff_seq+1,50)
            total_len = 2*self.coff_seq+1
            coff = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    coff_fea[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(coff_fea)
        if self.emb_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.emb_seq
            frame_end = frame+self.emb_seq

            S,B = emb_arr.shape
            frame = min(S-1,frame)
            #print(S,B)
            emb_fea = torch.zeros(2*self.emb_seq+1,16)
            #print(mfcc_arr.shape)
            total_len = 2*self.emb_seq+1
            embs = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    emb_fea[index,:] = emb_arr[f,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(emb_fea)

        if self.mfcc_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            #mfcc_file = os.path.join(self.mfcc_path,v_id)
            #mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            #mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame-self.mfcc_seq
            frame_end = frame+self.mfcc_seq
            s,f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2*self.mfcc_seq+1)
            #print(mfcc_arr.shape)
            total_len = 2*self.mfcc_seq+1
            if frame_start<0:
                if f-1<frame_end:
                    delta = frame_end-(f-1)
                    mfcc_fea[:,-frame_start:(total_len-delta)] = mfcc_arr[:,0:frame_end+1]
                else:
                    mfcc_fea[:,-frame_start:total_len] = mfcc_arr[:,0:frame_end+1]
            elif frame_end>=f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:,:total_len-(frame_end-f+1)] = mfcc_arr[:,frame_start:]
            else:
                mfcc_fea = mfcc_arr[:,frame_start:frame_end+1]
            mfcc_fea = mfcc_fea.permute(1,0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            #return anc_img,label_V,label_A,mfcc_fea,anc_list


        if self.word_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(anc_list.split("/")[1].split(".")[0])-1
            words = []
            frames_start = int(frame)-self.word_seq
            frames_end = int(frame)+self.word_seq
            #print(range(frames_start,frames_end))
            for f in range(frames_start,frames_end+1):

                if f<0:
                    words.append("-1")
                elif f>=len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            #print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)
        return anc_img,label_au_,feas,anc_list,au_type







class ABAW2_Exp_Tridata(data.dataset.Dataset):
    def __init__(self, csv_file,img_path,transform=None,has_emb=False,mfcc_dict = None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.img_path = img_path
        self.data = self.pd_data.to_dict("list")
        self.anchors = self.data['anchor']
        self.positives = self.data['positive']
        self.negatives = self.data['negative']
        self.exps = self.data['Expression']

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        a,p,n = self.anchors[index],self.positives[index],self.negatives[index]
        a_im = Image.open(os.path.join(self.img_path,a)).convert('RGB')
        p_im = Image.open(os.path.join(self.img_path,p)).convert('RGB')
        n_im = Image.open(os.path.join(self.img_path,n)).convert('RGB')

        if self.transform is not None:
            a_im  = self.transform(a_im)
            p_im  = self.transform(p_im)
            n_im  = self.transform(n_im)
        label = self.exps[index]
        return a_im,p_im,n_im,label



class ABAW2_Exp_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path, emb_dict=None,mfcc_dict = None,word_dict=None,
                 coff_dict = None, rig_dict=None, mfcc_seq = 15,
                 word_seq=15, emb_seq=15,coff_seq=15,transform=None, fewer_data=False,
                 resample=False, class_n=9, use_rig=False):
        print('loading data from: ',csv_file)
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        if 'training' in csv_file:
            self.pd_data = pd.concat([self.pd_data, pd.read_csv('./annos/ABAW3_new_exp_validation.csv')])

        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.class_n = class_n
        self.use_rig = use_rig
        # cols = self.pd_data.columns.va
        # self.types = self.data["type"]

        if "EXP" in self.pd_data.keys():
            self.labels_Exp = self.data['EXP']
        else:
            self.labels_Exp = self.data['Expression']
        total_imgs = len(self.imgs)
        if fewer_data:
            fewer_classes = [1,2,3]
            self.imgs = [self.imgs[e] for e in range(total_imgs) if self.labels_Exp[e] in fewer_classes]
            self.types = [self.types[e] for e in range(total_imgs) if self.labels_Exp[e] in fewer_classes]
            self.labels_Exp = [self.labels_Exp[e] for e in range(total_imgs) if self.labels_Exp[e] in fewer_classes]
        if resample:
            if class_n==4:
                self.resample_data_4()
            else:
                self.resample_data()
        assert len(self.imgs) == len(self.labels_Exp)

        self.labels_Exp = [int(j) for j in self.labels_Exp]
        self.img_path = img_path
        self.emb_dict = emb_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.coff_seq = coff_seq
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.coff_dict = coff_dict
        self.rig_dict = rig_dict
    def resample_data(self):
        data_class = []
        total_n = len(self.imgs)
        for i in range(8):
            data_class.append([[self.imgs[e] for e in range(total_n) if self.labels_Exp[e] == i],
                               [self.types[e] for e in range(total_n) if self.labels_Exp[e] == i],
                               [self.labels_Exp[e] for e in range(total_n) if self.labels_Exp[e] == i]
                              ])
        n_max = max([len(dc[0]) for dc in data_class])
        imgs = []
        types =[]
        labels_Exp =[]
        for i in range(8):
            imgs += data_class[i][0]*int(n_max/len(data_class[i][0]))
            types += data_class[i][1]*int(n_max/len(data_class[i][0]))
            labels_Exp += data_class[i][2]*int(n_max/len(data_class[i][0]))
        self.imgs = imgs
        self.types = types
        self.labels_Exp = labels_Exp

    def resample_data_4(self):
        data_class = []
        total_n = len(self.imgs)
        for i in range(8):
            data_class.append([[self.imgs[e] for e in range(total_n) if self.labels_Exp[e] == i],
                               [self.types[e] for e in range(total_n) if self.labels_Exp[e] == i],
                               [self.labels_Exp[e] for e in range(total_n) if self.labels_Exp[e] == i]
                              ])
        n_max = max([len(dc[0]) for dc in data_class])
        imgs = []
        types =[]
        labels_Exp =[]
        for i in range(8):
            if i in [1,2,3]:
                imgs += data_class[i][0]
                types += data_class[i][1]
                labels_Exp += data_class[i][2]
            else:
                imgs += data_class[i][0][::5]
                types += data_class[i][1][::5]
                labels_Exp += data_class[i][2][::5]
        self.imgs = imgs
        self.types = types
        self.labels_Exp = labels_Exp

    def get_labels(self):
        return self.labels_Exp

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        # ty = self.types[index]
        # if ty == "aff3":
        anc_img = Image.open(os.path.join(self.img_path,anc_list))
        # elif ty == "affectnet":
        #     anc_img = Image.open(os.path.join("/root/workspace/data/affectnet/",anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
        if self.labels_Exp:
            label_exp = int(self.labels_Exp[index])
        else:
            label_exp = -1
        if self.transform is not None:
            anc_img = self.transform(anc_img)

        feas = []
        if self.coff_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            frame_start = frame-self.coff_seq
            frame_end = frame+self.coff_seq
            S, B = coff_arr.shape
            frame = min(S-1,frame)
            coff_fea = torch.zeros(2*self.coff_seq+1,B)
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    coff_fea[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                    index+=1

            # feas.append(coff_fea)

            if self.use_rig:
                v_id = anc_list.split("/")[0] + ".txt"
                coff_arr = self.rig_dict[v_id]
                coff_arr = torch.from_numpy(coff_arr).float()
                frame = int(anc_list.split("/")[1].split(".")[0]) - 1
                frame_start = frame-self.coff_seq
                frame_end = frame+self.coff_seq
                S, B = coff_arr.shape
                frame = min(S-1,frame)
                coff_fea_rig = torch.zeros(2*self.coff_seq+1,B)
                index = 0
                for f in range(frame_start,frame_end+1):
                    if f<0:
                        index+=1
                        continue
                    elif f>=S:
                        index+=1
                        continue
                    else:
                        coff_fea_rig[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                        index+=1
                coff_fea = torch.cat([coff_fea, coff_fea_rig],dim=1)
            feas.append(coff_fea)

        if self.emb_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.emb_seq
            frame_end = frame+self.emb_seq

            S,B = emb_arr.shape
            frame = min(S-1,frame)
            #print(S,B)
            emb_fea = torch.zeros(2*self.emb_seq+1,B)
            #print(mfcc_arr.shape)
            total_len = 2*self.emb_seq+1
            embs = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    emb_fea[index,:] = emb_arr[f,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(emb_fea)

        if self.mfcc_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            #mfcc_file = os.path.join(self.mfcc_path,v_id)
            #mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            #mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame-self.mfcc_seq
            frame_end = frame+self.mfcc_seq
            s,f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2*self.mfcc_seq+1)
            #print(mfcc_arr.shape)
            total_len = 2*self.mfcc_seq+1
            if frame_start<0:
                if f-1<frame_end:
                    delta = frame_end-(f-1)
                    mfcc_fea[:,-frame_start:(total_len-delta)] = mfcc_arr[:,0:frame_end+1]
                else:
                    mfcc_fea[:,-frame_start:total_len] = mfcc_arr[:,0:frame_end+1]
            elif frame_end>=f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:,:total_len-(frame_end-f+1)] = mfcc_arr[:,frame_start:]
            else:
                mfcc_fea = mfcc_arr[:,frame_start:frame_end+1]
            mfcc_fea = mfcc_fea.permute(1,0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            #return anc_img,label_V,label_A,mfcc_fea,anc_list


        if self.word_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(anc_list.split("/")[1].split(".")[0])-1
            words = []
            if self.word_seq==-1:
                prev = (word_arr[:frame-60][::-1]+['-1']).index('-1')
                post = (word_arr[frame+60:]+['-1']).index('-1')
                frames_start = frame - 60 - prev + 1
                frames_end = frame + 60 + post - 1
            else:
                frames_start = int(frame)-self.word_seq
                frames_end = int(frame)+self.word_seq
            #print(range(frames_start,frames_end))
            for f in range(frames_start,frames_end+1):

                if f<0:
                    words.append("-1")
                elif f>=len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            #print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)

        if self.class_n == 4 and label_exp not in [1,2,3]:
            label_exp = 0
        elif self.class_n == 6 and label_exp in [1,2,3]:
            label_exp = 1
        elif self.class_n == 7 and label_exp == 2:
            label_exp= 5

        return anc_img,label_exp,feas,anc_list


class ABAW2_Exp_data_test(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """

    def __init__(self, csv_file, img_path, emb_dict=None, mfcc_dict=None, word_dict=None, coff_dict=None, mfcc_seq=15,
                 word_seq=15, emb_seq=15, coff_seq=15, transform=None, fewer_data=False, resample=False, class_n=9):
        print('loading data from: ', csv_file)
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.class_n = class_n

        total_imgs = len(self.imgs)

        self.img_path = img_path
        self.emb_dict = emb_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.coff_seq = coff_seq
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.coff_dict = coff_dict

    def get_labels(self):
        return self.labels_Exp

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path, anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
        if self.transform is not None:
            anc_img = self.transform(anc_img)
        feas = []
        if self.coff_dict != None:
            v_id = anc_list.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame - self.coff_seq
            frame_end = frame + self.coff_seq

            S, B = coff_arr.shape
            frame = min(S - 1, frame)

            coff_fea = torch.zeros(2 * self.coff_seq + 1, B)
            total_len = 2 * self.coff_seq + 1
            coff = []
            index = 0
            for f in range(frame_start, frame_end + 1):
                if f < 0:
                    index += 1
                    continue
                elif f >= S:
                    index += 1
                    continue
                else:
                    # print(f)
                    # print(emb_arr[f,:])
                    coff_fea[index, :] = coff_arr[f, :] - coff_arr[frame, :]
                    # print(emb_fea[index,:])
                    index += 1
            feas.append(coff_fea)

        if self.emb_dict != None:
            v_id = anc_list.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame - self.emb_seq
            frame_end = frame + self.emb_seq

            S, B = emb_arr.shape
            frame = min(S - 1, frame)
            # print(S,B)
            emb_fea = torch.zeros(2 * self.emb_seq + 1, 16)
            # print(mfcc_arr.shape)
            total_len = 2 * self.emb_seq + 1
            embs = []
            index = 0
            for f in range(frame_start, frame_end + 1):
                if f < 0:
                    index += 1
                    continue
                elif f >= S:
                    index += 1
                    continue
                else:
                    # print(f)
                    # print(emb_arr[f,:])
                    emb_fea[index, :] = emb_arr[f, :]
                    # print(emb_fea[index,:])
                    index += 1
            feas.append(emb_fea)

        if self.mfcc_dict != None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            # mfcc_file = os.path.join(self.mfcc_path,v_id)
            # mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            # mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame - self.mfcc_seq
            frame_end = frame + self.mfcc_seq
            s, f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2 * self.mfcc_seq + 1)
            # print(mfcc_arr.shape)
            total_len = 2 * self.mfcc_seq + 1
            if frame_start < 0:
                if f - 1 < frame_end:
                    delta = frame_end - (f - 1)
                    mfcc_fea[:, -frame_start:(total_len - delta)] = mfcc_arr[:, 0:frame_end + 1]
                else:
                    mfcc_fea[:, -frame_start:total_len] = mfcc_arr[:, 0:frame_end + 1]
            elif frame_end >= f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:, :total_len - (frame_end - f + 1)] = mfcc_arr[:, frame_start:]
            else:
                mfcc_fea = mfcc_arr[:, frame_start:frame_end + 1]
            mfcc_fea = mfcc_fea.permute(1, 0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            # return anc_img,label_V,label_A,mfcc_fea,anc_list

        if self.word_dict != None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            words = []
            if self.word_seq == -1:
                prev = (word_arr[:frame - 60][::-1] + ['-1']).index('-1')
                post = (word_arr[frame + 60:] + ['-1']).index('-1')
                frames_start = frame - 60 - prev + 1
                frames_end = frame + 60 + post - 1
            else:
                frames_start = int(frame) - self.word_seq
                frames_end = int(frame) + self.word_seq
            # print(range(frames_start,frames_end))
            for f in range(frames_start, frames_end + 1):

                if f < 0:
                    words.append("-1")
                elif f >= len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            # print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)


        return anc_img, -1, feas, anc_list


class BBN_ABAW2_Exp_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file1,csv_file2,img_path,transform=None):
        self.transform = transform
        self.pd_data1 = pd.read_csv(csv_file1)
        self.data1 = self.pd_data1.to_dict("list")
        self.imgs1 = self.data1['img']
        self.labels_Exp1 = self.data1['Expression']

        self.pd_data2 = pd.read_csv(csv_file2)
        self.data2 = self.pd_data2.to_dict("list")
        self.imgs2 = self.data2['img']
        self.labels_Exp2 = self.data2['Expression']

        self.img_path = img_path

    def __len__(self):
        return max(len(self.data1["img"]),len(self.data2["img"]))

    def __getitem__(self, index):

        if index >= len(self.imgs1) and index < 2 * len(self.imgs1):
            exp_index1 = index - len(self.imgs1)
        elif index >= 2 * len(self.imgs1) and index < 3 * len(self.imgs1):
            exp_index1 = index - 2 * len(self.imgs1)
        elif index >= 3 * len(self.imgs1) and index < 4 * len(self.imgs1):
            exp_index1 = index - 3 * len(self.imgs1)
        elif index >= 4 * len(self.imgs1) and index < 5 * len(self.imgs1):
            exp_index1 = index - 4 * len(self.imgs1)
        elif index >= 5 * len(self.imgs1) and index < 6 * len(self.imgs1):
            exp_index1 = index - 5 * len(self.imgs1)
        else:
            exp_index1 = index

        if index >= len(self.imgs2) and index < 2 * len(self.imgs2):
            exp_index2 = index - len(self.imgs2)
        elif index >= 2 * len(self.imgs2) and index < 3 * len(self.imgs2):
            exp_index2 = index - 2 * len(self.imgs2)
        elif index >= 3 * len(self.imgs2) and index < 4 * len(self.imgs2):
            exp_index2 = index - 3 * len(self.imgs2)
        elif index >= 4 * len(self.imgs2) and index < 5 * len(self.imgs2):
            exp_index2 = index - 4 * len(self.imgs2)
        elif index >= 5 * len(self.imgs2) and index < 6 * len(self.imgs2):
            exp_index2 = index - 5 * len(self.imgs2)
        else:
            exp_index2 = index

        anc_list1 = self.imgs1[exp_index1]
        anc_img1 = Image.open(os.path.join(self.img_path,anc_list1))
        if anc_img1.getbands()[0] != 'R':
            anc_img1 = anc_img1.convert('RGB')
        label_exp1 = int(self.labels_Exp1[exp_index1])
        if self.transform is not None:
            anc_img1 = self.transform(anc_img1)

        anc_list2 = self.imgs2[exp_index2]
        anc_img2 = Image.open(os.path.join(self.img_path,anc_list2))
        if anc_img2.getbands()[0] != 'R':
            anc_img2 = anc_img2.convert('RGB')
        label_exp2 = int(self.labels_Exp2[exp_index2])
        if self.transform is not None:
            anc_img2 = self.transform(anc_img2)

        return anc_img1,label_exp1,anc_list1,anc_img2,label_exp2,anc_list2

class BBN_ABAW2_Exp_data_class(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self,csv_file0, csv_file1,csv_file2,csv_file3,csv_file4,csv_file5,csv_file6,img_path,transform=None):
        self.transform = transform

        self.pd_data0 = pd.read_csv(csv_file0)
        self.data0 = self.pd_data0.to_dict("list")
        self.imgs0 = self.data0['img']
        self.labels_Exp0 = self.data0['Expression']

        self.pd_data1 = pd.read_csv(csv_file1)
        self.data1 = self.pd_data1.to_dict("list")
        self.imgs1 = self.data1['img']
        self.labels_Exp1 = self.data1['Expression']

        self.pd_data2 = pd.read_csv(csv_file2)
        self.data2 = self.pd_data2.to_dict("list")
        self.imgs2 = self.data2['img']
        self.labels_Exp2 = self.data2['Expression']

        self.pd_data3 = pd.read_csv(csv_file3)
        self.data3 = self.pd_data3.to_dict("list")
        self.imgs3 = self.data3['img']
        self.labels_Exp3 = self.data3['Expression']

        self.pd_data4 = pd.read_csv(csv_file4)
        self.data4 = self.pd_data4.to_dict("list")
        self.imgs4 = self.data4['img']
        self.labels_Exp4 = self.data4['Expression']

        self.pd_data5 = pd.read_csv(csv_file5)
        self.data5 = self.pd_data5.to_dict("list")
        self.imgs5 = self.data5['img']
        self.labels_Exp5 = self.data5['Expression']

        self.pd_data6 = pd.read_csv(csv_file6)
        self.data6 = self.pd_data6.to_dict("list")
        self.imgs6 = self.data6['img']
        self.labels_Exp6 = self.data6['Expression']

        self.img_path = img_path

    def __len__(self):
        return max(len(self.data0["img"]),len(self.data1["img"]),len(self.data2["img"]),len(self.data3["img"]),len(self.data4["img"]),len(self.data5["img"]),len(self.data6["img"]))

    def __getitem__(self, index):

        exp_index0 = random.randint(0,len(self.img0))
        exp_index1 = random.randint(0, len(self.img1))
        exp_index2 = random.randint(0, len(self.img2))
        exp_index3 = random.randint(0, len(self.img3))
        exp_index4 = random.randint(0, len(self.img4))
        exp_index5 = random.randint(0, len(self.img5))
        exp_index6 = random.randint(0, len(self.img6))


        anc_list0 = self.imgs0[exp_index0]
        anc_img0 = Image.open(os.path.join(self.img_path,anc_list0))
        if anc_img0.getbands()[0] != 'R':
            anc_img0 = anc_img0.convert('RGB')
        label_exp0 = int(self.labels_Exp0[exp_index0])
        if self.transform is not None:
            anc_img0 = self.transform(anc_img0)

        anc_list1 = self.imgs1[exp_index1]
        anc_img1 = Image.open(os.path.join(self.img_path,anc_list1))
        if anc_img1.getbands()[0] != 'R':
            anc_img1 = anc_img1.convert('RGB')
        label_exp1 = int(self.labels_Exp1[exp_index1])
        if self.transform is not None:
            anc_img1 = self.transform(anc_img1)

        anc_list2 = self.imgs2[exp_index2]
        anc_img2 = Image.open(os.path.join(self.img_path,anc_list2))
        if anc_img2.getbands()[0] != 'R':
            anc_img2 = anc_img2.convert('RGB')
        label_exp2 = int(self.labels_Exp2[exp_index2])
        if self.transform is not None:
            anc_img2 = self.transform(anc_img2)

        anc_list3 = self.imgs3[exp_index3]
        anc_img3 = Image.open(os.path.join(self.img_path,anc_list3))
        if anc_img3.getbands()[0] != 'R':
            anc_img3 = anc_img3.convert('RGB')
        label_exp3 = int(self.labels_Exp3[exp_index3])
        if self.transform is not None:
            anc_img3 = self.transform(anc_img3)


        anc_list4 = self.imgs4[exp_index4]
        anc_img4 = Image.open(os.path.join(self.img_path,anc_list4))
        if anc_img4.getbands()[0] != 'R':
            anc_img4 = anc_img4.convert('RGB')
        label_exp4 = int(self.labels_Exp4[exp_index4])
        if self.transform is not None:
            anc_img4 = self.transform(anc_img4)

        anc_list5 = self.imgs5[exp_index5]
        anc_img5 = Image.open(os.path.join(self.img_path,anc_list5))
        if anc_img5.getbands()[0] != 'R':
            anc_img5 = anc_img5.convert('RGB')
        label_exp5 = int(self.labels_Exp5[exp_index5])
        if self.transform is not None:
            anc_img5 = self.transform(anc_img5)

        anc_list6 = self.imgs6[exp_index6]
        anc_img6 = Image.open(os.path.join(self.img_path,anc_list6))
        if anc_img6.getbands()[0] != 'R':
            anc_img6 = anc_img6.convert('RGB')
        label_exp6 = int(self.labels_Exp6[exp_index6])
        if self.transform is not None:
            anc_img6 = self.transform(anc_img6)

        return anc_img0,label_exp0,anc_list0,anc_img1,label_exp1,anc_list1,anc_img2,label_exp2,anc_list2,anc_img3,label_exp3,anc_list3,anc_img4,label_exp4,anc_list4,anc_img5,label_exp5,anc_list5,anc_img6,label_exp6,anc_list6


class ABAW2_Exp_3dcnn_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,transform=None):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        self.labels_Exp = self.data['Expression']
        self.img_path = img_path
        #self.embs = self.data["emb"]

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_data = np.load(os.path.join(self.img_path,anc_list))
        anc_data = np.transpose(anc_data,(3,0,1,2))
        anc_data = torch.tensor(anc_data/255,dtype=torch.float32)
        label_exp = int(self.labels_Exp[index])

        return anc_data,label_exp,anc_list


def compute_class_weight(csv_data,type="AU"):
    csv_data = pd.read_csv(csv_data)
    if type=="AU":
        labels = csv_data["AU"]
        c = [0 for i in range(12)]
        N = len(labels)
        for i in range(len(labels)):
            l = labels[i].split(" ")
            for j in range(len(l)):
                if int(l[j]) > 0:
                    c[j] += 1
        r = [N / c[i] for i in range(12)]
        s = sum(r)
        r = [r[i] / s for i in range(12)]
        return torch.as_tensor(r, dtype=torch.float)
    elif type=="Exp":
        if "EXP" in csv_data.columns:
            labels = csv_data["EXP"].to_list()
        else:
            labels = csv_data["Expression"].to_list()
        counts = []
        for i in range(8):
            counts.append(labels.count(i))
        N = len(labels)
        r = [N / counts[i] for i in range(8)]
        s = sum(r)
        r = [r[i] / s for i in range(8)]
        return torch.as_tensor(r, dtype=torch.float)


class ABAW2_test_data(data.dataset.Dataset):
    """
    Args:
        transform(callabe, optional);A Function/transform that transform one PIL image.
    """
    def __init__(self, csv_file,img_path,emb_dict=None,mfcc_dict = None,word_dict=None,coff_dict=None, mfcc_seq = 15, word_seq=15, emb_seq=15, coff_seq=15,transform=None,is_test=False):
        self.transform = transform
        self.pd_data = pd.read_csv(csv_file)
        self.data = self.pd_data.to_dict("list")
        self.imgs = self.data['img']
        cols = self.pd_data.columns.values
        self.use_pred=False
        if "prob" in cols:
            self.pred = self.data["prob"]
            self.labels = self.data["label"]
            self.use_pred = True

        self.img_path = img_path
        self.emb_dict = emb_dict
        self.mfcc_seq = mfcc_seq
        self.word_seq = word_seq
        self.emb_seq = emb_seq
        self.mfcc_dict = mfcc_dict
        self.word_dict = word_dict
        self.is_test = is_test
        self.coff_seq = coff_seq
        self.coff_dict = coff_dict
        #self.embs = self.data["emb"]
        self.simple_trans = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()])
        self.hard_trans =transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(30),
            transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        # print(len(self.data["img"]))
        if self.is_test:
            return 100
        return len(self.data["img"])

    def __getitem__(self, index):
        anc_list = self.imgs[index]
        anc_img = Image.open(os.path.join(self.img_path,anc_list))
        if anc_img.getbands()[0] != 'R':
            anc_img = anc_img.convert('RGB')
       #label_exp = int(self.labels_Exp[index])
        if self.transform is not None:
            anc_img1 = self.simple_trans(anc_img)
            anc_img2 = self.hard_trans(anc_img)


        feas = []
        if self.coff_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            coff_arr = self.coff_dict[v_id]
            coff_arr = torch.from_numpy(coff_arr).float()
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.coff_seq
            frame_end = frame+self.coff_seq

            S,B = coff_arr.shape
            frame = min(S-1,frame)

            coff_fea = torch.zeros(2*self.coff_seq+1,50)
            total_len = 2*self.coff_seq+1
            coff = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    coff_fea[index,:] = coff_arr[f,:] - coff_arr[frame,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(coff_fea)

        if self.emb_dict!=None:
            v_id = anc_list.split("/")[0] + ".txt"
            emb_arr = self.emb_dict[v_id]
            emb_arr = torch.from_numpy(emb_arr).float()
            # print(emb_arr)
            frame = int(anc_list.split("/")[1].split(".")[0]) - 1

            frame_start = frame-self.emb_seq
            frame_end = frame+self.emb_seq

            S,B = emb_arr.shape
            frame = min(S-1,frame)
            #print(S,B)
            emb_fea = torch.zeros(2*self.emb_seq+1,16)
            #print(mfcc_arr.shape)
            total_len = 2*self.emb_seq+1
            embs = []
            index = 0
            for f in range(frame_start,frame_end+1):
                if f<0:
                    index+=1
                    continue
                elif f>=S:
                    index+=1
                    continue
                else:
                    #print(f)
                    #print(emb_arr[f,:])
                    emb_fea[index,:] = emb_arr[f,:]-emb_arr[frame,:]
                    #print(emb_fea[index,:])
                    index+=1
            feas.append(emb_fea)

        if self.mfcc_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            frame = int(anc_list.split("/")[1].split(".")[0]) - 1
            #mfcc_file = os.path.join(self.mfcc_path,v_id)
            #mfcc_arr = np.loadtxt(mfcc_file,delimiter=',')
            #mfcc_arr = torch.from_numpy(mfcc_arr).float()
            mfcc_arr = self.mfcc_dict[v_id]
            mfcc_arr = torch.from_numpy(mfcc_arr).float()
            frame_start = frame-self.mfcc_seq
            frame_end = frame+self.mfcc_seq
            s,f = mfcc_arr.shape
            mfcc_fea = torch.zeros(40, 2*self.mfcc_seq+1)
            #print(mfcc_arr.shape)
            total_len = 2*self.mfcc_seq+1
            if frame_start<0:
                if f-1<frame_end:
                    delta = frame_end-(f-1)
                    mfcc_fea[:,-frame_start:(total_len-delta)] = mfcc_arr[:,0:frame_end+1]
                else:
                    mfcc_fea[:,-frame_start:total_len] = mfcc_arr[:,0:frame_end+1]
            elif frame_end>=f:
                # print(f)
                # print(frame_start,frame_end)
                mfcc_fea[:,:total_len-(frame_end-f+1)] = mfcc_arr[:,frame_start:]
            else:
                mfcc_fea = mfcc_arr[:,frame_start:frame_end+1]
            mfcc_fea = mfcc_fea.permute(1,0)
            feas.append(mfcc_fea)
            # print(mfcc_fea.shape)

            #return anc_img,label_V,label_A,mfcc_fea,anc_list


        if self.word_dict !=None:
            v_id = anc_list.split("/")[0].split("_aligned")[0] + ".txt"
            if "left" in v_id:
                v_id = v_id.split("_left")[0] + ".txt"
            if "right" in v_id:
                v_id = v_id.split("_right")[0] + ".txt"

            word_arr = self.word_dict[v_id]
            frames_num = len(word_arr)
            frame = int(anc_list.split("/")[1].split(".")[0])-1
            words = []
            frames_start = int(frame)-self.word_seq
            frames_end = int(frame)+self.word_seq
            #print(range(frames_start,frames_end))
            for f in range(frames_start,frames_end+1):

                if f<0:
                    words.append("-1")
                elif f>=len(word_arr):
                    words.append("-1")
                else:
                    words.append(word_arr[f].strip().lower())
            #print(len(words))
            str_words = ",".join(words)
            feas.append(str_words)

        if self.use_pred:
            pred = self.pred[index].split(" ")
            pred = np.array([float(k) for k in pred])
            pred = torch.from_numpy(pred).float()
            label = int(self.labels[index])
            return anc_img1,anc_img2,feas,anc_list,pred,label
        return anc_img1,anc_img2,feas,anc_list
# transform = transforms.Compose([
#     transforms.Resize([224, 224]),
#     transforms.ToTensor()])
# from matplotlib import pyplot as plt
# import cv2
# from PIL import Image
# import cv2
# import torch
#
# if __name__ == '__main__':
#     trainset = ABAW2_multitask_data("multi_data.csv",r"D:\xiangmu\emotion\openface",transform)
#     trainloader = data.DataLoader(trainset, batch_size=1, num_workers=1)
#     unloader = transforms.ToPILImage()
#
#     for batch_idx, (img,label_au_,label_V,label_A,label_exp,name) in enumerate(trainloader):
#         print(len(label_au_))
#         print(label_au_)
#         img = img[0]
#         print(img.shape)
#         print(label_exp)
#         print(label_V)
#         print(label_A)
#         print(name)
#         image1 = img.cpu().clone()
#         image1 = image1.squeeze(0)
#         image1 = unloader(image1)
#         plt.imshow(image1)
#         plt.show()
#         cv2.waitKey()