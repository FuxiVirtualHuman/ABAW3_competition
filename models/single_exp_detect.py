import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
try:
    from code_new.models.pipeline_student_InceptionResnet import Pipeline_Incep
except:
    from models.pipeline_student_InceptionResnet import Pipeline_Incep

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from models.MISA import *

# Naive Transformer Layers as Classifier (For Ablation Study)
class Classifier_T(nn.Module):
    def __init__(self, dataset="BP4D", feature_dim=32):
        super(Classifier_T, self).__init__()
        self.dataset = dataset
        self.norm_layer = nn.BatchNorm1d(16, affine=False)
        self.constant = np.log(1 / np.sqrt(2 * np.pi))

        self.d_model = 16
        drop_out = 0.5
        self.pos_embed = PositionalEncoding(self.d_model, dropout=drop_out)
        self.encoder_layer1 = TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=16, dropout=drop_out)
        self.encoder_layer2 = TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=16, dropout=drop_out)
        self.encoder_layer3 = TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=16, dropout=drop_out)

        if dataset == "BP4D":
            self.linear_p1 = nn.Linear(feature_dim, 16)
            self.linear_p2 = nn.Linear(feature_dim, 16)
            self.linear_p3 = nn.Linear(feature_dim, 16)
            self.linear_p4 = nn.Linear(feature_dim, 16)
            self.linear_p5 = nn.Linear(feature_dim, 16)
            self.linear_p6 = nn.Linear(feature_dim, 16)
            self.linear_p7 = nn.Linear(feature_dim, 16)
            self.linear_p8 = nn.Linear(feature_dim, 16)
            self.linear_p9 = nn.Linear(feature_dim, 16)
            self.linear_p10 = nn.Linear(feature_dim, 16)
            self.linear_p11 = nn.Linear(feature_dim, 16)
            self.linear_p12 = nn.Linear(feature_dim, 16)
            self.linear_last1 = nn.Linear(16, 2, bias=False)
            self.linear_last2 = nn.Linear(16, 2, bias=False)
            self.linear_last3 = nn.Linear(16, 2, bias=False)
            self.linear_last4 = nn.Linear(16, 2, bias=False)
            self.linear_last5 = nn.Linear(16, 2, bias=False)
            self.linear_last6 = nn.Linear(16, 2, bias=False)
            self.linear_last7 = nn.Linear(16, 2, bias=False)
            self.linear_last8 = nn.Linear(16, 2, bias=False)
            self.linear_last9 = nn.Linear(16, 2, bias=False)
            self.linear_last10 = nn.Linear(16, 2, bias=False)
            self.linear_last11 = nn.Linear(16, 2, bias=False)
            self.linear_last12 = nn.Linear(16, 2, bias=False)
            self.final_linear = nn.Linear(24, 24)
        else:
            self.linear_p1 = nn.Linear(feature_dim, 16)
            self.linear_p2 = nn.Linear(feature_dim, 16)
            self.linear_p3 = nn.Linear(feature_dim, 16)
            self.linear_p4 = nn.Linear(feature_dim, 16)
            self.linear_p5 = nn.Linear(feature_dim, 16)
            self.linear_p6 = nn.Linear(feature_dim, 16)
            self.linear_p7 = nn.Linear(feature_dim, 16)
            self.linear_p8 = nn.Linear(feature_dim, 16)
            self.linear_last1 = nn.Linear(16, 2, bias=False)
            self.linear_last2 = nn.Linear(16, 2, bias=False)
            self.linear_last3 = nn.Linear(16, 2, bias=False)
            self.linear_last4 = nn.Linear(16, 2, bias=False)
            self.linear_last5 = nn.Linear(16, 2, bias=False)
            self.linear_last6 = nn.Linear(16, 2, bias=False)
            self.linear_last7 = nn.Linear(16, 2, bias=False)
            self.linear_last8 = nn.Linear(16, 2, bias=False)
            self.final_linear = nn.Linear(16, 16)

    def forward(self, x):

        # # Add Norm Gate
        # normed_x = self.norm_layer(x)
        # weight = - self.constant + 0.5 * normed_x**2
        # x = x * weight

        if self.dataset == "BP4D":
            x1 = self.linear_p1(x).unsqueeze(0)
            x2 = self.linear_p2(x).unsqueeze(0)
            x3 = self.linear_p3(x).unsqueeze(0)
            x4 = self.linear_p4(x).unsqueeze(0)
            x5 = self.linear_p5(x).unsqueeze(0)
            x6 = self.linear_p6(x).unsqueeze(0)
            x7 = self.linear_p7(x).unsqueeze(0)
            x8 = self.linear_p8(x).unsqueeze(0)
            x9 = self.linear_p9(x).unsqueeze(0)
            x10 = self.linear_p10(x).unsqueeze(0)
            x11 = self.linear_p11(x).unsqueeze(0)
            x12 = self.linear_p12(x).unsqueeze(0)
            # print('1:x12.size():')
            # print(x12.size())
            x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=0)
            # print('2:x.size():')
            # print(x.size())

            x = self.pos_embed(x)
            x = self.encoder_layer1(x)
            x = self.encoder_layer2(x)
            x = self.encoder_layer3(x)
            # print('3:x.size():')
            # print(x.size())
            x1 = self.linear_last1(x[0,:,:]).unsqueeze(1)
            x2 = self.linear_last1(x[1,:,:]).unsqueeze(1)
            x3 = self.linear_last1(x[2,:,:]).unsqueeze(1)
            x4 = self.linear_last1(x[3,:,:]).unsqueeze(1)
            x5 = self.linear_last1(x[4,:,:]).unsqueeze(1)
            x6 = self.linear_last1(x[5,:,:]).unsqueeze(1)
            x7 = self.linear_last1(x[6,:,:]).unsqueeze(1)
            x8 = self.linear_last1(x[7,:,:]).unsqueeze(1)
            x9 = self.linear_last1(x[8,:,:]).unsqueeze(1)
            x10 = self.linear_last1(x[9,:,:]).unsqueeze(1)
            x11 = self.linear_last1(x[10,:,:]).unsqueeze(1)
            x12 = self.linear_last1(x[11,:,:]).unsqueeze(1)
            # print('4:x12.size():')
            # print(x12.size())
            x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
            # print('5:x.size():')
            # print(x.size())


        else:
            x1 = self.linear_p1(x).unsqueeze(0)
            x2 = self.linear_p2(x).unsqueeze(0)
            x3 = self.linear_p3(x).unsqueeze(0)
            x4 = self.linear_p4(x).unsqueeze(0)
            x5 = self.linear_p5(x).unsqueeze(0)
            x6 = self.linear_p6(x).unsqueeze(0)
            x7 = self.linear_p7(x).unsqueeze(0)
            x8 = self.linear_p8(x).unsqueeze(0)
            x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=0)

            x = self.pos_embed(x)
            x = self.encoder_layer1(x)
            x = self.encoder_layer2(x)
            x = self.encoder_layer3(x)

            x1 = self.linear_last1(x[0,:,:]).unsqueeze(1)
            x2 = self.linear_last1(x[1,:,:]).unsqueeze(1)
            x3 = self.linear_last1(x[2,:,:]).unsqueeze(1)
            x4 = self.linear_last1(x[3,:,:]).unsqueeze(1)
            x5 = self.linear_last1(x[4,:,:]).unsqueeze(1)
            x6 = self.linear_last1(x[5,:,:]).unsqueeze(1)
            x7 = self.linear_last1(x[6,:,:]).unsqueeze(1)
            x8 = self.linear_last1(x[7,:,:]).unsqueeze(1)
            x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1)


        # current best
        x_ = x.view(x.shape[0], -1)
        final = self.final_linear(x_)
        final = final.view(x.shape[0], -1, 2)
        return x, final


class Classifier_T_for_exp(nn.Module):
    def __init__(self, dataset="BP4D", feature_dim=32):
        super(Classifier_T_for_exp, self).__init__()
        self.dataset = dataset
        self.norm_layer = nn.BatchNorm1d(16, affine=False)
        self.constant = np.log(1 / np.sqrt(2 * np.pi))

        self.d_model = 16
        drop_out = 0.5
        self.pos_embed = PositionalEncoding(self.d_model, dropout=drop_out)
        self.encoder_layer1 = TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=16, dropout=drop_out)
        self.encoder_layer2 = TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=16, dropout=drop_out)
        self.encoder_layer3 = TransformerEncoderLayer(self.d_model, nhead=4, dim_feedforward=16, dropout=drop_out)

        
        self.linear_p1 = nn.Linear(feature_dim, 16)
        self.linear_p2 = nn.Linear(feature_dim, 16)
        self.linear_p3 = nn.Linear(feature_dim, 16)
        self.linear_p4 = nn.Linear(feature_dim, 16)
        self.linear_p5 = nn.Linear(feature_dim, 16)
        self.linear_p6 = nn.Linear(feature_dim, 16)
        self.linear_p7 = nn.Linear(feature_dim, 16)
        self.linear_p8 = nn.Linear(feature_dim, 16)
        self.linear_last1 = nn.Linear(16, 1, bias=False)
        self.linear_last2 = nn.Linear(16, 1, bias=False)
        self.linear_last3 = nn.Linear(16, 1, bias=False)
        self.linear_last4 = nn.Linear(16, 1, bias=False)
        self.linear_last5 = nn.Linear(16, 1, bias=False)
        self.linear_last6 = nn.Linear(16, 1, bias=False)
        self.linear_last7 = nn.Linear(16, 1, bias=False)
        self.linear_last8 = nn.Linear(16, 1, bias=False)
        

    def forward(self, x):

        # # Add Norm Gate
        # normed_x = self.norm_layer(x)
        # weight = - self.constant + 0.5 * normed_x**2
        # x = x * weight

        
        x1 = self.linear_p1(x).unsqueeze(0)
        x2 = self.linear_p2(x).unsqueeze(0)
        x3 = self.linear_p3(x).unsqueeze(0)
        x4 = self.linear_p4(x).unsqueeze(0)
        x5 = self.linear_p5(x).unsqueeze(0)
        x6 = self.linear_p6(x).unsqueeze(0)
        x7 = self.linear_p7(x).unsqueeze(0)
        x8 = self.linear_p8(x).unsqueeze(0)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=0)

        x = self.pos_embed(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)

        x1 = self.linear_last1(x[0,:,:]).unsqueeze(1)
        x2 = self.linear_last1(x[1,:,:]).unsqueeze(1)
        x3 = self.linear_last1(x[2,:,:]).unsqueeze(1)
        x4 = self.linear_last1(x[3,:,:]).unsqueeze(1)
        x5 = self.linear_last1(x[4,:,:]).unsqueeze(1)
        x6 = self.linear_last1(x[5,:,:]).unsqueeze(1)
        x7 = self.linear_last1(x[6,:,:]).unsqueeze(1)
        x8 = self.linear_last1(x[7,:,:]).unsqueeze(1)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), dim=1).squeeze(-1)

        #prob = nn.Softmax(dim=1)(x)
        # print(prob)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# class Single_exp_detect(nn.Module):
#     def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False):
#         super(Single_exp_detect, self).__init__()
#         self.fea_extractor = Pipeline_Incep()
#         #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
#         self.use_mfcc = use_mfcc
#         if not self.use_mfcc:
#             self.Classifier = nn.Linear(16,8)
#         else:
#             self.gru1 = nn.GRU(40,10,2,batch_first=True,bidirectional=True)
#             self.gru2 = nn.GRU(20,5,1,batch_first=True,bidirectional=True)
#             self.linear1 = nn.Linear(16,10)
#             self.linear2 = nn.Linear(20,8)
            
#     def forward(self,x,mfcc_fea=None):
#         x = self.fea_extractor.forward_no_norm2(x)
#         if not self.use_mfcc:
#             prob = self.Classifier(x)
            
#         else:
#             vis_fea = self.linear1(x)
#             mfcc_out,_ = self.gru1(mfcc_fea)
#             mfcc_out,_ = self.gru2(mfcc_out)
#             mfcc_out = torch.mean(mfcc_out,dim=1)
#             fea = torch.cat((vis_fea,mfcc_out),dim=1)
#             prob = self.linear2(fea)
            
#         return prob

class Single_exp_detect(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb =False,use_coff=False):
        super(Single_exp_detect, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 32
        if self.use_coff:
            self.coff_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,16,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32
        if use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32
        if use_wordemb:
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32

        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            self.inter = nn.Sequential(
                nn.Linear(self.emb_dim,self.emb_dim),
                nn.Tanh(),
                nn.BatchNorm1d(self.emb_dim)
            )

        self.Classifier = nn.Linear(self.emb_dim,8)



    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp


    def forward(self,x,mfcc=None,words=None,embs=None,coff=None,is_drop=False,is_wordemb=False):
        emb = self.exp_emb_net.forward_fea(x)
        emb = self.vis_linear(emb)
        if self.use_coff:
            coff_out,_ = self.coff_gru1(coff)
            coff_out,_ = self.coff_gru2(coff_out)
            coff_out = torch.mean(coff_out,dim=1)
            coff_out = self.coff_linear(coff_out)
            emb = torch.cat((emb,coff_out),dim=1)

        if self.use_exp_emb:
            emb_out,_ = self.emb_gru1(embs)
            emb_out,_ = self.emb_gru2(emb_out)
            emb_out = torch.mean(emb_out,dim=1)
            emb_out = self.emb_linear(emb_out)
            emb = torch.cat((emb,emb_out),dim=1)
            
        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = torch.mean(mfcc_out,dim=1)
            mfcc_out = self.mfcc_linear(mfcc_out)
            emb = torch.cat((emb,mfcc_out),dim=1)
        
        if self.use_wordemb:
            if is_wordemb:
                word_emb = words
            else:
                word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = torch.mean(word_out,dim=1)
            word_out = self.word_linear(word_out)
            emb = torch.cat((emb,word_out),dim=1)
        if is_drop:
            mask = torch.zeros(x.shape[0],self.emb_dim).cuda()
            indexes = np.random.randint(0,30,x.shape[0])
            for i in range(x.shape[0]):
                mask[i] = torch.cuda.FloatTensor(1,self.emb_dim).uniform_() > indexes[i]/100
            emb = emb.mul(mask)
        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            emb = self.inter(emb)
        
        prob = self.Classifier(emb)
        return prob


class Single_exp_detect2(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb =False):
        super(Single_exp_detect2, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,8),
            nn.Tanh(),
            nn.BatchNorm1d(8)
        )
        self.emb_dim = 8
        

        if self.use_exp_emb:
            self.emb_static_linear = nn.Linear(50,8)
            self.emb_gru1 = nn.GRU(50,16,1,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 16
        if use_mfcc:
            self.mfcc_static_linear = nn.Linear(40,8)
            self.gru1 = nn.GRU(40,16,1,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 16
        if use_wordemb:
            self.word_static_linear = nn.Linear(50,8)
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,1,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 16

        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            self.inter = nn.Sequential(
                nn.Linear(self.emb_dim,self.emb_dim),
                nn.Tanh(),
                nn.BatchNorm1d(self.emb_dim)
            )

        self.Classifier = nn.Linear(self.emb_dim,8)



    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp


    def forward(self,x,mfcc=None,words=None,embs=None,is_drop=False):
        emb = self.exp_emb_net.forward_fea(x)
        emb = self.vis_linear(emb)
        
        if self.use_exp_emb:
            ll = (embs.shape[1]-1)//2    
            sta_emb_out = self.emb_static_linear(embs[:,ll,:])
            emb_out,_ = self.emb_gru1(embs)
            emb_out,_ = self.emb_gru2(emb_out)
            emb_out = torch.mean(emb_out,dim=1)
            emb_out = torch.tanh(self.emb_linear(emb_out))
            emb = torch.cat((emb,emb_out,sta_emb_out),dim=1)
            
        if self.use_mfcc:
            ll = (mfcc.shape[1]-1)//2    
            sta_mfcc_out = self.mfcc_static_linear(mfcc[:,ll,:])
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = torch.mean(mfcc_out,dim=1)
            mfcc_out = torch.tanh(self.mfcc_linear(mfcc_out))
            emb = torch.cat((emb,mfcc_out,sta_mfcc_out),dim=1)
        
        if self.use_wordemb:
            
            word_emb = self.get_word_emb(words)

            ll = (word_emb.shape[1]-1)//2    
            sta_words_out = self.word_static_linear(word_emb[:,ll,:])

            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = torch.mean(word_out,dim=1)
            word_out = torch.tanh(self.word_linear(word_out))
            emb = torch.cat((emb,word_out,sta_words_out),dim=1)
        if is_drop:
            mask = torch.zeros(x.shape[0],self.emb_dim).cuda()
            indexes = np.random.randint(0,30,x.shape[0])
            for i in range(x.shape[0]):
                mask[i] = torch.cuda.FloatTensor(1,self.emb_dim).uniform_() > indexes[i]/100
            emb = emb.mul(mask)
        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            emb = self.inter(emb)
        
        prob = self.Classifier(emb)
        return prob



class Single_exp_detect_seq(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb =False):
        super(Single_exp_detect_seq, self).__init__()
        #self.exp_emb_net = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb

        # self.vis_linear = nn.Sequential(
        #     nn.Linear(feature_dim,32),
        #     nn.Tanh(),
        #     nn.BatchNorm1d(32)
        # )
        self.emb_dim = 0

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,16,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32
        if use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32
        if use_wordemb:
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32

        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            self.inter = nn.Sequential(
                nn.Linear(self.emb_dim,self.emb_dim),
                nn.Tanh(),
                nn.BatchNorm1d(self.emb_dim)
            )

        self.Classifier = nn.Linear(self.emb_dim,8)



    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp


    def forward(self,mfcc=None,words=None,embs=None,is_drop=False):
        # emb = self.exp_emb_net.forward_fea(x)
        # emb = self.vis_linear(emb)
        if self.use_exp_emb:
            emb_out,_ = self.emb_gru1(embs)
            emb_out,_ = self.emb_gru2(emb_out)
            emb_out = torch.mean(emb_out,dim=1)
            emb_out = torch.tanh(emb_out)
            emb = emb_out
            #emb = torch.cat((emb,emb_out),dim=1)
            
        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = torch.mean(mfcc_out,dim=1)
            mfcc_out = torch.tanh(mfcc_out)
            emb = torch.cat((emb,mfcc_out),dim=1)
        
        if self.use_wordemb:
            word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = torch.mean(word_out,dim=1)
            word_out = torch.tanh(word_out)
            emb = torch.cat((emb,word_out),dim=1)
        if is_drop:
            mask = torch.zeros(x.shape[0],self.emb_dim).cuda()
            indexes = np.random.randint(0,30,x.shape[0])
            for i in range(x.shape[0]):
                mask[i] = torch.cuda.FloatTensor(1,self.emb_dim).uniform_() > indexes[i]/100
            emb = emb.mul(mask)

        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            emb = self.inter(emb)
        
        prob = self.Classifier(emb)
        return prob


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Single_exp_detect_multi_modal(nn.Module):
    def __init__(self,feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False,use_coff=True):
        super(Single_exp_detect_multi_modal, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,16),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )
        

        self.emb_dim = 64

        self.seq_dim = 0

        if self.use_coff:
            self.coff_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,8,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.coff_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.coff_inter = TransformerDecoder(self.coff_decoder_layer, num_layers=2)

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,10,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(20,8,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.emb_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.emb_inter = TransformerDecoder(self.emb_decoder_layer, num_layers=2)

        if self.use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,8,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.mfcc_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.mfcc_inter = TransformerDecoder(self.mfcc_decoder_layer, num_layers=2)
        
        if self.use_wordemb:
            import gensim.downloader
            print("start_downloads")
            glove_file = './glove_vector.pkl'
            if os.path.exists(glove_file):
                self.glove_vectors = load_pickle(glove_file)
            else:
                self.glove_vectors = gensim.downloader.load('glove-twitter-50')
                with open(glove_file, 'wb') as f:
                    pickle.dump(self.glove_vectors, f)
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,8,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.word_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.word_inter = TransformerDecoder(self.word_decoder_layer, num_layers=2)

        

        self.pos_embed1 = PositionalEncoding(16, dropout=0.5)
        self.pos_embed2 = PositionalEncoding(self.seq_dim, dropout=0.5)
        
        

        self.seq_inter1 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter2 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter3 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        
        

        self.seq_linear = nn.Linear(64,16)
        self.Classifier_final = nn.Linear(32,8)



    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp


    def forward(self,x,mfcc=None,words=None,embs=None,coff=None):
        emb = self.exp_emb_net.forward_fea(x)
        vis_fea = self.vis_linear(emb)
        seq_feas = []

        if self.use_coff:
            coff_out,_ = self.coff_gru1(coff)
            coff_out,_ = self.coff_gru2(coff_out)
            coff_out = coff_out.permute(1,0,2)
            coff_out = self.pos_embed1(coff_out)
            coff_out = self.coff_inter(coff_out,vis_fea.unsqueeze(0))
            seq_feas.append(coff_out)

        if self.use_exp_emb:
            exp_emb_out,_ = self.emb_gru1(embs)
            exp_emb_out,_ = self.emb_gru2(exp_emb_out)
            exp_emb_out = exp_emb_out.permute(1,0,2)
            exp_emb_out = self.pos_embed1(exp_emb_out)
            exp_emb_out = self.emb_inter(exp_emb_out,vis_fea.unsqueeze(0))

            seq_feas.append(exp_emb_out)
            
    
        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = mfcc_out.permute(1,0,2)
            mfcc_out = self.pos_embed1(mfcc_out)
            mfcc_out = self.mfcc_inter(mfcc_out,vis_fea.unsqueeze(0))
            seq_feas.append(mfcc_out)
           
        
        if self.use_wordemb:
            word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = word_out.permute(1,0,2)
            word_out = self.pos_embed1(word_out)
            word_out = self.word_inter(word_out,vis_fea.unsqueeze(0))
            seq_feas.append(word_out)
            

        seq_fea = torch.cat(seq_feas,dim=2)
        seq_fea = self.pos_embed2(seq_fea)

        seq_fea = self.seq_inter1(seq_fea)
        seq_fea = self.seq_inter2(seq_fea)
        seq_fea = self.seq_inter3(seq_fea)
        

        seq_fea = torch.mean(seq_fea,dim=0)
        seq_fea = self.seq_linear(seq_fea)
        
        fea = torch.cat((seq_fea,vis_fea),dim=1)
        prob2 = self.Classifier_final(fea)
            
        return prob2


class Single_exp_detect_trans(nn.Module):
    def __init__(self,feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False,
                 use_coff=True, use_feature3=False, word_emb='glove'):
        super(Single_exp_detect_trans, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff
        self.use_feature3 = use_feature3
        self.word_emb = word_emb

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,16),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )

        self.emb_dim = 64

        self.seq_dim = 0
        if use_feature3:
            self.net_3 = load_net3()

        if self.use_coff:
            self.coff_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,8,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.coff_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.coff_inter = TransformerDecoder(self.coff_decoder_layer, num_layers=2)

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,10,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(20,8,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.emb_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.emb_inter = TransformerDecoder(self.emb_decoder_layer, num_layers=2)

        if self.use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,8,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.mfcc_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.mfcc_inter = TransformerDecoder(self.mfcc_decoder_layer, num_layers=2)
        
        if self.use_wordemb:
            if self.word_emb.lower() == 'glove':
                import gensim.downloader
                glove_file = './glove_vector.pkl'
                print('begin lood GLOVE!')
                if os.path.exists(glove_file):
                    print('load from :', glove_file)
                    self.glove_vectors = load_pickle(glove_file)
                else:
                    print("start_downloads")
                    self.glove_vectors = gensim.downloader.load('glove-twitter-50')
                    with open(glove_file, 'wb') as f:
                        pickle.dump(self.glove_vectors, f)
                word_emb_dim = 50

            elif self.word_emb.lower() == 'bert':
                from transformers import BertTokenizer, BertModel
                print('loading bert model & tokenizer')
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert = BertModel.from_pretrained("bert-base-uncased").eval()
                word_emb_dim = 768
                self.word_linear_bert = nn.Sequential(
                    nn.Linear(768, 16),
                    nn.Tanh(),
                    nn.BatchNorm1d(16)
                )
            print('load success!')
            self.word_gru1 = nn.GRU(word_emb_dim,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,8,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim+=16
            self.word_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.word_inter = TransformerDecoder(self.word_decoder_layer, num_layers=2)

        

        self.pos_embed1 = PositionalEncoding(16, dropout=0.5)
        self.pos_embed2 = PositionalEncoding(self.seq_dim, dropout=0.5)
        
        

        self.seq_inter1 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter2 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter3 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter4 = nn.GRU(64,8,2,batch_first=False,bidirectional=True)

        self.seq_linear = nn.Sequential(
            nn.Linear(121*16,16),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )
        

        self.Classifier_final = nn.Linear(32,8)



    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp

    def set_with_index_order(self,l,):
        l = [ll for ll in l if ll!='-1']
        return sorted(set(l), key=l.index)

    def get_word_emb_bert(self, lists):
        vecs = []
        words = [self.set_with_index_order(l.split(',')) for l in lists]
        sentence = [(' ').join(w) for w in words]
        for i in range(len(lists)):
            token = self.tokenizer(sentence[i], return_tensors='pt')
            tmp = self.bert(**token.to('cuda')).pooler_output
            # tmp = np.array(tmp).astype(float)
            # tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            # print(tmp.shape)
            vecs.append(tmp)
        # vecs = np.array(vecs).astype(float)
        # print(vecs.shape)
        inp = torch.cat(vecs, dim=0)
        # print(inp.shape)
        return inp

    def forward(self,x,mfcc=None,words=None,embs=None,coff=None):
        emb = self.exp_emb_net.forward_fea(x)
        vis_fea_inter = emb
        vis_fea = self.vis_linear(emb)
        seq_feas = []

        if self.use_coff:
            coff_out,_ = self.coff_gru1(coff)
            coff_out,_ = self.coff_gru2(coff_out)
            coff_out = coff_out.permute(1,0,2)
            coff_out = self.pos_embed1(coff_out)
            coff_out = self.coff_inter(coff_out,vis_fea.unsqueeze(0))
            seq_feas.append(coff_out)

        if self.use_exp_emb:
            exp_emb_out,_ = self.emb_gru1(embs)
            exp_emb_out,_ = self.emb_gru2(exp_emb_out)
            exp_emb_out = exp_emb_out.permute(1,0,2)
            exp_emb_out = self.pos_embed1(exp_emb_out)
            exp_emb_out = self.emb_inter(exp_emb_out,vis_fea.unsqueeze(0))

            seq_feas.append(exp_emb_out)
            
    
        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = mfcc_out.permute(1,0,2)
            mfcc_out = self.pos_embed1(mfcc_out)
            mfcc_out = self.mfcc_inter(mfcc_out,vis_fea.unsqueeze(0))
            seq_feas.append(mfcc_out)
           
        
        if self.use_wordemb:
            if self.word_emb.lower()=='glove':
                word_emb = self.get_word_emb(words)
                word_out,_ = self.word_gru1(word_emb)
                word_out,_ = self.word_gru2(word_out)
                word_out = word_out.permute(1,0,2)
                word_out = self.pos_embed1(word_out)
            elif self.word_emb.lower() == 'bert':
                word_emb = self.get_word_emb_bert(words)
                word_out = self.word_linear_bert(word_emb).unsqueeze(0)
                word_out = word_out.repeat(121, 1, 1)
            word_out = self.word_inter(word_out,vis_fea.unsqueeze(0))
            seq_feas.append(word_out)

        seq_fea = torch.cat(seq_feas,dim=2)
        seq_fea = self.pos_embed2(seq_fea)

        seq_fea = self.seq_inter1(seq_fea)
        seq_fea = self.seq_inter2(seq_fea)
        seq_fea = self.seq_inter3(seq_fea)
        seq_fea,_ = self.seq_inter4(seq_fea)
        
        seq_fea = seq_fea.permute(1,0,2)
        seq_fea = seq_fea.reshape(x.shape[0],121*16)
        # seq_fea = torch.mean(seq_fea,dim=0)
        seq_fea = self.seq_linear(seq_fea)
        fea = torch.cat((seq_fea,vis_fea),dim=1)
        self.fea = fea
        if self.use_feature3:
            out3 = self.net_3(x, mfcc, words, embs, coff)
            feature3 = self.net_3.fea
            fea = (fea+feature3)/2

        # fea = seq_fea + vis_fea_inter
        # fea = self.final_linear(fea)
       
        prob2 = self.Classifier_final(fea)

        return prob2

def load_net3():
    net_3 = Single_exp_detect_trans(use_mfcc=True,use_wordemb=True,use_exp_emb =True,use_coff=True, use_feature3=False)
    ck_save_path = './checkpoints/single_task/baseline_EXP:CE+baseline+fewer_data_20220315-074138'
    state_dict = torch.load( f'{ck_save_path}/Exp_best.pth')
    net_3.load_state_dict(state_dict)
    print('==> load pretrained model:', f'{ck_save_path}/Exp_best.pth')
    return net_3

class Single_exp_detect_MISA(nn.Module):
    def __init__(self, feature_dim=512, use_mfcc=False, use_wordemb=False, use_exp_emb=False, use_coff=True):
        super(Single_exp_detect_MISA, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )

        self.emb_dim = 64

        self.seq_dim = 0

        if self.use_coff:
            self.coff_gru1 = nn.GRU(50, 16, 2, batch_first=True, bidirectional=True)
            self.coff_gru2 = nn.GRU(32, 8, 2, batch_first=True, bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim += 16
            self.coff_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.coff_inter = TransformerDecoder(self.coff_decoder_layer, num_layers=2)

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16, 10, 2, batch_first=True, bidirectional=True)
            self.emb_gru2 = nn.GRU(20, 8, 2, batch_first=True, bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim += 16
            self.emb_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.emb_inter = TransformerDecoder(self.emb_decoder_layer, num_layers=2)

        if self.use_mfcc:
            self.gru1 = nn.GRU(40, 16, 2, batch_first=True, bidirectional=True)
            self.gru2 = nn.GRU(32, 8, 2, batch_first=True, bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim += 16
            self.mfcc_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.mfcc_inter = TransformerDecoder(self.mfcc_decoder_layer, num_layers=2)

        if self.use_wordemb:
            import gensim.downloader
            glove_file = './glove_vector.pkl'
            print('begin laod GLOVE!')
            if os.path.exists(glove_file):
                print('load from :', glove_file)
                self.glove_vectors = load_pickle(glove_file)
            else:
                print("start_downloads")
                self.glove_vectors = gensim.downloader.load('glove-twitter-50')
                with open(glove_file, 'wb') as f:
                    pickle.dump(self.glove_vectors, f)
            print('load success!')
            self.word_gru1 = nn.GRU(50, 16, 2, batch_first=True, bidirectional=True)
            self.word_gru2 = nn.GRU(32, 8, 2, batch_first=True, bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.seq_dim += 16
            self.word_decoder_layer = TransformerDecoderLayer(16, nhead=4)
            self.word_inter = TransformerDecoder(self.word_decoder_layer, num_layers=2)

        self.pos_embed1 = PositionalEncoding(16, dropout=0.5)
        self.pos_embed2 = PositionalEncoding(self.seq_dim, dropout=0.5)

        self.seq_inter1 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter2 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter3 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter4 = nn.GRU(64, 8, 2, batch_first=False, bidirectional=True)

        self.private_a = nn.Linear(16, 16)
        self.private_t = nn.Linear(16, 16)
        self.private_v = nn.Linear(16, 16)
        self.shared = nn.Linear(16, 16)
        self.discriminator = nn.Linear(16,16)
        self.recon_a = nn.Linear(16, 16)
        self.recon_t = nn.Linear(16, 16)
        self.recon_v = nn.Linear(16, 16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=16, nhead=2)
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=16 * 6,
                                                           out_features=16 * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(0.1))
        self.fusion.add_module('fusion_layer_1_activation', nn.Tanh())
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=16 * 3, out_features=8))

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.loss_ortho = Ortho()
        self.loss_inner = Inner()

        self.seq_linear = nn.Sequential(
            nn.Linear(121 * 16, 16),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )

        self.Classifier_final = nn.Linear(32, 8)

    def get_word_emb(self, lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w != "-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))

            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            # print(tmp.shape)
            vecs.append(tmp)
        # vecs = np.array(vecs).astype(float)
        # print(vecs.shape)
        inp = torch.cat(vecs, dim=0)
        # print(inp.shape)
        return inp

    def shared_private(self, utterance_t, utterance_v, utterance_a):

        # Private-shared components
        self.utt_t_orig = utterance_t
        self.utt_v_orig = utterance_v
        self.utt_a_orig = utterance_a

        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

        self.shared_or_private_p_t = self.discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.discriminator(self.utt_private_a)
        self.shared_or_private_s = self.discriminator(
            (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)

        #recons
        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t,
                         self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(torch.mean(h, dim=2))
        h = h.permute(1,0,2).reshape(-1,6*16)
        o = self.fusion(h)
        return o

    def get_MISA_Loss(self):
        # diff loss
        # b * feature_dim (16 * 128)
        shared_t =  self.utt_shared_t
        shared_v =  self.utt_shared_v
        shared_a =  self.utt_shared_a
        private_t =  self.utt_private_t
        private_v =  self.utt_private_v
        private_a =  self.utt_private_a

        # Between private and shared
        loss_t = self.loss_diff(private_t, shared_t)
        loss_v = self.loss_diff(private_v, shared_v)
        loss_a = self.loss_diff(private_a, shared_a)
        # Across privates
        loss_at = self.loss_diff(private_a, private_t)
        loss_av = self.loss_diff(private_a, private_v)
        loss_tv = self.loss_diff(private_t, private_v)
        loss = loss_t
        loss += loss_v
        loss += loss_a
        loss += loss_at
        loss += loss_av
        loss += loss_tv

        # recons loss
        loss_recon = self.loss_recon( self.utt_t_recon,  self.utt_t_orig)
        loss_recon += self.loss_recon( self.utt_v_recon,  self.utt_v_orig)
        loss_recon += self.loss_recon( self.utt_a_recon,  self.utt_a_orig)
        loss += loss_recon/3.0

        # cmd loss
        cmd_tv = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, 5)
        loss_cmd = cmd_tv
        cmd_ta = self.loss_cmd(self.utt_shared_t, self.utt_shared_a, 5)
        cmd_av = self.loss_cmd(self.utt_shared_a, self.utt_shared_v, 5)
        loss_cmd+= cmd_ta
        loss_cmd+= cmd_av
        loss += loss_cmd/3.0

        return loss

    def forward(self, x, mfcc=None, words=None, embs=None, coff=None):
        emb = self.exp_emb_net.forward_fea(x)
        vis_fea_inter = emb
        vis_fea = self.vis_linear(emb)
        seq_feas = []

        if self.use_coff:
            coff_out, _ = self.coff_gru1(coff)
            coff_out, _ = self.coff_gru2(coff_out)
            coff_out = coff_out.permute(1, 0, 2)
            coff_out = self.pos_embed1(coff_out)
            coff_out = self.coff_inter(coff_out, vis_fea.unsqueeze(0))
            seq_feas.append(coff_out)

        if self.use_exp_emb:
            exp_emb_out, _ = self.emb_gru1(embs)
            exp_emb_out, _ = self.emb_gru2(exp_emb_out)
            exp_emb_out = exp_emb_out.permute(1, 0, 2)
            exp_emb_out = self.pos_embed1(exp_emb_out)
            exp_emb_out = self.emb_inter(exp_emb_out, vis_fea.unsqueeze(0))

            seq_feas.append(exp_emb_out)

        if self.use_mfcc:
            mfcc_out, _ = self.gru1(mfcc)
            mfcc_out, _ = self.gru2(mfcc_out)
            mfcc_out = mfcc_out.permute(1, 0, 2)
            mfcc_out = self.pos_embed1(mfcc_out)
            mfcc_out = self.mfcc_inter(mfcc_out, vis_fea.unsqueeze(0))
            seq_feas.append(mfcc_out)

        if self.use_wordemb:
            word_emb = self.get_word_emb(words)
            word_out, _ = self.word_gru1(word_emb)
            word_out, _ = self.word_gru2(word_out)
            word_out = word_out.permute(1, 0, 2)
            word_out = self.pos_embed1(word_out)
            word_out = self.word_inter(word_out, vis_fea.unsqueeze(0))
            seq_feas.append(word_out)

        out = self.shared_private(seq_feas[0].permute(1,0,2), seq_feas[1].permute(1,0,2), seq_feas[2].permute(1,0,2))
        #
        # seq_fea = torch.cat(seq_feas, dim=2)
        # seq_fea = self.pos_embed2(seq_fea)
        #
        # seq_fea = self.seq_inter1(seq_fea)
        # seq_fea = self.seq_inter2(seq_fea)
        # seq_fea = self.seq_inter3(seq_fea)
        # seq_fea, _ = self.seq_inter4(seq_fea)
        #
        # seq_fea = seq_fea.permute(1, 0, 2)
        # seq_fea = seq_fea.reshape(x.shape[0], 121 * 16)
        # # seq_fea = torch.mean(seq_fea,dim=0)
        # seq_fea = self.seq_linear(seq_fea)
        # fea = torch.cat((seq_fea, vis_fea), dim=1)
        # # fea = seq_fea + vis_fea_inter
        # # fea = self.final_linear(fea)
        #
        # prob2 = self.Classifier_final(fea)

        return out


class Single_exp_detect_Eff(nn.Module):
    def __init__(self,feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False,
                 use_coff=True,use_feature3=False, word_emb='glove', use_bert=False, emb_dim=16, coef_dim=50):
        super(Single_exp_detect_Eff, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff
        self.use_feature3 = use_feature3
        self.word_emb = word_emb
        self.use_bert = use_bert

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,32),
            # nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        
        self.vis_enc_h = nn.Sequential(
            nn.Linear(32,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )

        self.vis_enc_i = nn.Sequential(
            nn.Linear(32,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )

        self.emb_dim = 32

        if use_feature3:
            self.net_3 = load_net3()

        if self.use_coff:
            self.coff_gru1 = nn.GRU(coef_dim,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,32),
                # nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim+=32
            self.coff_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.coff_inter = TransformerDecoder(self.coff_decoder_layer, num_layers=2)

            self.coff_enc_i =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

            self.coff_enc_h =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(emb_dim,10,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(20,16,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,32),
                # nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim+=32
            self.emb_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.emb_inter = TransformerDecoder(self.emb_decoder_layer, num_layers=2)

            self.emb_enc_i =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

            self.emb_enc_h =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

        if self.use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,32),
                # nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim+=32
            self.mfcc_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.mfcc_inter = TransformerDecoder(self.mfcc_decoder_layer, num_layers=2)
            self.mfcc_enc_i =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

            self.mfcc_enc_h =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )


        if self.use_wordemb:
            if self.word_emb.lower() == 'glove':
                import gensim.downloader
                glove_file = './glove_vector.pkl'
                print('begin lood GLOVE!')
                if os.path.exists(glove_file):
                    print('load from :', glove_file)
                    self.glove_vectors = load_pickle(glove_file)
                else:
                    print("start_downloads")
                    self.glove_vectors = gensim.downloader.load('glove-twitter-50')
                    with open(glove_file, 'wb') as f:
                        pickle.dump(self.glove_vectors, f)
                print("load word model success!")
                word_emb_dim = 50
            if self.word_emb.lower() == 'bert' or self.use_bert:
                from transformers import BertTokenizer, BertModel
                print('loading bert model & tokenizer')
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert = BertModel.from_pretrained("bert-base-uncased").eval()
                self.word_linear_bert = nn.Sequential(
                    nn.Linear(768, 32),
                    nn.Tanh(),
                    nn.BatchNorm1d(32)
                )
            if self.word_emb.lower()== 'bert':
                word_emb_dim = 768

            self.word_gru1 = nn.GRU(word_emb_dim,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,32),
                # nn.Tanh(),
                nn.BatchNorm1d(32)
            )


            self.emb_dim+=32
            self.word_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.word_inter = TransformerDecoder(self.word_decoder_layer, num_layers=2)
            self.word_enc_i =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

            self.word_enc_h =  nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

            if self.use_bert:
                self.emb_dim += 32
                self.bert_decoder_layer = TransformerDecoderLayer(32, nhead=4)
                self.bert_inter = TransformerDecoder(self.bert_decoder_layer, num_layers=2)
                self.bert_enc_i = nn.Sequential(
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.BatchNorm1d(32)
                )

                self.bert_enc_h = nn.Sequential(
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.BatchNorm1d(32)
                )

        self.pos_embed1 = PositionalEncoding(32, dropout=0.5)
        
        
        self.h_linear = nn.Sequential(
                nn.Linear(self.emb_dim,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )

        self.i_linear = nn.Sequential(
                nn.Linear(self.emb_dim,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )



        self.Classifier_final = nn.Linear(32*3,8)


    def get_word_emb_bert(self, lists):
        vecs = []
        words = [self.set_with_index_order(l.split(',')) for l in lists]
        sentence = [(' ').join(w) for w in words]
        for i in range(len(lists)):
            token = self.tokenizer(sentence[i], return_tensors='pt')
            tmp = self.bert(**token.to('cuda')).pooler_output
            # tmp = np.array(tmp).astype(float)
            # tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            # print(tmp.shape)
            vecs.append(tmp)
        # vecs = np.array(vecs).astype(float)
        # print(vecs.shape)
        inp = torch.cat(vecs, dim=0)
        # print(inp.shape)
        return inp

    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp

    def set_with_index_order(self,l,):
        l = [ll for ll in l if ll!='-1']
        return sorted(set(l), key=l.index)


    def forward(self,x,mfcc=None,words=None,embs=None,coff=None,is_wordemb=False):
        emb = self.exp_emb_net.forward_fea(x)
        vis_fea_inter = emb
        vis_fea = self.vis_linear(emb)
        vis_h_fea = self.vis_enc_h(vis_fea)
        h_feas = [vis_h_fea]
        i_feas = []

        seq_feas = [vis_fea]

        if self.use_coff:
            coff_out,_ = self.coff_gru1(coff)
            coff_out,_ = self.coff_gru2(coff_out)
            coff_out = coff_out.permute(1,0,2)
            coff_out = self.pos_embed1(coff_out)
            coff_out = self.coff_inter(coff_out,vis_fea.unsqueeze(0))
            coff_out = torch.mean(coff_out,dim=0)
            seq_feas.append(coff_out)

            coff_h_fea = self.coff_enc_h(coff_out)
            h_feas.append(coff_h_fea)
            

        if self.use_exp_emb:
            exp_emb_out,_ = self.emb_gru1(embs)
            exp_emb_out,_ = self.emb_gru2(exp_emb_out)
            exp_emb_out = exp_emb_out.permute(1,0,2)
            exp_emb_out = self.pos_embed1(exp_emb_out)
            exp_emb_out = self.emb_inter(exp_emb_out,vis_fea.unsqueeze(0))
            exp_emb_out = torch.mean(exp_emb_out,dim=0)
            seq_feas.append(exp_emb_out)
            exp_emb_h_fea = self.emb_enc_h(exp_emb_out)
            h_feas.append(exp_emb_h_fea)
            
            
    
        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = mfcc_out.permute(1,0,2)
            mfcc_out = self.pos_embed1(mfcc_out)
            mfcc_out = self.mfcc_inter(mfcc_out,vis_fea.unsqueeze(0))
            mfcc_out = torch.mean(mfcc_out,dim=0)
            seq_feas.append(mfcc_out)

            mfcc_h_fea = self.mfcc_enc_h(mfcc_out)
            h_feas.append(mfcc_h_fea)
            
           
        
        if self.use_wordemb:
            if self.word_emb.lower() == 'glove':
                if is_wordemb:
                    word_emb = words
                else:
                    word_emb = self.get_word_emb(words)
                word_out,_ = self.word_gru1(word_emb)
                word_out,_ = self.word_gru2(word_out)
                word_out = word_out.permute(1,0,2)
                word_out = self.pos_embed1(word_out)
            elif self.word_emb.lower() == 'bert':
                with torch.no_grad():
                    word_emb = self.get_word_emb_bert(words)
                word_out = self.word_linear_bert(word_emb).unsqueeze(0)
                word_out = word_out.repeat(121, 1, 1)
            word_out = self.word_inter(word_out,vis_fea.unsqueeze(0))
            word_out = torch.mean(word_out,dim=0)
            seq_feas.append(word_out)

            word_h_fea = self.word_enc_h(word_out)
            h_feas.append(word_h_fea)

        if self.use_bert:
            with torch.no_grad():
                word_emb = self.get_word_emb_bert(words)
            word_out = self.word_linear_bert(word_emb).unsqueeze(0)
            word_out = word_out.repeat(121, 1, 1)
            word_out = self.bert_inter(word_out,vis_fea.unsqueeze(0))
            word_out = torch.mean(word_out,dim=0)
            seq_feas.append(word_out)
            bert_h_fea = self.bert_enc_h(word_out)
            h_feas.append(bert_h_fea)

        mean = 0
        for v in seq_feas:
            mean = mean + v
        
        mean = mean/len(seq_feas)

        in_feas = [v-mean for v in seq_feas]
        vis_i_fea = self.vis_enc_i(in_feas[0])
        i_feas.append(vis_i_fea)
        index = 1
        
        if self.use_coff:
            coff_i_fea = in_feas[index]
            coff_i_fea = self.coff_enc_i(coff_i_fea)
            i_feas.append(coff_i_fea)
            index +=1
        if self.use_exp_emb:
            emb_i_fea = in_feas[index]
            emb_i_fea = self.emb_enc_i(emb_i_fea)
            i_feas.append(emb_i_fea)
            index+=1
        if self.use_mfcc:
            mfcc_i_fea = in_feas[index]
            mfcc_i_fea = self.mfcc_enc_i(mfcc_i_fea)
            i_feas.append(mfcc_i_fea)
            index+=1
        if self.use_wordemb:
            word_i_fea = in_feas[index]
            word_i_fea = self.word_enc_i(word_i_fea)
            i_feas.append(word_i_fea)
            index+=1
        if self.use_bert:
            bert_i_fea = in_feas[index]
            bert_i_fea = self.word_enc_i(bert_i_fea)
            i_feas.append(bert_i_fea)
            index+=1

        h_fea= torch.cat(h_feas,dim=1)
        h_fea= self.h_linear(h_fea)
        i_fea = torch.cat(i_feas,dim=1)
        i_fea = self.i_linear(i_fea)

        final_fea = torch.cat((mean,h_fea,i_fea),dim=1)
        # with torch.no_grad():

        if self.use_feature3:
            out3 = self.net_3(x, mfcc, words, embs, coff)
            feature3 = self.net_3.fea
            feature3 = feature3.repeat(1,3)
            final_fea = (final_fea+feature3)/2

        self.fea = final_fea

        prob = self.Classifier_final(final_fea)


        return prob


class Single_exp_detect_combine(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb =False,use_coff=False):
        super(Single_exp_detect_combine, self).__init__()
        self.exp_emb_net = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 32

        if self.use_coff and self.use_exp_emb:
            self.coff_gru1 = nn.GRU(50+16,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32

        # if self.use_exp_emb:
        #     self.emb_gru1 = nn.GRU(16,16,2,batch_first=True,bidirectional=True)
        #     self.emb_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
        #     self.emb_linear = nn.Sequential(
        #         nn.Linear(32,32),
        #         nn.Tanh(),
        #         nn.BatchNorm1d(32)
        #     )
        #     self.emb_dim += 32
        if use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32
        if use_wordemb:
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.emb_dim += 32

        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            self.inter = nn.Sequential(
                nn.Linear(self.emb_dim,self.emb_dim),
                nn.Tanh(),
                nn.BatchNorm1d(self.emb_dim)
            )

        self.Classifier = nn.Linear(self.emb_dim,8)



    def get_word_emb(self,lists):
        vecs = []
        for i in range(len(lists)):
            tmp = []
            words = lists[i].split(",")
            # print(len(words))
            for w in words:
                if w!="-1":
                    tmp.append(self.glove_vectors[w])
                    # print(len(self.glove_vectors[w]))
                else:
                    tmp.append(np.array([0 for i in range(50)]))
            
            tmp = np.array(tmp).astype(float)
            tmp = torch.from_numpy(tmp).float().cuda().unsqueeze(0)
            #print(tmp.shape)
            vecs.append(tmp)
        #vecs = np.array(vecs).astype(float) 
        #print(vecs.shape)
        inp = torch.cat(vecs,dim=0)
       # print(inp.shape)
        return inp


    def forward(self,x,mfcc=None,words=None,embs=None,coff=None,is_drop=False,is_wordemb=False):
        emb = self.exp_emb_net.forward_fea(x)
        emb = self.vis_linear(emb)

        if self.use_coff and self.use_exp_emb:
            coff = torch.cat((coff,embs),dim=2)
            coff_out,_ = self.coff_gru1(coff)
            coff_out,_ = self.coff_gru2(coff_out)
            coff_out = torch.mean(coff_out,dim=1)
            coff_out = self.coff_linear(coff_out)
            emb = torch.cat((emb,coff_out),dim=1)

       
            
        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = torch.mean(mfcc_out,dim=1)
            mfcc_out = self.mfcc_linear(mfcc_out)
            emb = torch.cat((emb,mfcc_out),dim=1)
        
        if self.use_wordemb:
            if is_wordemb:
                word_emb = words
            else:
                word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = torch.mean(word_out,dim=1)
            word_out = self.word_linear(word_out)
            emb = torch.cat((emb,word_out),dim=1)
        if is_drop:
            mask = torch.zeros(x.shape[0],self.emb_dim).cuda()
            indexes = np.random.randint(0,30,x.shape[0])
            for i in range(x.shape[0]):
                mask[i] = torch.cuda.FloatTensor(1,self.emb_dim).uniform_() > indexes[i]/100
            emb = emb.mul(mask)
        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            emb = self.inter(emb)
        
        prob = self.Classifier(emb)
        return prob