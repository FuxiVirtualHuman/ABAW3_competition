import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.crop_model_relu_local_feature_16d import crop_model
from models.pipeline5 import Pipeline
import math
import torch
from models.pipeline_student_InceptionResnet import Pipeline_Incep

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

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

class Single_va_detect(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False,use_coff=False):
        super(Single_va_detect, self).__init__()
        self.fea_extractor = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff


        self.vis_linear = nn.Sequential(
            nn.Linear(512,16),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )
        self.emb_dim = 16

        if self.use_coff:
            self.coff_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 8

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,16,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 8
        if use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 8
        if use_wordemb:
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,8),
                nn.Tanh(),
                nn.BatchNorm1d(8)
            )
            self.emb_dim += 8

        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            self.inter = nn.Sequential(
                nn.Linear(self.emb_dim,self.emb_dim),
                nn.Tanh(),
                nn.BatchNorm1d(self.emb_dim)
            )
        
        self.out_linear1 = nn.Linear(self.emb_dim,16)
        self.out_linear2 = nn.Linear(16,2)
        
    
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

    def forward(self,x,embs=None,mfcc=None,words=None,coff=None,is_drop=False,is_wordemb=False):
        x = self.fea_extractor.forward_fea(x)
        emb = self.vis_linear(x)
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
        # print(emb.shape)
        # print(self.emb_dim)
        if is_drop:
            mask = torch.zeros(x.shape[0],self.emb_dim).cuda()
            indexes = np.random.randint(0,30,x.shape[0])
            for i in range(x.shape[0]):
                mask[i] = torch.cuda.FloatTensor(1,self.emb_dim).uniform_() > indexes[i]/100
            emb = emb.mul(mask)
        if self.use_wordemb or self.use_mfcc or self.use_exp_emb:
            emb = self.inter(emb)

        out = self.out_linear1(emb)
        out = self.out_linear2(out)
        prob = torch.tanh(out)
            
        return prob


class Single_va_detect_multi_modal(nn.Module):
    def __init__(self,feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False):
        super(Single_va_detect_multi_modal, self).__init__()
        self.exp_emb_net = Pipeline_Incep()

        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb

        self.vis_linear = nn.Sequential(
            nn.Linear(feature_dim,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 64
        self.seq_dim = 0
        self.sta_dim = 0
        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,10,1,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(20,16,1,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.sta_dim+=16
            self.seq_dim+=16

        if self.use_mfcc:
            self.gru1 = nn.GRU(40,16,1,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.sta_dim+=40
            self.seq_dim+=16
        
        if self.use_wordemb:
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,1,batch_first=True,bidirectional=True)
            self.word_gru2 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)
            self.word_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.sta_dim+=50
            self.seq_dim+=16

        
        
        
        self.seq_linear = nn.Linear(self.seq_dim,16)
        self.sta_linear = nn.Linear(self.sta_dim,16)
        self.Classifier1 = nn.Linear(32,2)
            
        
        
        self.Classifier_final = nn.Linear(64,2)



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


    def forward(self,x,mfcc=None,words=None,embs=None):
        emb = self.exp_emb_net.forward_fea(x)
        vis_fea = self.vis_linear(emb)
        seq_feas = []
        sta_feas = []
        if self.use_exp_emb:
            exp_emb_out,_ = self.emb_gru1(embs)
            exp_emb_out,_ = self.emb_gru2(exp_emb_out)
            exp_emb_out = torch.mean(exp_emb_out,dim=1)
            exp_emb_out = torch.tanh(self.emb_linear(exp_emb_out))
            seq_feas.append(exp_emb_out)
            l1 = (embs.shape[1]-1)//2
            sta_feas.append(embs[:,l1,:])

        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = torch.mean(mfcc_out,dim=1)
            mfcc_out = torch.tanh(self.mfcc_linear(mfcc_out))
            seq_feas.append(mfcc_out)
            l2 = (mfcc.shape[1]-1)//2
            sta_feas.append(mfcc[:,l2,:])
        
        if self.use_wordemb:
            word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = torch.mean(word_out,dim=1)
            word_out = torch.tanh(self.word_linear(word_out))
            seq_feas.append(word_out)
            l3 = (word_emb.shape[1]-1)//2
            sta_feas.append(word_emb[:,l3,:])

        seq_fea = torch.cat(seq_feas,dim=1)
        seq_fea = self.seq_linear(seq_fea)
        sta_fea = torch.cat(sta_feas,dim=1)
        sta_fea = self.sta_linear(sta_fea)
        fea1 = torch.cat((seq_fea,sta_fea),dim=1)
        prob1 = torch.tanh(self.Classifier1(fea1))

        fea2 = torch.cat((fea1,vis_fea),dim=1)
        prob2 = torch.tanh(self.Classifier_final(fea2))
            
        return prob1,prob2



class Single_va_detect_Eff(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False,use_coff=False):
        super(Single_va_detect_Eff, self).__init__()
        self.fea_extractor = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff


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


        if self.use_coff:
            self.coff_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
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
            self.emb_gru1 = nn.GRU(16,10,2,batch_first=True,bidirectional=True)
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
            import gensim.downloader
            print("start_downloads")
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
            print("load word model success!")
            self.word_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
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

        self.Classifier_final = nn.Linear(32*3,2)

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

    def forward(self,x,embs=None,mfcc=None,words=None,coff=None,is_drop=False,is_wordemb=False):
        emb = self.fea_extractor.forward_fea(x)
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
            if is_wordemb:
                word_emb = words
            else:
                word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = word_out.permute(1,0,2)
            word_out = self.pos_embed1(word_out)
            word_out = self.word_inter(word_out,vis_fea.unsqueeze(0))
            word_out = torch.mean(word_out,dim=0)
            seq_feas.append(word_out)

            word_h_fea = self.word_enc_h(word_out)
            h_feas.append(word_h_fea)

        
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

        h_fea= torch.cat(h_feas,dim=1)
        h_fea= self.h_linear(h_fea)
        i_fea = torch.cat(i_feas,dim=1)
        i_fea = self.i_linear(i_fea)

        final_fea = torch.cat((mean,h_fea,i_fea),dim=1)
        prob = self.Classifier_final(final_fea)


        return prob


class Single_va_detect_trans(nn.Module):
    def __init__(self, dataset="Aff2", feature_dim=512,use_mfcc=False,use_wordemb=False,use_exp_emb=False,use_coff=False):
        super(Single_va_detect_trans, self).__init__()
        self.fea_extractor = Pipeline_Incep()
        #self.Classifier = Classifier_T_for_exp(dataset="DISFA", feature_dim=512)
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
            self.glove_vectors = gensim.downloader.load('glove-twitter-50')
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
        self.seq_inter4 = nn.GRU(64,8,2,batch_first=False,bidirectional=True)
        

        self.seq_linear = nn.Sequential(
            nn.Linear(121*16,16),
            nn.Dropout(0.3),
            nn.Tanh(),
            nn.BatchNorm1d(16)
        )

        self.Classifier_final = nn.Linear(32,2)

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

    def forward(self,x,embs=None,mfcc=None,words=None,coff=None,is_drop=False,is_wordemb=False):
        emb = self.fea_extractor.forward_fea(x)
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
        seq_fea,_ = self.seq_inter4(seq_fea)
        
        seq_fea = seq_fea.permute(1,0,2)
        seq_fea = seq_fea.reshape(x.shape[0],121*16)
        # seq_fea = torch.mean(seq_fea,dim=0)
        seq_fea = self.seq_linear(seq_fea)
        fea = torch.cat((seq_fea,vis_fea),dim=1)
        # fea = seq_fea + vis_fea_inter
        # fea = self.final_linear(fea)
       
        prob2 = self.Classifier_final(fea)
            
        return prob2
       