from models.pipeline5 import Pipeline
import torch.nn as nn
import torch
from models.pipeline_student_InceptionResnet import Pipeline_Incep
import numpy as np
import math
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
class Multi_task_series_model(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",use_mfcc=False,use_wordemb=False,use_exp_emb =False):
        super(Multi_task_series_model,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

      
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb

        self.vis_linear = nn.Sequential(
            nn.Linear(512,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 32


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
            

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 8)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

        # self.resnet_linear = nn.Linear(128,512)

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
        
    def forward_baseline(self,x,embs=None,mfcc=None,words=None,is_drop=False):
        emb = self.exp_emb_net.forward_fea(x)
        emb = self.vis_linear(emb)
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

        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out






class Multi_task_series_model_Eff(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",use_mfcc=False,use_wordemb=False,use_exp_emb =False,use_coff=False):
        super(Multi_task_series_model_Eff,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

      
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb
        self.use_coff = use_coff


        self.vis_linear = nn.Sequential(
            nn.Linear(512,32),
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
        
        self.emb_dim = 32*3

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 8)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

        # self.resnet_linear = nn.Linear(128,512)

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
        
    def forward_baseline(self,x,embs=None,mfcc=None,words=None,coff=None,is_drop=False,is_wordemb=False):
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

        emb = torch.cat((mean,h_fea,i_fea),dim=1)

        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out




class Multi_task_series_model_trans(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",use_mfcc=False,use_wordemb=False,use_exp_emb =False,use_coff=False):
        super(Multi_task_series_model_trans,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

        self.use_coff = use_coff
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb

        self.vis_linear = nn.Sequential(
            nn.Linear(512,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 64

        self.seq_dim = 0

        if self.use_coff:
            self.coff_gru1 = nn.GRU(50,16,2,batch_first=True,bidirectional=True)
            self.coff_gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.coff_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.seq_dim+=32
            self.coff_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.coff_inter = TransformerDecoder(self.coff_decoder_layer, num_layers=2)

        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,10,2,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(20,16,2,batch_first=True,bidirectional=True)
            self.emb_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.seq_dim+=32
            self.emb_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.emb_inter = TransformerDecoder(self.emb_decoder_layer, num_layers=2)

        if self.use_mfcc:
            self.gru1 = nn.GRU(40,16,2,batch_first=True,bidirectional=True)
            self.gru2 = nn.GRU(32,16,2,batch_first=True,bidirectional=True)
            self.mfcc_linear = nn.Sequential(
                nn.Linear(32,32),
                nn.Tanh(),
                nn.BatchNorm1d(32)
            )
            self.seq_dim+=32
            self.mfcc_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.mfcc_inter = TransformerDecoder(self.mfcc_decoder_layer, num_layers=2)
        
        if self.use_wordemb:
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
            self.seq_dim+=32
            self.word_decoder_layer = TransformerDecoderLayer(32, nhead=4)
            self.word_inter = TransformerDecoder(self.word_decoder_layer, num_layers=2)

        

        self.pos_embed1 = PositionalEncoding(32, dropout=0.5)
        self.pos_embed2 = PositionalEncoding(self.seq_dim, dropout=0.5)
        
        

        self.seq_inter1 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter2 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        self.seq_inter3 = TransformerEncoderLayer(self.seq_dim, nhead=4, dim_feedforward=16, dropout=0.5)
        
        

        self.seq_linear = nn.Sequential(
            nn.Linear(self.seq_dim,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 32

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 8)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

        # self.resnet_linear = nn.Linear(128,512)

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
        
    def forward_baseline(self,x,embs=None,mfcc=None,words=None,coff=None,is_drop=False):
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
        emb = seq_fea+vis_fea
        # emb = torch.cat((seq_fea,vis_fea),dim=1)

        if is_drop:
            mask = torch.zeros(x.shape[0],self.emb_dim).cuda()
            indexes = np.random.randint(0,30,x.shape[0])
            for i in range(x.shape[0]):
                mask[i] = torch.cuda.FloatTensor(1,self.emb_dim).uniform_() > indexes[i]/100
            emb = emb.mul(mask)

        

        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out

class Multi_task_series_model2(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",use_mfcc=False,use_wordemb=False,use_exp_emb =False):
        super(Multi_task_series_model2,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model

      
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb

        self.vis_linear = nn.Sequential(
            nn.Linear(512,32),
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

        
        print(self.sta_dim,self.seq_dim)
        self.seq_linear = nn.Linear(self.seq_dim,16)
        self.sta_linear = nn.Linear(self.sta_dim,16)
        self.AU_Classifier1 = nn.Linear(32,12)
        self.VA_Classifier1 = nn.Linear(32,2)
        self.EXP_Classifier1 = nn.Linear(32,8)

            

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 8)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

        # self.resnet_linear = nn.Linear(128,512)

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
        
    def forward_baseline(self,x,embs=None,mfcc=None,words=None,is_drop=False):
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
        AU_inter_prob = self.AU_Classifier1(fea1)
        EXP_inter_prob = self.EXP_Classifier1(fea1)
        VA_inter_prob = torch.tanh(self.VA_Classifier1(fea1))

        emb = torch.cat((fea1,vis_fea),dim=1)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out,VA_inter_prob,AU_inter_prob,EXP_inter_prob






class Multi_task_series_model_emo_VA(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",freeze=False):
        super(Multi_task_series_model_emo_VA,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

        #if pretrained:
        #    state_dict = torch.load(pretrained,"cuda:0")
        #    self.exp_emb_net.load_state_dict(state_dict)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)

        self.VA_out1 = nn.Linear(hidden_size,2)
        
        self.VA_linear2 = nn.Linear(hidden_size*2,hidden_size)
        self.VA_out2 = nn.Linear(hidden_size,2)

        self.VA_inter = nn.Linear(hidden_size,hidden_size)


        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_out1 = nn.Linear(hidden_size,8)


        self.Exp_linear2 = nn.Linear(hidden_size*2, hidden_size)
        self.Exp_out2 = nn.Linear(hidden_size,8)


        #self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(hidden_size,hidden_size)


    def forward_baseline(self,x,output_VA=True,output_Exp=True): 
        emb = self.exp_emb_net.forward_no_norm2(x)
        #VA_out,Exp_out = None,None,None,None

        Exp_fea = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_out1 = self.Exp_out1(Exp_fea)

        Exp_inter = torch.relu(self.Exp_inter(Exp_fea))

        VA_fea = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_out1 = torch.tanh(self.VA_out1(VA_fea))

        VA_inter = torch.relu(self.VA_inter(VA_fea))

        Exp_out2 = torch.cat((Exp_fea,VA_inter),dim=1)
        Exp_out2 = torch.relu(self.Exp_linear2(Exp_out2))
        Exp_out2 = self.Exp_out2(Exp_out2)

        VA_out2 = torch.cat((VA_fea,Exp_inter),dim=1)
        VA_out2 = torch.relu(self.VA_linear2(VA_out2))
        VA_out2 = torch.tanh(self.VA_out2(VA_out2))

        return Exp_out1,Exp_out2,VA_out1,VA_out2




class Multi_task_series_model_affectnet(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",freeze=False):
        super(Multi_task_series_model_affectnet,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

        #if pretrained:
        #    state_dict = torch.load(pretrained,"cuda:0")
        #    self.exp_emb_net.load_state_dict(state_dict)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 8)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 8)

        # self.resnet_linear = nn.Linear(128,512)


    def forward_baseline(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out




class Multi_task_series_model_full(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",freeze=False):
        super(Multi_task_series_model_full,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model
        self.emb_dim = emb_dim

        #if pretrained:
        #    state_dict = torch.load(pretrained,"cuda:0")
        #    self.exp_emb_net.load_state_dict(state_dict)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 7)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p13 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p14 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p15 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last13 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last14 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last15 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(30, 30)
        self.AU_inter = nn.Linear(240,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

        # self.resnet_linear = nn.Linear(128,512)


    def forward_baseline(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        x13 = self.AU_linear_p13(emb)
        x13_inter = x13
        x13 = self.AU_linear_last13(x13).unsqueeze(1)
        x14 = self.AU_linear_p14(emb)
        x14_inter = x14
        x14 = self.AU_linear_last14(x14).unsqueeze(1)
        x15 = self.AU_linear_p15(emb)
        x15_inter = x15
        x15 = self.AU_linear_last15(x15).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,x13,x14,x15), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter, x13_inter, x14_inter, x15_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out




    def forward_dropout(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        final_AU_out = AU_out
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        # AU_out_ = AU_out.view(AU_out.shape[0], -1)
        # final_AU_out = self.AU_final_linear(AU_out_)
        # final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)


        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_inter = self.Exp_dropout(Exp_inter)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))

        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_inter = self.VA_dropout(VA_inter)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))
        VA_out = self.tanh1(VA_out)

        return VA_out,AU_out,final_AU_out,Exp_out


    def forward(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        x13 = self.AU_linear_p13(emb)
        x13_inter = x13
        x13 = self.AU_linear_last13(x13).unsqueeze(1)
        x14 = self.AU_linear_p14(emb)
        x14_inter = x14
        x14 = self.AU_linear_last14(x12).unsqueeze(1)
        x15 = self.AU_linear_p15(emb)
        x15_inter = x15
        x15 = self.AU_linear_last15(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,x13,x14,x15), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter, x13_inter, x14_inter, x15_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out

    def forward_resnet(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward(x,return_128d=True)
        emb = self.resnet_linear(emb)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        final_AU_out = AU_out
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        # AU_out_ = AU_out.view(AU_out.shape[0], -1)
        # final_AU_out = self.AU_final_linear(AU_out_)
        # final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out

    def forward_efficientnet(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        final_AU_out = AU_out
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        # AU_out_ = AU_out.view(AU_out.shape[0], -1)
        # final_AU_out = self.AU_final_linear(AU_out_)
        # final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out


    def forward_mutual(self,x,output_VA=True,output_AU=True,output_Exp=True):
        emb = self.exp_emb_net.forward_no_norm2(x)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)

        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out_inter = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out_inter),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        Exp_VA_inter = torch.cat((Exp_out,VA_out_inter),dim=1)
        Exp_VA_inter = torch.relu(self.AU_linear_mutual(self.EXP_VA_BN(Exp_VA_inter)))
        Exp_VA_AU_inter = torch.cat((Exp_VA_inter,AU_out_),dim=1)
        final_AU_out = self.AU_final_linear_mutual(Exp_VA_AU_inter)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_inter_mutual = torch.cat((Exp_inter,VA_out_inter),dim=1)
        Exp_out = self.Exp_linear_mutual(self.Exp_BN_mutual(Exp_inter_mutual))

        return VA_out,AU_out,final_AU_out,Exp_out




class Multi_task_series_model3(nn.Module):
    def __init__(self,emb_model,emb_dim=512,hidden_size=64,pretrained="checkpoints/minus_pipeline_best.pth",use_mfcc=False,use_wordemb=False,use_exp_emb =False):
        super(Multi_task_series_model3,self).__init__()
        # expression embedding network
        self.exp_emb_net = emb_model

      
        self.use_mfcc = use_mfcc
        self.use_wordemb = use_wordemb
        self.use_exp_emb = use_exp_emb

        self.vis_linear = nn.Sequential(
            nn.Linear(512,32),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.emb_dim = 64

        self.seq_dim = 0
        self.sta_dim = 0
        if self.use_exp_emb:
            self.emb_gru1 = nn.GRU(16,10,1,batch_first=True,bidirectional=True)
            self.emb_gru2 = nn.GRU(20,16,1,batch_first=True,bidirectional=True)

            self.emb_gru3 = nn.GRU(16,10,1,batch_first=True,bidirectional=True)
            self.emb_gru4 = nn.GRU(20,16,1,batch_first=True,bidirectional=True)

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

            self.gru3 = nn.GRU(40,16,1,batch_first=True,bidirectional=True)
            self.gru4 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)

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

            self.word_gru3 = nn.GRU(50,16,1,batch_first=True,bidirectional=True)
            self.word_gru4 = nn.GRU(32,16,1,batch_first=True,bidirectional=True)

            self.word_linear = nn.Sequential(
                nn.Linear(32,16),
                nn.Tanh(),
                nn.BatchNorm1d(16)
            )
            self.sta_dim+=50
            self.seq_dim+=16

        
        print(self.sta_dim,self.seq_dim)
        self.seq_linear = nn.Linear(self.seq_dim,16)
        self.sta_linear = nn.Linear(self.sta_dim,16)
        self.AU_Classifier1 = nn.Linear(32,12)
        self.VA_Classifier1 = nn.Linear(32,2)
        self.EXP_Classifier1 = nn.Linear(32,8)

            

        #VA branch
        self.VA_linear1 = nn.Linear(self.emb_dim,hidden_size)
        self.VA_dropout = nn.Dropout(p=0.1)
        self.VA_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.VA_linear2 = nn.Linear(hidden_size*2,2)
        self.tanh1 = nn.Tanh()

        self.VA_BN2 = nn.BatchNorm1d(hidden_size*2)

        #Exp branch
        self.Exp_linear1 = nn.Linear(self.emb_dim, hidden_size)
        self.Exp_dropout = nn.Dropout(p=0.1)
        self.Exp_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.Exp_linear2 = nn.Linear(hidden_size*2, 8)


        self.Exp_BN2 = nn.BatchNorm1d(hidden_size*2)
        self.Exp_inter = nn.Linear(128,64)

        #AU branch
        self.AU_BN1 = nn.BatchNorm1d(self.emb_dim)
        self.AU_linear_p1 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p2 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p3 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p4 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p5 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p6 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p7 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p8 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p9 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p10 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p11 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_p12 = nn.Linear(self.emb_dim, 16)
        self.AU_linear_last1 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last2 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last3 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last4 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last5 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last6 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last7 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last8 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last9 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last10 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last11 = nn.Linear(16, 2, bias=False)
        self.AU_linear_last12 = nn.Linear(16, 2, bias=False)
        self.AU_final_linear = nn.Linear(24, 24)
        self.AU_inter = nn.Linear(192,64)

        #MUTUAL
        self.EXP_VA_BN = nn.BatchNorm1d(hidden_size*2)
        self.AU_linear_mutual = nn.Linear(hidden_size*2, 24)
        self.AU_final_linear_mutual = nn.Linear(48, 24)

        self.Exp_BN_mutual = nn.BatchNorm1d(hidden_size*3)
        self.Exp_linear_mutual = nn.Linear(hidden_size*3, 7)

        # self.resnet_linear = nn.Linear(128,512)

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
        
    def forward_baseline(self,x,embs=None,mfcc=None,words=None,is_drop=False):
        emb = self.exp_emb_net.forward_fea(x)
        vis_fea = self.vis_linear(emb)
        seq_feas = []
        sta_feas = []
        if self.use_exp_emb:
            exp_emb_out,_ = self.emb_gru1(embs)
            exp_emb_out,_ = self.emb_gru2(exp_emb_out)
            exp_emb_out = torch.mean(exp_emb_out,dim=1)

            exp_emb_out2,_ = self.emb_gru3(embs[:,30:-30,:])
            exp_emb_out2,_ = self.emb_gru4(exp_emb_out2)
            exp_emb_out2 = torch.mean(exp_emb_out2,dim=1)

            exp_emb_out = exp_emb_out+ exp_emb_out2

            exp_emb_out = torch.tanh(self.emb_linear(exp_emb_out))
            seq_feas.append(exp_emb_out)
            l1 = (embs.shape[1]-1)//2
            sta_feas.append(embs[:,l1,:])

        if self.use_mfcc:
            mfcc_out,_ = self.gru1(mfcc)
            mfcc_out,_ = self.gru2(mfcc_out)
            mfcc_out = torch.mean(mfcc_out,dim=1)

            mfcc_out2,_ = self.gru3(mfcc[:,30:-30,:])
            mfcc_out2,_ = self.gru4(mfcc_out2)
            mfcc_out2 = torch.mean(mfcc_out2,dim=1)
            mfcc_out = mfcc_out + mfcc_out2

            mfcc_out = torch.tanh(self.mfcc_linear(mfcc_out))
            seq_feas.append(mfcc_out)
            l2 = (mfcc.shape[1]-1)//2
            sta_feas.append(mfcc[:,l2,:])
        
        if self.use_wordemb:
            word_emb = self.get_word_emb(words)
            word_out,_ = self.word_gru1(word_emb)
            word_out,_ = self.word_gru2(word_out)
            word_out = torch.mean(word_out,dim=1)

            word_out2,_ = self.word_gru3(word_emb[:,30:-30,:])
            word_out2,_ = self.word_gru4(word_out2)
            word_out2 = torch.mean(word_out2,dim=1)
            word_out = word_out + word_out2


            word_out = torch.tanh(self.word_linear(word_out))
            seq_feas.append(word_out)
            l3 = (word_emb.shape[1]-1)//2
            sta_feas.append(word_emb[:,l3,:])

        seq_fea = torch.cat(seq_feas,dim=1)
        seq_fea = self.seq_linear(seq_fea)
        sta_fea = torch.cat(sta_feas,dim=1)
        sta_fea = self.sta_linear(sta_fea)
        fea1 = torch.cat((seq_fea,sta_fea),dim=1)
        AU_inter_prob = self.AU_Classifier1(fea1)
        EXP_inter_prob = self.EXP_Classifier1(fea1)
        VA_inter_prob = torch.tanh(self.VA_Classifier1(fea1))

        emb = torch.cat((fea1,vis_fea),dim=1)
        VA_out,AU_out,final_AU_out,Exp_out = None,None,None,None
        emb = self.AU_BN1(emb)
        x1 = self.AU_linear_p1(emb)
        x1_inter = x1
        x1 = self.AU_linear_last1(x1).unsqueeze(1)
        x2 = self.AU_linear_p2(emb)
        x2_inter = x2
        x2 = self.AU_linear_last2(x2).unsqueeze(1)
        x3 = self.AU_linear_p3(emb)
        x3_inter = x3
        x3 = self.AU_linear_last3(x3).unsqueeze(1)
        x4 = self.AU_linear_p4(emb)
        x4_inter = x4
        x4 = self.AU_linear_last4(x4).unsqueeze(1)
        x5 = self.AU_linear_p5(emb)
        x5_inter = x5
        x5 = self.AU_linear_last5(x5).unsqueeze(1)
        x6 = self.AU_linear_p6(emb)
        x6_inter = x6
        x6 = self.AU_linear_last6(x6).unsqueeze(1)
        x7 = self.AU_linear_p7(emb)
        x7_inter = x7
        x7 = self.AU_linear_last7(x7).unsqueeze(1)
        x8 = self.AU_linear_p8(emb)
        x8_inter = x8
        x8 = self.AU_linear_last8(x8).unsqueeze(1)
        x9 = self.AU_linear_p9(emb)
        x9_inter = x9
        x9 = self.AU_linear_last9(x9).unsqueeze(1)
        x10 = self.AU_linear_p10(emb)
        x10_inter = x10
        x10 = self.AU_linear_last10(x10).unsqueeze(1)
        x11 = self.AU_linear_p11(emb)
        x11_inter = x11
        x11 = self.AU_linear_last11(x11).unsqueeze(1)
        x12 = self.AU_linear_p12(emb)
        x12_inter = x12
        x12 = self.AU_linear_last12(x12).unsqueeze(1)
        AU_out = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12), dim=1)
        AU_inter_out = torch.cat((x1_inter, x2_inter, x3_inter, x4_inter, x5_inter, x6_inter, x7_inter, x8_inter, x9_inter, x10_inter, x11_inter, x12_inter),dim=1)
        AU_inter_out = self.AU_inter(AU_inter_out)
        AU_out_ = AU_out.view(AU_out.shape[0], -1)
        final_AU_out = self.AU_final_linear(AU_out_)
        final_AU_out = final_AU_out.view(AU_out.shape[0], -1, 2)

        Exp_out = torch.relu(self.Exp_linear1(self.Exp_BN1(emb)))
        Exp_inter = torch.cat((AU_inter_out,Exp_out),dim=1)
        Exp_out = self.Exp_linear2(self.Exp_BN2(Exp_inter))
        Exp_inter_out = self.Exp_inter(Exp_inter)

        VA_out = torch.relu(self.VA_linear1(self.VA_BN1(emb)))
        VA_inter = torch.cat((Exp_inter_out,VA_out),dim=1)
        VA_out = self.VA_linear2(self.VA_BN2(VA_inter))

        return VA_out,AU_out,final_AU_out,Exp_out,VA_inter_prob,AU_inter_prob,EXP_inter_prob