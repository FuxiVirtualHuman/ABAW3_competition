import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.facenet2 import InceptionResnetV1
import math
import torch
from torch import optim
from torch.nn.modules.distance import PairwiseDistance


class Pipeline_Incep(nn.Module):
    def __init__(self):
        super(Pipeline_Incep, self).__init__()
        self.type = type
        self.student = InceptionResnetV1(pretrained="vggface2")
        self.linear1 = nn.Linear(512, 16, bias=False)
        print("===============load pretrained parameters===============")
        state_dict = torch.load("minus_pipeline_student_InceptionResnet_0.861.pth")
        self.load_state_dict(state_dict)
        print("===============load success===============")

        self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0002, momentum=0.9,
                               weight_decay=5e-4)
        self.crit1 = nn.MSELoss()
        self.crit2 = nn.TripletMarginLoss(margin=0.1)
        self.crit3 = nn.TripletMarginLoss(margin=0.2)
        self.crit4 = nn.TripletMarginLoss(margin=0.05)
        self.l2_dist = PairwiseDistance(2)
        self.l1_loss = nn.L1Loss()


    def forward(self, x):
        stu_exp_fea = self.student(x)
        stu_exp_emb = self.linear1(stu_exp_fea)
        stu_exp_emb = F.normalize(stu_exp_emb, dim=1)
        return stu_exp_emb

    def forward_fea(self, x):
        stu_exp_fea = self.student(x)
        
        return stu_exp_fea 
    
    def forward_no_norm2(self,x):
        stu_exp_fea = self.student(x)
        stu_exp_emb = self.linear1(stu_exp_fea)
        return stu_exp_emb



# print("resnet50 have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))
# with torch.no_grad():
#     flops, params = profile(net, inputs=(torch.randn(1, 3, 224,224),))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')



