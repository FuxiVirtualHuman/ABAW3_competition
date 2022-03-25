import os

import torch
# from models.pipeline5 import Pipeline
# from models.resnet_pipeline import Pipeline
from models.pipeline_student_InceptionResnet import Pipeline_Incep
# from models.minus_pipeline_512d_multitask import Multi_task_model
from models.multi_model_series import Multi_task_series_model3,Multi_task_series_model_trans
from data_new.ABAW2_data import compute_class_weight, ABAW2_Exp_data, ABAW2_VA_data, ABAW2_AU_data, \
    ABAW2_multitask_data2, ABAW3_multitask_data,ABAW2_test_data
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import numpy as np
import torch.nn.functional as F
from eval_metrics import metric_for_AU, metric_for_Exp, metric_for_VA, metric_for_AU_mlce
import torchvision.transforms.transforms as transforms
from torch.utils import data
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
#from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.autograd import Variable
import random
import pandas as pd
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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


def CCC_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 1e-10)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2) + 1e-10)
    return 1 - ccc


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            for i in range(class_num):
                self.alpha[i, :] = 0.25
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def sCE_and_focal_loss(y_pred, y_true):
    loss1 = LabelSmoothingCrossEntropy()(y_pred, y_true)
    loss2 = FocalLoss(class_num=7)(y_pred, y_true)
    return loss1 + loss2


def multilabel_categorical_crossentropy(y_pred, y_true):

    y_pred = y_pred[:, :, 1]
    y_true = y_true[:, :, 1]
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)

    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return torch.mean(neg_loss + pos_loss)


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon * 2) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1).long()
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)
    size.append(N)
    return ones.view(*size)


def train(epoch, loader, net, optimizer,best_score,best_au_score,best_exp_score,best_va_score):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
    tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (img,label_V,label_A,label_AU,label_exp,feas,name) in t:
        coff,embs,mfcc,words = feas 
        if use_cuda:
            img, mfcc,embs,coff = img.cuda(),mfcc.cuda(),embs.cuda(),coff.cuda()
            label_AU, label_V, label_A, label_exp = label_AU.long().cuda(), label_V.float().cuda(), label_A.float().cuda(), label_exp.cuda()
       
        optimizer.zero_grad()
        
        if model_name == 'baseline':
            VA_out, AU_out, final_AU_out, Exp_out = net.forward_baseline(img, embs=embs,mfcc=mfcc,words=words,coff=coff)
            AU_loss =0
            # print(AU_inter.shape,label_AU.shape)
            if AU_LOSS == 'ML_CE':
                one_hot_label_AU = get_one_hot(label_AU.long(), 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            elif AU_LOSS == 'ML_CE_and_sCE':
                for i in range(12):
                    t_target = label_AU[:, i]
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * LabelSmoothingCrossEntropy()(t_input, t_target)
                    AU_loss += t_loss
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            else:
                for i in range(12):
                    t_target = label_AU[:, i]
                    t_input2 = AU_out[:, i, :]
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target)+crit_AU(t_input2, t_target))
                    AU_loss = t_loss + AU_loss
            VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A) 
            Exp_loss = crit_Exp(Exp_out, label_exp) 
           
       
        
        
        loss = VA_loss + Exp_loss + AU_loss
        
        t.set_postfix(train_loss=loss.item(), VA_loss=VA_loss.item(), Exp_loss=Exp_loss.item(), AU_loss=AU_loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step(step)
            

        loss_sum += loss.item()
        step += 1



        VA_out = VA_out.detach().cpu().numpy()
        tmp_V_prob.extend(VA_out[:, 0])
        tmp_V_label.extend(label_V.cpu().numpy())

        tmp_A_prob.extend(VA_out[:, 1])
        tmp_A_label.extend(label_A.cpu().numpy())

        Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
        for i in range(Exp_out.shape[0]):
            v = max(Exp_prediction[i])
            index = np.where(Exp_prediction[i] == v)[0][0]
            tmp_exp_prob.append(index)
            tmp_exp_label.append(label_exp[i].cpu().numpy())

        
        if 'ML_CE' not in AU_LOSS:
            prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
        else:
            prob = final_AU_out[:, :, 1]
        tmp_AU_prob.extend(prob.data.cpu().numpy())
        tmp_AU_label.extend(label_AU.data.cpu().numpy())

        
        if step % 200 == 0:
            avg_loss = loss_sum / 200

            # VA metric
            ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a) / 2
            # Exp metric
            Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_exp_label, tmp_exp_prob,8)
            final_Exp_score = 1 * Exp_F1 + 0* Exp_acc
            # AU metric
            if 'ML_CE' not in AU_LOSS:
                AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU(tmp_AU_label, tmp_AU_prob)
            else:
                AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU_mlce(tmp_AU_label, tmp_AU_prob)
            final_AU_score = 1 * AU_F1 + 0* AU_acc
            final_cate_score = 1 * AU_F1 +0 * cate_acc
            final_score = final_AU_score + final_Exp_score + final_VA_score
            print(final_score)

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print(
                '  train set - VA, Exp ,AU_strict,AU_cate score     = {:.8f} , {:.8f} , {:.8f}, {:.8f}'.format(final_VA_score, final_Exp_score,
                                                                                        final_AU_score, final_cate_score))
            with open(f"{log_save_path}/train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {0}, step: {1},  Loss: {2}, ccc_v: {3} ccc_a: {4} Exp_F1: {5} Exp_acc: {6} AU_F1: {7} AU_acc: {8} VA_score: {9} Exp_score: {10} AU_socre: {11} AU_cate_acc:{12} AU_cate_score:{13}\n".format(
                        epoch, step,
                        avg_loss,
                        ccc_v, ccc_a, Exp_F1, Exp_acc, AU_F1, AU_acc, final_VA_score, final_Exp_score, final_AU_score,cate_acc,final_cate_score))
            tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
            tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
            loss_sum = 0.0

        if step % 6000 == 0:
            net = net.eval()
            best_score,best_au_score,best_exp_score,best_va_score = test_multi(epoch, multi_testloader, net, best_score,best_au_score,best_exp_score,best_va_score)
            net = net.train()
    return best_score,best_au_score,best_exp_score,best_va_score



def test_multi(epoch, loader, net, best_score,best_au_score,best_exp_score,best_va_score):
    print("test {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
    tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
    loss_sum = 0.0
    step = 1
    net = net.eval()
    t = tqdm(enumerate(loader))
    with torch.no_grad():
        for batch_idx, (img,label_V,label_A,label_AU,label_exp,feas,name) in t:
            coff,embs,mfcc,words = feas
            if use_cuda:
                img,mfcc,embs,coff = img.cuda(),mfcc.cuda(),embs.cuda(),coff.cuda()
                label_AU, label_V, label_A, label_exp = label_AU.long().cuda(), label_V.float().cuda(), label_A.float().cuda(), label_exp.cuda()
        
            VA_out, AU_out, final_AU_out, Exp_out = net.forward_baseline(img, embs=embs,mfcc=mfcc,words=words,coff=coff)
          
            VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)
            Exp_loss = crit_Exp(Exp_out, label_exp)
            AU_loss = 0
            if AU_LOSS == 'ML_CE':
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            elif AU_LOSS == 'ML_CE_and_sCE':
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * LabelSmoothingCrossEntropy()(t_input, t_target)
                    AU_loss += t_loss
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            else:
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_input2 = AU_out[:,i,:]
                    
                    t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target)+crit_AU(t_input2, t_target))
                    AU_loss += t_loss
            loss_sum =  loss_sum + AU_loss +Exp_loss+VA_loss

            VA_out = VA_out.detach().cpu().numpy()
            tmp_V_prob.extend(VA_out[:, 0])
            tmp_V_label.extend(label_V.cpu().numpy())

            tmp_A_prob.extend(VA_out[:, 1])
            tmp_A_label.extend(label_A.cpu().numpy())

            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            for i in range(Exp_out.shape[0]):
                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                tmp_exp_prob.append(index)
                tmp_exp_label.append(label_exp[i].cpu().numpy())

        
            if 'ML_CE' not in AU_LOSS:
                prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
            else:
                prob = final_AU_out[:, :, 1]
            tmp_AU_prob.extend(prob.data.cpu().numpy())
            tmp_AU_label.extend(label_AU.data.cpu().numpy())

            loss = VA_loss + Exp_loss + AU_loss
            t.set_postfix(train_loss=loss.item(), VA_loss=VA_loss.item(), Exp_loss=Exp_loss.item(), AU_loss=AU_loss.item())

        ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
        final_VA_score = (ccc_v + ccc_a) / 2
        Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_exp_label, tmp_exp_prob,8)
        final_Exp_score = 1 * Exp_F1 + 0 * Exp_acc
        if 'ML_CE' not in AU_LOSS:
            AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU(tmp_AU_label, tmp_AU_prob)
        else:
            AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU_mlce(tmp_AU_label, tmp_AU_prob)
        final_AU_score = 1 * AU_F1 + 0* AU_acc
        final_score = final_AU_score + final_Exp_score + final_VA_score
    
    if final_score >  best_score:
        best_score =  final_score
        torch.save(net.state_dict(), f'{ck_save_path}/multi_best.pth')
        #pred_multi(epoch, net, pred_save_path,best_score,step)
    if final_AU_score > best_au_score:
        best_au_score =  final_AU_score
        torch.save(net.state_dict(), f'{ck_save_path}/AU_best.pth')
    if final_VA_score > best_va_score:
        best_va_score = final_VA_score
        torch.save(net.state_dict(), f'{ck_save_path}/VA_best.pth')
    if final_Exp_score > best_exp_score:
        best_exp_score = final_Exp_score
        torch.save(net.state_dict(), f'{ck_save_path}/Exp_best.pth')

    with open(f"{log_save_path}/multi_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, AU_F1: {2} AU_acc: {3}  AU_score: {4} EXP_F1: {5} , EXP_acc: {6} CCC_V: {7} CCC_A:{8} final score: {9} best final score: {10} best VA score: {11} best AU score: {12} best exp score: {13}\n".format(
                    epoch, loss_sum / len(loader),
                    AU_F1, AU_acc, final_AU_score,Exp_F1, Exp_acc , ccc_v, ccc_a, final_score, best_score,best_va_score,best_au_score,best_exp_score))
            AU_F1_per_class = [str(k) for k in AU_F1_per_class]
            log_file.write(" ".join(AU_F1_per_class))
            log_file.write("\n")

            EXP_F1_per_class = [str(k) for k in Exp_F1_per_class]
            log_file.write(" ".join(EXP_F1_per_class))
            log_file.write("\n")
    return best_score,best_au_score,best_exp_score,best_va_score

def pred_multi(epoch, net, save_path, best_score, step=0):
    t_set =  ABAW2_test_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_mtl_test.csv",img_path, emb_dict,mfcc_dict,word_dict,60,60,60,transform1)
    loader = torch.utils.data.DataLoader(t_set, batch_size=bz * 3, num_workers=8)
    print("Pred {} epoch".format(epoch))
    tmp_AU_prob, tmp_AU_label = [], []
    net = net.eval()
    save_data = pd.DataFrame()
    imgs = []
    AU_prob = []
    AU_labels = []
    exp_prob = []
    exp_labels = []
    V_pred = []
    A_pred = []
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, feas,name) in t:
            embs, mfcc,words = feas
            if use_cuda:
                img = img.cuda()
                mfcc,embs= mfcc.cuda(),embs.cuda()
            VA_out, AU_out, final_AU_out, Exp_out  = net.forward_baseline(img, embs=embs,mfcc=mfcc,words=words)
            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            VA_out = VA_out.detach().cpu().numpy()
            if 'ML_CE' not in AU_LOSS:
                prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
            else:
                prob = final_AU_out[:, :, 1]
            for i in range(len(name)):
                imgs.append(name[i])
                p = prob[i].data.cpu().numpy()
                str_p = [str(k) for k in p]
                str_p = " ".join(str_p)
                AU_prob.append(str_p)
                str_label = ""
                if 'ML_CE' not in AU_LOSS:
                    for m in range(12):
                        if p[m]>=0.5:
                            str_label+="1 "
                        else:
                            str_label+="0 "
                else:
                    for m in range(12):
                        if p[m]>=0:
                            str_label+="1 "
                        else:
                            str_label+="0 "
                AU_labels.append(str_label)

                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                exp_labels.append(index)
                str_prob = [str(k) for k in Exp_prediction[i]]
                str_prob = " ".join(str_prob)
                exp_prob.append(str_prob)
                V_pred.append(VA_out[i,0])
                A_pred.append(VA_out[i,1])

    data = pd.DataFrame()
    data["img"] = imgs
    data["AU_prob"] = AU_prob
    data["AU_label"] = AU_labels
    data["exp_prob"] = exp_prob
    data["exp_label"] = exp_labels
    data["V"] = V_pred
    data["A"] = A_pred
    data.to_csv(os.path.join(save_path,"Multi_pred_epoch"+str(epoch)+"_step"+str(step)+"_best_score"+str(best_score)[:5]+".csv"))    




def train_Exp(epoch, loader, net, optimizer, best_Exp_score):
    print("train {} epoch".format(epoch))
    tmp_exp_prob = []
    tmp_exp_label = []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (exp_img, label_exp, name) in t:
        if use_cuda:
            exp_img = exp_img.cuda()
            label_exp = label_exp.cuda()

        optimizer.zero_grad()

        _, _, _, Exp_out = net(exp_img, output_VA=False, output_AU=False, output_Exp=True)
        # Exp_loss
        Exp_loss = crit_Exp(Exp_out, label_exp)
        Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
        for i in range(Exp_out.shape[0]):
            v = max(Exp_prediction[i])
            index = np.where(Exp_prediction[i] == v)[0][0]
            tmp_exp_prob.append(index)
            tmp_exp_label.append(label_exp[i].cpu().numpy())

        loss = Exp_loss
        t.set_postfix(loss=loss.item())
        # print(loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 100

            # Exp metric
            Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_exp_label, tmp_exp_prob,8)
            final_Exp_score = 0.67 * Exp_F1 + 0.33 * Exp_acc

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - Exp     = {:.8f}'.format(final_Exp_score))
            with open(f"./log/ABAW2/Exp_{task_type}_task_train_2nd.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {}, Exp_F1: {} Exp_acc: {}  Exp_score: {} \n".format(epoch, step,
                                                                                                      avg_loss, Exp_F1,
                                                                                                      Exp_acc,
                                                                                                      final_Exp_score))
            tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
            tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
            loss_sum = 0.0

        if step % 2500 == 0:
            net = net.eval()
            best_Exp_score = test_Exp(epoch, Exp_testloader, net, best_Exp_score, step)
            net = net.train()
    return best_Exp_score


def train_VA(epoch, loader, net, optimizer, best_VA_score):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (va_img, label_V, label_A, name) in t:

        if use_cuda:
            va_img = va_img.cuda()
            label_V, label_A = label_V.float().cuda(), label_A.float().cuda()

        optimizer.zero_grad()
        VA_out, _, _, _ = net(va_img, output_VA=True, output_AU=False, output_Exp=False)

        # VA_loss
        VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)
        VA_out = VA_out.detach().cpu().numpy()
        tmp_V_prob.extend(VA_out[:, 0])
        tmp_V_label.extend(label_V.cpu().numpy())

        tmp_A_prob.extend(VA_out[:, 1])
        tmp_A_label.extend(label_A.cpu().numpy())

        loss = VA_loss
        t.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 100

            # VA metric
            ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
            final_VA_score = (ccc_v + ccc_a) / 2

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - VA     = {:.8f}'.format(final_VA_score))
            with open(f"./log/ABAW2/VA_{task_type}_task_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {}, ccc_v: {} ccc_a: {} VA_score: {} \n".format(epoch, step,
                                                                                                 avg_loss, ccc_v, ccc_a,
                                                                                                 final_VA_score))
            tmp_V_prob, tmp_A_prob = [], []
            tmp_V_label, tmp_A_label = [], []
            loss_sum = 0.0

        if step % 2500 == 0:
            net = net.eval()
            best_VA_score = test_VA(epoch, VA_testloader, net, best_VA_score, step)
            net = net.train()
    return best_VA_score


def train_AU(epoch, loader, net, optimizer, best_AU_score):
    print("train {} epoch".format(epoch))
    tmp_AU_prob = []
    tmp_AU_label = []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (au_img, label_AU, name) in t:
        if use_cuda:
            au_img = au_img.cuda()
            label_AU = label_AU.cuda()

        optimizer.zero_grad()

        # AU_loss
        AU_loss = 0
        _, AU_out, final_AU_out, _ = net(au_img, output_VA=False, output_AU=True, output_Exp=False)
        for i in range(12):
            t_input = AU_out[:, i, :]
            t_target = label_AU[:, i].long()
            t_input2 = final_AU_out[:, i, :]
            t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target) + crit_AU(t_input2, t_target))
            AU_loss += t_loss
        prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
        tmp_AU_prob.extend(prob.data.cpu().numpy())
        tmp_AU_label.extend(label_AU.data.cpu().numpy())

        loss = AU_loss
        t.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        step += 1

        if step % 200 == 0:
            avg_loss = loss_sum / 100

            # AU metric
            AU_F1, AU_acc, AU_F1_per_class = metric_for_AU(tmp_AU_label, tmp_AU_prob)
            final_AU_score = 0.5 * AU_F1 + 0.5 * AU_acc

            print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            print('  train set - AU score     =  {:.8f}'.format(final_AU_score))
            with open(f"./log/ABAW2/AU_{task_type}_task_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {},  AU_F1: {} AU_acc: {}  AU_socre: {}\n".format(epoch, step,
                                                                                                   avg_loss,
                                                                                                   AU_F1, AU_acc,
                                                                                                   final_AU_score))
            tmp_AU_prob = []
            tmp_AU_label = []
            loss_sum = 0.0

        if step % 2500 == 0:
            net = net.eval()
            best_AU_score = test_AU(epoch, AU_testloader, net, best_AU_score, step)
            net = net.train()
    return best_AU_score


def test_VA(epoch, loader, net, best_acc, step=0):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []
    net = net.eval()
    VA_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_V, label_A, name) in t:

            if use_cuda:
                img = img.cuda()
                label_V, label_A = label_V.float().cuda(), label_A.float().cuda()
            if model_name == 'baseline':
                VA_out, _, _, _ = net.forward_resnet(img, output_AU=False, output_Exp=False)
            elif model_name == 'mutual':
                VA_out, _, _, _ = net.forward_mutual(img, output_AU=False, output_Exp=False)
            elif model_name == 'dropout':
                VA_out, _, _, _ = net.forward_dropout(img, output_AU=False, output_Exp=False)
            VA_loss = crit_VA(VA_out[:, 0], label_V) + crit_VA(VA_out[:, 1], label_A)
            VA_out = VA_out.detach().cpu().numpy()
            tmp_V_prob.extend(VA_out[:, 0])
            tmp_V_label.extend(label_V.cpu().numpy())

            tmp_A_prob.extend(VA_out[:, 1])
            tmp_A_label.extend(label_A.cpu().numpy())
            VA_loss_sum += VA_loss.item()
            t.set_postfix(test_VA_loss=VA_loss.item())

        ccc_v, ccc_a = metric_for_VA(tmp_V_label, tmp_A_label, tmp_V_prob, tmp_A_prob)
        final_VA_score = (ccc_v + ccc_a) / 2

        if final_VA_score > best_acc:
            best_acc = final_VA_score
            torch.save(net.state_dict(), f'{ck_save_path}/VA_best.pth')

        # torch.save(net.state_dict(), f'{ck_step_save_path}/VA_epoch{epoch}_step{step}.pth')

        with open(f"{log_save_path}/VA_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, ccc_v: {2} ccc_a: {3}  VA_score: {4} \n".format(
                    epoch, VA_loss_sum / len(loader),
                    ccc_v, ccc_a, final_VA_score))
        return best_acc


def test_Exp(epoch, loader, net, best_acc, step=0):
    print("train {} epoch".format(epoch))
    tmp_Exp_prob, tmp_Exp_label = [], []
    net = net.eval()
    Exp_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_exp, name) in t:

            if use_cuda:
                img = img.cuda()
                label_exp = label_exp.cuda()
            if model_name == 'baseline':
                _, _, _, Exp_out = net.forward_resnet(img, output_VA=False, output_AU=False)
            elif model_name == 'mutual':
                _, _, _, Exp_out = net.forward_mutual(img, output_VA=False, output_AU=False)
            elif model_name == 'dropout':
                _, _, _, Exp_out = net.forward_dropout(img, output_VA=False, output_AU=False)
            Exp_loss = crit_Exp(Exp_out, label_exp)
            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            for i in range(len(name)):
                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                tmp_Exp_prob.append(index)
                tmp_Exp_label.append(label_exp[i].cpu().numpy())
            t.set_postfix(test_Exp_loss=Exp_loss.item())
            Exp_loss_sum += Exp_loss.item()

        Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_Exp_label, tmp_Exp_prob,8)
        final_Exp_score = 0.67 * Exp_F1 + 0.33 * Exp_acc

        if final_Exp_score > best_acc:
            best_acc = final_Exp_score
            torch.save(net.state_dict(), f'{ck_save_path}/Exp_best.pth')

        # torch.save(net.state_dict(), f'{ck_step_save_path}/Exp_epoch{epoch}_step{step}.pth')

        with open(f"{log_save_path}/Exp_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {}, Loss: {}, Exp_F1: {} Exp_acc: {}  Exp_score: {} \n".format(
                    epoch, Exp_loss_sum / len(loader),
                    Exp_F1, Exp_acc, final_Exp_score))
            Exp_F1_per_class = [str(k) for k in Exp_F1_per_class]
            log_file.write(" ".join(Exp_F1_per_class))
            log_file.write("\n")
        return best_acc


def test_AU(epoch, loader, net, best_acc,best_acc_2, step=0):
    print("train {} epoch".format(epoch))
    tmp_AU_prob, tmp_AU_label = [], []
    net = net.eval()
    AU_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_AU, name) in t:

            if use_cuda:
                img = img.cuda()
                label_AU = label_AU.cuda()
            if model_name == 'baseline':
                _, AU_out, final_AU_out, _ = net.forward_resnet(img, output_VA=False, output_Exp=False)
            elif model_name == 'mutual':
                _, AU_out, final_AU_out, _ = net.forward_mutual(img, output_VA=False, output_Exp=False)
            elif model_name == 'dropout':
                _, AU_out, final_AU_out, _ = net.forward_dropout(img, output_VA=False, output_Exp=False)
            AU_loss = 0

            if AU_LOSS == 'ML_CE':
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            elif AU_LOSS == 'ML_CE_and_sCE':
                for i in range(12):
                    t_target = label_AU[:, i].long()
                    t_input = final_AU_out[:, i, :]
                    t_loss = AU_class_weight[i] * LabelSmoothingCrossEntropy()(t_input, t_target)
                    AU_loss += t_loss
                one_hot_label_AU = get_one_hot(label_AU, 2)
                AU_loss += crit_AU(final_AU_out, one_hot_label_AU)
            else:
                for i in range(12):
                    t_target = label_AU[:, i].long()

                    t_input = final_AU_out[:, i, :]
                    t_input2 = AU_out[:,i,:]
                    t_loss = AU_class_weight[i] * (crit_AU(t_input, t_target) + crit_AU(t_input2, t_target))
                    AU_loss += t_loss

            if 'ML_CE' not in AU_LOSS:
                prob = torch.softmax(final_AU_out, dim=2)[:, :, 1]
            else:
                prob = final_AU_out[:, :, 1]
            tmp_AU_prob.extend(prob.data.cpu().numpy())
            tmp_AU_label.extend(label_AU.data.cpu().numpy())
            t.set_postfix(test_AU_loss=AU_loss)
            AU_loss_sum += AU_loss

        if 'ML_CE' not in AU_LOSS:
            AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU(tmp_AU_label, tmp_AU_prob)
        else:
            AU_F1, AU_acc, AU_F1_per_class,cate_acc = metric_for_AU_mlce(tmp_AU_label, tmp_AU_prob)
        final_AU_score = 0.5 * AU_F1 + 0.5 * AU_acc
        final_cate_score = 0.5 * AU_F1 + 0.5 * cate_acc


        if final_AU_score > best_acc:
            best_acc = final_AU_score
            torch.save(net.state_dict(), f'{ck_save_path}/AU_best.pth')

        if final_cate_score > best_acc_2:
            best_acc_2 = final_cate_score
            torch.save(net.state_dict(), f'{ck_save_path}/AU_best_cate_score.pth')

        # torch.save(net.state_dict(), f'{ck_step_save_path}/AU_epoch{epoch}_step{step}.pth')

        with open(f"{log_save_path}/AU_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, AU_F1: {2} AU_acc: {3}  AU_score: {4} AU_cate_acc:{5} AU_cate_score:{6}\n".format(
                    epoch, AU_loss_sum / len(loader),
                    AU_F1, AU_acc, final_AU_score, cate_acc, final_cate_score))
            AU_F1_per_class = [str(k) for k in AU_F1_per_class]
            log_file.write(" ".join(AU_F1_per_class))
            log_file.write("\n")
        return best_acc,best_acc_2

def load_mfcc(mfcc_path):
    mfcc_dict = {}
    lists = os.listdir(mfcc_path)
    for l in tqdm(lists):
        mfcc= np.loadtxt(os.path.join(mfcc_path,l),delimiter=',')
        # mfcc= torch.from_numpy(mfcc).float()  
        mfcc_dict[l] = mfcc
    return mfcc_dict

def load_emb(emb_path):
    emb_dict = {}
    lists = os.listdir(emb_path)
    for l in tqdm(lists):
        emb= np.loadtxt(os.path.join(emb_path,l),delimiter=',')
        # mfcc= torch.from_numpy(mfcc).float()  
        emb_dict[l] = emb
    return emb_dict

def load_word(word_path):
    word_dict = {}
    lists = os.listdir(word_path)
    uni = ['','abramoff', 'acidifying', 'adenoid', 'adenoids', 'adjacency', 'adroop', 'affordances', 'agitators', 'algic',
           'alithea', 'amortize', 'amplitudes', 'anaesthetized', 'archly', 'assortative', 'astronautics', 'audiotape',
           'babbled', 'bactericide', 'bipedalism', 'blaspheming', 'blithely', 'bodach', 'boggley', 'bombsight',
           'borlaug', 'brenta', 'britisher', 'brittler', 'brittles', 'bukovina', 'cantilevered', 'catheterization',
           'champnell', 'charioteer', 'chastened', 'chilmark', 'choppiness', 'chulanont', 'cingulate', 'clattered',
           'claymores', 'cleavers', 'coercing', 'colicky', 'communistic', 'concini', 'conjectures', 'conjunctive',
           'couched', 'creatinine', 'crispness', 'cuttingly', 'deadening', 'deared', 'debroff', 'deciliter',
           'delegitimize', 'deleterious', 'demonstrable', 'deviates', 'devilfish', 'diastolic', 'dickstein',
           'diffident', 'dishonors', 'dispraised', 'doglike', 'dominantly', 'dosia', 'dotcoms', 'easeful', 'ecliptic',
           'ecstatically', 'elegies', 'elfrey', 'emulex', 'enigmatical', 'espaliers', 'espiau', 'eurodollar',
           'exhorting', 'expediently', 'eyecatching', 'eyres', 'fainter', 'fatting', 'fecund', 'fecundity', 'filaments',
           'fineman', 'folkloric', 'fortysomething', 'fragmentary', 'freeboard', 'frizzes', 'fugger', 'fullfledged',
           'gaffany', 'gauzy', 'gebara', 'georgis', 'ghettoes', 'girded', 'girdled', 'globalised', 'gnomon', 'gouraud',
           'grasmick', 'gratulate', 'gravest', 'grayer', 'greying', 'gulfs', 'gyrus', 'haab', 'haltingly', 'handbills',
           'hardener', 'headman', 'heliet', 'hellyer', 'heterosis', 'hominid', 'hued', 'hugin', 'huijin', 'humpbacked',
           'hundredfold', 'hypothesize', 'hypothesizing', 'impassive', 'impulsivity', 'inaugurating', 'interrelated',
           'intubated', 'ivanovitch', 'japanee', 'jasko', 'jidda', 'jinkey', 'kallheim', 'kapolna', 'kilkane', 'kostya',
           'kowtowing', 'kumano', 'kuvera', 'labarre', 'laertes', 'lafite', 'lampson', 'legatum', 'levinas', 'lhamo',
           'lichtman', 'ligands', 'luding', 'lyglenson', 'macora', 'magadha', 'magnesite', 'majesties', 'maltwood',
           'managa', 'massaccio', 'mckelvey', 'medians', 'mediocrities', 'mesenteric', 'mineralization', 'moister',
           'mondamin', 'montagnais', 'mortgagee', 'mousers', 'muchmore', 'mumga', 'mustees', 'myelin', 'nagran',
           'nanchang', 'niihau', 'noncombatant', 'numberplate', 'oberea', 'observables', 'oiliness', 'oilseed', 'okola',
           'orbitofrontal', 'outlives', 'ovate', 'overwatch', 'pagiel', 'panamanians', 'patrimony', 'pendergast',
           'penicillins', 'perceiver', 'perkier', 'pessaries', 'philby', 'phoenicia', 'physiologically', 'piltdown',
           'pinprick', 'plickaman', 'preadolescent', 'precipitancy', 'prerace', 'presuppose', 'proclivities',
           'prologues', 'propitious', 'ptomaines', 'pugilist', 'pulliam', 'quickwitted', 'quillings', 'racicot',
           'rebury', 'refloat', 'retarding', 'rigors', 'ringley', 'roamer', 'roback', 'ruefully', 'ruses', 'sambre',
           'sanlu', 'satiable', 'savoye', 'seamy', 'seepage', 'shagoth', 'shipload', 'shortish', 'shriekers', 'shrikes',
           'sisera', 'slavemaster', 'solyom', 'spasmodic', 'spectres', 'spiderlike', 'spinbronn', 'spinello', 'squibs',
           'stabilised', 'standpoints', 'stereoscope', 'sternness', 'stigmatizing', 'stimulative', 'sunstein',
           'superyachts', 'tabulated', 'tarth', 'thirtytwo', 'tokuda', 'transcriber', 'transcribes', 'trenched',
           'trewin', 'trivialized', 'truisms', 'ugrin', 'ultranationalists', 'umbilicus', 'underestimation',
           'underlies', 'urhobo', 'volkow', 'wafted', 'wakeful', 'wartorn', 'westernized', 'wharfs', 'yokefellow',
           'zammah']

    for l in tqdm(lists):
        f = open(os.path.join(word_path,l),"r")
        lines = f.readlines()
       # word_dict[l] = [k.strip() for k in lines]
        news = []
        for ls in lines:
            ll = ls.strip()
            if "'" in ll:
                t = ll.strip("'")[0].lower()
            else:
                t = ll.lower()
            if t in uni:
                t = "<hashtag>"
            news.append(t)
        word_dict[l] = news
    
    return word_dict

if __name__ == '__main__':
    setup_seed(20)
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    use_cuda = True
    task = 'MULTI'  # ['EXP','AU','VA','MULTI']
    if task in ['EXP', 'AU', 'VA']:
        task_type = 'single'
    elif task == 'MULTI':
        task_type = 'multi'

    model_name = 'baseline'  # ['baseline','mutual','dropout']
    VA_LOSS = 'MSE'  # ['MAE','MSE','SmoothL1','CCC']
    AU_LOSS = 'CE'  # ['LabelSmoothCE','CE','WCE','ML_CE']
    EXP_LOSS = 'CE'  # ['LabelSmoothCE','CE','sCE_and_focal_loss']
    # comment = 'dropout:0.1'
    comment = 'trans2_re_emb_seq60_mfcc_seq60_word_seq60_coff_seq60'
    # fold = 0  #[5,6,7,8,9]

    remark = 'va:' + VA_LOSS + '_au:' + AU_LOSS + '_exp:' + EXP_LOSS + f'+{comment}'
    print(remark)

    ck_save_path = f'/data/checkpoints/{task_type}_task/{model_name}_{remark}'
    ck_step_save_path = f'/data/checkpoints/{task_type}_task/{model_name}_{remark}/epoch_weight'
    log_save_path = f'./log2/{task_type}_task/{model_name}_{remark}'
    pred_save_path = f'/data/test/{task_type}_task/{model_name}_{remark}'
    os.makedirs(pred_save_path,exist_ok=True)
    os.makedirs(ck_step_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    print(f'************** NOW IS {task} TASK. ******************')
    train_csv = "/home/zhangwei05/ABAW3/annos/ABAW3_new_mtl_training_re.csv"
    # img_path = "/home/zhangwei05/data/aff_in_the_wild/"
    img_path = "/data/crop_face_jpg/"

    AU_class_weight = compute_class_weight(train_csv, "AU").cuda()
    Exp_class_weight = compute_class_weight(train_csv, "Exp")
    print(AU_class_weight, Exp_class_weight)
    uni = ['','abramoff', 'acidifying', 'adenoid', 'adenoids', 'adjacency', 'adroop', 'affordances', 'agitators', 'algic',
           'alithea', 'amortize', 'amplitudes', 'anaesthetized', 'archly', 'assortative', 'astronautics', 'audiotape',
           'babbled', 'bactericide', 'bipedalism', 'blaspheming', 'blithely', 'bodach', 'boggley', 'bombsight',
           'borlaug', 'brenta', 'britisher', 'brittler', 'brittles', 'bukovina', 'cantilevered', 'catheterization',
           'champnell', 'charioteer', 'chastened', 'chilmark', 'choppiness', 'chulanont', 'cingulate', 'clattered',
           'claymores', 'cleavers', 'coercing', 'colicky', 'communistic', 'concini', 'conjectures', 'conjunctive',
           'couched', 'creatinine', 'crispness', 'cuttingly', 'deadening', 'deared', 'debroff', 'deciliter',
           'delegitimize', 'deleterious', 'demonstrable', 'deviates', 'devilfish', 'diastolic', 'dickstein',
           'diffident', 'dishonors', 'dispraised', 'doglike', 'dominantly', 'dosia', 'dotcoms', 'easeful', 'ecliptic',
           'ecstatically', 'elegies', 'elfrey', 'emulex', 'enigmatical', 'espaliers', 'espiau', 'eurodollar',
           'exhorting', 'expediently', 'eyecatching', 'eyres', 'fainter', 'fatting', 'fecund', 'fecundity', 'filaments',
           'fineman', 'folkloric', 'fortysomething', 'fragmentary', 'freeboard', 'frizzes', 'fugger', 'fullfledged',
           'gaffany', 'gauzy', 'gebara', 'georgis', 'ghettoes', 'girded', 'girdled', 'globalised', 'gnomon', 'gouraud',
           'grasmick', 'gratulate', 'gravest', 'grayer', 'greying', 'gulfs', 'gyrus', 'haab', 'haltingly', 'handbills',
           'hardener', 'headman', 'heliet', 'hellyer', 'heterosis', 'hominid', 'hued', 'hugin', 'huijin', 'humpbacked',
           'hundredfold', 'hypothesize', 'hypothesizing', 'impassive', 'impulsivity', 'inaugurating', 'interrelated',
           'intubated', 'ivanovitch', 'japanee', 'jasko', 'jidda', 'jinkey', 'kallheim', 'kapolna', 'kilkane', 'kostya',
           'kowtowing', 'kumano', 'kuvera', 'labarre', 'laertes', 'lafite', 'lampson', 'legatum', 'levinas', 'lhamo',
           'lichtman', 'ligands', 'luding', 'lyglenson', 'macora', 'magadha', 'magnesite', 'majesties', 'maltwood',
           'managa', 'massaccio', 'mckelvey', 'medians', 'mediocrities', 'mesenteric', 'mineralization', 'moister',
           'mondamin', 'montagnais', 'mortgagee', 'mousers', 'muchmore', 'mumga', 'mustees', 'myelin', 'nagran',
           'nanchang', 'niihau', 'noncombatant', 'numberplate', 'oberea', 'observables', 'oiliness', 'oilseed', 'okola',
           'orbitofrontal', 'outlives', 'ovate', 'overwatch', 'pagiel', 'panamanians', 'patrimony', 'pendergast',
           'penicillins', 'perceiver', 'perkier', 'pessaries', 'philby', 'phoenicia', 'physiologically', 'piltdown',
           'pinprick', 'plickaman', 'preadolescent', 'precipitancy', 'prerace', 'presuppose', 'proclivities',
           'prologues', 'propitious', 'ptomaines', 'pugilist', 'pulliam', 'quickwitted', 'quillings', 'racicot',
           'rebury', 'refloat', 'retarding', 'rigors', 'ringley', 'roamer', 'roback', 'ruefully', 'ruses', 'sambre',
           'sanlu', 'satiable', 'savoye', 'seamy', 'seepage', 'shagoth', 'shipload', 'shortish', 'shriekers', 'shrikes',
           'sisera', 'slavemaster', 'solyom', 'spasmodic', 'spectres', 'spiderlike', 'spinbronn', 'spinello', 'squibs',
           'stabilised', 'standpoints', 'stereoscope', 'sternness', 'stigmatizing', 'stimulative', 'sunstein',
           'superyachts', 'tabulated', 'tarth', 'thirtytwo', 'tokuda', 'transcriber', 'transcribes', 'trenched',
           'trewin', 'trivialized', 'truisms', 'ugrin', 'ultranationalists', 'umbilicus', 'underestimation',
           'underlies', 'urhobo', 'volkow', 'wafted', 'wakeful', 'wartorn', 'westernized', 'wharfs', 'yokefellow',
           'zammah']
    # LOSS
    if VA_LOSS == 'MAE':
        crit_VA = L1Loss()
    elif VA_LOSS == 'MSE':
        crit_VA = MSELoss()
    elif VA_LOSS == 'SmoothL1':
        crit_VA = SmoothL1Loss()
    elif VA_LOSS == 'CCC':
        crit_VA = CCC_loss

    if AU_LOSS == 'CE':
        AU_class_weight = torch.ones_like(AU_class_weight) * 0.1
        crit_AU = CrossEntropyLoss()
    elif AU_LOSS == 'WCE':
        crit_AU = CrossEntropyLoss()
    elif AU_LOSS == 'LabelSmoothCE':
        crit_AU = LabelSmoothingCrossEntropy()
    elif AU_LOSS == 'ML_CE':
        crit_AU = multilabel_categorical_crossentropy

    if EXP_LOSS == 'CE':
        crit_Exp = CrossEntropyLoss()
        # crit_Exp = CrossEntropyLoss(Exp_class_weight.cuda())
    elif EXP_LOSS == 'LabelSmoothCE':
        # crit_Exp = CrossEntropyLoss_label_smooth(num_classes=7,epsilon=0.1)
        crit_Exp = LabelSmoothingCrossEntropy()
    elif EXP_LOSS == 'sCE_and_focal_loss':
        crit_Exp = sCE_and_focal_loss

    # data
    transform1 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])
    transform2 = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])

    bz = 100
    # model
    emb_net = Pipeline_Incep()
    # net = Multi_task_model(emb_net)
    # net = Multi_task_series_model3(emb_net,512, pretrained="checkpoints/resnet18_best.pth",use_mfcc=True,use_wordemb=True,use_exp_emb=True)
    net = Multi_task_series_model_trans(emb_net,512, pretrained="checkpoints/resnet18_best.pth",use_mfcc=True,use_wordemb=True,use_exp_emb=True,use_coff=True)
    
    if use_cuda:
        net = net.cuda()

    if task == 'EXP':
        Exp_trainset = ABAW2_Exp_data("data_new/new_Exp_training_1.csv", img_path, transform)
        trainset = Exp_trainset
        Exp_testset = ABAW2_Exp_data("data_new/new_Exp_validation_1.csv", img_path, transform1)
        Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=4)
    elif task == 'AU':
        AU_trainset = ABAW2_AU_data("data_new/new_AU_training_1.csv", img_path, transform)
        trainset = AU_trainset
        AU_testset = ABAW2_AU_data("data_new/new_AU_validation_1.csv", img_path, transform1)
        AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=4)
    elif task == 'VA':
        VA_trainset = ABAW2_VA_data("data_new/new_VA_training_1.csv", img_path, transform)
        trainset = VA_trainset
        VA_testset = ABAW2_VA_data("data_new/new_VA_validation_1.csv", img_path, transform1)
        VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=4)
    elif task == 'MULTI':
        word_path = "/data/wordlist/"
        word_dict = load_word(word_path)
        print("+++++ load word_dict ++++")
        mfcc_path = "/data/mfcc/"
        mfcc_dict = load_mfcc(mfcc_path)
        print("+++++ load mfcc_dict ++++")
        emb_path = "/data/exp_emb_chazhi/"
        emb_dict = load_emb(emb_path)
        print("+++++ load emb_dict ++++")
        coff_path = "/data/crop_face_3dmm_coff_processed_chazhi"
        coff_dict = load_emb(coff_path)

        trainset = ABAW3_multitask_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_mtl_training_re.csv",img_path,emb_dict,mfcc_dict,word_dict,coff_dict,60,60,60,60,transform=transform)
        testset = ABAW3_multitask_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_mtl_validation.csv",img_path,emb_dict,mfcc_dict,word_dict,coff_dict,60,60,60,60,transform=transform1)
        multi_testloader = data.DataLoader(testset, batch_size=bz * 3, num_workers=4)

    trainloader = data.DataLoader(trainset, batch_size=bz, num_workers=8, shuffle=True)

   

    # training parameters
    best_au_score, best_va_score, best_exp_score=  0, 0,0
    best_score=0
    lr = 0.002
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, 2, 1e-5)

    #scaler = GradScaler()
    for i in range(60):
        if task == 'EXP':
            best_Exp_score = train_Exp(i, trainloader, net, optimizer, best_Exp_score)
            best_Exp_score = test_Exp(i, Exp_testloader, net, best_Exp_score)
        elif task == 'VA':
            best_VA_score = train_VA(i, trainloader, net, optimizer, best_VA_score)
            best_VA_score = test_VA(i, VA_testloader, net, best_VA_score)
        elif task == 'AU':
            best_AU_score = train_AU(i, trainloader, net, optimizer, best_AU_score)
            best_AU_score = test_AU(i, AU_testloader, net, best_AU_score)
        elif task == 'MULTI':
            #if i==0:
            #    pred_multi(i, net, pred_save_path,best_score,len(trainloader))
            best_score,best_au_score,best_exp_score,best_va_score = train(i, trainloader, net, optimizer, best_score,best_au_score,best_exp_score,best_va_score)
            best_score,best_au_score,best_exp_score,best_va_score = test_multi(i, multi_testloader, net, best_score,best_au_score,best_exp_score,best_va_score)
            






