import os
import pickle
import sys

import torch
from models.pipeline5 import Pipeline
# from models.minus_pipeline_512d_multitask import Multi_task_model
# from models.multi_model_series import Multi_task_series_model
#from models.BBN_resnet import resnet50
# from models.BBN_resnet import resnet18
from models.single_exp_detect import Single_exp_detect_trans, Single_exp_detect_Eff
from data.ABAW2_data import compute_class_weight, ABAW2_Exp_data, ABAW2_VA_data, ABAW2_AU_data, \
    ABAW2_multitask_data2,ABAW2_test_data,ABAW2_Exp_data_test
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import numpy as np
import torch.nn.functional as F
from eval_metrics import metric_for_AU, metric_for_Exp, metric_for_VA
import torchvision.transforms.transforms as transforms
from torch.utils import data
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import f1_score
import random
import pandas as pd
from copy import deepcopy
def metric_for_AU_mlce(gt, pred, class_num=12):
    # compute_F1,acc
    F1 = []
    gt = np.array(gt)
    pred = np.array(pred)
    # cate_acc = np.sum((np.array(pred>0,dtype=np.float))==gt)/(gt.shape[0]*gt.shape[1])
    # print(pred.shape)
    for type in range(class_num):
        gt_ = gt[:, type]
        pred_ = pred[:, type]
        new_pred = ((pred_ >= 0.) * 1).flatten()
        F1.append(f1_score(gt_.flatten(), new_pred))

    F1_mean = np.mean(F1)

    # compute total acc
    counts = gt.shape[0]
    accs = 0
    for i in range(counts):
        pred_label = ((pred[i, :] >= 0.) * 1).flatten()
        gg = gt[i].flatten()
        j = 0
        for k in range(12):
            if int(gg[k]) == int(pred_label[k]):
                j += 1
        if j == 12:
            accs += 1

    acc = 1.0 * accs / counts

    return F1_mean, acc, F1


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
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
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
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
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
    '''
    label:(h,w), dtype:torch.long，可使用torch.LongTensor(mask)将shape为(h,w)的索引mask转为torch.long类型
    N:num_class，0也算一个类别
    '''
    size = list(label.size())
    label = label.view(-1).long()  # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)





def train_Exp(epoch, loader, net, optimizer, best_Exp_score):
    print("train {} epoch".format(epoch))
    tmp_exp_prob = []
    tmp_exp_label = []

    loss_sum = 0.0
    step = 1
    net = net.train()
    print('Total batch',len(loader))
    t = tqdm(enumerate(loader))
    for batch_idx, (exp_img, label_exp,feas, name) in t:
        coff,embs,mfcc,words = feas
        if use_cuda:
            exp_img = exp_img.cuda()
            label_exp ,mfcc,embs,coff = label_exp.cuda(),mfcc.cuda(),embs.cuda(),coff.cuda()

        optimizer.zero_grad()


        Exp_out = net(exp_img,embs=embs,mfcc=mfcc,words=words,coff=coff)

            # Exp_loss
        Exp_loss = crit_Exp(Exp_out, label_exp)

        Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
        for i in range(Exp_out.shape[0]):
            v = max(Exp_prediction[i])
            index = np.where(Exp_prediction[i] == v)[0][0]
            tmp_exp_prob.append(index)
            tmp_exp_label.append(label_exp[i].cpu().numpy())

        loss = Exp_loss

        loss.backward()
        optimizer.step()
        scheduler.step(step)

        loss_sum += loss.item()
        step += 1

        step_margin = 200
        if step % step_margin == 0:
            avg_loss = loss_sum / step_margin

            # Exp metric
            Exp_F1, Exp_acc, Exp_F1_per_class = metric_for_Exp(tmp_exp_label, tmp_exp_prob,8)
            final_Exp_score = 1 * Exp_F1

            #print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            #print('  train set - Exp     = {:.8f}'.format(final_Exp_score))
            with open(f"{log_save_path}/Exp_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {}, Exp_F1: {} Exp_acc: {}  Exp_score: {} \n".format(epoch, step,
                                                                                                      avg_loss, Exp_F1,
                                                                                                      Exp_acc,
                                                                                                      final_Exp_score))
            tmp_V_prob, tmp_A_prob, tmp_AU_prob, tmp_exp_prob = [], [], [], []
            tmp_V_label, tmp_A_label, tmp_AU_label, tmp_exp_label = [], [], [], []
            loss_sum = 0.0

        if step % 5000 == 0:
            net = net.eval()
            best_Exp_score = test_Exp(epoch, Exp_testloader, net, best_Exp_score, step)
            net = net.train()
    return best_Exp_score




def test_Exp(epoch, loader, net, best_acc, step=0):
    print("testing.. ")
    tmp_Exp_pred, tmp_Exp_label, tmp_Exp_prob, tmp_out = [], [] ,[], []
    net = net.eval()
    print('total batch:', len(loader))
    t = tqdm(enumerate(loader))
    with torch.no_grad():
        for batch_idx, (img, label_exp, feas, name) in t:
            # if batch_idx>10:
            #     break
            coff, embs, mfcc, words = feas
            if use_cuda:
                img = img.cuda()
                mfcc,embs,coff = mfcc.cuda(),embs.cuda(),coff.cuda()
            Exp_out = net(img,embs=embs,mfcc=mfcc,words=words,coff=coff)
            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            for i in range(len(name)):
                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                tmp_Exp_pred.append(index)
                tmp_Exp_prob.append(v)
                tmp_out.append(Exp_prediction[i])

        results = {'pred':tmp_Exp_pred, 'prob':tmp_Exp_prob, 'out':tmp_out}
        # with open('./results/exp_pred_results6.pkl', 'wb') as f:
        #     pickle.dump(results, f)

        return results



def test_Exp_with_metric(epoch, loader, net, best_acc, step=0):
    print("train {} epoch, total iters: {}".format(epoch, len(loader)))
    tmp_Exp_prob, tmp_Exp_label = [], []
    crit_Exp = LabelSmoothingCrossEntropy()
    net = net.eval()
    Exp_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, label_exp, feas,name) in t:
            coff, embs,mfcc,words = feas
            if use_cuda:
                img = img.cuda()
                label_exp,mfcc,embs,coff = label_exp.cuda(),mfcc.cuda(),embs.cuda(),coff.cuda()

            Exp_out = net(img,embs=embs,mfcc=mfcc,words=words,coff=coff)

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
        final_Exp_score = 1 * Exp_F1
        print(
            "epoch: {}, Loss: {}, Exp_F1: {} Exp_acc: {}  Exp_score: {} \n".format(
                epoch, Exp_loss_sum / len(loader),
                Exp_F1, Exp_acc, final_Exp_score))
        Exp_F1_per_class = [str(k) for k in Exp_F1_per_class]
        print(" ".join(Exp_F1_per_class))
        print("\n")
        return Exp_F1

def pred_Exp(epoch, net, save_path, best_score,step=0):
    t_set =  ABAW2_test_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_exp_test.csv",img_path, emb_dict,mfcc_dict,word_dict,60,60,60,transform1)
    loader = data.DataLoader(t_set, batch_size=bz * 3, num_workers=8)
    print("Pred {} epoch".format(epoch))
    tmp_Exp_prob, tmp_Exp_label = [], []
    net = net.eval()
    Exp_loss_sum = 0
    save_data = pd.DataFrame()
    imgs = []
    prob = []
    labels = []
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, feas,name) in t:
            # if batch_idx>10:
            #     break
            embs, mfcc,words = feas
            if use_cuda:
                img = img.cuda()
                mfcc,embs= mfcc.cuda(),embs.cuda()
            Exp_out = net(img, embs=embs, mfcc=mfcc, words=words)
            Exp_prediction = F.softmax(Exp_out, dim=1).detach().cpu().numpy()
            for i in range(len(name)):
                v = max(Exp_prediction[i])
                index = np.where(Exp_prediction[i] == v)[0][0]
                labels.append(index)
                imgs.append(name[i])
                str_prob = [str(k) for k in Exp_prediction[i]]
                str_prob = " ".join(str_prob)
                prob.append(str_prob)
    save_data["img"] = imgs
    save_data["prob"] = prob
    save_data["label"] = labels
    save_data.to_csv(os.path.join(save_path,"Exp_pred_epoch"+str(epoch)+"_step"+str(step)+"_best_score"+str(best_score)[:5]+".csv"))




def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

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
    preload_file = word_path+'.pkl'
    if os.path.exists(preload_file):
        print('load preloaded file:', preload_file)
        return load_pickle(preload_file)

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
    with open(preload_file, 'wb') as f:
        pickle.dump(word_dict, f)
    return word_dict
def load_mfcc(mfcc_path):
    mfcc_dict = {}
    lists = os.listdir(mfcc_path)
    preload_file = mfcc_path+'.pkl'
    if os.path.exists(preload_file):
        print('load preload file:',preload_file)
        return load_pickle(preload_file)
    for l in tqdm(lists):
        mfcc= np.loadtxt(os.path.join(mfcc_path,l),delimiter=',')
        # mfcc= torch.from_numpy(mfcc).float()
        mfcc_dict[l] = mfcc
    with open(preload_file, 'wb') as f:
        pickle.dump(mfcc_dict, f)
    return mfcc_dict

def load_emb(emb_path):
    emb_dict = {}
    lists = os.listdir(emb_path)
    preload_file = emb_path+'.pkl'
    if os.path.exists(preload_file):
        print('load preload file:',preload_file)
        return load_pickle(preload_file)

    for l in tqdm(lists):
        emb= np.loadtxt(os.path.join(emb_path,l),delimiter=',')
        # mfcc= torch.from_numpy(mfcc).float()
        emb_dict[l] = emb
    with open(preload_file, 'wb') as f:
        pickle.dump(emb_dict, f)
    return emb_dict

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

from transformers import BertTokenizer, BertModel
def Load_Bert_feature():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").eval()

    wordlist_path = r"F:\Aff2\wordlist"
    lists = os.listdir(wordlist_path)
    word_set = set()
    for l in tqdm.tqdm(lists):
        f = open(os.path.join(wordlist_path,l),"r")
        lines = f.readlines()
        # t1 = tqdm.tqdm(lines)
        for ls in lines:
            t = ls.strip().lower()
            if t!="-1":
                word_set.add(t)
    print(len(word_set))
    ABAW3_bert_dict = {}
    for t in tqdm.tqdm(list(word_set)):
        text_id = tokenizer(t,return_tensors='pt')
        ret = model(**text_id)
            # print(ret)
        emb = ret.last_hidden_state[0,1:-1]

        if emb.shape[1]>1:
            emb = torch.mean(emb,dim=0)
        ABAW3_bert_dict[t] = emb

def average_weight(weight1, weight2):
    weight_new = deepcopy(weight1)
    for key in weight1.keys():
        weight_new[key] = (weight1[key] + weight2[key]) / 2.
    return weight_new

if __name__ == '__main__':
    import time
    setup_seed(20)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = True
    # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    timestamp = ''
    use_greedy = True
    task = 'EXP'  # ['EXP','AU','VA','MULTI']
    if task in ['EXP', 'AU', 'VA']:
        task_type = 'single'
    elif task == 'MULTI':
        task_type = 'multi'

    print(f'************** NOW IS {task} TASK. ******************')
    img_path = "/data/data/ABAW/crop_face_jpg/"

    # data
    transform1 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])

    bz = 512 # memory:4.2G

    word_path = "/project/ABAW2_dataset/original_dataset/wordlist"
    word_dict = load_word(word_path)
    print("+++++ load word_dict ++++")
    mfcc_path = "/project/ABAW2_dataset/original_dataset/mfcc"
    mfcc_dict = load_mfcc(mfcc_path)
    print("+++++ load mfcc_dict ++++")
    emb_path = "/project/ABAW2_dataset/original_dataset/exp_emb_chazhi"
    emb_dict = load_emb(emb_path)
    print("+++++ load emb_dict ++++")
    coff_path = "/project/ABAW2_dataset/original_dataset/crop_face_3dmm_coff_processed_chazhi"
    coff_dict = load_emb(coff_path)

    test_csv = "./annos/ABAW3_EXP_test.csv"
    Exp_testset = ABAW2_Exp_data_test(test_csv, img_path, emb_dict,mfcc_dict,word_dict,
                                 coff_dict,60,60,60,60,transform1)
    Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz, num_workers=16, shuffle=False)
    print(f'data number: test: {Exp_testset.__len__()}')

    # net = Single_exp_detect_trans(use_mfcc=True,use_wordemb=True,use_exp_emb =True,use_coff=True)
    net = Single_exp_detect_Eff(use_mfcc=True,use_wordemb=True,use_exp_emb =True,use_coff=True,
                                use_feature3=True, word_emb='glove', use_bert=False)
    net = net.cuda().eval()

    timestamp_fold = ['20220321-061823','20220321-061928', '20220321-062006','20220321-062024','20220321-062038']

    if use_greedy:
        f1_fold = [0.266, 0.202, 0.262, 0.204, 0.227]
        timestamp_fold = sorted(timestamp_fold,key=lambda x: f1_fold[timestamp_fold.index(x)])
        ckpt_paths = [f'./checkpoints/single_task/baseline_EXP:CE+baseline+fewer_data_{timestamp}' for timestamp in
                      timestamp_fold]
        f1_best = max(f1_fold)
        print('==> init score:', f1_best)
        val_csv = "./annos/random_EXP_cls_test.csv"
        Exp_valset = ABAW2_Exp_data(val_csv, img_path, emb_dict, mfcc_dict, word_dict,
                                          coff_dict, None, 60, 60, 60, 60, transform1)
        Exp_valloader = data.DataLoader(Exp_valset, batch_size=bz, num_workers=16, shuffle=False)

        ck_save_path = ckpt_paths[0]
        if os.path.exists(f'{ck_save_path}/Exp_best.pth'):
            state_dict = torch.load( f'{ck_save_path}/Exp_best.pth')
            net.load_state_dict(state_dict)
            print('==> load pretrained model:', f'{ck_save_path}/Exp_best.pth')
        else:
            print('checkpoint does not exits:', ck_save_path)
            sys.exit()
        for ck_save_path in ckpt_paths[1:]:
            state_dict_tmp = torch.load(f'{ck_save_path}/Exp_best.pth')
            state_dict_new = average_weight(state_dict, state_dict_tmp)
            net.load_state_dict(state_dict_new)
            f1_new = test_Exp_with_metric(0, Exp_valloader, net, 0)
            print(f'before greedy:{f1_best}, after greedy: {f1_new}')
            if f1_new>f1_best:
                print('=====UPDATE STATE DICT======')
                state_dict = state_dict_new
                f1_best = f1_new
        net.load_state_dict(state_dict)
        torch.save(net.state_dict(), f'./checkpoints/single_task/fusion_061823.pth')
        results = test_Exp(0, Exp_testloader, net, 0)
        timestamp = ck_save_path.split('_')[-1]
        save_csv = test_csv.replace('annos', 'results').replace('.csv',f'_{timestamp}_greedy.csv')
        save_pkl = test_csv.replace('annos', 'results').replace('.csv',f'_{timestamp}_greedy.pkl')
        with open(save_pkl, 'wb') as f:
            pickle.dump(results, f)
        pd_data= pd.read_csv(test_csv)
        pd_data['EXP'] = results['pred']
        pd_data.to_csv(save_csv)
        print('saving result to:', save_csv)

    else:
        ckpt_paths = [f'./checkpoints/single_task/baseline_EXP:CE+baseline+fewer_data_{timestamp}' for timestamp in timestamp_fold]
        for ck_save_path in ckpt_paths:
            if os.path.exists(f'{ck_save_path}/Exp_best.pth'):
                state_dict = torch.load( f'{ck_save_path}/Exp_best.pth')
                net.load_state_dict(state_dict)
                print('==> load pretrained model:', f'{ck_save_path}/Exp_best.pth')
            else:
                print('checkpoint does not exits:', ck_save_path)
                sys.exit()

            scaler = GradScaler()
            results = test_Exp(0, Exp_testloader, net, 0)
            timestamp = ck_save_path.split('_')[-1]
            save_csv = os.path.join('results',ck_save_path.split('/')[-1]+'.csv')
            save_pkl = os.path.join('results',ck_save_path.split('/')[-1]+'.pkl')
            with open(save_pkl, 'wb') as f:
                pickle.dump(results, f)
            pd_data= pd.read_csv(test_csv)
            pd_data['EXP'] = results['pred']
            print('saving result to:', save_csv)
            pd_data.to_csv(save_csv, index=False)