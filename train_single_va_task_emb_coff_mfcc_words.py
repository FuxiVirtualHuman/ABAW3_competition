import os

import torch
from models.pipeline5 import Pipeline
# from models.minus_pipeline_512d_multitask import Multi_task_model
# from models.multi_model_series import Multi_task_series_model
#from models.BBN_resnet import resnet50
# from models.BBN_resnet import resnet18
from models.single_VA_detect import Single_va_detect_multi_modal,Single_va_detect,Single_va_detect_Eff,Single_va_detect_trans
from models.BBN_resnet import BBN_resnet18
from data_new.ABAW2_data import compute_class_weight, ABAW2_Exp_data, ABAW2_VA_data, ABAW2_AU_data, \
    ABAW2_multitask_data2,ABAW2_test_data
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
import pandas as pd


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





def train_VA(epoch, loader, net, optimizer, best_VA_score):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []

    loss_sum = 0.0
    step = 1
    net = net.train()
    t = tqdm(enumerate(loader))
    for batch_idx, (va_img, label_V, label_A,feas,name) in t:
        coff,embs, mfcc,words = feas
        if use_cuda:
            va_img = va_img.cuda()
            label_V, label_A ,mfcc,embs,coff = label_V.float().cuda(), label_A.float().cuda(),mfcc.cuda(),embs.cuda(),coff.cuda()

        optimizer.zero_grad()
        VA_out = net(va_img,embs=embs, mfcc=mfcc,words=words,coff=coff)

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

            #print('  train set - Total Loss       = {:.8f}'.format(avg_loss))
            #print('  train set - VA     = {:.8f}'.format(final_VA_score))
            with open(f"{log_save_path}/VA_{task_type}_task_train.log", "a+") as log_file:
                log_file.write(
                    "epoch: {}, step: {},  Loss: {}, ccc_v: {} ccc_a: {} VA_score: {} \n".format(epoch, step,
                                                                                                 avg_loss, ccc_v, ccc_a,
                                                                                                 final_VA_score))
            tmp_V_prob, tmp_A_prob = [], []
            tmp_V_label, tmp_A_label = [], []
            loss_sum = 0.0

        if step % 4000 == 0:
            net = net.eval()
            best_VA_score = test_VA(epoch, VA_testloader, net, best_VA_score, step)
            net = net.train()
    return best_VA_score





def test_VA(epoch, loader, net, best_acc, step=0):
    print("train {} epoch".format(epoch))
    tmp_V_prob, tmp_A_prob = [], []
    tmp_V_label, tmp_A_label = [], []
    net = net.eval()
    VA_loss_sum = 0
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (va_img, label_V, label_A,feas,name) in t:
            coff,embs, mfcc,words = feas
            if use_cuda:
                va_img = va_img.cuda()
                label_V, label_A ,mfcc,embs,coff = label_V.float().cuda(), label_A.float().cuda(),mfcc.cuda(),embs.cuda(),coff.cuda()

            # optimizer.zero_grad()
            VA_out = net(va_img,embs=embs, mfcc=mfcc,words=words,coff=coff)

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
            # pred_VA(epoch, net, save_path,best_acc,step)
        # torch.save(net.state_dict(), f'{ck_step_save_path}/VA_epoch{epoch}_step{step}.pth')

        with open(f"{log_save_path}/VA_test.log", "a+") as log_file:
            log_file.write(
                "epoch: {0}, Loss: {1}, ccc_v: {2} ccc_a: {3}  VA_score: {4} \n".format(
                    epoch, VA_loss_sum / len(loader),
                    ccc_v, ccc_a, final_VA_score))
        return best_acc



def pred_VA(epoch, net, save_path,best_score, step=0):
    t_set =  ABAW2_test_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_va_test.csv",img_path, emb_dict,mfcc_dict,word_dict,60,60,60,transform1)
    loader = data.DataLoader(t_set, batch_size=bz * 3, num_workers=8)
    print("Pred {} epoch".format(epoch))
    net = net.eval()
    tmp_V_prob, tmp_A_prob = [], []
    names = []
    with torch.no_grad():
        t = tqdm(enumerate(loader))
        for batch_idx, (img, feas,name) in t:
            embs, mfcc,words = feas
            if use_cuda:
                img = va_img.cuda()
                mfcc,embs= mfcc.cuda(),embs.cuda()
            _,VA_out = net(img,embs, mfcc, words)
            VA_out = VA_out.detach().cpu().numpy()
            tmp_V_prob.extend(VA_out[:, 0])
            tmp_A_prob.extend(VA_out[:, 1])
            names.extend(name)
    data = pd.DataFrame()
    data["img"] = names
    data["V"] = tmp_V_prob
    data["A"] = tmp_V_prob
    save_data.to_csv(os.path.join(save_path,"VA_pred_epoch"+str(epoch)+"_step"+str(step)+"_best_score"+str(best_score)[:5]+".csv"))  


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
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = True

    task = 'VA'  # ['EXP','AU','VA','MULTI']
    if task in ['EXP', 'AU', 'VA']:
        task_type = 'single'
    elif task == 'MULTI':
        task_type = 'multi'

    model_name = 'baseline'  # ['baseline','mutual','dropout','resnet']
    VA_LOSS = 'CCC'  # ['MAE','MSE','SmoothL1','CCC']
    AU_LOSS = 'CE'  # ['LabelSmoothCE','CE','WCE','ML_CE']
    # EXP_LOSS = 'LabelSmoothCE'  # ['LabelSmoothCE','CE','sCE_and_focal_loss']
    EXP_LOSS = 'CE'
    comment = 'trans_mfcc_seq30_words_seq30_embs_seq30_coff_seq30'
    # comment = 'BBN'

    # remark = 'va:' + VA_LOSS + '_au:' + AU_LOSS + '_exp:' + EXP_LOSS + f'+{model_name}+{comment}'
    remark = 'va:'+VA_LOSS + f'+{model_name}+{comment}'
    

    ck_save_path = f'./checkpoints2/ABAW2/{task_type}_task/{model_name}_{remark}'
    ck_step_save_path = f'./checkpoints2/ABAW2/{task_type}_task/{model_name}_{remark}/epoch_weight'
    log_save_path = f'./log2/ABAW2/{task_type}_task/{model_name}_{remark}'
    pred_save_path = f'./test/{task_type}_task/{model_name}_{remark}'
    os.makedirs(pred_save_path,exist_ok=True)
    if os.path.exists(ck_save_path):
        print('exits:{}'.format(ck_save_path))
    else:
        print('no exits')

    os.makedirs(ck_step_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)

    print(f'************** NOW IS {task} TASK. ******************')
    train_csv = "data_new/multi_data_new_1.csv"
    # img_path = "/home/zhangwei05/data/aff_in_the_wild/"
    img_path = "/data/crop_face_jpg/"

    AU_class_weight = compute_class_weight("/home/zhangwei05/ABAW3/annos/ABAW3_new_AU_training.csv", "AU").cuda()
    Exp_class_weight = compute_class_weight("/home/zhangwei05/ABAW3/annos/ABAW3_new_exp_training.csv", "Exp")
    print(AU_class_weight, Exp_class_weight)

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
        # crit_Exp = CrossEntropyLoss()
        crit_Exp = CrossEntropyLoss(Exp_class_weight.cuda())
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
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(30),
        transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5),
        transforms.ToTensor()])

    bz = 80


    if task == 'EXP':
        Exp_trainset = ABAW2_Exp_data("/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_exp_training.csv", img_path, transform2)
        trainset = Exp_trainset
        Exp_testset = ABAW2_Exp_data("/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_exp_validation.csv", img_path, transform1)
        Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=8)
    elif task == 'AU':
        AU_trainset = ABAW2_AU_data("/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_training.csv", img_path, False,transform)
        trainset = AU_trainset
        AU_testset = ABAW2_AU_data("/project/ABAW2_dataset/gzh/aff_in_the_wild/data_init/Third ABAW Annotations/AU_Set/ABAW3_new_AU_validation.csv", img_path, False,transform1)
        AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=8)
    elif task == 'VA':
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

        VA_trainset = ABAW2_VA_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_va_training.csv", img_path,emb_dict, mfcc_dict, word_dict,coff_dict,60,60,60,60,transform)
        trainset = VA_trainset
        VA_testset = ABAW2_VA_data("/home/zhangwei05/ABAW3/annos/ABAW3_new_va_validation.csv", img_path, emb_dict, mfcc_dict,word_dict,coff_dict,60,60,60,60,transform1)
        VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=16)
    elif task == 'MULTI':
        # trainset = ABAW2_multitask_data2( Exp_csv_file="data_new/new_Exp_training_1.csv",VA_csv_file="data_new/new_VA_training_1.csv",AU_csv_file="data_new/new_AU_training_1.csv", img_path=img_path,transform=transform)
        trainset = ABAW2_multitask_data2(Exp_csv_file="data_new/new_processed_Exp_training+emb.csv",
                                         VA_csv_file="data_new/new_processed_VA_training+emb.csv",
                                         AU_csv_file="data_new/new_processed_AU_training+emb.csv", img_path=img_path,
                                         Exp_VA_transform=transform, AU_transform=transform)
        Exp_testset = ABAW2_Exp_data("data_new/new_Exp_validation_1.csv", img_path, transform=transform1)
        Exp_testloader = data.DataLoader(Exp_testset, batch_size=bz * 3, num_workers=8)
        AU_testset = ABAW2_AU_data("data_new/new_AU_validation_1.csv", img_path, transform=transform1)
        AU_testloader = data.DataLoader(AU_testset, batch_size=bz * 3, num_workers=8)
        VA_testset = ABAW2_VA_data("data_new/new_VA_validation_1.csv", img_path, transform=transform1)
        VA_testloader = data.DataLoader(VA_testset, batch_size=bz * 3, num_workers=8)

    trainloader = data.DataLoader(trainset, batch_size=bz, num_workers=16, shuffle=True)

    # model
    # emb_net = Pipeline()
    ## net = Multi_task_model(emb_net)
    # net = Multi_task_series_model(emb_net, pretrained='checkpoints/minus_pipeline_affectnet+aff2_triplet_best.pth')
    # net = resnet50(num_classes=7)
    # net = Single_va_detect(use_mfcc=True,use_wordemb=True,use_exp_emb=True,use_coff=True)
    net = Single_va_detect_trans(use_mfcc=True,use_wordemb=True,use_exp_emb=True,use_coff=True)
    
    if use_cuda:
        net = net.cuda()

    # training parameters
    best_AU_score, best_VA_score, best_Exp_score = 0, 0, 0
    lr = 0.002
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, 2, 1e-5)

    # state_dict = torch.load("/tmp/pycharm_project_326/aff_in_the_wild/checkpoints/ABAW2/AU_single_task_best.pth")
    # net.load_state_dict(state_dict)

    # ck_file = ck_save_path+'/Exp_best.pth'
    # if len(ck_file) == 0:
    #     print('there is no ck restart')
    # state_dict = torch.load(ck_file)
    # net.load_state_dict(state_dict)

    scaler = GradScaler()
    for i in range(100):
        if task == 'EXP':
            best_Exp_score = train_Exp(i, trainloader, net, optimizer, best_Exp_score)
            best_Exp_score = test_Exp(i, Exp_testloader, net, best_Exp_score)
        elif task == 'VA':
            best_VA_score = train_VA(i, trainloader, net, optimizer, best_VA_score)
            best_VA_score = test_VA(i, VA_testloader, net, best_VA_score)
            # pred_VA(i, net, pred_save_path,best_VA_score,len(trainloader))
        elif task == 'AU':
            best_AU_score = train_AU(i, trainloader, net, optimizer, best_AU_score)
            best_AU_score = test_AU(i, AU_testloader, net, best_AU_score)
        elif task == 'MULTI':
            best_AU_score, best_VA_score, best_Exp_score = train(i, trainloader, net, optimizer, best_AU_score,
                                                                 best_VA_score, best_Exp_score)
            best_Exp_score = test_Exp(i, Exp_testloader, net, best_Exp_score)
            best_VA_score = test_VA(i, VA_testloader, net, best_VA_score)
            best_AU_score = test_AU(i, AU_testloader, net, best_AU_score)
