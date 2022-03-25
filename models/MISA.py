import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from utils import ReverseLayerF
import math
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

def to_gpu(x, on_cpu=False, gpu_id=None):
    """Tensor => Variable"""
    if torch.cuda.is_available() and not on_cpu:
        x = x.cuda(gpu_id)
    return x

def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)



"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class Ortho(nn.Module):
    '''
        在instance层面上做这个事情， 对每个instance输入都是一维向量， 先做norm在做内积。
    '''

    def __init__(self):
        super(Ortho, self).__init__()

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        loss_ortho = torch.mean(torch.einsum('ij,ij->i', [x1_norm, x2_norm]))
        return loss_ortho

    def norm(self, x):
        # mean = torch.mean(x, dim=1, keepdim=True)
        # std = torch.std(x, dim=1, keepdim=True)
        # x_norm = (x-mean)/std
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        return x_norm


class Inner(nn.Module):
    '''
        在instance层面上做这个事情， 对每个instance输入都是一维向量， 直接做内积。
    '''

    def __init__(self):
        super(Inner, self).__init__()

    def forward(self, x1, x2):
        # x1_norm = self.norm(x1)
        # x2_norm = self.norm(x2)
        dot_product = torch.einsum('ij,ij->i', [x1, x2])
        loss = torch.mean(dot_product)
        return loss


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self):
        super(MISA, self).__init__()

        self.text_size = config.embedding_size
        self.visual_size = 16
        self.acoustic_size = 40
        self.hidden_sizes=64

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = 8
        self.dropout_rate = dropout_rate = 0.1
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.loss_ortho = Ortho()
        self.loss_inner = Inner()

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between


        self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
        self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=self.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t',
                                      nn.Linear(in_features=hidden_sizes[0] * 4, out_features=self.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[1] * 4, out_features=self.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(self.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[2] * 4, out_features=self.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(self.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=self.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=self.hidden_size, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.self.hidden_size * 6,
                                                           out_features=self.self.hidden_size * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.self.hidden_size * 3, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        self.shared_feat = {'a': [], 't': [], 'v': []}  # P
        self.private_feat = {'a': [], 't': [], 'v': []}  # J
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.self.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu()
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def extract_features_ptflops(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu()

        padded_h1, (final_h1, _) = rnn1(sequence)
        normed_h1 = layer_norm(padded_h1)
        _, (final_h2, _) = rnn2(normed_h1)

        return final_h1, final_h2

    def alignment(self, sentences, visual, acoustic, lengths=121):

        batch_size = lengths.size(0)




        final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)

        # For reconstruction
        self.reconstruct()

        self.private_feat['a'].append(self.utt_private_a)
        self.private_feat['v'].append(self.utt_private_v)
        self.private_feat['t'].append(self.utt_private_t)
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t,
                         self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        return o

    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):

        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o

    def get_domain_loss(self, ):

        if self.train_config.use_cmd_sim:
            return 0.0

        # Predicted domain labels
        domain_pred_t = self.domain_label_t
        domain_pred_v = self.domain_label_v
        domain_pred_a = self.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0] * domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1] * domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2] * domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self, ):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        cmd_tv = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, 5)
        loss = cmd_tv
        cmd_ta = self.loss_cmd(self.utt_shared_t, self.utt_shared_a, 5)
        cmd_av = self.loss_cmd(self.utt_shared_a, self.utt_shared_v, 5)
        loss += cmd_ta
        loss += cmd_av
        loss = loss / 3.0
        metric_cmd = {'cmd_tv': cmd_tv, 'cmd_ta': cmd_ta, 'cmd_av': cmd_av}
        return loss, metric_cmd

    def get_diff_loss(self):
        # b * feature_dim (16 * 128)
        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        # Between private and shared
        loss_t = self.loss_diff(private_t, shared_t)
        loss_v = self.loss_diff(private_v, shared_v)
        loss_a = self.loss_diff(private_a, shared_a)
        # Across privates
        loss_at = self.loss_diff(private_a, private_t)
        loss_av = self.loss_diff(private_a, private_v)
        loss_tv = self.loss_diff(private_t, private_v)
        loss_diff = {'loss_t': loss_t, 'loss_v': loss_v, 'loss_a': loss_a,
                     'loss_at': loss_at, 'loss_av': loss_av, 'loss_tv': loss_tv}

        loss = loss_t
        loss += loss_v
        loss += loss_a
        loss += loss_at
        loss += loss_av
        loss += loss_tv
        return loss, loss_diff

    def get_recon_loss(self):

        loss = self.loss_recon(self.utt_t_recon, self.utt_t_orig)
        loss += self.loss_recon(self.utt_v_recon, self.utt_v_orig)
        loss += self.loss_recon(self.utt_a_recon, self.utt_a_orig)
        loss = loss / 3.0
        return loss

    def get_ortho_loss(self):

        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        loss = {}

        loss['t'] = self.loss_ortho(private_t, shared_t)
        loss['v'] = self.loss_ortho(private_v, shared_v)
        loss['a'] = self.loss_ortho(private_a, shared_a)

        loss['at'] = self.loss_ortho(private_a, private_t)
        loss['av'] = self.loss_ortho(private_a, private_v)
        loss['vt'] = self.loss_ortho(private_t, private_v)

        return loss