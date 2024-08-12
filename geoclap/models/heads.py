""" 
Adapted from: https://github.com/naver-ai/pcmepp/blob/main/pcmepp/models/img_encoder.py

Original and reference code: https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py
"""
import torch
import torch.nn as nn
import numpy as np
import os
from .gpo import GPO, AvgPool
from .MLP import pcmeppMLP


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderAggr(nn.Module):
    def __init__(self, feat_dim, fc_dim, precomp_enc_type='basic', no_imgnorm=False, no_sigma_ln=False, bias_init=-4, aggr='gpo'):
        super(EncoderAggr, self).__init__()
        self.fc_dim = fc_dim
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(feat_dim, fc_dim)
        self.precomp_enc_type = precomp_enc_type
        self.aggr = aggr
        if precomp_enc_type == 'basic':
            self.mlp = pcmeppMLP(feat_dim, fc_dim // 2, fc_dim, 2)
        if self.aggr == 'no_gpo':
            self.gpool = AvgPool()
        elif self.aggr == 'gpo':
            self.gpool = GPO(32, 32)
        self.init_weights(no_sigma_ln, bias_init)
        # FC => 512, 1024
        print("using gpo?",self.aggr=="gpo")
        
    def init_weights(self, no_sigma_ln, bias_init):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        if no_sigma_ln:
            # As there is no layer norm, set the bias of the linear layer to -4 to prevent too large std
            # nn.init.constant_(self.fc.bias, -4)
            self.fc.bias.data.fill_(bias_init)
        else:
            self.fc.bias.data.fill_(0)

    def forward(self, features_orig, feature_lengths):
        """Extract feature vectors."""
        features = self.fc(features_orig)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(features_orig) + features

        features, _ = self.gpool(features, feature_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


class EncoderHead(nn.Module):
    def __init__(self, feat_dim, fc_dim, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, for_landcover=False,**kwargs):
        super(EncoderHead, self).__init__()
        self.feat_encoder = EncoderAggr(feat_dim, fc_dim, precomp_enc_type, no_imgnorm, **kwargs)
        self.size_augment = size_augment
        self.for_landcover = for_landcover
    def forward(self, feats):
        """Extract feature vectors."""
        base_features = feats

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > self.size_augment:
                    feat_i = base_features[i][np.where(rand_list_1[i] > self.size_augment * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.feat_encoder(base_features, feat_lengths)
        if self.for_landcover:
            return features
        else:
            return {'mean': features, 'std': None}


class ProbEncoderHead(nn.Module):
    def __init__(self, feat_dim, fc_dim, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, **kwargs):
        super(ProbEncoderHead, self).__init__()
        self.mean_encoder = EncoderAggr(feat_dim, fc_dim, precomp_enc_type, no_imgnorm, **kwargs)
        self.std_encoder = EncoderAggr(feat_dim, fc_dim, precomp_enc_type, no_imgnorm, **kwargs)
        self.size_augment = size_augment

    def forward(self, feats):
        """Extract feature vectors."""
        base_features = feats
        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > self.size_augment:
                    feat_i = base_features[i][np.where(rand_list_1[i] > self.size_augment * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.mean_encoder(base_features, feat_lengths)
        std_features = self.std_encoder(base_features, feat_lengths)

        return {'mean': features, 'std': std_features}

if __name__ == '__main__':
    x = torch.randn(5,72,1024)
    head1 = EncoderHead(feat_dim=x.shape[-1], fc_dim=1024, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, aggr="no_gpo")
    head2 = EncoderHead(feat_dim=x.shape[-1], fc_dim=1024, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, aggr="gpo")
    head3 = ProbEncoderHead(feat_dim=x.shape[-1], fc_dim=1024, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, aggr="gpo")
    head4 = ProbEncoderHead(feat_dim=x.shape[-1], fc_dim=1024, precomp_enc_type='basic', no_imgnorm=False, size_augment=0.2, aggr="no_gpo")
    
    print(head1(x)['mean'].shape)
    print(head2(x)['mean'].shape)
    print(head3(x)['mean'].shape, head3(x)['std'].shape)
    print(head4(x)['mean'].shape, head4(x)['std'].shape)
