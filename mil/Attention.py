"""
Attention Based MIL
reference: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

class AttentionBase(nn.Module):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None, weight=None):
        super(AttentionBase, self).__init__()
        # parameters
        self.M = n_features
        self.L = hidden_layer
        self.ATTENTION_BRANCHES = attention_branches
        self.label_smoothing=label_smoothing
        # attention
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_labels),
            #nn.Sigmoid()
        )
        # criterion
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward_single(self, X):
        """for single WSI (instances x features)"""
        A = self.calc_attention(X, [1,0])  # KxATTENTION_BRANCHES
        Z = torch.mm(A, X)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        #Y_hat = (Y_prob>.5).float()
        return Y_prob, A

    def forward_multiwsi(self, X):
        """for Multi WSI (batch x instances x features)"""
        A = self.calc_attention(X, [2,1])  # KxATTENTION_BRANCHES
        Z = torch.squeeze(torch.bmm(A, X))  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        #Y_hat = (Y_prob>.5).float()
        return Y_prob, A

    def calc_attention(self, X, position):
        A = self.attention(X)
        A = torch.transpose(A, *position)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=position[0])  # softmax over K
        return A

    def calc_loss(self, X, Y):
        Y = Y.float()
        #Y_prob, _, _ = self.forward(X)
        Y_prob, _ = self.forward(X)
        if self.label_smoothing:
            Y=torch.clamp(Y, min=self.label_smoothing, max=1.-self.label_smoothing,)
        return self.criterion(Y_prob, Y)

    def predict_proba(self, X):
        return torch.squeeze(torch.sigmoid(self.forward(X)[0]))

class GatedAttentionBase(nn.Module):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None, weight=None):
        super(GatedAttentionBase, self).__init__()
        # parameters
        self.M = n_features
        self.L = hidden_layer
        self.ATTENTION_BRANCHES = attention_branches
        self.label_smoothing=label_smoothing
        # attention
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_labels),
            #nn.Sigmoid(),
        )
        # criterion
        self.criterion=nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward_single(self, X):
        """for single WSI (instances x features)"""
        A = self.calc_attention(X, [1,0])
        Z = torch.mm(A, X)  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        #Y_hat = (Y_prob>.5).float()
        return Y_prob, A

    def forward_multiwsi(self, X):
        """for Multi WSI (batch x instances x features)"""
        A = self.calc_attention(X, [2,1])
        Z = torch.squeeze(torch.bmm(A, X))  # ATTENTION_BRANCHESxM
        Y_prob = self.classifier(Z)
        #Y_hat = (Y_prob>.5).float()
        return Y_prob, A

    def calc_attention(self, X, position):
        A_V = self.attention_V(X)  # KxL
        A_U = self.attention_U(X)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, *position)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=position[0])  # softmax over K
        return A

    def calc_loss(self, X, Y):
        Y = Y.float()
        #Y_prob, _, _ = self.forward(X)
        Y_prob, _ = self.forward(X)
        if self.label_smoothing:
            Y=torch.clamp(Y, min=self.label_smoothing, max=1.-self.label_smoothing,)
        return self.criterion(Y_prob, Y)
        
    def predict_proba(self, X):
        return torch.squeeze(torch.sigmoid(self.forward(X)[0]))

class AttentionWSI(AttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None, weight:float=None):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing, weight=weight)

    def forward(self, X):
        return self.forward_single(X)

class AttentionMultiWSI(AttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None, weight:float=None):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing, weight=weight)

    def forward(self, X):
        return self.forward_multiwsi(X)

class GatedAttentionWSI(GatedAttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None, weight:float=None):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing, weight=weight)

    def forward(self, X):
        return self.forward_single(X)

class GatedAttentionMultiWSI(GatedAttentionBase):
    def __init__(self, n_features:int=512, hidden_layer:int=128, n_labels:int=1, attention_branches:int=1, label_smoothing:float=None, weight:float=None):
        super().__init__(n_features=n_features, hidden_layer=hidden_layer, n_labels=n_labels, attention_branches=attention_branches, label_smoothing=label_smoothing, weight=weight)

    def forward(self, X):
        return self.forward_multiwsi(X)

