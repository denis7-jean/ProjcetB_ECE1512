import torch.nn as nn
import torch
import torch.nn.functional as F


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class ABMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0.0):
        super(ABMIL, self).__init__()

        # feature dim reduction: D_feat -> D_inner
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.classifier   = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

        # pooling mode: "attention" (baseline) or "max" (ablation)
        self.pooling_mode = getattr(conf, "pooling_mode", "attention")

        if self.pooling_mode == "attention":
            # standard gated attention as in Ilse et al.
            self.attention = Attention_Gated(conf.D_inner, D, 1)

    def forward(self, x):
        """
        x: bag of patch features, shape [B, N, D_feat]
        In our setup B is usually 1 (one bag / slide per batch).
        """
        # take the single bag
        bag_feats = x[0]                         # [N, D_feat]
        med_feat  = self.dimreduction(bag_feats) # [N, D_inner]

        if self.pooling_mode == "max":
            # ===== Max Pooling Ablation =====
            afeat, _ = torch.max(med_feat, dim=0, keepdim=True)   # [1, D_inner]
        else:
            # ===== Baseline Attention Pooling =====
            A = self.attention(med_feat)        # [1, N]
            A = F.softmax(A, dim=1)             # softmax over instances
            afeat = torch.mm(A, med_feat)       # [1, D_inner]

        logits = self.classifier(afeat)         # [1, n_class]
        return logits

    


    