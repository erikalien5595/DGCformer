__all__ = ['DGC_backbone']

# Cell
from typing import Callable, Optional
import torch
import os
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from collections import OrderedDict
from layers.DGC_layers import *
from layers.RevIN import RevIN

# Cell
class DGC_backbone(nn.Module):
    def __init__(self, cn:int, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        self.cn = cn

        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in = c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    def forward(self, df, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        device = next(self.parameters()).device

        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1) 
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        latent_dim = 10  
        
        input_dim = z.shape[2]
        dgc = DGC(df, input_dim, latent_dim, 5, self.n_vars).to(device) 
        
        z1 = z
        x_bar, q, predict, z_temp, z_cls = dgc(z)

        p = target_distribution(q)


        latent_variables = []

        latent_variables = z_cls.detach().cpu().numpy()
        
        def cal_k_shape(data,num_cluster):
            """
            use best of cluster
            :param df: time series dataset
            :param num_cluster:
            :return:cluster label
            """
            ks = KMeans(n_clusters=num_cluster)
            y_pred = ks.fit_predict(data)
            label = ks.labels_
            return y_pred, label
        def shape_score(data,labels):

            score=silhouette_score(data,labels)
            
            return score

        k_shape, k_score = [], []
        for i in range(2,7):
            shape_pred, label1 = cal_k_shape(latent_variables,i)
            score = shape_score(latent_variables, label1)
            k_score.append(score)
            k_shape.append(i)

        dict_shape = dict(zip(k_shape, k_score))
        best_shape = sorted(dict_shape.items(), key=lambda x: x[1], reverse=True)[0][0]
        fin_label, label = cal_k_shape(latent_variables,best_shape) 
        fin_cluster = pd.DataFrame({ "cluster_label": fin_label})

        labeled_tensor_dict = {}

        for i in range(np.shape(latent_variables)[0]):
            labeled_tensor_dict[i] = {
                'data': torch.from_numpy(latent_variables[i]).unsqueeze(0),
                'label': label[i]
            }

        key = len(labeled_tensor_dict)
        
        mask = torch.zeros((self.n_vars,self.n_vars)).to(device)
        def cluster_and_visualize(time_series, n_clusters=best_shape):

            plt.figure(figsize=(12, 6))

            for i, labels in enumerate(set(label)):
                plt.subplot(n_clusters, 1, i + 1)
                for series in time_series[label == labels]:
                    plt.plot(series.cpu(), color='C'+str(i), alpha=0.5)
                plt.title(f"Cluster {i+1}")


            plt.tight_layout()
            
            plt.savefig('clustered_time_series.pdf')
            
            plt.close()
        cluster_and_visualize(z[0], n_clusters=best_shape)

        for i, value_i in labeled_tensor_dict.items():
            label_i = value_i['label']
            for j, value_j in labeled_tensor_dict.items():
                label_j = value_j['label']
                if label_i == label_j:
                    mask[i][j] = 1

        mask = torch.where(mask == 0, torch.tensor([-float('inf')]).to(device), torch.tensor([0.0]).to(device))
        
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        # print(z.shape,80*"-")
        # model
        # print(z.shape)
        z, attn = self.backbone(z, mask)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        # print(80*"_")
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)

        return x_bar, z1, q, predict, p, z, attn
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )
        
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.matmul(features, self.weight)
        output = torch.matmul(adj, support)
        if active:
            output = F.relu(output)
        return output
    
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder1 = nn.GRU(input_dim, 32, batch_first=True)
        self.encoder2 = nn.Linear(32, latent_dim)  
        self.decoder1 = nn.Linear(latent_dim, 32)  
        self.decoder2 = nn.GRU(32, input_dim, batch_first=True)
        self.Linear_mu = nn.Linear(latent_dim * 128, latent_dim) #########################################

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        z_temp1, _ = self.encoder1(x)
        z_temp1 = F.relu(z_temp1)
        z_temp2 = self.encoder2(z_temp1)
        z = F.relu(z_temp2)
        z_cls = self.Linear_mu(z.permute(1,2,0).flatten(1))
        dec_1 = self.decoder1(z)
        dec_1 = F.relu(dec_1)
        dec_2, _ = self.decoder2(dec_1)
        dec_2 = F.relu(dec_2)

        return dec_2, z_temp1, z, z_cls
def threshold_tensor(tensor):
    thresholded_tensor = torch.where(torch.abs(tensor) > 0.6, torch.tensor(1.0), torch.tensor(0.0))
    return thresholded_tensor   
class DGC(nn.Module):

    def __init__(self, df, input_dim, latent_dim, n_clusters, nvars, v=1):
        super(DGC, self).__init__()

        # autoencoder for intra information
        self.ae = VAE(input_dim, latent_dim)

        # GCN for inter information
        self.gnn_1 = GNNLayer(input_dim, 32)
        self.gnn_2 = GNNLayer(32, latent_dim)
        self.gnn_3 = GNNLayer(latent_dim, n_clusters)
        
        # cluster layer
        self.cluster_layer = nn.Parameter(torch.Tensor(128, n_clusters, latent_dim)) 
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        att = df.corr()
        att = torch.tensor(att.values)
        att = att.float()
        self.adj = torch.nn.Parameter(threshold_tensor(att)) 
        # degree
        self.v = v

    def forward(self, x):
        # DNN Module
        x_bar, z1, z, z_cls = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, self.adj)

        h = self.gnn_2((1-sigma)*h + sigma*z1, self.adj)

        h = self.gnn_3((1-sigma)*h + sigma*z, self.adj, active=False)

        predict = F.softmax(h, dim=2)
        # print(z.shape,self.cluster_layer.shape)
        # Dual Self-supervised Module
        
        q1 = []
        for i in range(z.shape[0]):
            q = 1.0 / (1.0 + torch.sum(torch.pow(z[i].unsqueeze(1) - self.cluster_layer[i], 2), 2) / 1)
            # print(q.shape)
            q = q.pow((1 + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            q1.append(q.unsqueeze(0))
        q = torch.cat(q1, dim=0)
        
        return x_bar, q, predict, z, z_cls
    
def target_distribution(q):
    target_tensor = []
    for i in range(q.shape[0]):
        weight = q[i]**2 / q[i].sum(0)
        target = (weight.t() / weight.sum(1)).t().unsqueeze(0)
        target_tensor.append(target)
    target_tensor = torch.cat(target_tensor, dim=0)
    return target_tensor

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(c_in, q_len, patch_num, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x, mask) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, ((x.shape[0]*x.shape[1]),x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z, attn = self.encoder(mask, u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z, attn
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, c_in, q_len, seg_num, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.c_in = c_in
        self.q_len = q_len
        self.d_model = d_model
        self.layers = nn.ModuleList([CROSSTSTEncoderLayer(c_in, q_len, seg_num, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, factor=3, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, mask, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        # print(output.shape)
        scores = None

        labels = torch.arange(output.size(1))
        
        output_dict = {}
            
        if self.res_attention:
            for mod in self.layers: output, attn, scores = mod(mask, output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output, attn
        else:
            for mod in self.layers: output, attn = mod(mask, output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output, attn

class CROSSTSTEncoderLayer(nn.Module):
    def __init__(self, c_in, q_len, seg_num, d_model, n_heads, d_k=None, d_v=None, factor=3, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        
        # self.mask = att

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.dim_sender = _MaskMultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.dim_receiver = _MaskMultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
            
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.c_in = c_in


    def forward(self, mask, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        # print(80*"=")
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)
        # print(80*"+")
        device = next(self.parameters()).device
        mask1 = torch.ones(mask.shape[0],1).to(device)
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[0]):
        #         if mask[i, j] != -float('inf'):
        #             mask1[i, 0] = 1
        #             mask1[j, 0] = 1
        # print(src.shape,80*"+")
        bs = src.shape[0] // self.c_in
        src = rearrange(src, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = bs)
        
        # print(src.shape)
        # if self.c_in < 3000:
        dim_buffer = src
        src2,attn_weights,_ = self.dim_receiver(mask, src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # else:
        #     batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=bs)
        #     # print(batch_router.shape)
        #     dim_buffer,_,_ = self.dim_sender(mask_2, batch_router, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # first router
        #     src2,attn_weights,_ = self.dim_receiver(mask_1, src, dim_buffer, dim_buffer, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)  # second router

        # if self.res_attention:
        #     src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # else:
        #     src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # print(80*"-")
        # Add & Norm
        src3 = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src3 = self.norm_attn(src3)

        # Feed-forward sublayer
        if self.pre_norm:
            src3 = self.norm_ffn(src3)
        ## Position-wise Feed-Forward
        src4 = self.ff(src3)
        ## Add & Norm
        src5 = src3 + self.dropout_ffn(src4) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src5 = self.norm_ffn(src5)
        
        src5 = src * (1-mask1) + src5 * mask1
        # print(src5.shape,80*"=")
        src5 = rearrange(src5, '(b seg_num) ts_d d_model -> (b ts_d) seg_num d_model', b=bs)
    

        if self.res_attention:
            return src5, attn, scores
        else:
            return src5, attn

class _MaskMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _MaskScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, mask, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(mask, q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(mask, q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        # print(Q.shape,80*"=")
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores 

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights
        
class _MaskScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, mask, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
    

        
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights