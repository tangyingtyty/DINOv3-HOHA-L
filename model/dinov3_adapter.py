import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p


class AdaHyperedgeGen(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads
        self.context = context

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_hyperedges * node_dim)  
        elif context == "both":
            self.context_net = nn.Linear(2*node_dim, num_hyperedges * node_dim)
        else:
            raise ValueError(
                f"Unsupported context '{context}'. "
                "Expected one of: 'mean', 'max', 'both'."
            )

        self.pre_head_proj = nn.Linear(node_dim, node_dim)
    
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)
 
    def forward(self, X):
        B, N, D = X.shape
        if self.context == "mean":
            context_cat = X.mean(dim=1)          
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)          
        else:
            avg_context = X.mean(dim=1)           
            max_context, _ = X.max(dim=1)           
            context_cat = torch.cat([avg_context, max_context], dim=-1) 
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D)  
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets           
        
        X_proj = self.pre_head_proj(X) 
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling 
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        
        logits = self.dropout(logits)  

        return F.softmax(logits, dim=1)


class AdaptiveSparseHyperedgeGenerator(nn.Module):                                                                                                                                                                                                                                                                                    
    def __init__(self, node_dim, num_hyperedges, num_subprototypes=4,                                                                                                                                                                      
               sparsity=0.2, tau=1.0, dropout=0.1):                                                                                                                                                                                                    
        super().__init__()             
        self.M = num_hyperedges                                                                                                                                                                                                            
        self.k = num_subprototypes                                                                                                                                                                                                         
        self.D = node_dim
        self.sparsity = sparsity                                                                                  
        self.tau = tau                                                                                                                                                                                                                     
                    
        self.proto_base = nn.Parameter(
            torch.empty(num_hyperedges, num_subprototypes, node_dim)
            )                                                                                                                                                                      
        nn.init.trunc_normal_(self.proto_base, std=0.02)

        hidden = max(node_dim // 4, 32)
        self.ctx_global = nn.Sequential(
            nn.Linear(2 * node_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_hyperedges * num_subprototypes * node_dim)
            )

        self.ctx_local  = nn.Linear(node_dim, num_subprototypes * node_dim)

        nn.init.normal_(self.ctx_global[-1].weight, std=1e-4)
        nn.init.zeros_(self.ctx_global[-1].bias)
        nn.init.normal_(self.ctx_local.weight, std=1e-4)
        nn.init.zeros_(self.ctx_local.bias)  

        self.node_proj = nn.Linear(node_dim, node_dim)
        self.dropout   = nn.Dropout(dropout)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    def lse_similarity(self, X_proj, P):
        proj = torch.einsum('bnd,bmkd->bnmk', X_proj, P) / math.sqrt(self.D)                                                                                                                                                               
        return self.tau * torch.logsumexp(proj / self.tau, dim=-1)                                                                                                                                                                         
                                                                                                                                                                                                                                            
    @staticmethod                                                                                         
    def topk_softmax(logits, k, dim):                                                                                                                                                                                                      
        if k >= logits.size(dim):                                                                                                                                                                                                          
            return F.softmax(logits, dim=dim)
        vals, idx = torch.topk(logits, k, dim=dim)
        probs_topk = F.softmax(vals, dim=dim)  
        out = torch.zeros_like(logits)
        out.scatter_(dim, idx, probs_topk)
        return out                                                                                                                                                                                                    
                                                                                                        
    def _initial_assignment(self, X, X_proj):                                                                                                                                                                                                     
        B = X.size(0)
        global_ctx = torch.cat([X.mean(1), X.max(1).values], dim=-1)                                                                                                                                                                       
        delta_P0   = self.ctx_global(global_ctx).view(B, self.M, self.k, self.D)                                                                                                                                                           
        P0         = self.proto_base.unsqueeze(0) + delta_P0                                                                                                                                                                               
        S0         = self.lse_similarity(X_proj, P0)                                                                                                                                                                                       
        H0         = F.softmax(S0, dim=1)                                                                                                                                                                            
        return H0, P0                                                                                                                                                                                                                        
                                                                                                        
    def _refine_prototypes(self, X, H0, P0):                                                                                                                                                                                                              
        B = X.size(0)
        edge_ctx = torch.einsum('bnm,bnd->bmd', H0, X)
        delta_P1 = self.ctx_local(edge_ctx).view(B, self.M, self.k, self.D)                                                                                                                                                                
        P1       = P0 + delta_P1                                                                                                                                                                                 
        return P1                                                                                         
                                                                                                                                                                                                                                            
    def _final_assignment(self, X_proj, P1, N):
        S1 = self.lse_similarity(X_proj, P1)                                                                                                                                                                                               
        K = max(1, int(N * self.sparsity))                                                     
        return self.topk_softmax(S1, k=K, dim=1)                                                                                                                                                                                           
                                                                                                                                                                                                                                            
    # ------------------------------------------------------------------                                  
    def forward(self, X):                                                                                                                                                                                                                  
        B, N, D = X.shape                                                                                                                                                                                                                  
        X_proj  = self.dropout(self.node_proj(X))
                                                                                                                                                                                                                                            
        H0, P0 = self._initial_assignment(X, X_proj)                                                                                                                                                                       
        P1 = self._refine_prototypes(X, H0, P0)                                             
        H  = self._final_assignment(X_proj, P1, N)                                                                                                                                                               
                               
        return H 

    
class AdaHGConv(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, dropout=0.1):
        super().__init__()
        self.edge_generator = AdaptiveSparseHyperedgeGenerator(embed_dim, num_hyperedges, dropout=dropout)
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        
    def forward(self, X):
        A = self.edge_generator(X)  
        
        He = torch.bmm(A.transpose(1, 2), X) 
        He = self.edge_proj(He)
        
        X_new = torch.bmm(A, He)  
        X_new = self.node_proj(X_new)
        
        return X_new + X
    

class AdaHGComputation(nn.Module):
    def __init__(self, embed_dim, num_hyperedges=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hgnn = AdaHGConv(
            embed_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            dropout=dropout
        )
        
    def forward(self, x):
        tokens = x.transpose(1, 2)
        tokens = self.hgnn(tokens)
        tokens = tokens.transpose(1, 2)
        return tokens


class Conv1d(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HyperGraphModule(nn.Module):
    def __init__(self, c1=32, c2=64, e=1.0, num_hyperedges=8):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 16 == 0, "Dimension of AdaHGComputation should be a multiple of 16."
        self.cv1 = Conv1d(c1, c_, 1, 1)
        self.cv2 = Conv1d(c1, c_, 1, 1)
        self.m = AdaHGComputation(embed_dim=c_, 
                          num_hyperedges=num_hyperedges, 
                          dropout=0.1
                          )
        self.cv3 = Conv1d(2 * c_, c1, 1)  

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class HyperGraphExtractor(nn.Module):
    def __init__(self, dim, graph_dim=64):
        super().__init__()
        self.graph = HyperGraphModule(c1=dim, c2=graph_dim, num_hyperedges=8)
        
    def forward(self, x, c):

        channels = x.shape[1]

        out = torch.cat([x, c], dim=1).transpose(1, 2)

        out = self.graph(out).transpose(1, 2)

        return out[:, :channels, :], out[:, channels:, :]

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4



class DINOv3_Adapter(nn.Module):
    def __init__(
        self,
        backbone,
        interaction_indexes=[9, 19, 29, 39],
        pretrain_size=1024,
        conv_inplane=64,
        graph_dim=256,
        add_vit_feature=True
    ):
        super(DINOv3_Adapter, self).__init__()
        self.backbone = backbone
        # Important: we freeze the backbone
        self.backbone.requires_grad_(False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        print("embed dim", self.embed_dim)
        print("interaction_indexes", self.interaction_indexes)
        print("patch_size", self.patch_size)

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        # self.interactions = nn.Sequential(
        #     *[
        #         None,
        #         None,
        #         None,
        #         HyperGraphExtractor(dim=self.embed_dim, graph_dim=graph_dim)
        #     ]
        # )
        self.interactions = nn.Sequential(
            *[
                HyperGraphExtractor(dim=self.embed_dim, graph_dim=graph_dim)
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        torch.nn.init.normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c = torch.cat([c2, c3, c4], dim=1)

        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                all_layers = self.backbone.get_intermediate_layers(
                    x, n=self.interaction_indexes, return_class_token=False
                )

        x_for_shape  = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        outs = list()
        for i, layer in enumerate(self.interactions):
            x = all_layers[i]
            if layer is not None:
                x, c = layer(x, c)
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # Split & Reshape
        c2 = c[:, 0: c2.size(1), :]
        c3 = c[:, c2.size(1): c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            # c1, c2, c3, c4 = x1, x2, x3, x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return [f1, f2, f3, f4]