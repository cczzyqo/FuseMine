import math
import torch
import torch.nn.functional as F
from torch import nn

from model.rotary import apply_rotary_emb
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from model.rms_norm import RMSNorm

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class SRADiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth,  # current layer index
        num_heads,
        num_kv_heads=None,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.num_heads = num_heads
        
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv1d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)
        
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(self, x, attn_mask=None):
        # x: [B, N, D]
        B, N, D = x.shape
        src_len = N
        q = self.q_proj(x).reshape(B, N, 2 * self.num_heads, self.head_dim)
        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        q = q.permute(0, 2, 1, 3)  # [B, 2*h, N, d]

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)  # [B, D, N]
            x_ = self.sr(x_)        # [B, D, N/r]
            x_ = x_.transpose(1, 2)  # [B, N/r, D]
            x_ = self.norm(x_)
            
            k = self.k_proj(x_).reshape(B, -1, 2 * self.num_kv_heads, self.head_dim)
            v = self.v_proj(x_).reshape(B, -1, self.num_kv_heads, 2 * self.head_dim)
        else:
            k = self.k_proj(x).reshape(B, N, 2 * self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).reshape(B, N, self.num_kv_heads, 2 * self.head_dim)

        
        k = repeat_kv(k.transpose(1, 2), self.n_rep)  # [B, 2*h, N/r, d]
        v = repeat_kv(v.transpose(1, 2), self.n_rep)  # [B, h, N/r, 2*d]

        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [B, 2*h, N, N/r]
        


        offset = src_len - N
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([N, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )

        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_weights = self.attn_drop(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn_weights = attn_weights.view(B, self.num_heads, 2, N, -1)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        
        attn = attn.transpose(1, 2).reshape(B, N, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)
        attn = self.proj_drop(attn)
        
        return attn
