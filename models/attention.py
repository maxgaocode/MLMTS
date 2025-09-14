# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange, repeat
#
#
# ########################################################################################
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x
#
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0.):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x, mask=None):
#         b, n, _, h = *x.shape, self.heads#32,18,100,8
#         qkv = self.to_qkv(x).chunk(3, dim=-1)#QKV: 32,18,100
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)#QKV: 32,4,18,25
#
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value=True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, float('-inf'))
#             del mask
#
#         attn = dots.softmax(dim=-1)
#
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, mlp_dim, dropout):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
#                 Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
#             ]))
#
#     def forward(self, x, mask=None):
#         for attn, ff in self.layers:
#             x = attn(x, mask=mask)
#             x = ff(x)
#         return x
#
#
# class Seq_Transformer(nn.Module):
#     def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
#         super().__init__()
#         patch_dim = channels * patch_size
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.c_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
#         self.to_c_token = nn.Identity()
#
#
#     def forward(self, forward_seq):
#         x = self.patch_to_embedding(forward_seq)#32,17,100
#         b, n, _ = x.shape
#         c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)#32,1,100
#         x = torch.cat((c_tokens, x), dim=1)#32,18,100
#         x = self.transformer(x)
#         c_t = self.to_c_token(x[:, 0])
#         return c_t



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


########################################################################################

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_k, mask=None):
        b, n, _, h = *x.shape, self.heads#32,18,100,8
        qkv = self.to_qkv(x).chunk(3, dim=-1)#QKV: 32,18,100
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)#QKV: 32,4,18,25

        # k = rearrange(x_k, 'b n (h d) -> b h n d',h=h)
        q = torch.tanh(q)
        k = torch.tanh(k)
        # v = torch.tanh(v)


        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, x_k, mask=None):
        for attn, ff in self.layers:
            x = attn(x, x_k, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq, x_k):
        x = self.patch_to_embedding(forward_seq)#32,17,100
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)#32,1,100
        x = torch.cat((c_tokens, x), dim=1)#32,18,100

        x_k = self.patch_to_embedding(x_k)#32,17,100
        b, n, _ = x_k.shape
        x_k = torch.cat((c_tokens, x_k), dim=1)#32,18,100

        x = self.transformer(x, x_k)
        c_t = self.to_c_token(x[:, 0])
        return c_t