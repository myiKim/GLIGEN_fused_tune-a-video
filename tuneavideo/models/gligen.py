# Adpated from 0.20.0 version of diffuser lib : https://github.com/huggingface/diffusers/blob/v0.20.0-release/src/diffusers/models/attention.py
import torch
from torch import nn
from diffusers.models.attention import CrossAttention, FeedForward
from einops import rearrange

class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head, video_length =24):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.video_length = video_length
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = CrossAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True


    def forward(self, x, objs):
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        print("n_visual: ", n_visual)
        obj_orig_size = objs.size()
        objs = self.linear(objs)
        print("objs1 size: ", objs.size())
        # objs = rearrange(objs, "(b f) d c -> b f d c", f=video_length)
        objs = objs.repeat(self.video_length, 1, 1)
        
        
        print("SIZE CHECK!")
        print("x size: ", x.size())
        print("objs2 size: ", objs.size())
        
        print("self.alpha_attn.tanh().size: ", self.alpha_attn.tanh().size())
        print("concat size : ", torch.cat([x, objs], dim=1).size())
        print("normed concat size : ",self.norm1(torch.cat([x, objs], dim=1)).size())
        print("attention size: ", self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :].size())
        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        print("SIZE CHECK FINISHED!")
  
        
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x