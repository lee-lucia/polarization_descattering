## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stxl
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange

from basicsr.utils.registry import ARCH_REGISTRY


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        if group == 1:
            self.ln1 = nn.Linear(embed_dim*4, dim)
            self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x, prior=None):
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # prior
        if group == 1:
            self.ln1 = nn.Linear(embed_dim*4, dim)
            self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x, prior=None):
        b,c,h,w = x.shape
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Hierarchical Integration Module
class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
        super(HIM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)
        
        q = self.q(_x)
        kv = self.kv(prior)
        k,v = kv.chunk(2, dim=-1)   

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, embed_dim, group)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, prior=None):
        x = x + self.attn(self.norm1(x), prior)
        x = x + self.ffn(self.norm2(x), prior)

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, num_blocks, group):

        super().__init__()
        self.group = group

        # build blocks
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
        if self.group > 1:
            self.him = HIM(dim, num_heads, bias, embed_dim, LayerNorm_type)

    def forward(self, x, prior=None):
        if prior is not None and self.group > 1:
            x = self.him(x, prior)
            prior = None

        for blk in self.blocks:
            x = blk(x, prior)
                
        return x


class Illumination_Estimator(nn.Module):
    # 光照估计器模块：用于从输入图像中估计光照分量
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        # n_fea_middle: 中间层特征通道数
        # n_fea_in: 输入特征通道数 (默认为4，因为输入是图像3通道+1个均值通道)
        # n_fea_out: 输出光照图通道数 (默认为3，彩色光照图)
        super(Illumination_Estimator, self).__init__()

        # 1x1卷积，将输入(图像+均值通道)映射到中间特征维度
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 深度卷积，在每个中间特征通道上独立进行空间卷积
        # groups=n_fea_middle 使其成为一个标准的深度卷积
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_middle) 

        # 1x1卷积，将深度卷积提取的特征映射回所需的光照图通道数
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        # self.act = nn.Sigmoid()

    def forward(self, img):
        # img: 输入图像, 形状 [B, C_img=3, H, W]
        
        # 计算图像在通道维度上的均值，并扩展维度以进行拼接
        mean_c = img.mean(dim=1).unsqueeze(1) # 形状 [B, 1, H, W]
        
        # 将原始图像和其均值通道在通道维度上拼接
        input_tensor = torch.cat([img, mean_c], dim=1) # 形状 [B, C_img+1, H, W]

        x_1 = self.conv1(input_tensor) # 形状 [B, n_fea_middle, H, W]
        illu_fea = self.depth_conv(x_1) # 光照特征, 形状 [B, n_fea_middle, H, W]
        illu_map = self.conv2(illu_fea) # 估计的光照图, 形状 [B, n_fea_out, H, W]
        # illu_map = self.act(illu_map)
        
        return illu_fea, illu_map

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
    
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head * heads, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        # print(q_inp.shape)
        # print(k_inp.shape)
        # print(v_inp.shape)
        # print(illu_attn.shape)
        # q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
        #                          (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))


        illu_flat = illu_fea_trans.flatten(1, 2)          # -> (b, n, C_head) == (1,1024,96)
        illu_attn = illu_flat.unsqueeze(1)                # -> (b, 1, n, 96)
        illu_attn = illu_attn.expand(-1, self.num_heads, -1, -1)
        # print(v.shape)
        # print(illu_attn.shape)
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        # print('v_inp',v_inp.shape)
        # print('v_inp.reshape(b, h, w, c)',v_inp.reshape(b, h, w, c*self.num_heads).permute(
        #     0, 3, 1, 2).shape)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c*self.num_heads).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out
    
class OurFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, OurFeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

    
class OursBasicLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, num_blocks, group):

        super().__init__()
        self.group = group

        # build blocks
        # self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
        #                             bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])

        # self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
        #                             bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
        self.blocks = IGAB(dim=dim, num_blocks=num_blocks, dim_head=dim, heads=num_heads)
        self.end = TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group)
        # self.estimator = Illumination_Estimator(dim)

        if self.group > 1:
            self.him = HIM(dim, num_heads, bias, embed_dim, LayerNorm_type)

    def forward(self, x, illu_fea, prior=None):

        x = self.blocks(x, illu_fea)

        if prior is not None and self.group > 1:
            x = self.him(x, prior)
            prior = None
        
        x = self.end(x)

        # illu_fea, illu_map = self.estimator(x)
        # x = x * illu_map + x

        # for blk in self.blocks:
        #     x = blk(x, illu_fea)

                
        return x
    
@ARCH_REGISTRY.register()
class OurTransformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 64,
        num_blocks = [3,4,4,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        embed_dim = 64,
        group=4,
    ):

        super(OurTransformer, self).__init__()

        # multi-scale
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = OursBasicLayer(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0], group=group)

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.ourdown1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = OursBasicLayer(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1], group=group//2)

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.ourdown2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = OursBasicLayer(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2], group=group//2)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.ourdown3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = OursBasicLayer(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[3], group=1)

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = OursBasicLayer(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2], group=group//2)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = OursBasicLayer(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1], group=group//2)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim**1), kernel_size=1, bias=bias)
        self.decoder_level1 = OursBasicLayer(dim=int(dim**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0], group=group)

        self.refinement = OursBasicLayer(dim=int(dim**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_refinement_blocks, group=group)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*1**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.Illumination_Estimator = Illumination_Estimator(dim)
    def forward(self, inp_img, prior=None):

        # multi-scale prior
        prior_1 = prior
        prior_2 = self.down_1(prior_1)
        prior_3 = self.down_2(prior_2).flatten(1)

        illu_fea, illu_map = self.Illumination_Estimator(inp_img)
        inp_img = inp_img * illu_map + inp_img
        
        inp_enc_level1 = self.patch_embed(inp_img)
        # print(inp_enc_level1.shape)
        # print(illu_fea.shape)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, illu_fea, prior_1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        illu_fea_level2 = self.ourdown1_2(illu_fea)
        # print(inp_enc_level2.shape)
        # print(illu_fea_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, illu_fea_level2, prior_2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        illu_fea_level3 = self.ourdown2_3(illu_fea_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, illu_fea_level3, prior_2) 

        inp_enc_level4 = self.down3_4(out_enc_level3)
        illu_fea_level4 = self.ourdown3_4(illu_fea_level3)             
        latent = self.latent(inp_enc_level4, illu_fea_level4, prior_3) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, illu_fea_level3, prior_2) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, illu_fea_level2, prior_2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        # print(inp_dec_level1.shape)
        # print(illu_fea_level2.shape)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, illu_fea, prior_1)
        
        out_dec_level1 = self.refinement(out_dec_level1, illu_fea, prior_1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


##########################################################################
# The implementation builds on Restormer code https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
@ARCH_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [3,4,4,4], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        embed_dim = 64,
        group=4,
    ):

        super(Transformer, self).__init__()

        # multi-scale
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = BasicLayer(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0], group=group)

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = BasicLayer(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1], group=group//2)

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = BasicLayer(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2], group=group//2)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = BasicLayer(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[3], group=1)

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicLayer(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2], group=group//2)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1], group=group//2)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0], group=group)

        self.refinement = BasicLayer(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_refinement_blocks, group=group)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.Illumination_Estimator = Illumination_Estimator(embed_dim)
    def forward(self, inp_img, prior=None):

        # multi-scale prior
        # prior_1 = prior
        # prior_2 = self.down_1(prior_1)
        # prior_3 = self.down_2(prior_2).flatten(1)

        illu_fea, illu_map = self.Illumination_Estimator(inp_img)
        inp_img = inp_img * illu_map + inp_img
        
        inp_enc_level1 = self.patch_embed(inp_img)
        # out_enc_level1 = self.encoder_level1(inp_enc_level1, prior_1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        # out_enc_level2 = self.encoder_level2(inp_enc_level2, prior_2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        # out_enc_level3 = self.encoder_level3(inp_enc_level3, prior_2) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        # latent = self.latent(inp_enc_level4, prior_3) 
        latent = self.latent(inp_enc_level4 ) 

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3, prior_2) 
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # out_dec_level2 = self.decoder_level2(inp_dec_level2, prior_2) 
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # out_dec_level1 = self.decoder_level1(inp_dec_level1, prior_1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # out_dec_level1 = self.refinement(out_dec_level1, prior_1)
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
    

if __name__ == '__main__':
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = OurTransformer(        
        inp_channels= 3,
        out_channels= 3,
        dim= 64,
        num_blocks= [2,2,4,4],
        num_refinement_blocks= 2,
        heads= [1,2,4,8],
        ffn_expansion_factor= 2.66,
        bias= False,
        LayerNorm_type= 'WithBias',
        dual_pixel_task= False,
        embed_dim= 64,
        group= 4 # N=4*4
    ).to(device) 
    
    input = torch.randn(1,3,64,64).to(device) 
    prior = torch.randn(1,16,256).to(device) 
    # print(input.shape)
    with torch.no_grad():
        out = model(input,prior)
    print(out.shape)
    print('qqqqqqqqqqqqqqqqqqqqq')

    from thop.profile import profile
    
    name = "our"
    with torch.no_grad():
        total_ops, total_params = profile(model, (input,prior, ))
    print("%s         | %.4f(M)      | %.4f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
    


