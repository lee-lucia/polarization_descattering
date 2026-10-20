import torch
import torch.nn as nn
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

class MLP(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 **kwargs):

        super(MLP, self).__init__()

        patch_mix_dims = int(patch_expansion * num_patches)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))

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

    def forward(self, img):
        # img: 输入图像, 形状 [B, C_img=3, H, W]
        
        # 计算图像在通道维度上的均值，并扩展维度以进行拼接
        mean_c = img.mean(dim=1).unsqueeze(1) # 形状 [B, 1, H, W]
        
        # 将原始图像和其均值通道在通道维度上拼接
        input_tensor = torch.cat([img, mean_c], dim=1) # 形状 [B, C_img+1, H, W]

        x_1 = self.conv1(input_tensor) # 形状 [B, n_fea_middle, H, W]
        illu_fea = self.depth_conv(x_1) # 光照特征, 形状 [B, n_fea_middle, H, W]
        illu_map = self.conv2(illu_fea) # 估计的光照图, 形状 [B, n_fea_out, H, W]
        
        return illu_fea, illu_map


@ARCH_REGISTRY.register()
class latent_encoder_gelu(nn.Module):

    def __init__(self, in_chans=6, embed_dim=64, block_num=4, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_gelu, self).__init__()

        # assert in_chans == int(6//stage), "in chanel size is wrong"

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_chans*16, embed_dim, 3, 1, 1), 
                # nn.Conv2d(20*16, embed_dim, 3, 1, 1), 
                nn.GELU(),
                )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            self.blocks.append(block)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group*group, embed_dims=embed_dim, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*4),
                nn.GELU(),)
        # self.Illumination_Estimator = Illumination_Estimator(embed_dim)

    def forward(self, inp_img, gt=None):
        # prior = self.net_le([self.img_I, self.i_polar,self.i_olp], self.img_I_gt)
        # illu_fea, illu_map = self.Illumination_Estimator(inp_img[0])
        # inp_img[0] = inp_img[0] * illu_map + inp_img[0]
        inp_img = torch.cat(inp_img,1) # b, 17, hw
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1) # b, 20, hw
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x

@ARCH_REGISTRY.register()
class latent_encoder_lrelu(nn.Module):

    def __init__(self, in_chans=6, embed_dim=64, block_num=4, stage=1, group=4, patch_expansion=0.5, channel_expansion=4):
        super(latent_encoder_lrelu, self).__init__()

        assert in_chans == int(6//stage), "in chanel size is wrong"

        self.group = group

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_chans*16, embed_dim, 3, 1, 1), 
                nn.LeakyReLU(0.1, True),
                )

        self.blocks = nn.ModuleList()
        for i in range(block_num):
            block = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), 
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1))
            self.blocks.append(block)

        self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True),
        )
        self.pool = nn.AdaptiveAvgPool2d((group, group))
        self.mlp = MLP(num_patches=group*group, embed_dims=embed_dim*4, patch_expansion=patch_expansion, channel_expansion=channel_expansion)
        self.end = nn.Sequential(
                nn.Linear(embed_dim*4, embed_dim*4),
                nn.LeakyReLU(0.1, True),)
        

    def forward(self, inp_img, gt=None):
        if gt is not None:
            x = torch.cat([gt, inp_img], dim=1)
        else:
            x = inp_img

        x = self.pixel_unshuffle(x)
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x) + x
        x = self.pool(self.conv2(x))
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.mlp(x)
        x = self.end(x)
        return x


if __name__ == '__main__':
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = latent_encoder_gelu(in_chans=12, embed_dim=128, block_num=6, group=4, stage=1,patch_expansion=0.5,channel_expansion=4).to(device) 
    
    lq_rgb = torch.randn(1,3,32,32).to(device) 
    lq_img0 = torch.randn(1,1,32,32).to(device)
    lq_img45 = torch.randn(1,1,32,32).to(device)
    lq_img90 = torch.randn(1,1,32,32).to(device)
    lq_img135 = torch.randn(1,1,32,32).to(device)
    lq_aolp = torch.randn(1,1,32,32).to(device)
    lq_dolp = torch.randn(1,1,32,32).to(device)
    gt_rgb = torch.randn(1,3,32,32).to(device) 
    input_features = [
    lq_rgb, lq_img0, lq_img45, 
    lq_img90, lq_img135, lq_aolp, lq_dolp
]
    print(input_features[0])
    # print(input_features)
    with torch.no_grad():
        out = model(input_features,gt_rgb)
    print(out.shape)
    print('qqqqqqqqqqqqqqqqqqqqq')

    from thop.profile import profile
    
    name = "our"
    with torch.no_grad():
        total_ops, total_params = profile(model, (input_features,gt_rgb, ))
    print("%s         | %.4f(M)      | %.4f(G)         |" % (name, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
   