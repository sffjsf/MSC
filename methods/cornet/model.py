import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

mode = 'bilinear' # 'nearest' # 

# --------------------------
# 1. 实现 EPM 依赖的 RCAB 残差通道注意力块（论文公式与结构对应）
# --------------------------
class RCAB(nn.Module):
    """残差通道注意力块（Residual Channel Attention Block），用于抑制非边界信息"""
    def __init__(self, in_channels, reduction=16):
        super(RCAB, self).__init__()
        # 通道注意力分支
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化获取通道特征
            nn.Conv2d(in_channels, in_channels // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, padding=0),
            nn.Sigmoid()  # 输出通道注意力权重
        )
        # 残差卷积分支
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels//2, in_channels),  # 与现有代码保持一致的归一化方式
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels//2, in_channels)
        )

    def forward(self, x):
        residual = x  # 残差连接
        # 通道注意力加权
        attn = self.attention(x)
        x = x * attn
        # 残差卷积
        x = self.conv(x)
        x += residual  # 残差融合
        return x

# --------------------------
# 2. 实现 WBNet 边界预测模块（EPM）
# --------------------------
class EPM(nn.Module):
    """边界预测模块（Edge Prediction Module），对应论文图7与描述"""
    def __init__(self, in_feat_dims, C_e=32):
        """
        Args:
            in_feat_dims: 输入特征块的通道数列表（对应 B1, B2, B3 的通道数）
            C_e: 通道映射后的统一通道数（论文设为32）
        """
        super(EPM, self).__init__()
        self.C_e = C_e
        # 1. 通道映射层：将 B1, B2, B3 通道数统一映射为 C_e
        self.channel_mapper = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, C_e, 1, padding=0),
                nn.Conv2d(C_e, C_e, 3, padding=1),
                nn.GroupNorm(C_e//2, C_e), 
                nn.ReLU(inplace=True)
            ) for in_dim in in_feat_dims
        ])
        # 2. RCAB 模块：抑制非边界信息（输入通道数 = 3*C_e = 96）
        self.rcab = RCAB(in_channels=3 * C_e, reduction=16)
        # 3. 分类器：输出 1 通道边界图（像素值∈[0,1]，1表示边界）
        self.classifier = nn.Conv2d(3 * C_e, 1, 1, padding=0)  # 1×1卷积压缩通道

    def forward(self, feat_list, x_size):
        """
        Args:
            feat_list: 输入特征块列表 [B1, B2, B3]（分辨率 B1 > B2 > B3）
            x_size: 输入图像的原始尺寸 (H, W)，用于上采样对齐
        Returns:
            S_c: 预测边界图 (batch, 1, H, W)，像素值∈[0,1]
        """
        # 步骤1：通道映射 + 上采样（统一分辨率至输入图像尺寸）
        mapped_feats = []
        for i, (feat, mapper) in enumerate(zip(feat_list, self.channel_mapper)):
            # 通道映射
            feat_mapped = mapper(feat)
            # 上采样至输入图像尺寸（与现有代码上采样模式一致）
            feat_up = F.interpolate(feat_mapped, size=x_size, mode=mode, align_corners=False)
            mapped_feats.append(feat_up)
       
        # 步骤2：特征拼接（3个 C_e 通道特征 → 96通道）
        concat_feat = torch.cat(mapped_feats, dim=1)  # (batch, 3*C_e, H, W)
        
        # 步骤3：RCAB 抑制非边界信息
        rcab_feat = self.rcab(concat_feat)
        
        # 步骤4：分类器输出边界图（Sigmoid激活确保输出∈[0,1]）
        S_c = torch.sigmoid(self.classifier(rcab_feat))
        
        return S_c
    


def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')

def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1)
    yield nn.GroupNorm(cout, cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')
    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')

def up_conv(cin, cout, up=True):
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')


class RAM(nn.Module):
    def __init__(self, config, feat, tar_feat):
        super(RAM, self).__init__()
        #self.conv2 = nn.Sequential(*list(up_conv(feat[1], tar_feat)))
        #self.conv1 = nn.Sequential(*list(up_conv(feat[0], tar_feat, False)))
        self.gconv = nn.Sequential(*list(up_conv(tar_feat, tar_feat, False)))
        #self.conv0 = nn.Sequential(*list(up_conv(tar_feat * 3, tar_feat, False)))
        self.res_conv1 = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)
        self.res_conv2 = nn.Conv2d(tar_feat, tar_feat, 3, padding=1)

        self.fuse = nn.Conv2d(tar_feat * 3, tar_feat, 3, padding=1)

    def forward(self, xs, glob_x):
        glob_x0 = nn.functional.interpolate(self.gconv(glob_x), size=xs[0].size()[2:], mode=mode)
        
        loc_x1 = xs[0]
        res_x1 = torch.sigmoid(self.res_conv1(loc_x1 - glob_x0)) #
        loc_x2 = nn.functional.interpolate(xs[1], size=xs[0].size()[2:], mode=mode)
        res_x2 = torch.sigmoid(self.res_conv2(loc_x2 - glob_x0)) #
        loc_x = self.fuse(torch.cat([loc_x1 * res_x1, loc_x2 * res_x2, glob_x0], dim=1))
        
        return loc_x, res_x1, res_x2
        '''
        
        loc_x1 = xs[0]
        loc_x2 = nn.functional.interpolate(xs[1], size=xs[0].size()[2:], mode=mode)
        loc_x = self.fuse(torch.cat([loc_x1, loc_x2, glob_x0], dim=1))
        return loc_x, None, None
        '''
class ResidualFeatureEnhancer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 保持通道数不变：3x3卷积（same padding, stride=1）
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(in_channels // reduction, in_channels)  # 归一化不改变shape
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(in_channels // reduction, in_channels)
        
    def forward(self, x):
        residual = x  # 保留原始特征（残差连接的核心）
        # 第一层卷积+归一化+激活
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        # 第二层卷积+归一化
        x = self.conv2(x)
        x = self.norm2(x)
        # 残差融合：增强特征但不改变shape
        x += residual
        return x

class decoder(nn.Module):
    def __init__(self, config, encoder, feat):
        super(decoder, self).__init__()
        self.epm = EPM(in_feat_dims=[feat[0], feat[1], feat[2]], C_e=32)
        #self.adapter = [nn.Sequential(*list(up_conv(feat[i], feat[0], False))).cuda() for i in range(5)]
        self.adapter0 = nn.Sequential(*list(up_conv(feat[0], feat[0], False)))
        self.adapter1 = nn.Sequential(*list(up_conv(feat[1], feat[0], False)))
        self.adapter2 = nn.Sequential(*list(up_conv(feat[2], feat[0], False)))
        self.adapter3 = nn.Sequential(*list(up_conv(feat[3], feat[0], False)))
        self.adapter4 = nn.Sequential(*list(up_conv(feat[4], feat[0], False)))
        # self.aa=ResidualFeatureEnhancer(feat[0])
        #self.gconv = nn.Sequential(*list(up_conv(feat[-1], feat[0], False)))
        #self.gconv = glob_block(config, feat)
        
        self.region = RAM(config, feat[2:4], feat[0])
        self.local = RAM(config, feat[0:2], feat[0])
        
        self.gb_conv = nn.Sequential(*list(local_conv(feat[0], feat[0])))
        
        #self.fuse = nn.Conv2d(feat[0] * 2, 1, 3, padding=1)
        # --------------------------
        # 新增：初始化 EPM 模块
        # 输入为编码器前3个特征块（B1, B2, B3），通道数对应 feat[0], feat[1], feat[2]
        # --------------------------
        
        
    def forward(self, xs, x_size):
        #xs = [self.adapter[i](xs[i]) for i in range(5)]
                # --------------------------
        # 新增：调用 EPM 生成边界图
        # 输入 EPM 的特征块：xs[0] (B1), xs[1] (B2), xs[2] (B3)（适配器处理后）
        # --------------------------
        pred_bdy = self.epm(feat_list=[xs[0], xs[1], xs[2]], x_size=x_size)
        xs[0] = self.adapter0(xs[0])
        xs[1] = self.adapter1(xs[1])
        xs[2] = self.adapter2(xs[2])
        xs[3] = self.adapter3(xs[3])
        xs[4] = self.adapter4(xs[4])
        
        glob_x = xs[4]
        reg_x, r3, r4 = self.region(xs[2:4], glob_x)
        
        glob_x = self.gb_conv(glob_x)
        loc_x, r1, r2 = self.local(xs[0:2], glob_x)

        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode=mode)

        # aaa=self.aa(xs[0])
        pred = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        #pred = self.fuse(torch.cat([loc_x, reg_x], dim=1))
        #print(loc_x.size(), reg_x.size())
        #pred = torch.sum(loc_x + reg_x, dim=1, keepdim=True)
        #pred = (nn.functional.cosine_similarity(loc_x, reg_x, dim=1) + 1) / 2.
        #pred = pred.unsqueeze(1)
        #print(pred.size())
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')
        #print(pred.size())


        OutDict = {}
        #OutDict['atten'] = [r1, r2, r3, r4]
        #OutDict['feat'] = [loc_x, reg_x]
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred
        OutDict['bdy'] = pred_bdy  # 新增：EPM 输出的边界图
        return OutDict
        

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.decoder = decoder(config, encoder, feat)

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        xs = self.encoder(x)
        out = self.decoder(xs, x_size)
        return out
# class Network(nn.Module):
#     def __init__(self, config, encoder, feat):
#         super(Network, self).__init__()

#         self.encoder_1 = encoder
#         self.decoder_1 = decoder(config, encoder, feat)
#         self.encoder_2 = encoder
#         self.decoder_2 = decoder(config, encoder, feat)

#     def forward(self, x, phase='test'):
#         x_size = x.size()[2:]
#         xs = self.encoder_1(x)
#         out1 = self.decoder_1(xs, x_size)
#         xs1 = self.encoder_2(x)
#         out2 = self.decoder_2(xs1, x_size)
#         return out1,out2


