import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TEstimationModule(nn.Module):
    """t-estimation 模块，处理 Dark Channel Prior、Bright Channel Prior 和 Y，输出 t(x)。"""
    def __init__(self, channels=32):
        super(TEstimationModule, self).__init__()
        self.channels = channels

        # 定义五个卷积块
        self.block1 = self._make_block(3, self.channels)  # 输入3通道（Dark, Bright, Y）
        self.block2 = self._make_block(self.channels, self.channels)
        self.block3 = self._make_block(self.channels, self.channels)
        self.block4 = self._make_block(self.channels, self.channels)
        self.block5 = self._make_block(self.channels, 3)  # 输出1通道的 t(x)

        self.sigmoid = nn.Sigmoid()

    def _make_block(self, in_channels, out_channels):
        """创建单个卷积块：两层卷积 + ReLU 激活。"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dark_channel, bright_channel, y):
        """前向传播：合并输入并通过五个卷积块处理。"""
        # 输入形状：[B, 1, H, W] -> 合并为 [B, 3, H, W]
        inputs = torch.cat([dark_channel, bright_channel, y], dim=1)
        x = self.block1(inputs)
        x = self.block2(x)+x
        x = self.block3(x)+x
        x = self.block4(x)+x
        t_x = self.block5(x)+inputs  # 输出形状：[B, 3, H, W]
        t_x = self.sigmoid(t_x)
        return t_x

class LightFieldProcessor(nn.Module):
    """光场图像处理器，处理输入 lr 并生成 t(x)。"""
    def __init__(self):
        super(LightFieldProcessor, self).__init__()
        self.t_estimation = TEstimationModule()

    def extract_priors(self, lr):
        """从光场图像 lr 中提取 Dark Channel Prior、Bright Channel Prior 和 Y。"""
        # lr 形状：[B, U, V, C, H, W]
        # 重排为 [B, C, U*H, V*W] 以便计算先验
        lr_reshaped = rearrange(lr, 'b u v c h w -> b c (u h) (v w)', u=5, v=5)

        # Dark Channel Prior：所有通道中的最小值
        dark_channel = torch.min(lr_reshaped, dim=1, keepdim=True)[0]  # [B, 1, U*H, V*W]

        # Bright Channel Prior：所有通道中的最大值
        bright_channel = torch.max(lr_reshaped, dim=1, keepdim=True)[0]  # [B, 1, U*H, V*W]

        # Y：RGB 通道的平均值作为亮度通道
        y = torch.mean(lr_reshaped, dim=1, keepdim=True)  # [B, 1, U*H, V*W]

        return dark_channel, bright_channel, y

    def forward(self, lr):
        """前向传播：提取先验并通过 t-estimation 模块处理。"""
        # 提取先验
        [b, u, v, c, h, w] = lr.size()
        dark_channel, bright_channel, y = self.extract_priors(lr)

        # 通过 t-estimation 模块处理
        t_x = self.t_estimation(dark_channel, bright_channel, y)

        # 重排输出为 [B, U, V, 1, H, W]
        t_x = rearrange(t_x, 'b c (u h) (v w) -> b u v c h w', u=u, v=v, h=h, w=w)

        return t_x

# 测试代码
if __name__ == "__main__":
    # 创建模型实例
    model = LightFieldProcessor()

    # 生成随机输入张量：[B=1, U=5, V=5, C=3, H=64, W=64]
    lr = torch.randn(1, 5, 5, 3, 64, 64)

    # 前向传播
    t_x = model(lr)

    # 输出形状验证
    print("Output shape:", t_x.shape)  # 预期输出：[1, 5, 5, 1, 64, 64]