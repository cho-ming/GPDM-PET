import torch
from torch import nn, Tensor
from .common import TemporalEmbedding, LinearAttention, LabelEmbedding
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, n_cls=1, n_filters=16,dim_emb: int = 1024):
        super(GeneratorUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        #
        # self.linear = nn.Sequential(nn.Linear(in_channels * in_channels * in_channels, n_filters * 8 * 9 * 9 * 9),
        #                             nn.ReLU(inplace=True))
        self.embedding1 = TemporalEmbedding(dim_emb, 7)
        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.embedding2 = TemporalEmbedding(dim_emb, n_filters)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.embedding3 = TemporalEmbedding(dim_emb, 2*n_filters)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.embedding4 = TemporalEmbedding(dim_emb, 4*n_filters)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.attention1 = LinearAttention(8*n_filters)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.embedding5 = TemporalEmbedding(dim_emb, 8*n_filters)
        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.embedding6 = TemporalEmbedding(dim_emb, 4 * n_filters)
        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.embedding7 = TemporalEmbedding(dim_emb, 2 * n_filters)
        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:

        x = self.embedding1(x, t)
        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds0 = self.embedding2(ds0,t)
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds1 = self.embedding3(ds1, t)
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        ds2 = self.embedding4(ds2, t)
        x = self.block_4_2_left(self.attention1(self.block_4_1_left(self.pool_3(ds2))))

        x = self.block_3_2_right(self.block_3_1_right(self.embedding5(torch.cat([self.upconv_3(x), ds2], 1), t)))
        x = self.block_2_2_right(self.block_2_1_right(self.embedding6(torch.cat([self.upconv_2(x), ds1], 1), t)))
        x = self.block_1_2_right(self.block_1_1_right(self.embedding7(torch.cat([self.upconv_1(x), ds0], 1), t)))

        x = self.conv1x1(x)
        return x


class ResConvGroupNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        batch1 = nn.BatchNorm2d(out_channels)
        relu1 = nn.LeakyReLU()

        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        batch2 = nn.BatchNorm2d(out_channels)
        relu2 = nn.LeakyReLU()

        layers = [batch1, relu1, conv2, batch2, relu2]

        self.feat = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return x + self.feat(x)


class UNet(nn.Module):
    def __init__(self, dim_emb: int = 1024):
        super().__init__()
        ch = [32, 64, 128, 256, 256, 128, 64, 32]
        self.ch = ch
        # Positional Embedding
        self.embedding1 = TemporalEmbedding(dim_emb, 7)
        # Input is 1x28x28
        self.block1 = ResConvGroupNorm(7, ch[0])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding2 = TemporalEmbedding(dim_emb, ch[0])
        self.block2 = ResConvGroupNorm(ch[0], ch[1])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding3 = TemporalEmbedding(dim_emb, ch[1])
        self.block3 = ResConvGroupNorm(ch[1], ch[2])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding4 = TemporalEmbedding(dim_emb, ch[2])
        self.block4 = ResConvGroupNorm(ch[2], ch[3])
        self.down4 = nn.Conv2d(ch[3], ch[3],4, stride=2, padding=1, bias=False)

        self.embedding5 = TemporalEmbedding(dim_emb, ch[3])
        self.block5 = ResConvGroupNorm(ch[3], ch[4])
        self.attention1 = LinearAttention(ch[4])
        self.up1 = nn.ConvTranspose2d(ch[4], ch[4], 4, stride=2, padding=1, bias=False)

        new_ch = ch[3] + ch[4]
        self.embedding6 = TemporalEmbedding(dim_emb, new_ch)
        self.block6 = ResConvGroupNorm(new_ch, ch[5])
        self.up2 = nn.ConvTranspose2d(ch[5], ch[5],4, stride=2, padding=1, bias=False)

        new_ch = ch[2] + ch[5]
        self.embedding7 = TemporalEmbedding(dim_emb, new_ch)
        self.block7 = ResConvGroupNorm(new_ch, ch[6])
        self.up3 = nn.ConvTranspose2d(ch[6], ch[6], 4, stride=2, padding=1, bias=False)

        new_ch = ch[1] + ch[6]
        self.embedding8 = TemporalEmbedding(dim_emb, new_ch)
        self.block8 = ResConvGroupNorm(new_ch, ch[7])
        self.up4 = nn.ConvTranspose2d(ch[7], ch[7], 4, stride=2, padding=1, bias=False)

        new_ch = ch[0] + ch[7]
        self.embedding9 = TemporalEmbedding(dim_emb, new_ch)
        self.block9 = ResConvGroupNorm(new_ch, 7)
        self.out = nn.Conv2d(7, 7, 1)







        # # Now input is 64x7x7
        # self.embedding3 = TemporalEmbedding(dim_emb, ch[1])
        # self.block3 = ResConvGroupNorm(ch[1], ch[2])
        # self.attention1 = LinearAttention(ch[2])
        # self.up1 = nn.ConvTranspose2d(ch[2], ch[2], 4, stride=2, padding=1, bias=False)
        #
        # # Now input is 64x14x14
        # new_ch = ch[2] + ch[1]
        # self.embedding4 = TemporalEmbedding(dim_emb, new_ch)
        # self.block4 = ResConvGroupNorm(new_ch, ch[3])
        # self.up2 = nn.ConvTranspose2d(ch[3], ch[3], 4, stride=2, padding=1, bias=False)
        #
        # # Now input is 16x28x28
        # new_ch = ch[3] + ch[0]
        # self.embedding5 = TemporalEmbedding(dim_emb, new_ch)
        # self.block5 = ResConvGroupNorm(new_ch, 1)
        # self.out = nn.Conv2d(1, 1, 1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        num = x.shape[0]
        before = x.detach().cpu().numpy()

        x0 = self.embedding1(x, t)
        print(x0.shape)
        x1 = self.block1(x0)  # 128x64x100x100
        print(x1.shape)
        x1 = self.embedding2(x1, t)  # 128x64x100x100
        print(x1.shape)
        x2 = self.block2(self.down1(x1))  # 128x128x50x50
        x2 = self.embedding3(x2, t)
        x3 = self.block3(self.down2(x2))
        x3 = self.embedding4(x3, t)
        x4 = self.block4(self.down3(x3))
        x4 = self.embedding5(x4, t)
        print(x4.shape)
        tt = self.block5(self.down4(x4))
        print(tt.shape)
        print(self.attention1(tt).shape)
        print('))))')
        x5 = self.up1(self.attention1(self.block5(self.down4(x4))))

        x6 = torch.cat([x4, x5], dim=1)
        x6 = self.embedding6(x6, t)
        x7 = self.up2(self.block6(x6))
        x8 = torch.cat([x7,x3],dim=1)
        x8 = self.embedding7(x8, t)
        x9 = self.up3(self.block7(x8))
        x10 = torch.cat([x9,x2],dim=1)
        x10 = self.embedding8(x10, t)
        x11 = self.up4(self.block8(x10))
        print('d',x11.shape)
        x12 = torch.cat([x11,x1],dim=1)
        print(x12.shape)
        x12 = self.embedding9(x12, t)
        out = self.out(self.block9(x12))

        after = out.detach().cpu().numpy()

        # x6 = torch.cat([x5, x1], dim=1)
        # x6 = self.embedding5(x6, t)
        # out = self.out(self.block5(x6))
        #
        #
        #
        # x0 = self.embedding1(x, t) #128x1x100x100
        # x1 = self.block1(x0) #128x64x100x100
        # x1 = self.embedding2(x1, t) #128x64x100x100
        # x2 = self.block2(self.down1(x1)) #128x128x50x50
        # x2 = self.embedding3(x2, t) #128x128x50
        # x3 = self.up1(self.attention1(self.block3(self.down2(x2)))) #128x256x50x50
        #
        # x4 = torch.cat([x2, x3], dim=1)
        # x4 = self.embedding4(x4, t)
        # x5 = self.up2(self.block4(x4))
        # x6 = torch.cat([x5, x1], dim=1)
        # x6 = self.embedding5(x6, t)
        # out = self.out(self.block5(x6))
        return out

class UNet2(nn.Module):
    def __init__(self, dim_emb: int = 1024):
        super().__init__()
        ch = [64, 128, 256, 512, 512, 256, 128, 64]
        self.ch = ch
        # Positional Embedding
        self.embedding1 = TemporalEmbedding(dim_emb, 2)
        # Input is 1x28x28
        self.block1 = ResConvGroupNorm(2, ch[0])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding2 = TemporalEmbedding(dim_emb, ch[0])
        self.block2 = ResConvGroupNorm(ch[0], ch[1])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding3 = TemporalEmbedding(dim_emb, ch[1])
        self.block3 = ResConvGroupNorm(ch[1], ch[2])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding4 = TemporalEmbedding(dim_emb, ch[2])
        self.block4 = ResConvGroupNorm(ch[2], ch[3])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, stride=2, padding=1, bias=False)

        self.embedding5 = TemporalEmbedding(dim_emb, ch[3])
        self.block5 = ResConvGroupNorm(ch[3], ch[4])
        self.attention1 = LinearAttention(ch[4])
        self.up1 = nn.ConvTranspose2d(ch[4], ch[4], 4, stride=2, padding=1, bias=False)

        new_ch = ch[3] + ch[4]
        self.embedding6 = TemporalEmbedding(dim_emb, new_ch)
        self.block6 = ResConvGroupNorm(new_ch, ch[5])
        self.up2 = nn.ConvTranspose2d(ch[5], ch[5], 4, stride=2, padding=1, bias=False)

        new_ch = ch[2] + ch[5]
        self.embedding7 = TemporalEmbedding(dim_emb, new_ch)
        self.block7 = ResConvGroupNorm(new_ch, ch[6])
        self.up3 = nn.ConvTranspose2d(ch[6], ch[6], 4, stride=2, padding=1, bias=False)

        new_ch = ch[1] + ch[6]
        self.embedding8 = TemporalEmbedding(dim_emb, new_ch)
        self.block8 = ResConvGroupNorm(new_ch, ch[7])
        self.up4 = nn.ConvTranspose2d(ch[7], ch[7], 4, stride=2, padding=1, bias=False)

        new_ch = ch[0] + ch[7]
        self.embedding9 = TemporalEmbedding(dim_emb, new_ch)
        self.block9 = ResConvGroupNorm(new_ch, 1)
        self.out = nn.Conv2d(1, 1, 1)

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        x00 = torch.cat([x,y],dim=1)
        x0 = self.embedding1(x, t)
        x1 = self.block1(x0)  # 128x64x100x100
        x1 = self.embedding2(x1, t)  # 128x64x100x100
        x2 = self.block2(self.down1(x1))  # 128x128x50x50
        x2 = self.embedding3(x2, t)
        x3 = self.block3(self.down2(x2))
        x3 = self.embedding4(x3, t)
        x4 = self.block4(self.down3(x3))
        x4 = self.embedding5(x4, t)
        x5 = self.up1(self.attention1(self.block5(self.down4(x4))))

        x6 = torch.cat([x4, x5], dim=1)
        x6 = self.embedding6(x6, t)
        x7 = self.up2(self.block6(x6))
        x8 = torch.cat([x7,x3],dim=1)
        x8 = self.embedding7(x8, t)
        x9 = self.up3(self.block7(x8))
        x10 = torch.cat([x9,x2],dim=1)
        x10 = self.embedding8(x10, t)
        x11 = self.up4(self.block8(x10))
        x12 = torch.cat([x11,x1],dim=1)
        x12 = self.embedding9(x12, t)
        out = self.out(self.block9(x12))
        return out

class UNet3(nn.Module):
    def __init__(self, dim_emb: int = 1024):
        super().__init__()
        ch = [64, 128, 256, 512, 512, 256, 128, 64]
        self.ch = ch
        # Positional Embedding
        self.embedding1 = TemporalEmbedding(dim_emb, 1)
        # Input is 1x28x28
        self.block1 = ResConvGroupNorm(1, ch[0])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1, bias=False)
        self.down11 = nn.Conv2d(ch[0], ch[0], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding2 = TemporalEmbedding(dim_emb, ch[0])
        self.block2 = ResConvGroupNorm(ch[0], ch[1])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1, bias=False)
        self.down22 = nn.Conv2d(ch[1], ch[1], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding3 = TemporalEmbedding(dim_emb, ch[1])
        self.block3 = ResConvGroupNorm(ch[1], ch[2])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, stride=2, padding=1, bias=False)
        self.down33 = nn.Conv2d(ch[2], ch[2], 4, stride=2, padding=1, bias=False)

        # Now input is 32x14x14
        self.embedding4 = TemporalEmbedding(dim_emb, ch[2])
        self.block4 = ResConvGroupNorm(ch[2], ch[3])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, stride=2, padding=1, bias=False)
        self.down44 = nn.Conv2d(ch[3], ch[3], 4, stride=2, padding=1, bias=False)

        self.embedding5 = TemporalEmbedding(dim_emb, ch[3])
        self.block5 = ResConvGroupNorm(ch[3], ch[4])
        self.attention1 = LinearAttention(ch[4])
        self.up1 = nn.ConvTranspose2d(ch[4], ch[4], 4, stride=2, padding=1, bias=False)

        new_ch = ch[3] + ch[4]
        self.embedding6 = TemporalEmbedding(dim_emb, new_ch)
        self.block6 = ResConvGroupNorm(new_ch, ch[5])
        self.up2 = nn.ConvTranspose2d(ch[5], ch[5], 4, stride=2, padding=1, bias=False)

        new_ch = ch[2] + ch[5]
        self.embedding7 = TemporalEmbedding(dim_emb, new_ch)
        self.block7 = ResConvGroupNorm(new_ch, ch[6])
        self.up3 = nn.ConvTranspose2d(ch[6], ch[6], 4, stride=2, padding=1, bias=False)

        new_ch = ch[1] + ch[6]
        self.embedding8 = TemporalEmbedding(dim_emb, new_ch)
        self.block8 = ResConvGroupNorm(new_ch, ch[7])
        self.up4 = nn.ConvTranspose2d(ch[7], ch[7], 4, stride=2, padding=1, bias=False)

        new_ch = ch[0] + ch[7]
        self.embedding9 = TemporalEmbedding(dim_emb, new_ch)
        self.block9 = ResConvGroupNorm(new_ch, 1)
        self.out = nn.Conv2d(1, 1, 1)

    def forward(self, x: Tensor,y: Tensor, t: Tensor) -> Tensor:
        x0 = self.embedding1(x, t)
        x1 = self.block1(x0)  # 128x64x100x100
        x11 = self.block11(y)
        x1 = self.embedding2(x1, t)  # 128x64x100x100
        x2 = self.block2(self.down1(x1))  # 128x128x50x50
        x2 = self.embedding3(x2, t)
        x3 = self.block3(self.down2(x2))
        x3 = self.embedding4(x3, t)
        x4 = self.block4(self.down3(x3))
        x4 = self.embedding5(x4, t)
        x5 = self.up1(self.attention1(self.block5(self.down4(x4))))

        x6 = torch.cat([x4, x5], dim=1)
        x6 = self.embedding6(x6, t)
        x7 = self.up2(self.block6(x6))
        x8 = torch.cat([x7,x3],dim=1)
        x8 = self.embedding7(x8, t)
        x9 = self.up3(self.block7(x8))
        x10 = torch.cat([x9,x2],dim=1)
        x10 = self.embedding8(x10, t)
        x11 = self.up4(self.block8(x10))
        x12 = torch.cat([x11,x1],dim=1)
        x12 = self.embedding9(x12, t)
        out = self.out(self.block9(x12))

        after = out.detach().cpu().numpy()
        return out


class ConditionalUNet(UNet):
    def __init__(self, dim_emb: int = 1024):
        super().__init__(dim_emb)
        self.label_emb1 = LabelEmbedding(dim_emb, self.ch[0])
        self.label_emb2 = LabelEmbedding(dim_emb, self.ch[1])
        self.label_emb3 = LabelEmbedding(dim_emb, self.ch[2])
        self.label_emb4 = LabelEmbedding(dim_emb, self.ch[3])
        self.label_emb5 = LabelEmbedding(dim_emb, self.ch[3] + self.ch[0])

    def forward(self, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
        x0 = self.embedding1(x, t)
        x1 = self.block1(x0)
        x1 = self.label_emb1(x1, label)
        x1 = self.embedding2(x1, t)
        x2 = self.block2(self.down1(x1))
        x2 = self.label_emb2(x2, label)
        x2 = self.embedding3(x2, t)
        crossed = self.label_emb3(self.block3(self.down2(x2)), label)
        x3 = self.up1(self.attention1(crossed))
        x4 = torch.cat([x2, x3], dim=1)
        x4 = self.embedding4(x4, t)
        x5 = self.up2(self.label_emb4(self.block4(x4), label))
        x6 = torch.cat([x5, x1], dim=1)
        x6 = self.label_emb5(x6, label)
        x6 = self.embedding5(x6, t)
        out = self.out(self.block5(x6))
        return out


class UNet_Gan(nn.Module):
    def __init__(self, in_channels=7, n_cls=7, n_filters=32):
        super(UNet_Gan, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters
        #
        # self.linear = nn.Sequential(nn.Linear(in_channels * in_channels * in_channels, n_filters * 8 * 9 * 9 * 9),
        #                             nn.ReLU(inplace=True))

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose2d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose2d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv2d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        ds0 = self.block_1_2_left(self.block_1_1_left(x))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        x = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1))

        x = self.conv1x1(x)
        return x
        # if self.n_cls == 1:
        #     return torch.sigmoid(x)
        # else:
        #     return F.softmax(x, dim=1)



if __name__ == '__main__':
    mode = UNet().cuda()
    intt = torch.randn((1,7,192,192)).cuda()
    t = torch.randint(0, 1000, (intt.shape[0],)).cuda()
    out = mode(intt,t)
    print(out.shape)