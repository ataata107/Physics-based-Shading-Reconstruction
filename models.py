from functools import partial
import torch
import torch.nn as nn
from networks import ContextualAttention

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

def conv_bn(in_channels, out_channels, conv, activation, *args, **kwargs):
    return nn.Sequential(nn.BatchNorm2d(in_channels), activation_func(activation),conv(in_channels, out_channels, *args, **kwargs))


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =   self.dilation # (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        # x = self.activate(x) Removed activation for now as not mentioned in paper
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels,dilation=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.dilation = dilation
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, activation = self.activation, bias=False, stride=self.downsampling, dilation = self.dilation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, activation = self.activation, bias=False, dilation = self.dilation)
        )

class EncoderBlockRGB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,kernel_size=3, stride=2,padding=1),
            ResNetBasicBlock(self.out_channels,self.out_channels),
            ResNetBasicBlock(self.out_channels,self.out_channels),
            ResNetBasicBlock(self.out_channels,self.out_channels),
            ResNetBasicBlock(self.out_channels,self.out_channels)
        )
    def forward(self, x):
        x = self.block(x)
        return x


class EncoderBlockShadingF(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride=2,padding =1),
            ResNetBasicBlock(self.out_channels,self.out_channels)
        )
    def forward(self,x):
        x = self.block(x)
        return x


class EncoderBlockShadingL(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride=2,padding=1),
            ResNetBasicBlock(self.out_channels,self.out_channels,dilation=1),
            ResNetBasicBlock(self.out_channels,self.out_channels,dilation=1),
            ResNetBasicBlock(self.out_channels,self.out_channels,dilation=1),
            ResNetBasicBlock(self.out_channels,self.out_channels)
        )
    def forward(self,x):
        x = self.block(x)
        return x

class EncoderRGB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.block = nn.Sequential(
        #     EncoderBlockRGB(self.in_channels,64),
        #     EncoderBlockRGB(64,128),
        #     EncoderBlockRGB(128,128),
        #     EncoderBlockRGB(128,256)
        # )
        self.encoderRGB_1 = EncoderBlockRGB(self.in_channels,64)
        self.encoderRGB_2 = EncoderBlockRGB(64,128)
        self.encoderRGB_3 = EncoderBlockRGB(128,128)
        self.encoderRGB_4 = EncoderBlockRGB(128,256)

    def forward(self, x):
        x_1 = self.encoderRGB_1(x)
        x_2 = self.encoderRGB_2(x_1)
        x_3 = self.encoderRGB_3(x_2)
        x_4 = self.encoderRGB_4(x_3)
        return x_1,x_2,x_3,x_4


class EncoderShading(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels= in_channels
        # self.block = nn.Sequential(
        #     EncoderBlockShadingF(self.in_channels,32),
        #     EncoderBlockShadingF(32,32),
        #     EncoderBlockShadingF(32,32),
        #     EncoderBlockShadingL(32,32)
        # )
        self.encoderShading_1 = EncoderBlockShadingF(self.in_channels,32)
        self.encoderShading_2 = EncoderBlockShadingF(32,32)
        self.encoderShading_3 = EncoderBlockShadingF(32,32)
        self.encoderShading_4 = EncoderBlockShadingL(32,32)
    def forward(self, x):
        x_1 = self.encoderShading_1(x)
        x_2 = self.encoderShading_2(x_1)
        x_3 = self.encoderShading_3(x_2)
        x_4 = self.encoderShading_4(x_3)
        return x_1,x_2,x_3,x_4

class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_channels= in_channels
        # self.encoderRGB = EncoderRGB(self.in_channels)
        # self.encoderShading = EncoderShading(self.in_channels)
        self.attentionModule = ContextualAttention()
        self.conv2d1 = nn.Conv2d(256,32,kernel_size=1)
        self.conv2d2 = nn.Conv2d(288,64,kernel_size=1)
    def forward(self,rgb_features,shading_features):
        # rgb_features = self.encoderRGB(img)
        # shading_features = self.encoderShading(img)
        concat_features = torch.cat((rgb_features,shading_features),1)
        rgb_features_short = self.conv2d1(rgb_features)
        concat_features = self.conv2d2(concat_features)
        attention_features,_ = self.attentionModule(shading_features, rgb_features_short)
        attention_features = torch.cat((attention_features,concat_features),1)
        return attention_features
    
class ConvBlockDecoder(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.out_channels), 
            activation_func("leaky_relu")
            )
        self.upsample  = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self,x,counterpart):
        x = self.block(x)
        x = self.upsample(x)
        x = torch.cat((x,counterpart),dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels,out_channels,channel_1,channel_2):
        super().__init__()
        self.in_channels= in_channels
        self.out_channels = out_channels
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.decoder_1 = ConvBlockDecoder(self.in_channels,128)
        self.decoder_2 = ConvBlockDecoder(self.channel_1,128)
        self.decoder_3 = ConvBlockDecoder(self.channel_1,128)
        self.block = nn.Sequential(
            nn.Conv2d(self.channel_2,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128), 
            activation_func("leaky_relu"),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128,self.out_channels,kernel_size=3,stride=1,padding=1)
            )
    def forward(self,x4,x3,x2,x1):
        x = self.decoder_1(x4,x3)
        x = self.decoder_2(x,x2)
        x = self.decoder_3(x,x1)
        x = self.block(x)
        return x

class FinalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoderRGB = EncoderRGB(3)
        self.encoderShading = EncoderShading(1)
        self.fusionModule = FusionModule()
        self.DecoderRGB = Decoder(256,3,256,192)
        self.DecoderShading =Decoder(96,1,160,160)
    def forward(self,img, shade):
        x1,x2,x3,x4 = self.encoderRGB(img)
        y1,y2,y3,y4 = self.encoderShading(shade)
        z4 = self.fusionModule(x4,y4)
        albedo = self.DecoderRGB(x4,x3,x2,x1)
        shading = self.DecoderShading(z4,y3,y2,y1)
        return albedo,shading

# conv = conv3x3(in_channels=32, out_channels=64)
# print(conv)

# dummy = torch.ones((1, 3, 256, 256))
# EncoderRGB(3)(dummy).shape
# EncoderShading(3)(dummy).shape
# block = ResNetBasicBlock(32, 64)
# block(dummy).shape
# print(block)