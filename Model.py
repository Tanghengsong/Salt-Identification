import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Network import SegmentationNetwork
import MyResNet
from torchvision import models

class ELU_1(nn.ELU):
    def __init__(self, *args, **kwargs):
        super(ELU_1, self).__init__(*args, **kwargs)
    
    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False,
                              groups=groups, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
     
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
                                            
class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, SE=False):
        super(CenterBlock, self).__init__()
        self.SE = SE
        self.pool = pool
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        if self.pool:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        residual = self.conv_res(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        
        return x
        
class Decoder_v3(nn.Module):
    def __init__(self, in_channels, convT_channels, out_channels, convT_ratio=2, SE=False):
        super(Decoder_v3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.SE = SE
        self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)
        self.conv1 = ConvBn2d(in_channels + convT_channels // convT_ratio, out_channels)
        self.conv2 = ConvBn2d(out_channels, out_channels)
        
        self.conv_res = nn.Conv2d(convT_channels // convT_ratio, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x, skip):
        x = self.convT(x)
        residual = x
        x = torch.cat([x, skip], 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += self.conv_res(residual)
        x = self.relu(x)
        
        return x
                
class UNetResNet34_SE_Hyper_SPP(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34_SE_Hyper_SPP, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = ELU_1(inplace=True)

        self.resnet = MyResNet.resnet34(pretrained=pretrained, activation=self.activation, SE=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.activation,
        )  # 64

        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = CenterBlock(512, 64, pool=False, SE=True)

        self.decoder4 = Decoder_v3(256, 64,  64, convT_ratio=1,  SE=True)
        self.decoder3 = Decoder_v3(128, 64,  64, convT_ratio=1,  SE=True)
        self.decoder2 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)
        self.decoder1 = Decoder_v3(64,  64,  64, convT_ratio=1,  SE=True)

        self.reducer = ConvBn2d(64 * 5, 64, kernel_size=1, padding=0)

        self.ppm = PyramidPoolingModule([(128, 128), (64, 64), (42, 42), (21, 21)], 64 * 5)

        self.logit = nn.Sequential(
            ConvBn2d(64 * 5, 128, kernel_size=3, padding=1),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        #x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        #x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        #x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        c = self.center(e4)  # 8

        d4 = self.decoder4(c, e3)  # 16
        d3 = self.decoder3(d4, e2)  # 32
        d2 = self.decoder2(d3, e1)  # 64
        d1 = self.decoder1(d2, x)   # 128

        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
            F.upsample(c,  scale_factor=16, mode='bilinear', align_corners=False)
            ], 1)
        logit = self.logit(f)
        return logit
    
    
class UNetResNet34(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet34, self).__init__(**kwargs)
        self.resnet = models.resnet34(pretrained=pretrained)
        self.activation = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.activation
        )  # 64


        self.encoder1 = self.resnet.layer1  # 64
        self.encoder2 = self.resnet.layer2  # 128
        self.encoder3 = self.resnet.layer3  # 256
        self.encoder4 = self.resnet.layer4  # 512

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBn2d(512, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(512 + 512, 512, 512, convT_channels=1024)
        self.decoder4 = Decoder(256 + 256, 256, 256, convT_channels=512)
        self.decoder3 = Decoder(128 + 128, 128, 128, convT_channels=256)
        self.decoder2 = Decoder(64 + 64, 64, 64, convT_channels=128)
        self.decoder1 = Decoder(32 + 64, 64, 32, convT_channels=64)

        self.logit = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        # batch_size,C,H,W = x.shape
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
        x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
        x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]

        x = self.conv1(x) # 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64

        e1 = self.encoder1(p)   # 64
        e2 = self.encoder2(e1)  # 32
        e3 = self.encoder3(e2)  # 16
        e4 = self.encoder4(e3)  # 8

        f = self.center(e4)  # 4

        f = self.decoder5(f, e4)  # 8
        f = self.decoder4(f, e3)  # 16
        f = self.decoder3(f, e2)  # 32
        f = self.decoder2(f, e1)  # 64
        f = self.decoder1(f, x)

        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)  # ; print('logit',logit.size())
        return logit   
        
class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))
        
        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))
        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))
        
    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=8, mode='bilinear', align_corners=True)  # 256, 8, 8

        d2 = self.down2_1(x)  # 512, 4, 4
        d3 = self.down3_1(d2)  # 512, 2, 2

        d2 = self.down2_2(d2)  # 256, 4, 4
        d3 = self.down3_2(d3)  # 256, 8, 8

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 4, 4
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        x = self.conv1(x)  # 256, 8, 8
        x = x * d2

        x = x + x_glob

        return x
    
def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
                         nn.BatchNorm2d(output_dim),
                         nn.ELU(True))

class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z
    
class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z 

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output   

class Decoder_Resnet50(nn.Module):
    def __init__(self, in_channels, channels, out_channels, convT_channels=0, convT_ratio=2, SE=False, activation=None):
        super(Decoder_Resnet50, self).__init__()
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

        self.SE = SE
        self.convT_channels = convT_channels
        self.conv1 = ConvBn2d(in_channels, channels,
                              kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels,
                              kernel_size=3, padding=1)
        if convT_channels:
            self.convT = nn.ConvTranspose2d(convT_channels, convT_channels // convT_ratio, kernel_size=2, stride=2)

        if SE:
            self.scSE = SCse(out_channels)

    def forward(self, x, skip):
        if self.convT_channels:
            x = self.convT(x)
        else:
            x = F.upsample(x, scale_factor=2, mode='bilinear',
                           align_corners=True)  # False
        x = torch.cat([x, skip], 1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        if self.SE:
            x = self.scSE(x)

        return x
    
class Decoderv2(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(Decoderv2, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c   
    
class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)            
            
class Res34Unetv4(SegmentationNetwork):
    def __init__(self, pretrained=True, activation='relu', num_classes=1, mask_class=2, **kwargs):
        super(Res34Unetv4, self).__init__(**kwargs)
        
        self.num_classes = num_classes
        self.mask_class = mask_class
        self.conv_first = nn.Conv2d(4, 3,kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.resnet = models.resnet34(True)
        self.activation = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.activation)
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCse(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCse(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCse(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCse(512))
        
        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)
        
        self.center = nn.Sequential(FPAv2(512, 256), nn.MaxPool2d(2, 2))
        
        self.decoder5 = Decoderv2(256, 512, 64)
        self.decoder4 = Decoderv2(64, 256, 64)
        self.decoder3 = Decoderv2(64, 128, 64)
        self.decoder2 = Decoderv2(64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)
        
        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))
        
        
    def forward(self, x, z):
        # x: (batch_size, 3, 128, 128)
        # Z: (batch_size, 1)
        # initial depth dimension
        depth = np.ones_like(np.array(x[:, 0, :, :].cpu()))
        for i in range(len(x)):
            depth[i] = depth[i]*(np.array(z.cpu())[i])
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(1)
        depth = depth.cuda()
        x = torch.cat((x, depth), 1) # 4, 128, 128
        x = self.conv_first(x) # 3, 128, 128
        x = self.conv1(x) # 64, 64, 64
        e2 = self.encoder2(x) # 64, 64, 64
        e3 = self.encoder3(e2)  # 128, 32, 32
        e4 = self.encoder4(e3)  # 256, 16, 16
        e5 = self.encoder5(e4)  # 512, 8, 8
        
        center_512 = self.center_global_pool(e5)
        center_64 = self.center_conv1x1(center_512)  # 64
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten) # 2
        
        f = self.center(e5)  # 256, 4, 4
        
        d5 = self.decoder5(f, e5)  # 64, 8, 8
        d4 = self.decoder4(d5, e4)  # 64, 16, 16
        d3 = self.decoder3(d4, e3)  # 64, 32, 32
        d2 = self.decoder2(d3, e2)  # 64, 64, 64
        d1 = self.decoder1(d2)  # 64, 128, 128

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 128, 128
        f = F.dropout2d(f, p=0.5)
        
        x_no_empty = self.logits_no_empty(f)
        
        hypercol_add_center = torch.cat((
            f,
            F.upsample(center_64, scale_factor=128,mode='bilinear')),1)  # 320+64, 128, 128
        
        x_final = self.logits_final(hypercol_add_center)

        return center_fc, x_no_empty, x_final
            

class UNetResNet50_SE(SegmentationNetwork):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def __init__(self, pretrained=True, activation='relu', **kwargs):
        super(UNetResNet50_SE, self).__init__(**kwargs)
        
        self.conv_first = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.convtranspose = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        self.resnet = models.resnet50(pretrained=pretrained)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )  # 64


        self.encoder1 = nn.Sequential(self.resnet.layer1, SCse(256))# 256
        self.encoder2 = nn.Sequential(self.resnet.layer2, SCse(512))# 512
        self.encoder3 = nn.Sequential(self.resnet.layer3, SCse(1024))# 1024
        self.encoder4 = nn.Sequential(self.resnet.layer4, SCse(2048))# 2048

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBn2d(2048, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(1024, 1024,
                     kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder_Resnet50(512 + 1024*2, 512, 512, convT_channels=1024, SE=True)
        self.decoder4 = Decoder_Resnet50(256 + 512*2, 256, 256, convT_channels=512, SE=True)
        self.decoder3 = Decoder_Resnet50(128 + 256*2, 128, 128, convT_channels=256, SE=True)
        self.decoder2 = Decoder_Resnet50(64 + 128*2, 64, 64, convT_channels=128, SE=True)
        self.decoder1 = Decoder_Resnet50(32 + 64, 64, 32, convT_channels=64, SE=True)

        self.logit = nn.Sequential(nn.Conv2d(224, 64, kernel_size=3, padding=1),
                                   nn.ELU(True),
                                   nn.Conv2d(64, 1, kernel_size=1, bias=False))
        

    def forward(self, x, z):
        # batch_size,C,H,W = x.shape

        depth = np.ones_like(np.array(x[:, 0, :, :].cpu()))
        for i in range(len(x)):
            depth[i] = depth[i]*(np.array(z.cpu())[i])
        depth = torch.from_numpy(depth)
        depth = depth.unsqueeze(1)
        depth = depth.cuda()
        x = torch.cat((x, depth), 1) # 4, 128, 128
        x = self.conv_first(x) # 3, 128, 128
        x = self.convtranspose(x) # 3, 256, 256
        x = self.conv1(x) # 64, 128, 128
        p = F.max_pool2d(x, kernel_size=2, stride=2) # 64, 64, 64

        e1 = self.encoder1(x)   # 256, 64, 64 
        e2 = self.encoder2(e1)  # 512, 32, 32
        e3 = self.encoder3(e2)  # 1024, 16, 16
        e4 = self.encoder4(e3)  # 2048, 8, 8

        f = self.center(e4)  # 1024, 4, 4

        d5 = self.decoder5(f, e4)  # 512, 8, 8
        d4 = self.decoder4(d5, e3)  # 256, 16, 16
        d3 = self.decoder3(d4, e2)  # 128, 32, 32
        d2 = self.decoder2(d3, e1)  # 64, 64, 64
        d1 = self.decoder1(d2, x)   # 32, 128, 128
        
        ff = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True)), 1)  # 224, 128, 128

        # f = F.dropout2d(f, p=0.50)
        logit = self.logit(ff)
        return logit
