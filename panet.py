import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models.resnet import resnet101

class PAN(nn.Module):
    def __init__(self, pretrained=True, fixed_feature_weights=True):
        super(PAN, self).__init__()

        resnet = resnet101(pretrained=pretrained)

        # Freeze those weights
        if fixed_feature_weights:
            for p in resnet.parameters():
                p.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Downsample layers
        self.downlayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downlayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downlayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Smooth layers
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Smooth layers
        self.smooth7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # depth prediction layer
        self.depth = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.smooth2(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.smooth3(p2)

        # Bottom-up Path Augmentation (for instance segmentation, there is ReLU after every conv)
        n2 = p2
        n3 = p3 + self.downlayer1(n2)
        n3 = self.smooth4(n3)
        n4 = p4 + self.downlayer1(n3)
        n4 = self.smooth5(n4)
        n5 = p5 + self.downlayer1(n4)
        n5 = self.smooth6(n5)

        # Top-down merge again (Double check this design)
        m5 = n5
        m4 = self._upsample_add(m5, n4)
        m4 = self.smooth7(m4)
        m3 = self._upsample_add(m4, n3)
        m3 = self.smooth8(m3)
        m2 = self._upsample_add(m3, n2)
        m2 = self.smooth9(m2)

        depth = self.depth(m2)

        return F.sigmoid(depth)

def test():
    net = PAN()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())

test()