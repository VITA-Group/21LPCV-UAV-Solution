# YOLOv5 quantization module, specificly designed for quantization aware training
from .common import *
from .yolo import *
import sys
sys.path.append("..")
from utils.autoanchor import check_anchor_order

# This module is designed for the submitted version, for original YOLOv5, uncomment all lines
class YoloV5_quanted(nn.Module):

    def __init__(self, num_cls=2, ch=3, anchors=None):
        super(YoloV5_quanted, self).__init__()
        assert anchors != None, 'anchor must be provided'

        # divid by
        cd = 4
        wd = 3

        ch_ = [512// cd, 512 // cd, 1024 // cd]
        # ch_ = [256// cd, 512 // cd, 1024 // cd]
        out_ch_ = [ch_[i] for i in range(len(anchors))]

        self.focus = Focus(ch, 64 // cd, k=3, act=get_activation("Hardswish"))
        self.conv1 = Conv(64 // cd, 128 // cd, 3, 2, act=get_activation("Hardswish"))
        self.csp1 = C3(128 // cd, 128 // cd, n=3 // wd, act=get_activation("Hardswish"))
        self.conv2 = Conv(128 // cd, 256 // cd, 3, 2, act=get_activation("Hardswish"))
        self.csp2 = C3(256 // cd, 256 // cd, n=3 // wd, act=get_activation("Hardswish"))
        self.conv3 = Conv(256 // cd, 512 // cd, 3, 2, act=get_activation("Hardswish"))
        self.csp3 = C3(512 // cd, 512 // cd, n=3 // wd, act=get_activation("Hardswish"))
        self.conv4 = Conv(512 // cd, 1024 // cd, 3, 2, act=get_activation("Hardswish"))
        self.spp = SPP(1024 // cd, 1024 // cd)
        self.csp4 = C3(1024 // cd, 1024 // cd, n=3 // wd, shortcut=False, act=get_activation("Hardswish"))

        # PANet
        self.conv5 = Conv(1024 // cd, 512 // cd, act=get_activation("Hardswish"))
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = C3(1024 // cd, 512 // cd, n=3 // wd, shortcut=False, act=get_activation("Hardswish"))


        """
        self.conv6 = Conv(512 // cd, 256 // cd, act=get_activation("Hardswish"))
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = C3(512 // cd, 256 // cd, n=3 // wd, shortcut=False, act=get_activation("Hardswish"))

        self.conv7 = Conv(256 // cd, 256 // cd, 3, 2, act=get_activation("Hardswish"))
        self.csp7 = C3(512 // cd, 512 // cd, n=3 // wd, shortcut=False, act=get_activation("Hardswish"))

        self.conv8 = Conv(512 // cd, 512 // cd, 3, 2, act=get_activation("Hardswish"))
        self.csp8 = C3(1024 // cd, 1024 // cd, n=3 // wd, shortcut=False, act=get_activation("Hardswish"))
        """


        self.head = Detect_Q(nc = 2, anchors = anchors, ch = out_ch_)  # Change to 21
        self.head.anchors /= self.head.stride.view(-1, 1, 1)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.torch_cat = torch.nn.quantized.FloatFunctional()

    def _build_backbone(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x = self.conv2(x)
        x_p3 = self.csp2(x)   # P3
        x = self.conv3(x_p3)
        x_p4 = self.csp3(x)   # P4
        x = self.conv4(x_p4)
        x = self.spp(x)
        x = self.csp4(x)

        return x_p3, x_p4, x

    def _build_neck(self, p3, p4, feas):
        h_p5 = self.conv5(feas)  # head P5
        x = self.up1(h_p5)
        x_concat = self.torch_cat.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        """
        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = self.torch_cat.cat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = self.torch_cat.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        # print(x_concat.shape)
        x_large = self.csp8(x_concat)
        """



        return [x]
        # return [x_small, x_medium, x_large]

    def fuse_conv(self, model):
        torch.quantization.fuse_modules(model, ['conv', 'bn'], inplace=True)

    def fuse_model(self):
        self.fuse_conv(self.focus.conv)
        self.fuse_conv(self.conv1)
        self.fuse_conv(self.csp1.cv1)
        self.fuse_conv(self.csp1.cv2)
        self.fuse_conv(self.csp1.cv3)
        self.fuse_conv(self.csp1.m[0].cv1)
        self.fuse_conv(self.csp1.m[0].cv2)
        self.fuse_conv(self.conv2)
        self.fuse_conv(self.csp2.cv1)
        self.fuse_conv(self.csp2.cv2)
        self.fuse_conv(self.csp2.cv3)
        self.fuse_conv(self.csp2.m[0].cv1)
        self.fuse_conv(self.csp2.m[0].cv2)
        self.fuse_conv(self.conv3)
        self.fuse_conv(self.csp3.cv1)
        self.fuse_conv(self.csp3.cv2)
        self.fuse_conv(self.csp3.cv3)
        self.fuse_conv(self.csp3.m[0].cv1)
        self.fuse_conv(self.csp3.m[0].cv2)
        self.fuse_conv(self.conv4)
        self.fuse_conv(self.spp.cv1)
        self.fuse_conv(self.spp.cv2)
        self.fuse_conv(self.csp4.cv1)
        self.fuse_conv(self.csp4.cv2)
        self.fuse_conv(self.csp4.cv3)
        self.fuse_conv(self.csp4.m[0].cv1)
        self.fuse_conv(self.csp4.m[0].cv2)
        self.fuse_conv(self.conv5)
        self.fuse_conv(self.csp5.cv1)
        self.fuse_conv(self.csp5.cv2)
        self.fuse_conv(self.csp5.cv3)
        self.fuse_conv(self.csp5.m[0].cv1)
        self.fuse_conv(self.csp5.m[0].cv2)

        """
        self.fuse_conv(self.conv6)
        self.fuse_conv(self.csp6.cv1)
        self.fuse_conv(self.csp6.cv2)
        self.fuse_conv(self.csp6.cv3)
        self.fuse_conv(self.csp6.m[0].cv1)
        # self.fuse_conv(self.csp6.m[0].cv2)
        self.fuse_conv(self.csp6.m[0].cv2.cv1)
        self.fuse_conv(self.csp6.m[0].cv2.cv2)

        self.fuse_conv(self.conv7)
        # self.fuse_conv(self.conv7.dconv)
        # self.fuse_conv(self.conv7.pconv)
        self.fuse_conv(self.csp7.cv1)
        self.fuse_conv(self.csp7.cv2)
        self.fuse_conv(self.csp7.cv3)
        self.fuse_conv(self.csp7.m[0].cv1)
        self.fuse_conv(self.csp7.m[0].cv2)
        # self.fuse_conv(self.csp7.m[0].cv2.dconv)
        # self.fuse_conv(self.csp7.m[0].cv2.pconv)
        self.fuse_conv(self.conv8)
        # self.fuse_conv(self.conv8.dconv)
        # self.fuse_conv(self.conv8.pconv)
        self.fuse_conv(self.csp8.cv1)
        self.fuse_conv(self.csp8.cv2)
        self.fuse_conv(self.csp8.cv3)
        self.fuse_conv(self.csp8.m[0].cv1)
        self.fuse_conv(self.csp8.m[0].cv2)
        # self.fuse_conv(self.csp8.m[0].cv2.dconv)
        # self.fuse_conv(self.csp8.m[0].cv2.pconv)
        """

    def forward(self, x, augment=False):

        x = self.quant(x)
        p3, p4, feas = self._build_backbone(x)

        neck_out = self._build_neck(p3, p4, feas)
        res = self.head(neck_out)

        return res
