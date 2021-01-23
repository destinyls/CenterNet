import torch
from torch import nn
from torch.nn import functional as F

from models.decode import _topk, _nms
from models.utils import _gather_feat, _transpose_and_gather_feat

class DDDhead(nn.Module):
    def __init__(self, opt, input_channels):
        super(DDDhead, self).__init__()
        self.max_objs = 50
        self.max_detection = opt.K
        self.heads = opt.heads
        self.input_channels = input_channels

        for head in sorted(self.heads):
            if head == "hm":
                fc = nn.Sequential(
                    nn.Conv2d(self.input_channels,
                            opt.head_conv,
                            kernel_size=3,
                            padding=1,
                            bias=True),
                    nn.BatchNorm2d(opt.head_conv),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(opt.head_conv,
                              opt.num_classes,
                              kernel_size=1,
                              padding=1 // 2,
                              bias=True)
                )
                fc[-1].bias.data.fill_(-2.19)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(self.input_channels,
                            opt.head_conv,
                            kernel_size=3,
                            padding=1,
                            bias=True),
                    nn.BatchNorm2d(opt.head_conv),
                    nn.ReLU(inplace=True)
                )
                self._fill_fc_weights(fc)
            self.__setattr__(head, fc)

        for head in sorted(self.heads):
            if head == "hm":
                continue
            num_output = self.heads[head]
            fc = nn.Conv2d(
                opt.head_conv, 
                num_output, 
                kernel_size=1, 
                padding=1 // 2, 
                bias=True
            )
            self._fill_fc_weights(fc)
            self.__setattr__("reg_"+head, fc)

        self.reg_3dbox = nn.Conv2d(64, 8, kernel_size=1, padding=1 // 2, bias=True)
        self._fill_fc_weights(self.reg_3dbox)


    def _fill_fc_weights(self, layers):
            for m in layers.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, features, batch=None):
        up_level16, up_level8, up_level4 = features[0], features[1], features[2]
        results = {}
        for head in sorted(self.heads):
            results[head] = self.__getattr__(head)(up_level4)

        if self.training:
            proj_points = batch["ind"]
        if not self.training:
           heat = results['hm'].sigmoid_()
           heat = _nms(heat)
           scores, inds, clses, ys, xs = _topk(heat, K=self.max_detection)
           proj_points = inds
        
        _, _, h, w = up_level4.size()

        print("hw: ", h, w)
        proj_points = torch.clamp(proj_points, 0, w*h-1)

        proj_points_8 = proj_points // 2
        proj_points_16 = proj_points // 4
        # 1/8 [N, K, 256]
        up_level8_pois = _transpose_and_gather_feat(up_level8, proj_points_8)
        # 1/16 [N, K, 256]
        up_level16_pois = _transpose_and_gather_feat(up_level16, proj_points_16)
        up_level_pois = torch.cat((up_level8_pois, up_level16_pois), dim=-1)

        reg_pois = torch.ones([8, 64, 64, 320], device=up_level4.device)
        print("reg_pois: ", reg_pois.shape)
        reg_pois = self.reg_3dbox(reg_pois)
        print("reg_pois: ", reg_pois.shape)

        '''

        for head in sorted(self.heads):
            if head == "hm":
                continue
            reg_pois = _transpose_and_gather_feat(results[head], proj_points)
            # reg_pois = torch.cat((reg_pois, up_level_pois), dim=-1)
            reg_pois = reg_pois.permute(0, 2, 1).contiguous().unsqueeze(-1)
            reg_pois = torch.ones([8, 64, 64, 320], device=up_level4.device)
            print("reg_pois: ", reg_pois.shape)
            reg_pois = self.reg_3dbox(reg_pois)

            # reg_pois = self.__getattr__("reg_"+head)(reg_pois)
            print("reg_pois: ", reg_pois.shape)
            results[head] = reg_pois.permute(0, 2, 1, 3).contiguous().squeeze(-1)
        '''
        return [results]

def build_ddd_head(opt, input_channels=256):
    return DDDhead(opt, input_channels)
