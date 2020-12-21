import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#Feature Veränderungen in self.ce_loss, self.layer2 und self.fc_input_feature anpassen

class SCNN(nn.Module):
    def __init__(self, input_size, ms_ks=9, pretrained=True):
        """
        Argument
            ms_ks: kernel size in message passing conv
        """
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.net_init(input_size, ms_ks)
        if not pretrained:
            self.weight_init()

        self.scale_background = 0.4
        self.scale_seg = 1.0
        self.scale_exist = 0.1

        #self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, self.scale_background])) #For multi channel label
        self.ce_loss = nn.BCEWithLogitsLoss() #For single channel label
        #self.ce_loss = nn.BCELoss(weight=torch.tensor([1, 1, self.scale_background]))

    def forward(self, img, seg_gt=None, exist_gt=None):
        x = self.backbone(img)
        #print(x.shape)
        x = self.layer1(x)
        x = self.message_passing_forward(x)
        x = self.layer2(x)
        exist_pred = x
        #print(x.shape)
        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        #print(seg_pred.shape)
        #x = self.layer3(x)
        #x = x.view(-1, self.fc_input_feature)
        #exist_pred = self.fc(x)

        """
        if seg_gt is not None and exist_gt is not None:
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist
        else:
            loss_seg = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
            loss = torch.tensor(0, dtype=img.dtype, device=img.device)
        """

        #target = torch.argmax(seg_gt, dim=1)
        #print(target.shape, target.dim())
        #print(seg_gt.shape) #target
        #print(target.view(target.size(0), -1).unique(dim=1))
        #print(seg_pred.shape, seg_gt.shape)

        #seg_pred = seg_pred.squeeze()
        #seg_gt = seg_gt.unsqueeze(1)
        #seg_pred = torch.argmax(seg_pred, 1)
        #seg_gt = seg_gt.type_as(seg_pred)
        #seg_pred = seg_pred.type_as(seg_gt)
        #seg_gt = seg_gt.type(torch.FloatTensor)
        #seg_pred = seg_pred.type(torch.FloatTensor)
        print(seg_gt.shape, seg_pred.shape)
        print(torch.max(seg_pred[0]), torch.min(seg_pred[0]))
        #print(seg_pred[0])
        seg_gt = seg_gt.type_as(seg_pred)
        loss_seg = self.ce_loss(seg_pred, seg_gt) #seg_gt.shape = [4, 288, 512]
        loss_exist = torch.tensor(0, dtype=img.dtype, device=img.device)
        loss = loss_seg * self.scale_seg #+ loss_exist * self.scale_exist

        return seg_pred, exist_pred, loss_seg, loss_exist #, loss

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def net_init(self, input_size, ms_ks):
        input_w, input_h = input_size
        self.fc_input_feature = 3 * int(input_w/16) * int(input_h/16)
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features

        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False))
        self.message_passing.add_module('left_right', nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left', nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 3, 1)  # get (nB, 5, 36, 100) #änderung von 5 auf 3 für 3 klassen
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = 1.
                m.bias.data.zero_()

def main():
    scnn = SCNN((512,1024))
    
if __name__ == "__main__":
    main()