import torch
import torch.nn as nn
from torchinfo import summary

class attention_mask(nn.Module):
    def __init__(self, in_channel = 32):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.seq(x)
        # B, 1, 36, 36
        xsum = torch.sum(torch.abs(x.view(-1)))
        xshape = tuple(x.size()) # 1, 32, 36, 36
        return x / xsum * xshape[2] * xshape[3] * 0.5

class TSM(nn.Module):
    def __init__(self, frame_depth=10, fold_num = 3):
        super().__init__()
        self.frame_depth = frame_depth
        self.fold_num = fold_num
    
    def forward(self, x):
        """
        Args:
            x (tensor): shape is F x C x H x W
        """
        f, c, h, w = x.size()
        n_fold = c // self.fold_num
        segment_num = f // self.frame_depth
        x = x.view(segment_num, self.frame_depth, c, h, w)
        out = torch.zeros_like(x)
        out[:, 1:, :n_fold] = x[:, :-1, :n_fold] # down shift
        out[:, :-1, n_fold : 2 * n_fold] = x[:, 1:, n_fold: 2 * n_fold] # up shift
        out[:, :, 2 * n_fold: ] = x[:, :, 2 * n_fold: ]
        return out.view(-1, c, h, w)
    
class TSCAN(nn.Module):
    def __init__(self, out_channel = [32, 64], dropout_p = [0.25, 0.5], frame_depth = 20):
        super().__init__()
        self.tsm_list = nn.ModuleList([TSM(frame_depth) for _ in range(4)])
        self.motionBranch1 = nn.Sequential(
            self.tsm_list[0], # 
            nn.Conv2d(3, out_channel[0], kernel_size=3, padding=1),
            nn.Tanh(),
            self.tsm_list[1],
            nn.Conv2d(out_channel[0], out_channel[0], kernel_size=3, padding=1),
            nn.Tanh()
        )
        # the input is only one averaged image
        self.appearanceBranch1 = nn.Sequential(
            nn.Conv2d(3, out_channel[0], kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(out_channel[0], out_channel[0], kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.attentionMask = nn.ModuleList(attention_mask(out_channel[i]) for i in range(2))
        self.motionBranch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,2)),
            nn.Dropout(dropout_p[0]),
            self.tsm_list[2],
            nn.Conv2d(out_channel[0], out_channel[0], kernel_size=3, padding=1),
            nn.Tanh(),
            self.tsm_list[3],
            nn.Conv2d(out_channel[0], out_channel[1], kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.appearanceBranch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2,2)),
            nn.Dropout(dropout_p[0]),
            nn.Conv2d(out_channel[0], out_channel[0], kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(out_channel[0], out_channel[1], kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.avg = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(dropout_p[0]),
            nn.Flatten(),
        )
        self.output_bvp = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),
            nn.Tanh(),
            nn.Dropout(dropout_p[1]),
            nn.Linear(128, 1)
        )
        
        self.output_r = nn.Sequential(
            nn.Linear(64 * 9 * 9, 128),
            nn.Tanh(),
            nn.Dropout(dropout_p[1]),
            nn.Linear(128, 1)
        )
        
    def forward(self, raw_data, diff_data):
        diff_data = self.motionBranch1(diff_data)
        raw_data = self.appearanceBranch1(raw_data)
        mask = self.attentionMask[0](raw_data)
        diff_data = diff_data * mask
        diff_data = self.motionBranch2(diff_data)
        raw_data = self.appearanceBranch2(raw_data)
        mask = self.attentionMask[1](raw_data)
        diff_data = diff_data * mask
        diff_data = self.avg(diff_data)
        out_bvp = self.output_bvp(diff_data)
        out_r = self.output_r(diff_data)
        return out_bvp, out_r
  
tensor = torch.rand([100, 3, 36, 36])
model = TSCAN()
# print(model(tensor, tensor)[0].shape)
summary(model, input_size=[(100, 3, 36, 36), (100, 3, 36, 36)])


