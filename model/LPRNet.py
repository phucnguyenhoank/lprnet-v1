import torch
import torch.nn as nn

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        q_ch_out = ch_out // 4
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, q_ch_out, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(q_ch_out, q_ch_out, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(q_ch_out, q_ch_out, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(q_ch_out, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

# Create model
class LPRNet(nn.Module):
    def __init__(self, phase, lpr_max_len, class_num, dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128), # [-1, 128, 20, 90]
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(), # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)), # [-1, 64, 18, 44], use max pool 3D when we want max pool can change #channels
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(), # 13
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(), # 22
        )

        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )
        
    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            # normalize f
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits
    
    def show_num_layer(self):
        for i, layer in enumerate(self.backbone.children()):
            print(f"{i}: {layer}")

def build_lprnet(lpr_max_len=8, phase=False, class_num=37, dropout_rate=0.5):

    lprnet = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return lprnet.train()
    else:
        return lprnet.eval()