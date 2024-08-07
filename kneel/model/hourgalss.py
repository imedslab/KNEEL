
from torch import nn
from typing import Union
from huggingface_hub import PyTorchModelHubMixin


from .modules import Hourglass, MultiScaleHGResidual, conv_block_3x3, SoftArgmax2D, HGResidual


class HourglassNet(nn.Module, PyTorchModelHubMixin, tags=["kneel", "arXiv:1907.12237"]):
    def __init__(self, n_inputs=1, n_outputs=6, bw=64, hg_depth=4,
                 upmode = "bilinear", multiscale_hg_block=False, se=False,
                 se_ratio=16, dropout=0.25, stage="global_search", fold=0, spacing=1., crop=450, pad=520):

        super(HourglassNet, self).__init__()
        self.stage = stage
        self.fold = fold
        self.spacing = spacing
        self.crop = crop
        self.pad = pad
        self.multiscale_hg_block = multiscale_hg_block
        self.se = se
        self.se_ratio = se_ratio

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_inputs, bw, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(bw),
            nn.ReLU(inplace=True),
            self.__make_hg_block(bw, bw * 2),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            self.__make_hg_block(bw * 2, bw * 2),
            self.__make_hg_block(bw * 2, bw * 2),
            self.__make_hg_block(bw * 2, bw * 4)
        )

        self.hourglass = Hourglass(hg_depth, bw * 4, bw * 4, bw * 8, upmode, multiscale_hg_block,
                                   se=se, se_ratio=se_ratio)

        self.mixer = nn.Sequential(nn.Dropout2d(p=dropout),
                                   conv_block_3x3(bw * 8, bw * 8),
                                   nn.Dropout2d(p=dropout),
                                   conv_block_3x3(bw * 8, bw * 4))

        self.out_block = nn.Sequential(nn.Conv2d(bw * 4, n_outputs, kernel_size=1, padding=0))
        self.sagm = SoftArgmax2D()

    def __make_hg_block(self, inp, out):
        if self.multiscale_hg_block:
            return MultiScaleHGResidual(inp, out, se=self.se, se_ratio=self.se_ratio)
        else:
            return HGResidual(inp, out, se=self.se, se_ratio=self.se_ratio)

    def forward(self, x):
        o_layer_1 = self.layer1(x)
        o_layer_2 = self.layer2(o_layer_1)

        o_hg = self.hourglass(o_layer_2)
        o_mixer = self.mixer(o_hg)
        out = self.out_block(o_mixer)

        return self.sagm(out)