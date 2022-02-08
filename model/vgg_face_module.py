from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ksize, blocks,
                 init=nn.init.kaiming_uniform_, bn=True):
        super(VGGBlock, self).__init__()
        conv_blocks = []
        for _ in range(blocks):
            if in_channel:
                conv_layer = nn.Conv2d(in_channel, out_channel, ksize, padding=ksize//2)
                in_channel = 0  # disable in_channel
            else:
                conv_layer = nn.Conv2d(out_channel, out_channel, ksize, padding=ksize//2)
            init(conv_layer.weight)
            conv_blocks.append(conv_layer)
            if bn:
                conv_blocks.append(nn.BatchNorm2d(out_channel))
            conv_blocks.append(nn.ReLU())
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        return self.conv_blocks(x)
