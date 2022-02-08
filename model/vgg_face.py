import torch.nn as nn
from model.vgg_face_module import VGGBlock


class VGGFaceNet(nn.Module):
    def __init__(self, input_channel=3, num_of_classes=1000,
                 blocks=[2, 2, 3, 3, 3], dropout=0.5,
                 init=nn.init.kaiming_uniform_, bn=True):
        super(VGGFaceNet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv0 = VGGBlock(3, 64, 3, blocks[0], init=init, bn=bn)
        self.conv1 = VGGBlock(64, 128, 3, blocks[1], init=init, bn=bn)
        self.conv2 = VGGBlock(128, 256, 3, blocks[2], init=init, bn=bn)
        self.conv3 = VGGBlock(256, 512, 3, blocks[3], init=init, bn=bn)
        self.conv4 = VGGBlock(512, 512, 3, blocks[4], init=init, bn=bn)
        self.fc_conv1 = nn.Conv2d(512, 4096, 7)
        init(self.fc_conv1.weight)
        self.fc_conv2 = nn.Conv2d(4096, 4096, 1)
        init(self.fc_conv1.weight)
        self.fc_conv3 = nn.Conv2d(4096, num_of_classes, 1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        output = x
        for block in [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4]:
            output = self.pool(block(output))

        output = self.dropout(self.fc_conv1(output))
        output = self.dropout(self.fc_conv2(output))
        output = self.fc_conv3(output)
        return self.flatten(output)
