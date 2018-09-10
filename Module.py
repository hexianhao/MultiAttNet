import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttNet(nn.Module):

    def __init__(self):

        super(AttNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.AttConv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.AttForward1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.AttConv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.AttForward2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.dir = np.array([[1, -1], [1, 0], [1, 1],
                             [-1, -1], [-1, 0], [-1, 1],
                             [0, -1], [0, 1]], dtype=np.int32)

    def forward(self, image):

        output1 = self.layer1(image)
        output2 = self.layer2(output1)

        '''
        从第二层开始，每层的特征映射N*M*C
        将每层的特征映射视为N*M个C为特征
        并计算N*M个权重值作为attention
        '''
        output3 = self.layer3(output2)
        attconv1 = self.AttConv1(output3)
        features = attconv1.permute(0, 3, 1, 2).view(-1, 64)
        scores = self.AttForward1(features)
        Attention = F.softmax(scores.view(-1, attconv1.size(2) * attconv1.size(3)), dim=1)
        attfeat1 = self.AttFeat(output3, Attention)

        output4 = self.layer4(output3)
        attconv2 = self.AttConv2(output4)
        features = attconv2.permute(0, 3, 1, 2).view(-1, 64)
        scores = self.AttForward2(features)
        Attention = F.softmax(scores.view(-1, attconv2.size(2) * attconv2.size(3)), dim=1)
        attfeat2 = self.AttFeat(output4, Attention)

        attfeat = torch.cat([attfeat1, attfeat2], 1)


        return attfeat



    def AttFeat(self, feat_map, scores):

        AttFeature = torch.zeros([feat_map.size(0), feat_map.size(1)])

        for i in range(feat_map.size(2)):
            for j in range(feat_map.size(3)):

                feat = feat_map[:, :, i, j]

                for k in range(len(self.dir)):
                    dx = i + self.dir[k, 0]
                    dy = j + self.dir[k, 1]
                    if dx >= 0 and dx < feat_map.size(2) and \
                        dy >= 0 and dy < feat_map.size(3):
                        feat += feat_map[:, :, dx, dy]

                feat = torch.transpose(feat, 0, 1)
                AttFeature += torch.transpose(feat * scores[:, i, j], 0, 1)

        return AttFeature


def weight_init(m):

    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif isinstance(m, nn.Linear):
        n = m.in_features * m.out_features
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
