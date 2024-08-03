import torch
import torch.nn as nn


class ConvBA(nn.Module):
    # 这里CBA = conv + BN + action
    def __init__(self, c1, c2, k, s, p=0, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    def __init__(self, c, shortcut: bool):
        super(BottleNeck, self).__init__()
        self.conv1 = ConvBA(c, c, k=1, s=1)
        self.conv2 = ConvBA(c, c, k=3, s=1, p=1)

        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut is True:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, repeat, shortcut=True):
        super(C3, self).__init__()

        self.conv1 = ConvBA(c1, c2 // 2, k=1, s=1, p=0)
        self.conv2 = ConvBA(c1, c2 // 2, k=1, s=1, p=0)

        self.bottles = nn.Sequential(*[BottleNeck(c2 // 2, shortcut=shortcut) for _ in range(repeat)])

        self.conv3 = ConvBA(c2, c2, k=1, s=1, p=0)

    def forward(self, x):
        # x2 = self.conv2(x)
        # x1 = self.bottles(self.conv1(x))
        # return self.conv3(torch.cat((x1, x2), dim=1))
        return self.conv3(torch.cat([self.bottles(self.conv1(x)), self.conv2(x)], dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c_mid, c2):
        super(SPPF, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBA(c1, c_mid, k=1, s=1)
        self.conv2 = ConvBA(c_mid * 4, c2, k=1, s=1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class Backbone(nn.Module):
    def __init__(self, img_size=(600, 800), c1=3, c2=32, repeat=1):
        super(Backbone, self).__init__()
        self.img_size = img_size
        if img_size[0] == 600:
            self.focus = ConvBA(c1=c1, c2=c2, k=6, s=2, p=(22, 2))  # p1,图片的高从600padding至640
        else:
            self.focus = ConvBA(c1=c1, c2=c2, k=6, s=2, p=2)  # p1
        self.p2 = ConvBA(c1=c2, c2=c2 * 2, k=3, s=2, p=1)
        self.c3_1 = C3(c1=c2 * 2, c2=c2 * 2, repeat=repeat, shortcut=True)
        self.p3 = ConvBA(c1=c2 * 2, c2=c2 * 4, k=3, s=2, p=1)
        self.c3_2 = C3(c1=c2 * 4, c2=c2 * 4, repeat=repeat * 2, shortcut=True)
        self.p4 = ConvBA(c1=c2 * 4, c2=c2 * 8, k=3, s=2, p=1)
        self.c3_3 = C3(c1=c2 * 8, c2=c2 * 8, repeat=repeat * 3, shortcut=True)
        self.p5 = ConvBA(c1=c2 * 8, c2=c2 * 16, k=3, s=2, p=1)
        self.c3_4 = C3(c1=c2 * 16, c2=c2 * 16, repeat=repeat, shortcut=True)
        self.sppf = SPPF(c1=c2 * 16, c_mid=c2 * 8, c2=c2 * 16)

    def forward(self, x):
        y1 = self.c3_2(self.p3(self.c3_1(self.p2(self.focus(x)))))
        y2 = self.c3_3(self.p4(y1))
        y3 = self.sppf(self.c3_4(self.p5(y2)))

        return y1, y2, y3


class Neck(nn.Module):
    def __init__(self, c2=128):
        super(Neck, self).__init__()

        self.p6 = ConvBA(c1=c2 * 4, c2=c2 * 2, k=1, s=1)
        self.c3_5 = C3(c1=c2 * 4, c2=c2 * 2, repeat=1, shortcut=False)
        self.p7 = ConvBA(c1=c2 * 2, c2=c2, k=1, s=1)
        self.c3_6 = C3(c1=c2 * 2, c2=c2, repeat=1, shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2, x3):
        y3 = self.p6(x3)

        y2 = self.p7(self.c3_5(torch.cat((self.upsample(y3), x2), dim=1)))

        y1 = self.c3_6(torch.cat((self.upsample(y2), x1), dim=1))

        return y1, y2, y3


class Head(nn.Module):
    def __init__(self, c2=128, n_class=3):
        super(Head, self).__init__()

        self.p8 = ConvBA(c1=c2, c2=c2, k=3, s=2, p=1)
        self.c3_7 = C3(c1=c2 * 2, c2=c2 * 2, repeat=1, shortcut=False)
        self.p9 = ConvBA(c1=c2 * 2, c2=c2 * 2, k=3, s=2, p=1)
        self.c3_8 = C3(c1=c2 * 4, c2=c2 * 4, repeat=1, shortcut=False)

        self.head1 = nn.Conv2d(in_channels=c2, out_channels=3 * (5 + n_class), kernel_size=1, stride=1, padding=0)
        self.head2 = nn.Conv2d(in_channels=c2 * 2, out_channels=3 * (5 + n_class), kernel_size=1, stride=1, padding=0)
        self.head3 = nn.Conv2d(in_channels=c2 * 4, out_channels=3 * (5 + n_class), kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3):
        y1 = self.head1(x1)

        x2 = self.c3_7(torch.cat((self.p8(x1), x2), dim=1))

        y2 = self.head2(x2)

        y3 = self.head3(self.c3_8(torch.cat((self.p9(x2), x3), dim=1)))

        return y1, y2, y3


class YOLOv5s(nn.Module):
    def __init__(self, c1=3, img_size=(600, 800), channel=32, n_class=3):
        super(YOLOv5s, self).__init__()

        self.backbone = Backbone(c1=c1, img_size=img_size, c2=channel, repeat=1)
        self.neck = Neck(c2=channel * 4)
        self.head = Head(c2=channel * 4, n_class=n_class)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(*x)
        x = self.head(*x)
        return x


if __name__ == '__main__':
    x = torch.rand((7, 3, 512, 640), requires_grad=False)
    model = YOLOv5s(img_size=(512, 640), n_class=5)
    # model.load_state_dict(torch.load('./weight/YOLOv5s_epoch500.pth'))
    # print(model.state_dict())
    with torch.no_grad():
        y = model(x)
    for i in y:
        print(i.shape)
