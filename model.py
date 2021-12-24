import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Generator(nn.Module):
    def __init__(self, input_size=4096, n_out_channel=3):

        super(Generator, self).__init__()

        self.defc7 = nn.Linear(input_size, 4096)
        self.relu_defc7 = nn.LeakyReLU(0.3,inplace=True)

        self.defc6 = nn.Linear(4096, 4096)
        self.relu_defc6 = nn.LeakyReLU(0.3,inplace=True)

        self.defc5 = nn.Linear(4096, 4096)
        self.relu_defc5 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.deconv5.weight, a=0.3)
        self.relu_deconv5 = nn.LeakyReLU(0.3,inplace=True)

        self.conv5_1 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.conv5_1.weight, a=0.3)
        self.relu_conv5_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.deconv4.weight, a=0.3)
        self.relu_deconv4 = nn.LeakyReLU(0.3,inplace=True)

        self.conv4_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.conv4_1.weight, a=0.3)
        self.relu_conv4_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.deconv3.weight, a=0.3)
        self.relu_deconv3 = nn.LeakyReLU(0.3,inplace=True)

        self.conv3_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.conv3_1.weight, a=0.3)
        self.relu_conv3_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.deconv2.weight, a=0.3)
        self.relu_deconv2 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.deconv1.weight, a=0.3)
        self.relu_deconv1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv0 = nn.ConvTranspose2d(32, n_out_channel, kernel_size=4, stride=2, padding=1, bias=True)
        #torch.nn.init.kaiming_normal_(self.deconv0.weight, a=0.3)

        self.defc = nn.Sequential(
            self.defc7,
            self.relu_defc7,
            self.defc6,
            self.relu_defc6,
            self.defc5,
            self.relu_defc5,
        )

        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.deconv1,
            self.relu_deconv1,
            self.deconv0,
        )

    def forward(self, z):

        f = self.defc(z)
        f = f.view(-1, 256, 4, 4)
        g = self.deconv(f)
        g = transforms.functional.center_crop(g, (227,227))

        return g

class Comparator(nn.Module):
    def __init__(self, encoder, target_layer):
        super(Comparator, self).__init__()
        features = list(encoder.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)

    def forward(self, x):
        if self.target_layer == -1:
            return encoder(x)

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.target_layer:
                return x
            
class Comparator_v2(nn.Module):
    def __init__(self, encoder):
        super(Comparator_v2, self).__init__()
        features = list(encoder.features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x, target_layer):
        if int(target_layer) == -1:
            return encoder(x)

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == int(target_layer):
                return x
            
class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=7, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, std):
        noise = torch.randn(x.size(), requires_grad=False, device=x.device) * std
        x2 = x + noise

        x2 = self.features1(x2)
        x2 = self.avgpool(x2)
        h = torch.flatten(x2, 1)
        h = self.classifier(h)
        return h
