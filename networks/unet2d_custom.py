import torch
import torch.nn as nn


# === Base blocks ===

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25)
        )

    def forward(self, x):
        return self.conv(x)


class unet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2,
                 first_stage_fsizes=(3, 3), normalization='linear'):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, nb_filter[0], kernel_size=first_stage_fsizes[0], padding=first_stage_fsizes[0] // 2),
            ConvBlock(nb_filter[0], nb_filter[0], kernel_size=first_stage_fsizes[1], padding=first_stage_fsizes[1] // 2)
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(nb_filter[0], nb_filter[1], kernel_size=3, padding=1),
            ConvBlock(nb_filter[1], nb_filter[1], kernel_size=3, padding=1)
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(nb_filter[1], nb_filter[2], kernel_size=3, padding=1),
            ConvBlock(nb_filter[2], nb_filter[2], kernel_size=3, padding=1)
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(nb_filter[2], nb_filter[3], kernel_size=3, padding=1),
            ConvBlock(nb_filter[3], nb_filter[3], kernel_size=3, padding=1)
        )

        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(nb_filter[3], nb_filter[4], kernel_size=3, padding=1),
            ConvBlock(nb_filter[4], nb_filter[4], kernel_size=3, padding=1)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)

        self.dec4 = nn.Sequential(
            ConvBlock(nb_filter[4], nb_filter[3], kernel_size=3, padding=1),
            ConvBlock(nb_filter[3], nb_filter[3], kernel_size=3, padding=1)
        )

        self.up3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.dec3 = nn.Sequential(
            ConvBlock(nb_filter[3], nb_filter[2], kernel_size=3, padding=1),
            ConvBlock(nb_filter[2], nb_filter[2], kernel_size=3, padding=1)
        )
        self.up2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(nb_filter[2], nb_filter[1], kernel_size=3, padding=1),
            ConvBlock(nb_filter[1], nb_filter[1], kernel_size=3, padding=1)
        )
        self.up1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(nb_filter[1], nb_filter[0], kernel_size=3, padding=1),
            ConvBlock(nb_filter[0], nb_filter[0], kernel_size=3, padding=1)
        )
        self.final_conv1 = ConvBlock(nb_filter[0], nb_filter[0])
        self.final_conv2 = ConvBlock(nb_filter[0] * 2, nb_filter[0])

        self.output_layer = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        f1 = self.final_conv1(d1)
        f2 = self.final_conv2(torch.cat([d1, f1], dim=1))

        return self.output_layer(f2)

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']
        decoder_layers = ['up4', 'dec4', 'up3', 'dec3', 'up2', 'dec2', 'up1', 'dec1']
        out_layers = ['final_conv1', 'final_conv2', 'output_layer']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}
        return module_dict

    @staticmethod
    def get_adam_base_LRs():
        BASE_LRS = {
            "enc1": 1e-4, "enc2": 1e-4, "enc3": 1e-4, "enc4": 1e-4, "bottleneck": 1e-4,
            "up4": 1e-4, "dec4": 1e-4, "up3": 1e-4, "dec3": 1e-4,
            "up2": 1e-4, "dec2": 1e-4, "up1": 1e-4, "dec1": 1e-4,
            "final_conv1": 1e-4, "final_conv2": 1e-4, "output_layer": 1e-4,
        }
        return BASE_LRS

    @staticmethod
    def get_SGD_base_LRs():
        BASE_LRS = {
            "enc1": 0.01, "enc2": 0.01, "enc3": 0.01, "enc4": 0.01, "bottleneck": 0.01,
            "up4": 0.01, "dec4": 0.01, "up3": 0.01, "dec3": 0.01,
            "up2": 0.01, "dec2": 0.01, "up1": 0.01, "dec1": 0.01,
            "final_conv1": 0.01, "final_conv2": 0.01, "output_layer": 0.01,
        }
        return BASE_LRS


class unet_wo_skip(nn.Module):
    def __init__(self, in_channels=1, num_classes=2,
                 first_stage_fsizes=(3, 3), normalization='linear'):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, nb_filter[0], kernel_size=first_stage_fsizes[0], padding=first_stage_fsizes[0] // 2),
            ConvBlock(nb_filter[0], nb_filter[0], kernel_size=first_stage_fsizes[1], padding=first_stage_fsizes[1] // 2)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(nb_filter[0], nb_filter[1], kernel_size=3, padding=1),
            ConvBlock(nb_filter[1], nb_filter[1], kernel_size=3, padding=1)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(nb_filter[1], nb_filter[2], kernel_size=3, padding=1),
            ConvBlock(nb_filter[2], nb_filter[2], kernel_size=3, padding=1)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(nb_filter[2], nb_filter[3], kernel_size=3, padding=1),
            ConvBlock(nb_filter[3], nb_filter[3], kernel_size=3, padding=1)
        )
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(nb_filter[3], nb_filter[4], kernel_size=3, padding=1),
            ConvBlock(nb_filter[4], nb_filter[4], kernel_size=3, padding=1)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], kernel_size=2, stride=2)

        self.dec4 = nn.Sequential(
            ConvBlock(nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
            ConvBlock(nb_filter[3], nb_filter[3], kernel_size=3, padding=1)
        )
        self.up3 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], kernel_size=2, stride=2)

        self.dec3 = nn.Sequential(
            ConvBlock(nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
            ConvBlock(nb_filter[2], nb_filter[2], kernel_size=3, padding=1)
        )
        self.up2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBlock(nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
            ConvBlock(nb_filter[1], nb_filter[1], kernel_size=3, padding=1)
        )
        self.up1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBlock(nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
            ConvBlock(nb_filter[0], nb_filter[0], kernel_size=3, padding=1)
        )
        self.final_conv1 = ConvBlock(nb_filter[0], nb_filter[0])
        self.final_conv2 = ConvBlock(nb_filter[0] * 2, nb_filter[0])

        self.output_layer = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        f1 = self.final_conv1(d1)
        f2 = self.final_conv2(torch.cat([d1, f1], dim=1))

        return self.output_layer(f2)

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']
        decoder_layers = ['up4', 'dec4', 'up3', 'dec3', 'up2', 'dec2', 'up1', 'dec1']
        out_layers = ['final_conv1', 'final_conv2', 'output_layer']
        module_dict = {'encoder': encoder_layers,
                       'decoder': decoder_layers,
                       'out': out_layers}

        return module_dict


class unet_dense(nn.Module):
    def __init__(self, in_channels=1, num_classes=2,
                 first_stage_fsizes=(3, 3), normalization="softmax"):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(in_channels, nb_filter[0], kernel_size=first_stage_fsizes[0], padding=first_stage_fsizes[0] // 2),
            ConvBlock(nb_filter[0], nb_filter[0], kernel_size=first_stage_fsizes[1], padding=first_stage_fsizes[1] // 2)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBlock(nb_filter[0], nb_filter[1], kernel_size=3, padding=1),
            ConvBlock(nb_filter[1], nb_filter[1], kernel_size=3, padding=1)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBlock(nb_filter[1], nb_filter[2], kernel_size=3, padding=1),
            ConvBlock(nb_filter[2], nb_filter[2], kernel_size=3, padding=1)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ConvBlock(nb_filter[2], nb_filter[3], kernel_size=3, padding=1),
            ConvBlock(nb_filter[3], nb_filter[3], kernel_size=3, padding=1)
        )
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(nb_filter[3], nb_filter[4], kernel_size=3, padding=1),
            ConvBlock(nb_filter[4], nb_filter[4], kernel_size=3, padding=1)
        )
        self.Avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(2048, num_classes))

        if normalization == 'sigmoid':
            # assert self.num_class == 1
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            assert num_classes > 1
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def forward(self, x):
        features = []
        e1 = self.enc1(x)
        features.append(e1)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        features.append(e2)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        features.append(e3)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        features.append(e4)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)
        features.append(b)

        b_pool = self.Avgpool(b)
        # x5_pool = self.maxpool(x5)
        pre_out = torch.flatten(b_pool, 1)
        # out = self.projector(x6)
        out = self.fc(pre_out)

        return out

    @staticmethod
    def get_module_dicts():
        encoder_layers = ['enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']
        fc_layers = ['fc']
        module_dict = {'encoder': encoder_layers,
                       'fc': fc_layers}

        return module_dict
