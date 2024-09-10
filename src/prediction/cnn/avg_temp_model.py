import torch.nn as nn


class AvgTempModel(nn.Module):
    def __init__(self):
        super(AvgTempModel, self).__init__()

        # Formula :: NewImageSize = (PrevImageSize - KernelSize + 2PaddingSize) / Stride + 1

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # @Layer1 = 1x200x200 -> 32x200x200
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer2 = 32x200x200 -> 32x100x100
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # @Layer3 = 32x100x100 -> 64x100x100
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer4 = 64x100x100 -> 64x50x50
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            # @Layer5 = 64x50x50 -> 128x24x24
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer6 = 128x24x24 -> 128x12x12
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # @Layer7 = 128x12x12 -> 256x12x12
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer8 = 256x12x12 -> 256x6x6
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc_layer(x)

        return x
