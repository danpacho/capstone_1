import torch.nn as nn


class DragModel(nn.Module):
    def __init__(self):
        super(DragModel, self).__init__()

        # Formula :: NewImageSize = (PrevImageSize - KernelSize + 2PaddingSize) / Stride + 1

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            # @Layer1 = 1x200x200 -> 16x200x200
            nn.ReLU(),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # @Layer2 = 16x200x200 -> 32x200x200
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer3 = 32x200x200 -> 32x100x100
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # @Layer4 = 32x100x100 -> 64x100x100
            nn.ReLU(),
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # @Layer6 = 64x100x100 -> 128x100x100
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer7 = 128x100x100 -> 128x50x50
        )

        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # @Layer8 = 128x50x50 -> 256x50x50
            nn.ReLU(),
            # @Layer9 = 256x50x50 -> 256x50x50
        )

        self.conv_layer_6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # @Layer10 = 256x50x50 -> 512x50x50
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # @Layer11 = 512x50x50 -> 512x25x25
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 25 * 25, 100),
            nn.ReLU(),
            nn.Linear(100, 7),
            nn.ReLU(),
            nn.Linear(7, 1),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        x = self.conv_layer_6(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc_layer(x)

        return x
