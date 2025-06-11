# style_transfer/model.py
import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Define architecture
        def conv_layer(in_c, out_c, kernel, stride):
            return nn.Sequential(
                nn.ReflectionPad2d(kernel // 2),
                nn.Conv2d(in_c, out_c, kernel, stride),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU()
            )
        def residual_block(channels):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3),
                nn.InstanceNorm2d(channels, affine=True),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3),
                nn.InstanceNorm2d(channels, affine=True),
            )
        def upsample(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_c, out_c, 3),
                nn.InstanceNorm2d(out_c, affine=True),
                nn.ReLU()
            )
        self.model = nn.Sequential(
            conv_layer(3, 32, 9, 1),
            conv_layer(32, 64, 3, 2),
            conv_layer(64, 128, 3, 2),
            residual_block(128),
            residual_block(128),
            residual_block(128),
            upsample(128, 64),
            upsample(64, 32),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9),
        )

    def forward(self, x):
        return self.model(x)
